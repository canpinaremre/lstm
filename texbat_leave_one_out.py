#!/usr/bin/env python3
"""Leave-one-dataset-out TEXBAT spoof detection workflow.

For each spoofed dataset ds2_spoof through ds8_spoof:
  - train on a cleanDynamic + cleanStatic genuine split
  - train on all other ds*_spoof datasets as spoof samples
  - test on the excluded spoof dataset plus held-out clean samples

The default epoch count is intentionally short for script validation.
Increase --epochs for final experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from scipy.io import loadmat

from lstm import (
    Config,
    build_all_windows,
    build_model,
    build_tf_datasets,
    configure_accelerator,
    load_and_validate,
    standardize_apply,
    standardize_fit,
)


SPOOF_IDS = tuple(range(2, 9))
CLEAN_NAMES = ("cleanDynamic", "cleanStatic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TEXBAT ds2-ds8 spoof leave-one-out LSTM experiments."
    )
    parser.add_argument("--texbat-dir", default="TEXBAT", help="Folder containing TEXBAT .ubx.mat files.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "texbat_leave_one_out"),
        help="Folder for fold outputs and summary files.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Short default for smoke testing.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=580)
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction from train datasets.")
    parser.add_argument(
        "--spoof-ratio",
        type=float,
        default=0.50,
        help="Spoof fraction after downsampling train and evaluation data.",
    )
    parser.add_argument(
        "--eval-clean-frac",
        type=float,
        default=0.20,
        help="Fraction of clean windows reserved for balanced evaluation in each fold.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Prediction threshold used for confusion metrics.",
    )
    parser.add_argument(
        "--max-windows-per-dataset",
        type=int,
        default=0,
        help="Optional cap per loaded dataset for quick local checks. 0 means no cap.",
    )
    parser.add_argument(
        "--no-save-models",
        action="store_true",
        help="Skip saving per-fold .keras models and weights.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write the LaTeX table but skip pdflatex PDF compilation.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    return replace(
        Config(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        window=args.window,
        stride=args.stride,
        seed=args.seed,
        val_frac=args.val_frac,
        spoof_ratio=args.spoof_ratio,
    )


def mat_path(texbat_dir: Path, stem: str) -> Path:
    path = texbat_dir / f"{stem}.ubx.mat"
    if not path.exists():
        raise FileNotFoundError(f"Missing TEXBAT MAT file: {path}")
    return path


def limit_windows(X: np.ndarray, max_windows: int, seed: int) -> np.ndarray:
    if max_windows <= 0 or X.shape[0] <= max_windows:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_windows, replace=False)
    return X[idx]


def load_dataset_windows(path: Path, label: int, cfg: Config, max_windows: int = 0) -> np.ndarray:
    mat = loadmat(str(path))
    features, sat_count, _, _, sv_matrix = load_and_validate(mat, path.stem, cfg.feature_keys)
    X, _ = build_all_windows(features, sat_count, sv_matrix, label=label, cfg=cfg)
    X = limit_windows(X, max_windows=max_windows, seed=cfg.seed + label + len(path.name))
    print(f"{path.name}: windows={X.shape[0]} label={label}")
    return X


def load_all_windows(texbat_dir: Path, cfg: Config, max_windows: int) -> Tuple[Dict[str, np.ndarray], Dict[int, np.ndarray]]:
    clean = {
        name: load_dataset_windows(mat_path(texbat_dir, name), label=0, cfg=cfg, max_windows=max_windows)
        for name in CLEAN_NAMES
    }
    spoof = {
        ds_id: load_dataset_windows(
            mat_path(texbat_dir, f"ds{ds_id}_spoof"),
            label=1,
            cfg=cfg,
            max_windows=max_windows,
        )
        for ds_id in SPOOF_IDS
    }
    return clean, spoof


def concat_nonempty(arrays: Iterable[np.ndarray], name: str) -> np.ndarray:
    kept = [a for a in arrays if a.shape[0] > 0]
    if not kept:
        raise RuntimeError(f"No windows available for {name}.")
    return np.concatenate(kept, axis=0)


def downsample_classes(
    X_spoof: np.ndarray,
    X_clean: np.ndarray,
    spoof_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < spoof_ratio < 1.0):
        raise ValueError("--spoof-ratio must be between 0 and 1.")

    n_spoof = X_spoof.shape[0]
    n_clean = X_clean.shape[0]
    total = min(int(np.floor(n_spoof / spoof_ratio)), int(np.floor(n_clean / (1.0 - spoof_ratio))))
    use_spoof = int(np.floor(total * spoof_ratio))
    use_clean = total - use_spoof

    if use_spoof <= 1 or use_clean <= 1:
        raise RuntimeError(f"Not enough windows after balancing: spoof={use_spoof}, clean={use_clean}")

    rng = np.random.default_rng(seed)
    spoof_idx = rng.choice(n_spoof, size=use_spoof, replace=False)
    clean_idx = rng.choice(n_clean, size=use_clean, replace=False)
    X = np.concatenate([X_spoof[spoof_idx], X_clean[clean_idx]], axis=0)
    y = np.concatenate(
        [np.ones((use_spoof,), dtype=np.int32), np.zeros((use_clean,), dtype=np.int32)],
        axis=0,
    )
    order = rng.permutation(y.shape[0])
    return X[order], y[order]


def split_clean_for_fold(
    X_clean: np.ndarray,
    eval_clean_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < eval_clean_frac < 1.0):
        raise ValueError("--eval-clean-frac must be between 0 and 1.")
    if X_clean.shape[0] < 4:
        raise RuntimeError(f"Not enough clean windows to split: clean={X_clean.shape[0]}")

    rng = np.random.default_rng(seed)
    idx = np.arange(X_clean.shape[0])
    rng.shuffle(idx)
    n_eval = max(2, int(np.floor(X_clean.shape[0] * eval_clean_frac)))
    n_eval = min(n_eval, X_clean.shape[0] - 2)
    eval_idx = idx[:n_eval]
    train_idx = idx[n_eval:]
    return X_clean[train_idx], X_clean[eval_idx]


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < val_frac < 1.0):
        raise ValueError("--val-frac must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = max(1, int(np.floor(idx.shape[0] * val_frac)))
        n_val = min(n_val, idx.shape[0] - 1)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(np.mean(y_pred == y_true)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_probability": float(np.mean(y_prob)),
    }


def json_safe(row: Dict[str, object]) -> Dict[str, object]:
    safe: Dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, float) and not np.isfinite(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.2f}\\%"
    except (TypeError, ValueError):
        return "--"


def write_latex_table(path: Path, rows: List[Dict[str, object]], cfg: Config, threshold: float) -> None:
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{siunitx}",
        r"\usepackage{caption}",
        r"\begin{document}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{TEXBAT leave-one-out results with balanced training and evaluation sets.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Held-out & Eval N & Accuracy & AUC & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["excluded_dataset"]),
                    str(row["test_windows_after_balance"]),
                    pct(row["test_accuracy"]),
                    f"{float(row['test_auc']):.4f}",
                    pct(row["test_precision"]),
                    pct(row["test_recall_detection_rate"]),
                    pct(row["test_f1"]),
                ]
            )
            + r" \\"
        )

    mean_accuracy = np.mean([float(row["test_accuracy"]) for row in rows])
    mean_auc = np.mean([float(row["test_auc"]) for row in rows])
    mean_precision = np.mean([float(row["test_precision"]) for row in rows])
    mean_recall = np.mean([float(row["test_recall_detection_rate"]) for row in rows])
    mean_f1 = np.mean([float(row["test_f1"]) for row in rows])
    lines.extend(
        [
            r"\midrule",
            " & ".join(
                [
                    r"\textbf{Mean}",
                    "--",
                    pct(mean_accuracy),
                    f"{mean_auc:.4f}",
                    pct(mean_precision),
                    pct(mean_recall),
                    pct(mean_f1),
                ]
            )
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption*{Training spoof ratio = "
                f"{cfg.spoof_ratio:.2f}; evaluation spoof ratio = {cfg.spoof_ratio:.2f}; "
                f"threshold = {threshold:.2f}; epochs = {cfg.epochs}."
                r"}"
            ),
            r"\end{table}",
            r"\end{document}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def compile_latex_pdf(tex_path: Path) -> bool:
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; wrote LaTeX table but skipped PDF compilation.")
        return False

    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print("pdflatex failed; see output below.")
        print(result.stdout)
        return False
    return True


def run_fold(
    excluded_ds: int,
    clean_windows: Dict[str, np.ndarray],
    spoof_windows: Dict[int, np.ndarray],
    cfg: Config,
    output_dir: Path,
    threshold: float,
    eval_clean_frac: float,
    save_models: bool,
) -> Dict[str, object]:
    print(f"\n=== Fold: leave out ds{excluded_ds}_spoof ===")
    fold_dir = output_dir / f"leave_out_ds{excluded_ds}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    X_clean_all = concat_nonempty(clean_windows.values(), "clean datasets")
    X_clean_train, X_clean_eval = split_clean_for_fold(
        X_clean_all,
        eval_clean_frac=eval_clean_frac,
        seed=cfg.seed + 500 + excluded_ds,
    )
    X_spoof_train = concat_nonempty(
        (spoof_windows[ds_id] for ds_id in SPOOF_IDS if ds_id != excluded_ds),
        f"spoof train datasets excluding ds{excluded_ds}",
    )
    X_spoof_eval = spoof_windows[excluded_ds]
    if X_spoof_eval.shape[0] == 0:
        raise RuntimeError(f"No windows available for excluded ds{excluded_ds}_spoof.")

    X_balanced, y_balanced = downsample_classes(
        X_spoof_train,
        X_clean_train,
        spoof_ratio=cfg.spoof_ratio,
        seed=cfg.seed + excluded_ds,
    )
    X_test, y_test = downsample_classes(
        X_spoof_eval,
        X_clean_eval,
        spoof_ratio=cfg.spoof_ratio,
        seed=cfg.seed + 1000 + excluded_ds,
    )
    X_train, y_train, X_val, y_val = split_train_val(
        X_balanced,
        y_balanced,
        val_frac=cfg.val_frac,
        seed=cfg.seed + 100 + excluded_ds,
    )

    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_val = standardize_apply(X_val, mean, std)
    X_test = standardize_apply(X_test, mean, std)

    train_ds, val_ds, test_ds = build_tf_datasets(X_train, y_train, X_val, y_val, X_test, y_test, cfg)
    model = build_model(cfg, num_features=len(cfg.feature_keys))
    ckpt_path = fold_dir / "best_weights.weights.h5"
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor=cfg.monitor_metric,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=[ckpt],
        verbose=2,
    )
    if ckpt_path.exists():
        model.load_weights(str(ckpt_path))

    val_eval = model.evaluate(val_ds, verbose=0, return_dict=True)
    test_eval = model.evaluate(test_ds, verbose=0, return_dict=True)
    test_prob = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    test_metrics = binary_metrics(y_test, test_prob, threshold=threshold)

    np.savez(
        fold_dir / "norm_params.npz",
        mean=mean,
        std=std,
        window=np.array([cfg.window], dtype=np.int32),
        feature_keys=np.array(cfg.feature_keys),
    )
    np.savez(fold_dir / "train_history.npz", **{k: np.array(v) for k, v in history.history.items()})
    if save_models:
        model.save(str(fold_dir / "model.keras"))
        model.save_weights(str(fold_dir / "model_weights.weights.h5"))

    row: Dict[str, object] = {
        "excluded_dataset": f"ds{excluded_ds}_spoof",
        "train_spoof_windows": int(X_spoof_train.shape[0]),
        "train_clean_windows": int(X_clean_train.shape[0]),
        "train_windows_after_balance": int(X_train.shape[0]),
        "val_windows": int(X_val.shape[0]),
        "eval_clean_windows": int(X_clean_eval.shape[0]),
        "eval_spoof_windows": int(X_spoof_eval.shape[0]),
        "test_windows_after_balance": int(y_test.shape[0]),
        "test_spoof_windows_after_balance": int(np.sum(y_test == 1)),
        "test_clean_windows_after_balance": int(np.sum(y_test == 0)),
        "val_loss": float(val_eval.get("loss", np.nan)),
        "val_accuracy": float(val_eval.get("acc", np.nan)),
        "val_auc": float(val_eval.get("auc", np.nan)),
        "test_loss": float(test_eval.get("loss", np.nan)),
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": float(test_eval.get("auc", np.nan)),
        "test_precision": test_metrics["precision"],
        "test_recall_detection_rate": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_mean_probability": test_metrics["mean_probability"],
        "test_tp": test_metrics["tp"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
    }
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(row), f, indent=2, allow_nan=False)

    print(
        f"ds{excluded_ds}_spoof accuracy={row['test_accuracy']:.4f} "
        f"recall="
        f"{row['test_recall_detection_rate']:.4f} "
        f"mean_prob={row['test_mean_probability']:.4f} "
        f"TP={row['test_tp']} TN={row['test_tn']} "
        f"FP={row['test_fp']} FN={row['test_fn']}"
    )
    return row


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    texbat_dir = Path(args.texbat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    configure_accelerator()

    clean_windows, spoof_windows = load_all_windows(texbat_dir, cfg, args.max_windows_per_dataset)
    rows = [
        run_fold(
            excluded_ds=ds_id,
            clean_windows=clean_windows,
            spoof_windows=spoof_windows,
            cfg=cfg,
            output_dir=output_dir,
            threshold=args.threshold,
            eval_clean_frac=args.eval_clean_frac,
            save_models=not args.no_save_models,
        )
        for ds_id in SPOOF_IDS
    ]

    summary_csv = output_dir / "summary.csv"
    write_summary_csv(summary_csv, rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump([json_safe(row) for row in rows], f, indent=2, allow_nan=False)

    summary_tex = output_dir / "summary_table.tex"
    write_latex_table(summary_tex, rows, cfg=cfg, threshold=args.threshold)
    pdf_path = summary_tex.with_suffix(".pdf")
    pdf_ok = False
    if not args.skip_pdf:
        pdf_ok = compile_latex_pdf(summary_tex)

    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved LaTeX table: {summary_tex}")
    if pdf_ok:
        print(f"Saved PDF table: {pdf_path}")
    print("Leave-one-out run complete.")


if __name__ == "__main__":
    main()
