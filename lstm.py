from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # Allows --help without the training environment.
    np = None

try:
    import tensorflow as tf
except ModuleNotFoundError:  # Allows --help without the training environment.
    tf = None

try:
    from scipy.io import loadmat
except ModuleNotFoundError:  # Allows --help without the training environment.
    loadmat = None

# Optional GUI file picker (desktop). Used when CLI paths are not provided.
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:  # pragma: no cover
    tk = None
    filedialog = None


@dataclass(frozen=True)
class Config:
    # Feature names to read from the MAT files (each is sat_count x time_len)
    feature_keys: Tuple[str, ...] = (
        "cp_stdev",
        "cp_mes_delta",
        "do_stdev",
        "do_mes_delta",
        "pr_stdev",
        # "pr_mes_delta",
    )

    # Sliding-window parameters
    window: int = 20
    stride: int = 1

    # Random split ratios AFTER building all windows
    train_frac: float = 0.70
    val_frac: float = 0.15  # test = 1 - train_frac - val_frac

    # Class ratio in the final dataset (downsample only, no oversampling)
    spoof_ratio: float = 0.50

    # Training parameters
    batch_size: int = 32
    epochs: int = 300
    seed: int = 580
    shuffle_buffer: int = 500

    # Model parameters
    lstm_units: int = 64
    dense_units: int = 32
    dropout: float = 0.0
    learning_rate: float = 1e-3
    l2_regularization: float = 1e-4

    # Train all epochs, but keep best weights using ModelCheckpoint
    monitor_metric: str = "val_auc"  # or "val_prauc"

    # Report metrics at this threshold and optionally tune another on validation.
    prediction_threshold: float = 0.50
    tune_f1_threshold: bool = True

    # Small normalized subsets saved for the optional SHAP analysis script.
    shap_background_size: int = 100
    shap_explain_size: int = 100

    # Window validity rules (based on sv_id)
    drop_if_sv_contains_zero: bool = True
    require_prev_sample_valid: bool = True


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Train and evaluate an LSTM GNSS spoof detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("spoofed_file", nargs="?", help="Spoofed MAT file.")
    parser.add_argument("genuine_file", nargs="?", help="Genuine MAT file.")

    data = parser.add_argument_group("windowing and data")
    data.add_argument("--features", nargs="+", default=list(defaults.feature_keys))
    data.add_argument("--window", type=positive_int, default=defaults.window)
    data.add_argument("--stride", type=positive_int, default=defaults.stride)
    data.add_argument("--train-frac", type=float, default=defaults.train_frac)
    data.add_argument("--val-frac", type=float, default=defaults.val_frac)
    data.add_argument("--spoof-ratio", type=float, default=defaults.spoof_ratio)
    data.add_argument(
        "--allow-sv-zero",
        action="store_true",
        help="Keep windows containing sv_id == 0.",
    )
    data.add_argument(
        "--allow-invalid-previous-sample",
        action="store_true",
        help="Do not require a valid sample immediately before each window.",
    )

    model = parser.add_argument_group("model and training")
    model.add_argument("--lstm-units", type=positive_int, default=defaults.lstm_units)
    model.add_argument("--dense-units", type=positive_int, default=defaults.dense_units)
    model.add_argument("--dropout", type=float, default=defaults.dropout)
    model.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    model.add_argument("--l2-regularization", type=float, default=defaults.l2_regularization)
    model.add_argument("--batch-size", type=positive_int, default=defaults.batch_size)
    model.add_argument("--epochs", type=positive_int, default=defaults.epochs)
    model.add_argument("--seed", type=int, default=defaults.seed)
    model.add_argument("--shuffle-buffer", type=positive_int, default=defaults.shuffle_buffer)
    model.add_argument(
        "--monitor-metric",
        choices=("val_auc", "val_prauc", "val_acc"),
        default=defaults.monitor_metric,
    )

    evaluation = parser.add_argument_group("evaluation and explanation")
    evaluation.add_argument(
        "--prediction-threshold",
        type=float,
        default=defaults.prediction_threshold,
        help="Fixed threshold reported alongside the validation-tuned threshold.",
    )
    evaluation.add_argument(
        "--no-tune-f1-threshold",
        action="store_true",
        help="Use prediction-threshold instead of selecting a threshold on validation F1.",
    )
    evaluation.add_argument(
        "--shap-background-size",
        type=nonnegative_int,
        default=defaults.shap_background_size,
    )
    evaluation.add_argument(
        "--shap-explain-size",
        type=nonnegative_int,
        default=defaults.shap_explain_size,
    )
    evaluation.add_argument(
        "--no-shap-samples",
        action="store_true",
        help="Do not save the small normalized sample bundle used by explain_shap.py.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", default="outputs", help="Parent directory for this run.")
    output.add_argument("--run-name", help="Run folder name; timestamp when omitted.")
    output.add_argument("--no-keras", action="store_true", help="Skip the final .keras model export.")
    output.add_argument("--no-tflite", action="store_true", help="Skip TensorFlow Lite export.")
    output.add_argument("--no-saved-model", action="store_true", help="Skip SavedModel export.")

    args = parser.parse_args(argv)
    if bool(args.spoofed_file) != bool(args.genuine_file):
        parser.error("provide both spoofed_file and genuine_file, or neither to use the file picker")
    if not (0.0 < args.train_frac < 1.0):
        parser.error("--train-frac must be between 0 and 1")
    if not (0.0 <= args.val_frac < 1.0):
        parser.error("--val-frac must be between 0 and 1")
    if args.train_frac + args.val_frac >= 1.0:
        parser.error("--train-frac + --val-frac must be less than 1")
    if not (0.0 < args.spoof_ratio < 1.0):
        parser.error("--spoof-ratio must be between 0 and 1")
    if not (0.0 <= args.dropout < 1.0):
        parser.error("--dropout must be at least 0 and less than 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.l2_regularization < 0.0:
        parser.error("--l2-regularization must be non-negative")
    if not (0.0 <= args.prediction_threshold <= 1.0):
        parser.error("--prediction-threshold must be between 0 and 1")
    return args


def config_from_args(args: argparse.Namespace) -> Config:
    shap_background_size = 0 if args.no_shap_samples else args.shap_background_size
    shap_explain_size = 0 if args.no_shap_samples else args.shap_explain_size
    return Config(
        feature_keys=tuple(args.features),
        window=args.window,
        stride=args.stride,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        spoof_ratio=args.spoof_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        lstm_units=args.lstm_units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        monitor_metric=args.monitor_metric,
        prediction_threshold=args.prediction_threshold,
        tune_f1_threshold=not args.no_tune_f1_threshold,
        shap_background_size=shap_background_size,
        shap_explain_size=shap_explain_size,
        drop_if_sv_contains_zero=not args.allow_sv_zero,
        require_prev_sample_valid=not args.allow_invalid_previous_sample,
    )


def require_training_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if tf is None:
        missing.append("tensorflow")
    if loadmat is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "Missing training dependencies: " + ", ".join(missing) + ". "
            "Run this script from the Python environment used for model training."
        )


# -----------------------------
# Helpers
# -----------------------------

class Tee:
    """Write terminal output to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _to_2d(a) -> np.ndarray:
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {a.shape}")
    return a


def _mode_ignore_zeros(row: np.ndarray) -> int:
    row = np.asarray(row).astype(np.int32)
    row = row[row != 0]
    if row.size == 0:
        return 0
    vals, cnts = np.unique(row, return_counts=True)
    return int(vals[np.argmax(cnts)])


def extract_sv_per_sat(sv_id_2d: np.ndarray) -> np.ndarray:
    """Return one PRN per satellite row using mode ignoring zeros."""
    sv = _to_2d(sv_id_2d)
    return np.array([_mode_ignore_zeros(sv[i, :]) for i in range(sv.shape[0])], dtype=np.int32)


def standardize_fit(X_train: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def make_windows_for_series(
    series_TF: np.ndarray,
    label: int,
    window: int,
    stride: int,
    sv_row: Optional[np.ndarray],
    drop_if_sv_contains_zero: bool,
    require_prev_sample_valid: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_TF: (time_len, F)
    sv_row: (time_len,) int sv_id values for this satellite row

    Window rules:
      - If drop_if_sv_contains_zero: discard windows containing any sv_id == 0.
      - If require_prev_sample_valid: skip windows where s==0 or sv_row[s-1]==0.

    Returns:
      X: (N, window, F) float32
      y: (N,) int32
    """
    time_len, F = series_TF.shape
    if time_len < window:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X_list: List[np.ndarray] = []
    last_start = time_len - window

    for s in range(0, last_start + 1, stride):
        e = s + window

        if sv_row is not None and drop_if_sv_contains_zero:
            if require_prev_sample_valid:
                if s == 0:
                    continue
                if sv_row[s - 1] == 0:
                    continue
            if np.any(sv_row[s:e] == 0):
                continue

        X_list.append(series_TF[s:e, :])

    if not X_list:
        return np.zeros((0, window, F), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    X = np.stack(X_list).astype(np.float32)
    y = np.full((X.shape[0],), int(label), dtype=np.int32)
    return X, y


def pick_paths_from_user(
    spoofed_path: Optional[str] = None,
    genuine_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Use supplied CLI paths when provided, otherwise try the GUI picker."""
    if spoofed_path is not None and genuine_path is not None:
        return spoofed_path, genuine_path

    if tk is not None and filedialog is not None:
        try:
            root = tk.Tk()
            root.withdraw()

            spoofed_path = filedialog.askopenfilename(
                title="Select SPOOFED dataset (.mat file)",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
            )
            if not spoofed_path:
                raise RuntimeError("No spoofed file selected.")

            genuine_path = filedialog.askopenfilename(
                title="Select GENUINE dataset (.mat file)",
                filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
            )
            if not genuine_path:
                raise RuntimeError("No genuine file selected.")

            return spoofed_path, genuine_path
        except Exception:
            pass

    raise RuntimeError("Provide spoofed and genuine MAT files on the command line.")


def load_and_validate(
    mat_data: dict,
    dataset_name: str,
    feature_keys: Tuple[str, ...],
) -> Tuple[List[np.ndarray], int, int, np.ndarray, np.ndarray]:
    """Load features from a MAT dict and validate shapes."""
    feat0 = _to_2d(mat_data[feature_keys[0]]).astype(np.float32)
    sat_count, time_len = feat0.shape

    for k in feature_keys:
        a = _to_2d(mat_data[k])
        if a.shape != (sat_count, time_len):
            raise ValueError(f"{dataset_name}: {k} shape {a.shape} != ({sat_count},{time_len})")

    sv_id = _to_2d(mat_data["sv_id"])
    if sv_id.shape != (sat_count, time_len):
        raise ValueError(f"{dataset_name}: sv_id shape {sv_id.shape} != ({sat_count},{time_len})")

    features: List[np.ndarray] = []
    for k in feature_keys:
        a = _to_2d(mat_data[k]).astype(np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(a)

    sv_per_sat = extract_sv_per_sat(sv_id)
    sv_matrix = sv_id.astype(np.int32)

    print(f"{dataset_name} - sat_count: {sat_count}, time_len: {time_len}")
    print(f"{dataset_name} - sv_per_sat: {sv_per_sat.tolist()}")

    return features, sat_count, time_len, sv_per_sat, sv_matrix


def build_all_windows(
    features: List[np.ndarray],
    sat_count: int,
    sv_matrix: np.ndarray,
    label: int,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windows for all satellite rows and concatenate."""
    F = len(features)
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for i in range(sat_count):
        series = np.stack([features[f_idx][i, :] for f_idx in range(F)], axis=-1)  # (time_len, F)
        sv_row = sv_matrix[i, :]

        X, y = make_windows_for_series(
            series_TF=series,
            label=label,
            window=cfg.window,
            stride=cfg.stride,
            sv_row=sv_row,
            drop_if_sv_contains_zero=cfg.drop_if_sv_contains_zero,
            require_prev_sample_valid=cfg.require_prev_sample_valid,
        )

        if X.shape[0] > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        return (
            np.zeros((0, cfg.window, F), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


def stratified_sample_and_split(
    X_spoof: np.ndarray,
    X_genuine: np.ndarray,
    cfg: Config,
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - Enforce cfg.spoof_ratio by downsampling (no oversampling).
    - Random split AFTER windowing (no time split), stratified by class.

    NOTE: This can introduce temporal leakage across splits if adjacent windows
    are highly correlated. You requested this behavior for experimentation.
    """
    rng = np.random.default_rng(cfg.seed)

    n_s = int(X_spoof.shape[0])
    n_g = int(X_genuine.shape[0])

    if n_s == 0 or n_g == 0:
        raise RuntimeError(f"Insufficient windows. spoof={n_s}, genuine={n_g}")

    r = float(cfg.spoof_ratio)
    if not (0.0 < r < 1.0):
        raise ValueError("spoof_ratio must be between 0 and 1 (exclusive).")

    train_frac = float(cfg.train_frac)
    val_frac = float(cfg.val_frac)
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    # Max achievable total size without replacement for desired ratio
    total_max_from_s = int(np.floor(n_s / r))
    total_max_from_g = int(np.floor(n_g / (1.0 - r)))
    total = min(total_max_from_s, total_max_from_g)
    if total <= 0:
        raise RuntimeError("Cannot satisfy spoof_ratio with available windows.")

    n_s_des = int(np.floor(total * r))
    n_g_des = total - n_s_des
    if n_s_des <= 0 or n_g_des <= 0:
        raise RuntimeError("spoof_ratio produced empty class after downsampling.")

    idx_s = rng.choice(n_s, size=n_s_des, replace=False)
    idx_g = rng.choice(n_g, size=n_g_des, replace=False)
    rng.shuffle(idx_s)
    rng.shuffle(idx_g)

    def _split_counts(n: int) -> Tuple[int, int, int]:
        n_train = int(np.floor(n * train_frac))
        n_val = int(np.floor(n * val_frac))
        n_test = n - n_train - n_val
        if n_test <= 0:
            # Move one from train or val to test if possible
            if n_train > 1:
                n_train -= 1
                n_test += 1
            elif n_val > 1:
                n_val -= 1
                n_test += 1
            else:
                raise RuntimeError("Not enough samples to create train/val/test splits.")
        return n_train, n_val, n_test

    s_tr, s_va, s_te = _split_counts(n_s_des)
    g_tr, g_va, g_te = _split_counts(n_g_des)

    s_train = idx_s[:s_tr]
    s_val = idx_s[s_tr:s_tr + s_va]
    s_test = idx_s[s_tr + s_va:s_tr + s_va + s_te]

    g_train = idx_g[:g_tr]
    g_val = idx_g[g_tr:g_tr + g_va]
    g_test = idx_g[g_tr + g_va:g_tr + g_va + g_te]

    X_train = np.concatenate([X_spoof[s_train], X_genuine[g_train]], axis=0)
    y_train = np.concatenate([
        np.ones((len(s_train),), np.int32),
        np.zeros((len(g_train),), np.int32),
    ], axis=0)

    X_val = np.concatenate([X_spoof[s_val], X_genuine[g_val]], axis=0)
    y_val = np.concatenate([
        np.ones((len(s_val),), np.int32),
        np.zeros((len(g_val),), np.int32),
    ], axis=0)

    X_test = np.concatenate([X_spoof[s_test], X_genuine[g_test]], axis=0)
    y_test = np.concatenate([
        np.ones((len(s_test),), np.int32),
        np.zeros((len(g_test),), np.int32),
    ], axis=0)

    def _shuffle_pair(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return X[idx], y[idx]

    X_train, y_train = _shuffle_pair(X_train, y_train)
    X_val, y_val = _shuffle_pair(X_val, y_val)
    X_test, y_test = _shuffle_pair(X_test, y_test)

    print("\nDataset sizes after ratio + random stratified split:")
    print(f"  Train: {X_train.shape} spoof%={y_train.mean() * 100:.1f}")
    print(f"  Val  : {X_val.shape} spoof%={y_val.mean() * 100:.1f}")
    print(f"  Test : {X_test.shape} spoof%={y_test.mean() * 100:.1f}")

    dataset_params_path = output_dir / "dataset_params.npz"
    np.savez(
        dataset_params_path,
        seed=np.array([cfg.seed], dtype=np.int32),
        window=np.array([cfg.window], dtype=np.int32),
        stride=np.array([cfg.stride], dtype=np.int32),
        train_frac=np.array([cfg.train_frac], dtype=np.float32),
        val_frac=np.array([cfg.val_frac], dtype=np.float32),
        spoof_ratio=np.array([cfg.spoof_ratio], dtype=np.float32),
        n_spoof_raw=np.array([n_s], dtype=np.int64),
        n_genuine_raw=np.array([n_g], dtype=np.int64),
        n_spoof_used=np.array([n_s_des], dtype=np.int64),
        n_genuine_used=np.array([n_g_des], dtype=np.int64),
        train_size=np.array([len(y_train)], dtype=np.int64),
        val_size=np.array([len(y_val)], dtype=np.int64),
        test_size=np.array([len(y_test)], dtype=np.int64),
    )
    print(f"Saved: {dataset_params_path}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds
        .shuffle(min(len(X_train), cfg.shuffle_buffer), seed=cfg.seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_model(cfg: Config, num_features: int) -> tf.keras.Model:
    # Regularizers affect training only; they do not complicate inference graphs.
    reg = tf.keras.regularizers.l2(cfg.l2_regularization)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(cfg.window, num_features)),
        # Keep recurrent_dropout at 0 so TensorFlow can use the fast cuDNN LSTM kernel on GPU.
        tf.keras.layers.LSTM(
            cfg.lstm_units,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            recurrent_dropout=0.0,
        ),
        tf.keras.layers.Dropout(cfg.dropout),
        tf.keras.layers.Dense(cfg.dense_units, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="prauc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def print_confusion(
    model: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    title: str = "",
) -> dict:
    y_prob = model.predict(X, batch_size=256, verbose=0).reshape(-1)
    return print_confusion_from_probs(y_prob, y, threshold=threshold, title=title)


def confusion_from_probs(y_prob: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.int32).reshape(-1)
    if y_prob.shape[0] != y.shape[0]:
        raise ValueError("y_prob and y must contain the same number of samples.")

    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y == 1) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))

    def safe_ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    total = tp + tn + fp + fn
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    npv = safe_ratio(tn, tn + fn)
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = safe_ratio(tp + tn, total)
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc_denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    matthews_corrcoef = safe_ratio(tp * tn - fp * fn, mcc_denominator)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": total,
        "positive_support": tp + fn,
        "negative_support": tn + fp,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "f1": f1,
        "matthews_corrcoef": matthews_corrcoef,
        "false_positive_rate": 1.0 - specificity,
        "false_negative_rate": 1.0 - recall,
        "prevalence": safe_ratio(tp + fn, total),
        "predicted_positive_rate": safe_ratio(tp + fp, total),
        "mean_probability": float(np.mean(y_prob)) if total else 0.0,
    }


def print_confusion_from_probs(
    y_prob: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    title: str = "",
) -> dict:
    result = confusion_from_probs(y_prob, y, threshold)

    if title:
        print(f"{title}")
    print(
        f"Confusion @thr={threshold:.2f}: TP={result['tp']} FP={result['fp']} "
        f"TN={result['tn']} FN={result['fn']} | accuracy={result['accuracy']:.4f} "
        f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
        f"specificity={result['specificity']:.4f} f1={result['f1']:.4f}"
    )
    return result


def find_best_f1_threshold(y_prob: np.ndarray, y: np.ndarray) -> dict:
    """Choose the threshold that maximizes F1 on validation predictions."""
    best = None
    for threshold in np.linspace(0.01, 0.99, 99):
        result = confusion_from_probs(y_prob, y, float(threshold))
        if best is None or result["f1"] > best["f1"]:
            best = result

    return best


def sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a deterministic, approximately class-balanced subset."""
    if max_samples <= 0 or X.shape[0] == 0:
        return X[:0], y[:0]
    if X.shape[0] <= max_samples:
        return X.copy(), y.copy()

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // max(1, len(classes)))
    selected: List[int] = []
    for class_value in classes:
        class_indices = np.flatnonzero(y == class_value)
        take = min(per_class, class_indices.shape[0])
        selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())

    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(y.shape[0]), np.asarray(selected), assume_unique=False)
        take = min(max_samples - len(selected), remaining.shape[0])
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    selected_array = np.asarray(selected[:max_samples], dtype=np.int64)
    rng.shuffle(selected_array)
    return X[selected_array], y[selected_array]


def save_shap_samples(
    run_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
) -> Optional[Path]:
    if cfg.shap_background_size <= 0 or cfg.shap_explain_size <= 0:
        return None

    X_background, y_background = sample_rows(
        X_train, y_train, cfg.shap_background_size, cfg.seed + 1001
    )
    X_explain, y_explain = sample_rows(
        X_test, y_test, cfg.shap_explain_size, cfg.seed + 1002
    )
    output_path = run_dir / "shap_samples.npz"
    np.savez_compressed(
        output_path,
        X_background=X_background,
        y_background=y_background,
        X_explain=X_explain,
        y_explain=y_explain,
        feature_keys=np.asarray(cfg.feature_keys),
        window=np.asarray([cfg.window], dtype=np.int32),
    )
    print(f"Saved: {output_path}")
    return output_path


def save_tflite_model(model: tf.keras.Model, output_path: str = "lstm_spoof_detector.tflite") -> None:
    """Convert a trained Keras model to TensorFlow Lite and save it.

    The model is cloned on CPU before conversion to avoid tracing GPU-only
    CuDNN kernels (for example ``tf.CudnnRNNV3``), which TFLite cannot convert.
    """
    model_cfg = model.get_config()

    # Build a conversion-friendly config without relying on layer class-level
    # from_config calls from potentially revived/wrapped objects.
    if isinstance(model_cfg, dict) and "layers" in model_cfg:
        for layer_cfg in model_cfg["layers"]:
            if layer_cfg.get("class_name") != "LSTM":
                continue
            layer_inner_cfg = layer_cfg.get("config", {})

            # Prevent tracing GPU-only CuDNN kernels (tf.CudnnRNNV3) during export.
            # Unrolling also keeps execution in standard TF ops that TFLite can lower.
            layer_inner_cfg["unroll"] = True
            if "use_cudnn" in layer_inner_cfg:
                layer_inner_cfg["use_cudnn"] = False

    with tf.device("/CPU:0"):
        export_model = model.__class__.from_config(model_cfg)
        export_model.set_weights(model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    try:
        tflite_model = converter.convert()
    except Exception as err:
        print(f"Builtins-only TFLite conversion failed ({err}); retrying with Select TF Ops.")
        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Saved: {output_path}")


def configure_accelerator() -> None:
    """Prefer GPU execution for training when available."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected by TensorFlow; training will run on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    names = ", ".join(gpu.name for gpu in gpus)
    print(f"Using GPU for training: {names}")


def main(
    run_timestamp: str,
    run_dir: Path,
    cfg: Config,
    spoofed_path: str,
    genuine_path: str,
    save_keras: bool = True,
    save_tflite: bool = True,
    save_saved_model: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    configure_accelerator()

    spoofed_path = str(Path(spoofed_path).expanduser().resolve())
    genuine_path = str(Path(genuine_path).expanduser().resolve())
    run_config = {
        "timestamp": run_timestamp,
        "run_dir": str(run_dir.resolve()),
        "input_files": {
            "spoofed": spoofed_path,
            "genuine": genuine_path,
        },
        "config": asdict(cfg),
        "exports": {
            "keras": save_keras,
            "tflite": save_tflite,
            "saved_model": save_saved_model,
        },
    }
    run_config_path = run_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as config_file:
        json.dump(run_config, config_file, indent=2)
    print(f"Configuration:\n{json.dumps(run_config, indent=2)}")
    print(f"Saved: {run_config_path}")

    mat_spoofed = loadmat(spoofed_path)
    mat_genuine = loadmat(genuine_path)

    spoofed_features, spoofed_sat_count, _, _, spoofed_sv_matrix = load_and_validate(
        mat_spoofed, "SPOOFED", cfg.feature_keys
    )
    genuine_features, genuine_sat_count, _, _, genuine_sv_matrix = load_and_validate(
        mat_genuine, "GENUINE", cfg.feature_keys
    )

    # 1) Build ALL windows first (no time split)
    X_spoof, _ = build_all_windows(spoofed_features, spoofed_sat_count, spoofed_sv_matrix, label=1, cfg=cfg)
    X_genuine, _ = build_all_windows(genuine_features, genuine_sat_count, genuine_sv_matrix, label=0, cfg=cfg)

    print("\nRaw windows (before ratio/split):")
    print(f"  spoof  : {X_spoof.shape}")
    print(f"  genuine: {X_genuine.shape}")

    if X_spoof.shape[0] == 0 or X_genuine.shape[0] == 0:
        raise RuntimeError("No windows created. Reduce window, adjust stride, or check sv_id filtering.")

    # 2) Enforce spoof_ratio, then random stratified train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_sample_and_split(
        X_spoof, X_genuine, cfg, run_dir
    )

    # 3) Normalize (fit on train only)
    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_val = standardize_apply(X_val, mean, std)
    X_test = standardize_apply(X_test, mean, std)

    norm_params_path = run_dir / "norm_params.npz"
    np.savez(
        norm_params_path,
        mean=mean,
        std=std,
        window=np.array([cfg.window], dtype=np.int32),
        feature_keys=np.array(cfg.feature_keys),
    )
    print(f"Saved: {norm_params_path}")
    shap_samples_path = save_shap_samples(run_dir, X_train, y_train, X_test, y_test, cfg)

    # 4) tf.data
    train_ds, val_ds, test_ds = build_tf_datasets(X_train, y_train, X_val, y_val, X_test, y_test, cfg)

    # 5) Model + checkpoint (train full epochs, keep best weights)
    model = build_model(cfg, num_features=len(cfg.feature_keys))
    best_weights_path = str(run_dir / "best_weights.weights.h5")

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor=cfg.monitor_metric,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=2,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=[ckpt],
        verbose=2,
    )

    # 6) Load best weights and evaluate
    try:
        model.load_weights(best_weights_path)
        print(f"Loaded best weights: {best_weights_path} (monitor={cfg.monitor_metric})")
    except Exception as e:
        print(f"Warning: could not load best weights ({e}). Using last epoch weights.")

    print("\nEvaluation (VAL):")
    val_results = model.evaluate(val_ds, verbose=2, return_dict=True)

    print("\nEvaluation (TEST):")
    test_results = model.evaluate(test_ds, verbose=2, return_dict=True)

    # Choose threshold on validation only, then apply it to test.
    val_prob = model.predict(X_val, batch_size=256, verbose=0).reshape(-1)
    test_prob = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    fixed_threshold = cfg.prediction_threshold
    print("\nFixed-threshold metrics:")
    val_fixed_metrics = print_confusion_from_probs(
        val_prob, y_val, threshold=fixed_threshold, title="VAL"
    )
    test_fixed_metrics = print_confusion_from_probs(
        test_prob, y_test, threshold=fixed_threshold, title="TEST"
    )

    if cfg.tune_f1_threshold:
        best_threshold = find_best_f1_threshold(val_prob, y_val)
        threshold = best_threshold["threshold"]
        print(
            "\nBest threshold from VAL F1 sweep: "
            f"{threshold:.2f} | precision={best_threshold['precision']:.4f} "
            f"recall={best_threshold['recall']:.4f} f1={best_threshold['f1']:.4f}"
        )
    else:
        best_threshold = val_fixed_metrics
        threshold = fixed_threshold
        print(f"\nF1 threshold tuning disabled; using fixed threshold {threshold:.2f}.")

    val_selected_metrics = print_confusion_from_probs(
        val_prob, y_val, threshold=threshold, title="VAL confusion at selected threshold"
    )
    test_selected_metrics = print_confusion_from_probs(
        test_prob, y_test, threshold=threshold, title="TEST confusion at selected threshold"
    )

    # 7) Save final model (with best weights loaded)
    tflite_path = run_dir / "lstm_spoof_detector.tflite"
    keras_path = run_dir / "lstm_spoof_detector.keras"
    saved_model_path = run_dir / "lstm_saved_model"
    weights_path = run_dir / "lstm_weights.weights.h5"
    model.save_weights(str(weights_path))
    print(f"Saved: {weights_path}")
    artifacts: Dict[str, Any] = {
        "run_config": str(run_config_path),
        "normalization": str(norm_params_path),
        "best_weights": best_weights_path,
        "weights": str(weights_path),
    }
    if shap_samples_path is not None:
        artifacts["shap_samples"] = str(shap_samples_path)

    export_jobs = []
    if save_tflite:
        export_jobs.append(("tflite", tflite_path, lambda: save_tflite_model(model, str(tflite_path))))
    if save_keras:
        export_jobs.append(("keras", keras_path, lambda: model.save(str(keras_path))))
    if save_saved_model:
        export_jobs.append(("saved_model", saved_model_path, lambda: model.export(str(saved_model_path))))

    for artifact_name, artifact_path, export_job in export_jobs:
        try:
            export_job()
            artifacts[artifact_name] = str(artifact_path)
            print(f"Saved: {artifact_path}")
        except Exception as error:
            artifacts[f"{artifact_name}_error"] = str(error)
            print(f"Warning: {artifact_name} export failed: {error}")

    # Save training history for later plotting
    train_history_path = run_dir / "train_history.npz"
    np.savez(str(train_history_path), **{k: np.array(v) for k, v in history.history.items()})
    print(f"Saved: {train_history_path}")
    artifacts["training_history"] = str(train_history_path)

    monitor_values = history.history.get(cfg.monitor_metric, [])
    if monitor_values:
        best_epoch_index = int(np.argmax(np.asarray(monitor_values)))
        best_epoch = best_epoch_index + 1
        best_monitor_value = float(monitor_values[best_epoch_index])
    else:
        best_epoch = None
        best_monitor_value = None

    eval_results_path = run_dir / "evaluation_results.json"
    eval_results: Dict[str, Any] = {
        "timestamp": run_timestamp,
        "run_dir": str(run_dir.resolve()),
        "input_files": run_config["input_files"],
        "config": asdict(cfg),
        "model": {
            "lstm_units": cfg.lstm_units,
            "dense_units": cfg.dense_units,
            "trainable_parameters": int(model.count_params()),
        },
        "data": {
            "raw_spoof_windows": int(X_spoof.shape[0]),
            "raw_genuine_windows": int(X_genuine.shape[0]),
            "train_windows": int(y_train.shape[0]),
            "validation_windows": int(y_val.shape[0]),
            "test_windows": int(y_test.shape[0]),
            "train_spoof_windows": int(np.sum(y_train == 1)),
            "validation_spoof_windows": int(np.sum(y_val == 1)),
            "test_spoof_windows": int(np.sum(y_test == 1)),
        },
        "training": {
            "epochs_completed": len(history.epoch),
            "monitor_metric": cfg.monitor_metric,
            "best_epoch": best_epoch,
            "best_monitor_value": best_monitor_value,
        },
        "threshold_selection": {
            "metric": "f1" if cfg.tune_f1_threshold else "fixed",
            "selected_on": "validation" if cfg.tune_f1_threshold else "configuration",
            "selected_threshold": float(threshold),
            "validation_result": best_threshold,
        },
        "validation": {key: float(value) for key, value in val_results.items()},
        "test": {key: float(value) for key, value in test_results.items()},
        "validation_at_fixed_threshold": val_fixed_metrics,
        "test_at_fixed_threshold": test_fixed_metrics,
        "validation_at_selected_threshold": val_selected_metrics,
        "test_at_selected_threshold": test_selected_metrics,
        # Backward-compatible names retained for existing result readers.
        "validation_confusion": val_selected_metrics,
        "test_confusion": test_selected_metrics,
        "artifacts": artifacts,
        "duration_seconds": float(time.perf_counter() - started),
    }
    with eval_results_path.open("w", encoding="utf-8") as results_file:
        json.dump(eval_results, results_file, indent=2)
    print(f"Saved: {eval_results_path}")
    return eval_results


def run_with_log(args: argparse.Namespace) -> Path:
    cfg = config_from_args(args)
    spoofed_path, genuine_path = pick_paths_from_user(args.spoofed_file, args.genuine_file)
    outputs_dir = Path(args.output_dir).expanduser()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = outputs_dir / (args.run_name or run_timestamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_stdout = Tee(sys.stdout, log_file)
        tee_stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Run output directory: {run_dir}")
            print(f"Logging terminal output to: {log_path}")
            main(
                run_timestamp=run_timestamp,
                run_dir=run_dir,
                cfg=cfg,
                spoofed_path=spoofed_path,
                genuine_path=genuine_path,
                save_keras=not args.no_keras,
                save_tflite=not args.no_tflite,
                save_saved_model=not args.no_saved_model,
            )
    return run_dir


if __name__ == "__main__":
    parsed_args = parse_args()
    require_training_dependencies()
    run_with_log(parsed_args)
