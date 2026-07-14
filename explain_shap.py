#!/usr/bin/env python3
"""Explain a completed lstm.py run with SHAP expected gradients."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate global and local SHAP explanations for an lstm.py run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dir", help="Completed lstm.py run directory.")
    parser.add_argument("--model-path", help="Override .keras model path.")
    parser.add_argument("--samples-path", help="Override shap_samples.npz path.")
    parser.add_argument("--output-dir", help="Defaults to RUN_DIR/shap.")
    parser.add_argument("--background-size", type=positive_int, default=100)
    parser.add_argument("--explain-size", type=positive_int, default=20)
    parser.add_argument(
        "--nsamples",
        type=positive_int,
        default=200,
        help="Expected-gradient samples per explanation; larger is slower but less noisy.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=50)
    parser.add_argument("--seed", type=int, default=580)
    parser.add_argument("--local-plots", type=nonnegative_int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    selected = []
    per_class = max(1, max_samples // max(1, classes.shape[0]))
    for class_value in classes:
        indices = np.flatnonzero(y == class_value)
        take = min(per_class, indices.shape[0])
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())
    if len(selected) < max_samples:
        remaining = np.setdiff1d(np.arange(y.shape[0]), np.asarray(selected), assume_unique=False)
        take = min(max_samples - len(selected), remaining.shape[0])
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    selected_array = np.asarray(selected[:max_samples], dtype=np.int64)
    rng.shuffle(selected_array)
    return X[selected_array], y[selected_array]


def load_model(run_dir: Path, model_path: Path | None, config: Dict[str, Any], tf: Any) -> Any:
    candidates = []
    if model_path is not None:
        candidates.append(model_path)
    candidates.append(run_dir / "lstm_spoof_detector.keras")
    for candidate in candidates:
        if candidate.exists():
            print(f"Loading Keras model: {candidate}")
            return tf.keras.models.load_model(str(candidate))

    from lstm import Config, build_model

    valid_fields = {field.name for field in fields(Config)}
    config_values = {key: value for key, value in config.items() if key in valid_fields}
    if "feature_keys" in config_values:
        config_values["feature_keys"] = tuple(config_values["feature_keys"])
    lstm_config = Config(**config_values)
    model = build_model(lstm_config, num_features=len(lstm_config.feature_keys))

    weight_candidates = [
        run_dir / "best_weights.weights.h5",
        run_dir / "lstm_weights.weights.h5",
    ]
    for candidate in weight_candidates:
        if candidate.exists():
            model.load_weights(str(candidate))
            print(f"Rebuilt model and loaded weights: {candidate}")
            return model
    raise FileNotFoundError(
        "No .keras model or compatible weights found. Train without --no-keras, "
        "or keep best_weights.weights.h5 and run_config.json."
    )


def normalize_shap_values(raw_values: Any, input_shape: Tuple[int, ...]) -> np.ndarray:
    values = raw_values
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one model output, received {len(values)} SHAP outputs.")
        values = values[0]
    values = np.asarray(values)
    if values.shape == input_shape + (1,):
        values = values[..., 0]
    if values.shape != input_shape:
        raise ValueError(f"SHAP values have shape {values.shape}; expected {input_shape}.")
    return values.astype(np.float32)


def write_feature_importance(
    output_path: Path,
    feature_names: Sequence[str],
    mean_absolute: np.ndarray,
    mean_signed: np.ndarray,
) -> None:
    order = np.argsort(mean_absolute)[::-1]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["rank", "feature", "mean_absolute_shap", "mean_signed_shap"])
        for rank, index in enumerate(order, start=1):
            writer.writerow([
                rank,
                feature_names[index],
                float(mean_absolute[index]),
                float(mean_signed[index]),
            ])


def write_local_contributions(
    output_path: Path,
    feature_names: Sequence[str],
    contributions: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["sample", "true_label", "spoof_probability", *feature_names])
        for index in range(contributions.shape[0]):
            writer.writerow([
                index,
                int(labels[index]),
                float(predictions[index]),
                *[float(value) for value in contributions[index]],
            ])


def save_plots(
    output_dir: Path,
    shap_module: Any,
    plt: Any,
    values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    predictions: np.ndarray,
    labels: np.ndarray,
    local_plot_count: int,
) -> Dict[str, str]:
    plot_errors: Dict[str, str] = {}
    mean_absolute_by_feature = np.mean(np.abs(values), axis=(0, 1))
    feature_order = np.argsort(mean_absolute_by_feature)
    plt.figure(figsize=(8, max(3.5, 0.55 * len(feature_names))))
    plt.barh(
        np.asarray(feature_names)[feature_order],
        mean_absolute_by_feature[feature_order],
        color="#287271",
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global feature importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=180)
    plt.close()

    mean_absolute_by_timestep = np.mean(np.abs(values), axis=(0, 2))
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, values.shape[1] + 1), mean_absolute_by_timestep, color="#d1495b")
    plt.xlabel("Timestep within window")
    plt.ylabel("Mean absolute SHAP value")
    plt.title("Importance across the input window")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "timestep_importance.png", dpi=180)
    plt.close()

    feature_contributions = values.sum(axis=1)
    feature_values = X_explain.mean(axis=1)
    try:
        shap_module.summary_plot(
            feature_contributions,
            feature_values,
            feature_names=list(feature_names),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "feature_summary.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as error:
        plot_errors["feature_summary"] = str(error)
        plt.close("all")

    max_abs = float(np.max(np.abs(values))) if values.size else 1.0
    max_abs = max(max_abs, 1e-9)
    for index in range(min(local_plot_count, values.shape[0])):
        plt.figure(figsize=(10, max(3.5, 0.65 * len(feature_names))))
        image = plt.imshow(
            values[index].T,
            aspect="auto",
            cmap="coolwarm",
            vmin=-max_abs,
            vmax=max_abs,
        )
        plt.yticks(np.arange(len(feature_names)), feature_names)
        plt.xlabel("Timestep within window")
        plt.ylabel("Feature")
        plt.title(
            f"Sample {index}: label={int(labels[index])}, "
            f"spoof probability={predictions[index]:.4f}"
        )
        plt.colorbar(image, label="SHAP value")
        plt.tight_layout()
        plt.savefig(output_dir / f"sample_{index:03d}_heatmap.png", dpi=180)
        plt.close()
    return plot_errors


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    global np
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise SystemExit("NumPy is required to read the saved SHAP samples.") from error
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise SystemExit("TensorFlow is required to load the trained model.") from error
    try:
        import shap
    except ModuleNotFoundError as error:
        raise SystemExit("SHAP is not installed. Install it with: pip install shap matplotlib") from error

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run configuration: {run_config_path}")
    with run_config_path.open("r", encoding="utf-8") as input_file:
        run_config = json.load(input_file)
    config = run_config["config"]

    samples_path = Path(args.samples_path).expanduser() if args.samples_path else run_dir / "shap_samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(
            f"Missing {samples_path}. Train with positive --shap-background-size and "
            "--shap-explain-size values."
        )
    sample_data = np.load(samples_path, allow_pickle=False)
    X_background = sample_data["X_background"].astype(np.float32)
    y_background = sample_data["y_background"].astype(np.int32)
    X_explain = sample_data["X_explain"].astype(np.float32)
    y_explain = sample_data["y_explain"].astype(np.int32)
    feature_names = [str(value) for value in sample_data["feature_keys"].tolist()]
    X_background, y_background = sample_rows(
        X_background, y_background, args.background_size, args.seed
    )
    X_explain, y_explain = sample_rows(X_explain, y_explain, args.explain_size, args.seed + 1)

    model_path = Path(args.model_path).expanduser() if args.model_path else None
    model = load_model(run_dir, model_path, config, tf)
    predictions = model.predict(X_explain, batch_size=args.batch_size, verbose=0).reshape(-1)
    background_prediction = float(
        np.mean(model.predict(X_background, batch_size=args.batch_size, verbose=0))
    )

    print(
        f"Explaining {X_explain.shape[0]} windows using "
        f"{X_background.shape[0]} background windows."
    )
    explainer = shap.GradientExplainer(model, X_background, batch_size=args.batch_size)
    raw_values = explainer.shap_values(X_explain, nsamples=args.nsamples, rseed=args.seed)
    values = normalize_shap_values(raw_values, X_explain.shape)

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)
    mean_absolute_by_feature = np.mean(np.abs(values), axis=(0, 1))
    mean_signed_by_feature = np.mean(values, axis=(0, 1))
    mean_absolute_by_timestep = np.mean(np.abs(values), axis=(0, 2))
    feature_contributions = values.sum(axis=1)
    shap_sum = values.sum(axis=(1, 2))
    additivity_residual = predictions - background_prediction - shap_sum

    np.savez_compressed(
        output_dir / "shap_values.npz",
        shap_values=values,
        X_explain=X_explain,
        y_explain=y_explain,
        predictions=predictions,
        background_prediction=np.asarray([background_prediction], dtype=np.float32),
        feature_names=np.asarray(feature_names),
        mean_absolute_by_feature=mean_absolute_by_feature,
        mean_absolute_by_timestep=mean_absolute_by_timestep,
        feature_contributions=feature_contributions,
        additivity_residual=additivity_residual,
    )
    write_feature_importance(
        output_dir / "feature_importance.csv",
        feature_names,
        mean_absolute_by_feature,
        mean_signed_by_feature,
    )
    write_local_contributions(
        output_dir / "local_feature_contributions.csv",
        feature_names,
        feature_contributions,
        y_explain,
        predictions,
    )

    plot_errors: Dict[str, str] = {}
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            raise SystemExit("Plotting requires matplotlib: pip install matplotlib") from error
        plot_errors = save_plots(
            output_dir,
            shap,
            plt,
            values,
            X_explain,
            feature_names,
            predictions,
            y_explain,
            args.local_plots,
        )

    order = np.argsort(mean_absolute_by_feature)[::-1]
    summary = {
        "run_dir": str(run_dir),
        "model_output": "spoof probability",
        "positive_shap_meaning": "pushes prediction toward spoof (label 1)",
        "negative_shap_meaning": "pushes prediction toward genuine (label 0)",
        "background_windows": int(X_background.shape[0]),
        "explained_windows": int(X_explain.shape[0]),
        "nsamples": args.nsamples,
        "background_prediction_mean": background_prediction,
        "mean_absolute_additivity_residual": float(np.mean(np.abs(additivity_residual))),
        "feature_importance": [
            {
                "rank": rank,
                "feature": feature_names[index],
                "mean_absolute_shap": float(mean_absolute_by_feature[index]),
                "mean_signed_shap": float(mean_signed_by_feature[index]),
            }
            for rank, index in enumerate(order, start=1)
        ],
        "plot_errors": plot_errors,
    }
    with (output_dir / "shap_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print(f"SHAP outputs saved to: {output_dir}")
    print(f"Top feature: {feature_names[order[0]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
