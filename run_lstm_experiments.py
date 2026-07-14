#!/usr/bin/env python3
"""Run a grid of lstm.py trainings and collect comparable result tables."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
        description="Run window/stride/model-size LSTM experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("spoofed_file", help="Spoofed MAT file.")
    parser.add_argument("genuine_file", help="Genuine MAT file.")

    grid = parser.add_argument_group("experiment grid")
    grid.add_argument("--windows", nargs="+", type=positive_int, default=[20])
    grid.add_argument("--strides", nargs="+", type=positive_int, default=[1])
    grid.add_argument("--lstm-units", nargs="+", type=positive_int, default=[64])
    grid.add_argument("--dense-units", nargs="+", type=positive_int, default=[32])
    grid.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[580],
        help="Use several seeds to estimate run-to-run variance.",
    )

    training = parser.add_argument_group("shared training settings")
    training.add_argument("--features", nargs="+", default=None)
    training.add_argument("--epochs", type=positive_int, default=300)
    training.add_argument("--batch-size", type=positive_int, default=32)
    training.add_argument("--shuffle-buffer", type=positive_int, default=500)
    training.add_argument("--train-frac", type=float, default=0.70)
    training.add_argument("--val-frac", type=float, default=0.15)
    training.add_argument("--spoof-ratio", type=float, default=0.50)
    training.add_argument("--dropout", type=float, default=0.0)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--l2-regularization", type=float, default=1e-4)
    training.add_argument(
        "--monitor-metric",
        choices=("val_auc", "val_prauc", "val_acc"),
        default="val_auc",
    )
    training.add_argument("--prediction-threshold", type=float, default=0.50)
    training.add_argument("--no-tune-f1-threshold", action="store_true")
    training.add_argument("--allow-sv-zero", action="store_true")
    training.add_argument("--allow-invalid-previous-sample", action="store_true")

    explanation = parser.add_argument_group("SHAP sample artifacts")
    explanation.add_argument("--shap-background-size", type=nonnegative_int, default=100)
    explanation.add_argument("--shap-explain-size", type=nonnegative_int, default=100)
    explanation.add_argument("--no-shap-samples", action="store_true")

    output = parser.add_argument_group("execution and output")
    output.add_argument("--output-dir", default=str(Path("outputs") / "experiments"))
    output.add_argument("--experiment-name", help="Experiment folder name; timestamp when omitted.")
    output.add_argument("--python", default=sys.executable, help="Python executable used for lstm.py.")
    output.add_argument("--no-keras", action="store_true", help="Skip per-run .keras exports.")
    output.add_argument("--save-tflite", action="store_true", help="Export TFLite for every run.")
    output.add_argument("--save-saved-model", action="store_true", help="Export SavedModel for every run.")
    output.add_argument("--stop-on-error", action="store_true")
    output.add_argument("--dry-run", action="store_true", help="Print commands without training.")

    args = parser.parse_args(argv)
    if args.train_frac <= 0.0 or args.val_frac < 0.0 or args.train_frac + args.val_frac >= 1.0:
        parser.error("train/validation fractions must leave a positive test fraction")
    if not (0.0 < args.spoof_ratio < 1.0):
        parser.error("--spoof-ratio must be between 0 and 1")
    if not (0.0 <= args.dropout < 1.0):
        parser.error("--dropout must be at least 0 and less than 1")
    if args.learning_rate <= 0.0 or args.l2_regularization < 0.0:
        parser.error("learning rate must be positive and L2 regularization non-negative")
    if not (0.0 <= args.prediction_threshold <= 1.0):
        parser.error("--prediction-threshold must be between 0 and 1")
    return args


def flatten_dict(value: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, item in value.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flat.update(flatten_dict(item, flat_key))
        elif isinstance(item, (list, tuple)):
            flat[flat_key] = json.dumps(item)
        elif item is None:
            flat[flat_key] = ""
        else:
            flat[flat_key] = item
    return flat


def write_flat_csv(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    flat_records = [flatten_dict(record) for record in records]
    if not flat_records:
        path.write_text("", encoding="utf-8")
        return
    leading = ["run_index", "status", "return_code", "duration_seconds", "run_dir"]
    all_fields = {key for record in flat_records for key in record}
    fieldnames = [key for key in leading if key in all_fields]
    fieldnames.extend(sorted(all_fields.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_records)


def nested_get(value: Dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


AGGREGATE_METRICS = {
    "validation_loss": "evaluation.validation.loss",
    "validation_accuracy": "evaluation.validation.acc",
    "validation_roc_auc": "evaluation.validation.auc",
    "validation_pr_auc": "evaluation.validation.prauc",
    "validation_f1_selected": "evaluation.validation_at_selected_threshold.f1",
    "test_loss": "evaluation.test.loss",
    "test_accuracy": "evaluation.test.acc",
    "test_roc_auc": "evaluation.test.auc",
    "test_pr_auc": "evaluation.test.prauc",
    "test_accuracy_selected": "evaluation.test_at_selected_threshold.accuracy",
    "test_balanced_accuracy_selected": "evaluation.test_at_selected_threshold.balanced_accuracy",
    "test_precision_selected": "evaluation.test_at_selected_threshold.precision",
    "test_recall_selected": "evaluation.test_at_selected_threshold.recall",
    "test_specificity_selected": "evaluation.test_at_selected_threshold.specificity",
    "test_f1_selected": "evaluation.test_at_selected_threshold.f1",
    "test_mcc_selected": "evaluation.test_at_selected_threshold.matthews_corrcoef",
    "test_accuracy_fixed": "evaluation.test_at_fixed_threshold.accuracy",
    "test_f1_fixed": "evaluation.test_at_fixed_threshold.f1",
}


def aggregate_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    successful = [result for result in results if result.get("status") == "completed"]
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    group_fields = (
        "window",
        "stride",
        "lstm_units",
        "dense_units",
        "dropout",
        "learning_rate",
        "l2_regularization",
    )
    for result in successful:
        parameters = result["parameters"]
        key = tuple(parameters[field] for field in group_fields)
        grouped.setdefault(key, []).append(result)

    aggregates: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        row: Dict[str, Any] = dict(zip(group_fields, key))
        row["completed_runs"] = len(group)
        row["seeds"] = json.dumps([item["parameters"]["seed"] for item in group])
        for output_name, metric_path in AGGREGATE_METRICS.items():
            values = [nested_get(item, metric_path) for item in group]
            numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
            if numeric_values:
                row[f"{output_name}_mean"] = statistics.fmean(numeric_values)
                row[f"{output_name}_std"] = statistics.pstdev(numeric_values)
        aggregates.append(row)
    return aggregates


def save_summaries(experiment_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    with (experiment_dir / "results.json").open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    write_flat_csv(experiment_dir / "results.csv", results)

    aggregates = aggregate_results(results)
    with (experiment_dir / "aggregate_results.json").open("w", encoding="utf-8") as output_file:
        json.dump(aggregates, output_file, indent=2)
    write_flat_csv(experiment_dir / "aggregate_results.csv", aggregates)


def shared_lstm_args(args: argparse.Namespace) -> List[str]:
    values = [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--shuffle-buffer", str(args.shuffle_buffer),
        "--train-frac", str(args.train_frac),
        "--val-frac", str(args.val_frac),
        "--spoof-ratio", str(args.spoof_ratio),
        "--dropout", str(args.dropout),
        "--learning-rate", str(args.learning_rate),
        "--l2-regularization", str(args.l2_regularization),
        "--monitor-metric", args.monitor_metric,
        "--prediction-threshold", str(args.prediction_threshold),
        "--shap-background-size", str(args.shap_background_size),
        "--shap-explain-size", str(args.shap_explain_size),
    ]
    if args.features:
        values.extend(["--features", *args.features])
    if args.no_tune_f1_threshold:
        values.append("--no-tune-f1-threshold")
    if args.allow_sv_zero:
        values.append("--allow-sv-zero")
    if args.allow_invalid_previous_sample:
        values.append("--allow-invalid-previous-sample")
    if args.no_shap_samples:
        values.append("--no-shap-samples")
    if args.no_keras:
        values.append("--no-keras")
    if not args.save_tflite:
        values.append("--no-tflite")
    if not args.save_saved_model:
        values.append("--no-saved-model")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    lstm_script = script_dir / "lstm.py"
    experiment_name = args.experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_dir = Path(args.output_dir).expanduser() / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = experiment_dir / "runs"
    runs_dir.mkdir()

    combinations = list(itertools.product(
        args.windows,
        args.strides,
        args.lstm_units,
        args.dense_units,
        args.seeds,
    ))
    manifest = {
        "created_at": datetime.now().isoformat(),
        "spoofed_file": str(Path(args.spoofed_file).expanduser().resolve()),
        "genuine_file": str(Path(args.genuine_file).expanduser().resolve()),
        "total_runs": len(combinations),
        "arguments": vars(args),
    }
    with (experiment_dir / "experiment_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)

    print(f"Experiment directory: {experiment_dir}")
    print(f"Runs to execute: {len(combinations)}")
    results: List[Dict[str, Any]] = []
    common_args = shared_lstm_args(args)

    for run_index, (window, stride, lstm_units, dense_units, seed) in enumerate(combinations, start=1):
        run_name = (
            f"run_{run_index:04d}_w{window}_s{stride}_"
            f"lstm{lstm_units}_dense{dense_units}_seed{seed}"
        )
        run_dir = runs_dir / run_name
        parameters = {
            "window": window,
            "stride": stride,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "seed": seed,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
        }
        command = [
            args.python,
            str(lstm_script),
            args.spoofed_file,
            args.genuine_file,
            "--window", str(window),
            "--stride", str(stride),
            "--lstm-units", str(lstm_units),
            "--dense-units", str(dense_units),
            "--seed", str(seed),
            "--output-dir", str(runs_dir),
            "--run-name", run_name,
            *common_args,
        ]
        print(
            f"\n[{run_index}/{len(combinations)}] window={window} stride={stride} "
            f"lstm={lstm_units} dense={dense_units} seed={seed}"
        )

        if args.dry_run:
            print(subprocess.list2cmdline(command))
            results.append({
                "run_index": run_index,
                "status": "dry_run",
                "return_code": 0,
                "duration_seconds": 0.0,
                "run_dir": str(run_dir),
                "parameters": parameters,
                "command": command,
            })
            continue

        started = time.perf_counter()
        launch_error = None
        try:
            completed = subprocess.run(command, cwd=script_dir, check=False)
            return_code = completed.returncode
        except OSError as error:
            launch_error = str(error)
            return_code = -1
        duration = time.perf_counter() - started
        result: Dict[str, Any] = {
            "run_index": run_index,
            "return_code": return_code,
            "duration_seconds": duration,
            "run_dir": str(run_dir.resolve()),
            "parameters": parameters,
            "command": command,
        }
        evaluation_path = run_dir / "evaluation_results.json"
        if return_code == 0 and evaluation_path.exists():
            try:
                with evaluation_path.open("r", encoding="utf-8") as input_file:
                    result["evaluation"] = json.load(input_file)
                result["status"] = "completed"
            except (OSError, json.JSONDecodeError) as error:
                result["status"] = "failed"
                result["error"] = f"Could not read {evaluation_path}: {error}"
        if result.get("status") != "completed":
            result["status"] = "failed"
            if "error" not in result:
                detail = (
                    f"launch failed: {launch_error}"
                    if launch_error
                    else f"lstm.py exited with {return_code}"
                )
                result["error"] = f"{detail}; see {run_dir / 'run.log'}"
        results.append(result)
        save_summaries(experiment_dir, results)

        if result["status"] == "failed" and args.stop_on_error:
            print(result["error"])
            break

    save_summaries(experiment_dir, results)
    completed_count = sum(result.get("status") == "completed" for result in results)
    failed_count = sum(result.get("status") == "failed" for result in results)
    print(f"\nCompleted: {completed_count}; failed: {failed_count}")
    print(f"Per-run results: {experiment_dir / 'results.csv'}")
    print(f"Aggregate results: {experiment_dir / 'aggregate_results.csv'}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
