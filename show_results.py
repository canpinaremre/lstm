#!/usr/bin/env python3
"""Display run_lstm_experiments.py results.json files as readable tables."""

from __future__ import annotations

import argparse
import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


def nested_get(value: Dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_score(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:.4f}"


def format_integer(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{int(value):,}"


def format_duration(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    seconds = int(round(value))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def format_text(value: Any) -> str:
    return "-" if value is None else str(value)


@dataclass(frozen=True)
class Column:
    key: str
    title: str
    path: str
    formatter: Callable[[Any], str] = format_text


COLUMNS: Dict[str, Column] = {
    "run": Column("run", "Run", "run_index", format_integer),
    "status": Column("status", "Status", "status"),
    "window": Column("window", "Window", "parameters.window", format_integer),
    "stride": Column("stride", "Stride", "parameters.stride", format_integer),
    "lstm": Column("lstm", "LSTM", "parameters.lstm_units", format_integer),
    "dense": Column("dense", "Dense", "parameters.dense_units", format_integer),
    "seed": Column("seed", "Seed", "parameters.seed", format_integer),
    "parameters": Column(
        "parameters", "Parameters", "evaluation.model.trainable_parameters", format_integer
    ),
    "epochs": Column("epochs", "Epochs", "evaluation.training.epochs_completed", format_integer),
    "best_epoch": Column("best_epoch", "Best epoch", "evaluation.training.best_epoch", format_integer),
    "train_windows": Column(
        "train_windows", "Train N", "evaluation.data.train_windows", format_integer
    ),
    "test_windows": Column("test_windows", "Test N", "evaluation.data.test_windows", format_integer),
    "val_accuracy": Column("val_accuracy", "Val acc", "evaluation.validation.acc", format_score),
    "val_auc": Column("val_auc", "Val AUC", "evaluation.validation.auc", format_score),
    "val_prauc": Column("val_prauc", "Val PR-AUC", "evaluation.validation.prauc", format_score),
    "val_f1": Column(
        "val_f1", "Val F1", "evaluation.validation_at_selected_threshold.f1", format_score
    ),
    "test_loss": Column("test_loss", "Test loss", "evaluation.test.loss", format_score),
    "test_accuracy": Column(
        "test_accuracy",
        "Test acc",
        "evaluation.test_at_selected_threshold.accuracy",
        format_score,
    ),
    "test_auc": Column("test_auc", "Test AUC", "evaluation.test.auc", format_score),
    "test_prauc": Column("test_prauc", "Test PR-AUC", "evaluation.test.prauc", format_score),
    "test_f1": Column(
        "test_f1", "Test F1", "evaluation.test_at_selected_threshold.f1", format_score
    ),
    "precision": Column(
        "precision",
        "Precision",
        "evaluation.test_at_selected_threshold.precision",
        format_score,
    ),
    "recall": Column(
        "recall", "Recall", "evaluation.test_at_selected_threshold.recall", format_score
    ),
    "specificity": Column(
        "specificity",
        "Specificity",
        "evaluation.test_at_selected_threshold.specificity",
        format_score,
    ),
    "mcc": Column(
        "mcc", "MCC", "evaluation.test_at_selected_threshold.matthews_corrcoef", format_score
    ),
    "threshold": Column(
        "threshold", "Threshold", "evaluation.threshold_selection.selected_threshold", format_score
    ),
    "duration": Column("duration", "Duration", "duration_seconds", format_duration),
}


VIEWS = {
    "summary": (
        "run", "status", "window", "stride", "lstm", "dense", "seed",
        "test_accuracy", "test_auc", "test_prauc", "test_f1",
    ),
    "metrics": (
        "run", "status", "val_accuracy", "val_auc", "val_prauc", "val_f1",
        "test_loss", "test_accuracy", "test_auc", "test_prauc", "test_f1",
        "precision", "recall", "specificity", "mcc", "threshold",
    ),
    "config": (
        "run", "status", "window", "stride", "lstm", "dense", "seed",
        "parameters", "epochs", "best_epoch", "train_windows", "test_windows", "duration",
    ),
}
VIEWS["all"] = tuple(dict.fromkeys(VIEWS["config"] + VIEWS["metrics"]))


def normalize_results(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Every results.json entry must be an object.")
        return data
    if isinstance(data, dict):
        if "evaluation" in data or "status" in data:
            return [data]
        if "config" in data and "test" in data:
            return [{
                "run_index": 1,
                "status": "completed",
                "duration_seconds": data.get("duration_seconds"),
                "parameters": {
                    "window": nested_get(data, "config.window"),
                    "stride": nested_get(data, "config.stride"),
                    "lstm_units": nested_get(data, "config.lstm_units"),
                    "dense_units": nested_get(data, "config.dense_units"),
                    "seed": nested_get(data, "config.seed"),
                },
                "evaluation": data,
            }]
    raise ValueError("Expected experiment results (a JSON list) or one evaluation result object.")


def build_rows(
    results: Sequence[Dict[str, Any]],
    columns: Sequence[Column],
) -> List[Tuple[Dict[str, Any], List[str]]]:
    return [
        (result, [column.formatter(nested_get(result, column.path)) for column in columns])
        for result in results
    ]


def render_terminal(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    lines = [separator, render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    lines.append(separator)
    return "\n".join(lines)


def render_markdown(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    escaped = lambda value: value.replace("|", "\\|")
    lines = [
        "| " + " | ".join(escaped(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escaped(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_csv(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().rstrip("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show LSTM experiment results.json as a table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_file", help="results.json or evaluation_results.json path.")
    parser.add_argument("--view", choices=tuple(VIEWS), default="summary")
    parser.add_argument("--format", choices=("terminal", "markdown", "csv"), default="terminal")
    parser.add_argument("--sort-by", choices=tuple(COLUMNS), default="run")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows; 0 shows all.")
    parser.add_argument("--output", help="Write the table to a file instead of stdout.")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_path = Path(args.results_file).expanduser()
    with results_path.open("r", encoding="utf-8") as input_file:
        results = normalize_results(json.load(input_file))

    if args.completed_only:
        results = [result for result in results if result.get("status") == "completed"]

    sort_column = COLUMNS[args.sort_by]

    present_results = [
        result for result in results if nested_get(result, sort_column.path) is not None
    ]
    missing_results = [
        result for result in results if nested_get(result, sort_column.path) is None
    ]
    present_results.sort(
        key=lambda result: nested_get(result, sort_column.path),
        reverse=args.descending,
    )
    results = present_results + missing_results
    if args.limit:
        results = results[:args.limit]

    columns = [COLUMNS[key] for key in VIEWS[args.view]]
    table_rows = [row for _, row in build_rows(results, columns)]
    headers = [column.title for column in columns]
    renderers = {
        "terminal": render_terminal,
        "markdown": render_markdown,
        "csv": render_csv,
    }
    table = renderers[args.format](headers, table_rows)

    completed = sum(result.get("status") == "completed" for result in results)
    failed = sum(result.get("status") == "failed" for result in results)
    footer = f"Rows: {len(results)} | completed: {completed} | failed: {failed}"
    rendered = table if args.format == "csv" else f"{table}\n{footer}"

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved table: {output_path}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
