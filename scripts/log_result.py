#!/usr/bin/env python3
"""
Append a training result from run.log to results.tsv.

Examples:
    python scripts/log_result.py --description "baseline" --status keep
    python scripts/log_result.py --status crash --description "OOM at larger context"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


SCHEMA = [
    "commit",
    "val_sharpe",
    "test_sharpe",
    "val_macro_f1",
    "val_bal_acc",
    "val_loss",
    "test_loss",
    "training_seconds",
    "total_seconds",
    "num_steps",
    "num_params_M",
    "context_length",
    "num_features",
    "status",
    "description",
]

SUMMARY_KEYS = {
    "val_sharpe": "val_sharpe",
    "test_sharpe": "test_sharpe",
    "val_macro_f1": "val_macro_f1",
    "val_bal_acc": "val_bal_acc",
    "val_loss": "val_loss",
    "test_loss": "test_loss",
    "training_seconds": "training_seconds",
    "total_seconds": "total_seconds",
    "num_steps": "num_steps",
    "num_params_M": "num_params_M",
    "context_length": "context_length",
    "num_features": "num_features",
}

CRASH_DEFAULTS = {
    "val_sharpe": "0.000000",
    "test_sharpe": "0.000000",
    "val_macro_f1": "0.000000",
    "val_bal_acc": "0.000000",
    "val_loss": "0.000000",
    "test_loss": "0.000000",
    "training_seconds": "0.0",
    "total_seconds": "0.0",
    "num_steps": "0",
    "num_params_M": "0.00",
    "context_length": "0",
    "num_features": "0",
}


def git_short_commit(workdir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_summary(run_log: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in run_log.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in SUMMARY_KEYS:
            summary[SUMMARY_KEYS[key]] = value
    return summary


def ensure_results_header(results_path: Path) -> None:
    if results_path.exists():
        return
    results_path.write_text("\t".join(SCHEMA) + "\n")


def sanitize_description(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").strip()


def append_row(results_path: Path, row: dict[str, str]) -> None:
    ensure_results_header(results_path)
    with results_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA, delimiter="\t")
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append metrics from run.log to results.tsv.")
    parser.add_argument("--run-log", type=Path, default=Path("run.log"))
    parser.add_argument("--results", type=Path, default=Path("results.tsv"))
    parser.add_argument("--status", choices=["keep", "discard", "crash"], required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--commit", help="Override the short git commit hash.")
    args = parser.parse_args()

    workdir = Path.cwd()
    commit = args.commit or git_short_commit(workdir)
    description = sanitize_description(args.description)

    if args.status == "crash":
        metrics = dict(CRASH_DEFAULTS)
    else:
        if not args.run_log.exists():
            raise FileNotFoundError(f"Missing run log: {args.run_log}")
        metrics = parse_summary(args.run_log)
        missing = [column for column in SUMMARY_KEYS.values() if column not in metrics]
        if missing:
            raise ValueError(
                f"run.log is missing summary fields: {', '.join(missing)}. "
                "Only use --status crash when the run failed before summary output."
            )

    row = {column: "" for column in SCHEMA}
    row["commit"] = commit
    row["status"] = args.status
    row["description"] = description
    row.update(metrics)

    append_row(args.results, row)
    print("\t".join(row[column] for column in SCHEMA))


if __name__ == "__main__":
    main()
