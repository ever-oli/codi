from __future__ import annotations

import csv
from pathlib import Path

RESULTS_COLUMNS = [
    "run_id",
    "task",
    "dataset",
    "modality",
    "model_name",
    "params_m",
    "train_seconds",
    "best_val_metric",
    "test_metric",
    "peak_memory_mb",
    "batch_size",
    "seed",
    "status",
    "notes",
    "config_path",
    "checkpoint_path",
    "log_path",
    "timestamp",
]


def initialize_results_file(results_path: Path) -> None:
    if results_path.exists():
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writeheader()


def append_result_row(results_path: Path, row: dict[str, str]) -> None:
    initialize_results_file(results_path)
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writerow({column: row.get(column, "") for column in RESULTS_COLUMNS})
