from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path("/Volumes/Expansion/Data/autoresearch-av")


@dataclass(frozen=True)
class StoragePaths:
    root: Path
    datasets: Path
    cache: Path
    runs: Path
    results: Path
    models: Path
    logs: Path
    configs: Path


def get_storage_paths(root: str | None = None) -> StoragePaths:
    base = Path(root or os.environ.get("AUTORESEARCH_AV_ROOT") or DEFAULT_ROOT).expanduser()
    return StoragePaths(
        root=base,
        datasets=base / "datasets",
        cache=base / "cache",
        runs=base / "runs",
        results=base / "results",
        models=base / "models",
        logs=base / "logs",
        configs=base / "configs",
    )


def ensure_storage_layout(paths: StoragePaths) -> None:
    for directory in asdict(paths).values():
        Path(directory).mkdir(parents=True, exist_ok=True)


def create_run_dir(paths: StoragePaths, task_name: str, model_name: str, dry_run: bool) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "dryrun" if dry_run else "train"
    run_id = f"{timestamp}-{task_name}-{model_name}-{suffix}"
    run_dir = paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def write_run_config(run_dir: Path, payload: dict) -> Path:
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path
