"""
Persist workflow results (final state + debug) per run, per strategy, per query.
Run folder: {outputs_dir}/runs/{date}_{short_hash}/.
Files: {strategy}_query_{index}.json (full state + query_meta for debug loading).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_json(obj: Any) -> Any:
    """JSON encoder fallback for Path, numpy, etc."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def create_run_dir(parent_dir: str | Path) -> Path:
    """
    Create a new run directory: parent_dir/runs/YYYY-MM-DD_{8char_hash}.
    Returns the path to the new directory.
    """
    parent_dir = Path(parent_dir)
    runs_dir = parent_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    seed = f"{now.isoformat()}"
    short_hash = hashlib.sha256(seed.encode()).hexdigest()[:8]
    run_dir = runs_dir / f"{date_str}_{short_hash}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _result_filename(strategy_name: str, query_index: int) -> str:
    return f"{strategy_name}_query_{query_index}.json"


def save_result(
    run_dir: str | Path,
    strategy_name: str,
    query_index: int,
    state: dict[str, Any],
    *,
    query_meta: dict[str, Any] | None = None,
) -> Path:
    """
    Save workflow final state (and optional query meta) to run_dir.
    File: run_dir/{strategy_name}_query_{query_index}.json.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"state": state, "query_meta": query_meta or {}}
    path = run_dir / _result_filename(strategy_name, query_index)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default_json)
    return path


def load_result(
    run_dir: str | Path,
    strategy_name: str,
    query_index: int,
) -> dict[str, Any]:
    """
    Load state and query_meta from run_dir.
    Returns dict with keys "state" and "query_meta". Use result["state"] for debug sections.
    """
    run_dir = Path(run_dir)
    path = run_dir / _result_filename(strategy_name, query_index)
    if not path.exists():
        raise FileNotFoundError(f"No result file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_run_dirs(parent_dir: str | Path) -> list[Path]:
    """List run directories under parent_dir/runs/, newest first."""
    runs_dir = Path(parent_dir) / "runs"
    if not runs_dir.exists():
        return []
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def get_completed_tasks(
    run_dir: str | Path,
    strategy_names: list[str],
    num_queries: int,
) -> set[tuple[str, int]]:
    """
    Return set of (strategy_name, query_index) for which a result file exists.
    Used to resume batch: skip these pairs.
    """
    run_dir = Path(run_dir)
    completed: set[tuple[str, int]] = set()
    for name in strategy_names:
        for qi in range(num_queries):
            if (run_dir / _result_filename(name, qi)).exists():
                completed.add((name, qi))
    return completed


def run_dir_metadata(run_dir: str | Path) -> dict[str, Any]:
    """Read optional metadata from run_dir (e.g. run_meta.json)."""
    path = Path(run_dir) / "run_meta.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_run_dir_metadata(run_dir: str | Path, meta: dict[str, Any]) -> None:
    """Write run_meta.json (strategies, query_count, started_at, etc.)."""
    path = Path(run_dir) / "run_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=_default_json)
