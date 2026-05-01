"""
Shared API dependencies and helper functions.

Used by multiple routers to locate artifact files and resolve the latest run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.pipeline.registry import list_runs, load_run_manifest
from src.utils.config_loader import get_config


def _artifact_cfg() -> dict:
    """Return the artifacts section of pipeline.yaml."""
    return get_config("pipeline")["artifacts"]


def latest_completed_run() -> Optional[tuple[str, str]]:
    """Return (run_id, model_name) for the most recent completed pipeline run.

    Returns:
        Tuple of (run_id, model_name) or None if no completed runs exist.
    """
    df = list_runs()
    if df.empty:
        return None

    completed = df[df["status"] == "completed"]
    if completed.empty:
        return None

    row = completed.iloc[-1]
    run_id: str = row["run_id"]
    model: str = row.get("model", "ridge") or "ridge"
    return run_id, model


def predictions_path(run_id: str, model: str) -> Path:
    """Resolve the predictions CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["predictions_dir"]) / f"{run_id}_{model}_predictions.csv"


def weights_path(run_id: str, model: str) -> Path:
    """Resolve the portfolio weights CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["weights_dir"]) / f"{run_id}_{model}_weights.csv"


def metrics_path(run_id: str, model: str) -> Path:
    """Resolve the metrics CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["metrics_dir"]) / f"{run_id}_{model}_metrics.csv"


def returns_path(run_id: str) -> Path:
    """Resolve the processed returns CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["data_dir"]) / f"{run_id}_returns.csv"


def equity_curve_path(run_id: str, model: str) -> Path:
    """Resolve the equity curve CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["metrics_dir"]) / f"{run_id}_{model}_equity_curve.csv"


def resolve_run(run_id: Optional[str]) -> tuple[str, str]:
    """Resolve run_id and model_name, falling back to the latest completed run.

    Args:
        run_id: Explicit run ID from query param, or None to use latest.

    Returns:
        Tuple of (run_id, model_name).

    Raises:
        LookupError: If no completed runs are available.
        FileNotFoundError: If the given run_id has no manifest.
    """
    if run_id is not None:
        manifest = load_run_manifest(run_id)
        model: str = (
            manifest.get("pipeline_cfg", {}).get("models", {}).get("default", "ridge")
            or "ridge"
        )
        return run_id, model

    result = latest_completed_run()
    if result is None:
        raise LookupError("No completed pipeline runs found. Run POST /pipeline/run first.")
    return result
