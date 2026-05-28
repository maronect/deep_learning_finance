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


def _pick_model(manifest: dict, requested: Optional[str]) -> str:
    """Select a model name from a run manifest.

    If requested is provided and exists in the run, it is returned. Otherwise,
    the model with the highest Sharpe is returned. For legacy single-model
    manifests, only the trained model is available.

    Args:
        manifest: Run manifest dict.
        requested: Model name from the caller's query param, or None.

    Returns:
        The resolved model name string.

    Raises:
        LookupError: If requested model is not available in this run.
    """
    available: list[str] = manifest.get("models", [])
    if not available:
        legacy = (
            manifest.get("pipeline_cfg", {}).get("models", {}).get("default", "ridge")
            or "ridge"
        )
        if requested is not None and requested != legacy:
            raise LookupError(
                f"Model '{requested}' is not available in this run. "
                f"Only '{legacy}' was trained. "
                "Re-run the pipeline to generate multi-model artifacts."
            )
        return legacy

    if requested is not None:
        if requested not in available:
            raise LookupError(
                f"Model '{requested}' not found in this run. "
                f"Available: {available}."
            )
        return requested

    # Pick the model with the highest Sharpe across all models in this run.
    metrics: dict = manifest.get("metrics", {}) or {}
    return max(
        available,
        key=lambda m: (metrics.get(m) or {}).get("Sharpe") or float("-inf"),
    )


def latest_completed_run(model: Optional[str] = None) -> Optional[tuple[str, str]]:
    """Return (run_id, model_name) for the most recent completed pipeline run.

    Args:
        model: Optional model name to prefer. If None, returns the best-Sharpe model.

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
    manifest = load_run_manifest(run_id)
    resolved_model = _pick_model(manifest, model)
    return run_id, resolved_model


def predictions_path(run_id: str, model: str) -> Path:
    """Resolve the predictions CSV path for a given run and model."""
    cfg = _artifact_cfg()
    return Path(cfg["predictions_dir"]) / f"{run_id}_{model}_predictions.csv"


def weights_path(run_id: str, model: str) -> Path:
    """Resolve the portfolio weights CSV path for a given run and model."""
    cfg = _artifact_cfg()
    return Path(cfg["weights_dir"]) / f"{run_id}_{model}_weights.csv"


def metrics_path(run_id: str, model: str) -> Path:
    """Resolve the portfolio metrics CSV path for a given run and model."""
    cfg = _artifact_cfg()
    return Path(cfg["metrics_dir"]) / f"{run_id}_{model}_metrics.csv"


def model_metrics_path(run_id: str, model: str) -> Path:
    """Resolve the model diagnostics metrics CSV path for a given run and model."""
    cfg = _artifact_cfg()
    return Path(cfg["metrics_dir"]) / f"{run_id}_{model}_model_metrics.csv"


def returns_path(run_id: str) -> Path:
    """Resolve the processed returns CSV path for a given run."""
    cfg = _artifact_cfg()
    return Path(cfg["data_dir"]) / f"{run_id}_returns.csv"


def equity_curve_path(run_id: str, model: str) -> Path:
    """Resolve the equity curve CSV path for a given run and model."""
    cfg = _artifact_cfg()
    return Path(cfg["metrics_dir"]) / f"{run_id}_{model}_equity_curve.csv"


def resolve_run(
    run_id: Optional[str],
    model: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve run_id and model_name, falling back to the latest completed run.

    If model is not specified, returns the model with the highest Sharpe for
    the resolved run (or the only available model for legacy single-model runs).

    Args:
        run_id: Explicit run ID from query param, or None to use latest.
        model: Optional model name from query param.

    Returns:
        Tuple of (run_id, model_name).

    Raises:
        LookupError: If no completed runs are available.
        FileNotFoundError: If the given run_id has no manifest.
    """
    if run_id is not None:
        manifest = load_run_manifest(run_id)
        resolved_model = _pick_model(manifest, model)
        return run_id, resolved_model

    result = latest_completed_run(model)
    if result is None:
        raise LookupError("No completed pipeline runs found. Run POST /pipeline/run first.")
    return result
