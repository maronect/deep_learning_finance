"""
Run registry for experiment traceability and reproducibility.

Provides functions to list, load, and compare past pipeline executions
stored under artifacts/runs/. Each run is identified by its run_id
(ISO timestamp string) and stores a manifest JSON with config, metrics,
selected assets, and artifact paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config_loader import get_config
from src.utils.storage import is_s3_enabled


def _primary_metrics(metrics: dict, primary_model: str) -> dict:
    """Extract the flat metrics dict for the primary model.

    Handles both the new multi-model format (metrics is dict[str, dict]) and
    the legacy single-model format (metrics is a flat dict with "Sharpe" etc.).
    """
    if not metrics:
        return {}
    if primary_model and isinstance(metrics.get(primary_model), dict):
        return metrics[primary_model]
    # Legacy format: flat dict keyed by metric names.
    if any(isinstance(v, dict) for v in metrics.values()):
        return {}
    return metrics


def _runs_dir() -> Path:
    """Return the runs artifact directory from config."""
    cfg = get_config("pipeline")
    return Path(cfg["artifacts"]["runs_dir"])


def _summary_row(m: dict) -> dict:
    """Build a single list_runs() summary row from a manifest dict."""
    metrics = m.get("metrics", {}) or {}
    primary = m.get("primary_model") or m.get("pipeline_cfg", {}).get("models", {}).get("default", "")
    primary_metrics = _primary_metrics(metrics, primary)
    return {
        "run_id": m.get("run_id", ""),
        "started_at": m.get("started_at", ""),
        "status": m.get("status", ""),
        "model": primary,
        "selected_assets_count": len(m.get("selected_assets") or []),
        "sharpe": primary_metrics.get("Sharpe"),
        "annualized_return": primary_metrics.get("Annualized_Return"),
        "artifacts_written_count": len(m.get("artifacts_written") or []),
    }


def _load_manifests_from_s3() -> list[dict]:
    """Load all run manifests stored in S3 under the runs/ prefix.

    Used as a fallback when the local artifacts/runs/ directory is empty (e.g.
    after a container is recreated and the local volume starts empty).

    Returns:
        List of manifest dicts. Empty if S3 holds no manifests.
    """
    from src.utils.storage import s3_key_from_path, s3_list, s3_read_bytes

    prefix = s3_key_from_path(_runs_dir()) + "/"
    manifests: list[dict] = []
    for key in sorted(s3_list(prefix)):
        if not key.endswith("_manifest.json"):
            continue
        data = s3_read_bytes(key)
        if data is None:
            continue
        try:
            manifests.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return manifests


def list_runs() -> pd.DataFrame:
    """List all recorded pipeline runs with their key metadata.

    Scans artifacts/runs/ for manifest JSON files and returns a summary
    DataFrame sorted by run_id (chronological order). When the local directory
    is empty and S3 storage is enabled, manifests are read from S3 instead.

    Returns:
        DataFrame with columns: run_id, started_at, status, model,
        selected_assets_count, sharpe, annualized_return, artifacts_written_count.
        Returns an empty DataFrame if no runs have been recorded.
    """
    runs_path = _runs_dir()
    manifests = sorted(runs_path.glob("*_manifest.json"))

    rows = []
    if manifests:
        for manifest_file in manifests:
            try:
                with open(manifest_file) as f:
                    m = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(_summary_row(m))
    elif is_s3_enabled():
        for m in _load_manifests_from_s3():
            rows.append(_summary_row(m))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_run_manifest(run_id: str) -> dict[str, Any]:
    """Load the manifest of a specific run.

    Args:
        run_id: The run identifier (ISO timestamp string, e.g. "20260415T143000").

    Returns:
        Dict with the full run manifest content.

    Raises:
        FileNotFoundError: If no manifest exists for the given run_id.
    """
    manifest_path = _runs_dir() / f"{run_id}_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)

    if is_s3_enabled():
        from src.utils.storage import s3_key_from_path, s3_read_bytes

        data = s3_read_bytes(s3_key_from_path(manifest_path))
        if data is not None:
            return json.loads(data)

    raise FileNotFoundError(
        f"No manifest found for run_id='{run_id}' at {manifest_path}."
    )


def compare_runs(run_ids: Optional[list[str]] = None) -> pd.DataFrame:
    """Compare metrics and configuration across multiple runs.

    Args:
        run_ids: List of run IDs to compare. If None, all recorded runs are included.

    Returns:
        DataFrame indexed by run_id with columns for model, key config params,
        and all available metrics (Sharpe, Annualized_Return, Annualized_Volatility,
        Cumulative_Return).

    Raises:
        ValueError: If run_ids is provided but empty.
    """
    if run_ids is not None and len(run_ids) == 0:
        raise ValueError("run_ids must not be empty. Pass None to include all runs.")

    if run_ids is None:
        summary = list_runs()
        if summary.empty:
            return pd.DataFrame()
        run_ids = summary["run_id"].tolist()

    rows = []
    for run_id in run_ids:
        try:
            m = load_run_manifest(run_id)
        except FileNotFoundError:
            continue

        cfg = m.get("pipeline_cfg", {})
        metrics = m.get("metrics", {}) or {}
        primary = m.get("primary_model") or cfg.get("models", {}).get("default", "")
        pm = _primary_metrics(metrics, primary)

        rows.append(
            {
                "run_id": run_id,
                "started_at": m.get("started_at", ""),
                "status": m.get("status", ""),
                "model": primary,
                "blend_alpha": cfg.get("models", {}).get("blend_alpha"),
                "lag_window": cfg.get("features", {}).get("lag_window"),
                "train_ratio": cfg.get("models", {}).get("train_ratio"),
                "risk_free_rate": cfg.get("optimization", {}).get("risk_free_rate"),
                "frequency": cfg.get("data", {}).get("frequency"),
                "n_assets": len(m.get("selected_assets") or []),
                "sharpe": pm.get("Sharpe"),
                "annualized_return": pm.get("Annualized_Return"),
                "annualized_volatility": pm.get("Annualized_Volatility"),
                "cumulative_return": pm.get("Cumulative_Return"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("run_id")
    return df
