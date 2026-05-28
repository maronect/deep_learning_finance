"""
SQLite persistence layer for pipeline run records.

Stores run metadata, config, metrics, and artifact paths in a single
database file (artifacts/pipeline_runs.db). Complements the JSON manifests
in artifacts/runs/ with a queryable, indexed store that enables fast
comparison of multiple runs without scanning the filesystem.

Schema (table: runs)
--------------------
run_id          TEXT  PRIMARY KEY
started_at      TEXT
status          TEXT
model           TEXT
blend_alpha     REAL
train_ratio     REAL
lag_window      INTEGER
frequency       TEXT
risk_free_rate  REAL
n_assets        INTEGER
selected_assets TEXT  (JSON array of ticker strings)
sharpe          REAL
annualized_return       REAL
annualized_volatility   REAL
cumulative_return       REAL
mean_return     REAL
volatility      REAL
artifacts_written TEXT  (JSON array of paths)
error           TEXT
synced_at       TEXT  (ISO timestamp of last DB write)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from src.utils.config_loader import get_config

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT,
    status              TEXT,
    model               TEXT,
    blend_alpha         REAL,
    train_ratio         REAL,
    lag_window          INTEGER,
    frequency           TEXT,
    risk_free_rate      REAL,
    n_assets            INTEGER,
    selected_assets     TEXT,
    sharpe              REAL,
    annualized_return   REAL,
    annualized_volatility REAL,
    cumulative_return   REAL,
    mean_return         REAL,
    volatility          REAL,
    artifacts_written   TEXT,
    error               TEXT,
    synced_at           TEXT
);
"""

_UPSERT = """
INSERT INTO runs (
    run_id, started_at, status, model, blend_alpha, train_ratio,
    lag_window, frequency, risk_free_rate, n_assets, selected_assets,
    sharpe, annualized_return, annualized_volatility, cumulative_return,
    mean_return, volatility, artifacts_written, error, synced_at
) VALUES (
    :run_id, :started_at, :status, :model, :blend_alpha, :train_ratio,
    :lag_window, :frequency, :risk_free_rate, :n_assets, :selected_assets,
    :sharpe, :annualized_return, :annualized_volatility, :cumulative_return,
    :mean_return, :volatility, :artifacts_written, :error, :synced_at
)
ON CONFLICT(run_id) DO UPDATE SET
    started_at          = excluded.started_at,
    status              = excluded.status,
    model               = excluded.model,
    blend_alpha         = excluded.blend_alpha,
    train_ratio         = excluded.train_ratio,
    lag_window          = excluded.lag_window,
    frequency           = excluded.frequency,
    risk_free_rate      = excluded.risk_free_rate,
    n_assets            = excluded.n_assets,
    selected_assets     = excluded.selected_assets,
    sharpe              = excluded.sharpe,
    annualized_return   = excluded.annualized_return,
    annualized_volatility = excluded.annualized_volatility,
    cumulative_return   = excluded.cumulative_return,
    mean_return         = excluded.mean_return,
    volatility          = excluded.volatility,
    artifacts_written   = excluded.artifacts_written,
    error               = excluded.error,
    synced_at           = excluded.synced_at;
"""


def _db_path() -> Path:
    """Resolve the SQLite file path from pipeline config."""
    cfg = get_config("pipeline")
    base_dir = cfg["artifacts"]["base_dir"]
    return Path(base_dir) / "pipeline_runs.db"


@contextmanager
def _connect() -> Generator[sqlite3.Connection, None, None]:
    """Open a database connection, ensure the schema exists, and close on exit."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_CREATE_TABLE)
        conn.commit()
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the runs table if it does not exist.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    with _connect():
        pass  # table creation happens inside _connect


def _manifest_to_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """Convert a run manifest dict to a flat DB row dict."""
    pipeline_cfg = manifest.get("pipeline_cfg", {}) or {}
    models_cfg = pipeline_cfg.get("models", {}) or {}
    features_cfg = pipeline_cfg.get("features", {}) or {}
    opt_cfg = pipeline_cfg.get("optimization", {}) or {}
    data_cfg = pipeline_cfg.get("data", {}) or {}

    metrics_raw = manifest.get("metrics", {}) or {}
    primary_model = manifest.get("primary_model") or models_cfg.get("default", "")
    # Support both multi-model format (metrics is dict[str, dict]) and legacy format.
    if primary_model and isinstance(metrics_raw.get(primary_model), dict):
        metrics = metrics_raw[primary_model]
    elif any(isinstance(v, dict) for v in metrics_raw.values()):
        metrics = {}
    else:
        metrics = metrics_raw
    selected = manifest.get("selected_assets") or []

    return {
        "run_id": manifest.get("run_id", ""),
        "started_at": str(manifest.get("started_at", "")),
        "status": manifest.get("status", ""),
        "model": primary_model,
        "blend_alpha": models_cfg.get("blend_alpha"),
        "train_ratio": models_cfg.get("train_ratio"),
        "lag_window": features_cfg.get("lag_window"),
        "frequency": data_cfg.get("frequency", ""),
        "risk_free_rate": opt_cfg.get("risk_free_rate"),
        "n_assets": len(selected),
        "selected_assets": json.dumps(selected),
        "sharpe": metrics.get("Sharpe"),
        "annualized_return": metrics.get("Annualized_Return"),
        "annualized_volatility": metrics.get("Annualized_Volatility"),
        "cumulative_return": metrics.get("Cumulative_Return"),
        "mean_return": metrics.get("Mean_Return"),
        "volatility": metrics.get("Volatility"),
        "artifacts_written": json.dumps(manifest.get("artifacts_written") or []),
        "error": manifest.get("error"),
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
    d = dict(row)
    for field in ("selected_assets", "artifacts_written"):
        raw = d.get(field)
        if raw:
            try:
                d[field] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                d[field] = []
        else:
            d[field] = []
    return d


def upsert_run(manifest: dict[str, Any]) -> None:
    """Insert or update a run record from a manifest dict.

    Safe to call multiple times for the same run_id — updates all fields
    on conflict. Typically called at the end of stage_export_artifacts.

    Args:
        manifest: The run manifest dict (as produced by context.to_run_manifest()
            extended with a 'metrics' key).
    """
    row = _manifest_to_row(manifest)
    with _connect() as conn:
        conn.execute(_UPSERT, row)
        conn.commit()


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a single run record by run_id.

    Args:
        run_id: The run identifier.

    Returns:
        Dict with all run fields, or None if not found.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_runs(
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return all run records, optionally filtered by status.

    Args:
        status: Optional status filter ('completed', 'failed', 'running').
        limit: Maximum number of records to return (most recent first).

    Returns:
        List of run dicts sorted by run_id descending (newest first).
    """
    query = "SELECT * FROM runs"
    params: list[Any] = []

    if status is not None:
        query += " WHERE status = ?"
        params.append(status)

    query += " ORDER BY run_id DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_row_to_dict(r) for r in rows]


def compare_runs(run_ids: list[str]) -> list[dict[str, Any]]:
    """Return side-by-side records for the given run IDs.

    Args:
        run_ids: List of run identifiers to compare.

    Returns:
        List of run dicts for found run IDs (missing IDs are silently skipped).
        Ordered to match the input run_ids order.
    """
    if not run_ids:
        return []

    placeholders = ",".join("?" * len(run_ids))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM runs WHERE run_id IN ({placeholders})", run_ids
        ).fetchall()

    by_id = {_row_to_dict(r)["run_id"]: _row_to_dict(r) for r in rows}
    return [by_id[rid] for rid in run_ids if rid in by_id]


def sync_from_manifests() -> int:
    """Scan artifacts/runs/ and upsert all manifest files into the database.

    Useful for rebuilding the DB after a fresh deployment or if the DB was
    deleted but manifest files remain.

    Returns:
        Number of manifests successfully synced.
    """
    cfg = get_config("pipeline")
    runs_dir = Path(cfg["artifacts"]["runs_dir"])

    if not runs_dir.exists():
        return 0

    manifests = sorted(runs_dir.glob("*_manifest.json"))
    synced = 0

    for manifest_file in manifests:
        try:
            with open(manifest_file) as f:
                import json as _json
                manifest = _json.load(f)
            upsert_run(manifest)
            synced += 1
        except Exception:
            continue

    return synced
