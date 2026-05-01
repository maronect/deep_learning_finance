"""
Router for GET /history/runs, GET /history/runs/compare, POST /history/sync.

Provides queryable access to the SQLite-backed run registry (Stage 4).
Complements the file-based pipeline.py router with richer filtering,
side-by-side comparison, and a rebuild endpoint for DB recovery.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.responses import (
    CompareRunsResponse,
    RunRecord,
    RunsDbListResponse,
    SyncResponse,
)
from src.persistence.database import (
    compare_runs,
    get_run,
    list_runs,
    sync_from_manifests,
)

router = APIRouter(prefix="/history", tags=["history"])


def _to_run_record(row: dict) -> RunRecord:
    """Convert a DB row dict to a RunRecord response model."""
    return RunRecord(
        run_id=row.get("run_id", ""),
        started_at=str(row.get("started_at", "")),
        status=str(row.get("status", "")),
        model=str(row.get("model", "")),
        blend_alpha=row.get("blend_alpha"),
        train_ratio=row.get("train_ratio"),
        lag_window=row.get("lag_window"),
        frequency=row.get("frequency"),
        risk_free_rate=row.get("risk_free_rate"),
        n_assets=int(row.get("n_assets") or 0),
        selected_assets=row.get("selected_assets") or [],
        sharpe=row.get("sharpe"),
        annualized_return=row.get("annualized_return"),
        annualized_volatility=row.get("annualized_volatility"),
        cumulative_return=row.get("cumulative_return"),
        mean_return=row.get("mean_return"),
        volatility=row.get("volatility"),
        artifacts_written=row.get("artifacts_written") or [],
        error=row.get("error"),
        synced_at=row.get("synced_at"),
    )


@router.get(
    "/runs",
    response_model=RunsDbListResponse,
    summary="List all runs from the persistent database",
)
def list_db_runs(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: completed, failed, running.",
    ),
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=500,
        description="Maximum number of records to return (most recent first).",
    ),
) -> RunsDbListResponse:
    """Return run records from the SQLite database.

    Unlike GET /pipeline/runs (which scans manifest files), this endpoint
    queries the indexed database and supports filtering and pagination.
    """
    rows = list_runs(status=status, limit=limit)
    records = [_to_run_record(r) for r in rows]
    return RunsDbListResponse(runs=records, count=len(records))


@router.get(
    "/runs/compare",
    response_model=CompareRunsResponse,
    summary="Compare multiple runs side by side",
)
def compare_db_runs(
    run_ids: str = Query(
        description="Comma-separated list of run IDs to compare (e.g. '20260415T120000,20260416T080000').",
    ),
) -> CompareRunsResponse:
    """Return a side-by-side comparison of the requested runs.

    Metrics, model configuration, and asset selection are shown for each run,
    enabling direct comparison of experiments.

    Args:
        run_ids: Comma-separated run ID strings.

    Raises:
        400: If no run_ids are provided.
        404: If none of the requested run_ids are found in the database.
    """
    id_list = [r.strip() for r in run_ids.split(",") if r.strip()]
    if not id_list:
        raise HTTPException(status_code=400, detail="Provide at least one run_id.")

    rows = compare_runs(id_list)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"None of the requested run_ids were found: {id_list}. "
                   "Run POST /history/sync to import manifest files.",
        )

    records = [_to_run_record(r) for r in rows]
    return CompareRunsResponse(run_ids=id_list, found=len(records), runs=records)


@router.get(
    "/runs/{run_id}",
    response_model=RunRecord,
    summary="Retrieve a single run record from the database",
)
def get_db_run(run_id: str) -> RunRecord:
    """Return the full persisted record for a specific run.

    Raises:
        404: If the run_id is not found in the database.
    """
    row = get_run(run_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{run_id}' not found in the database. "
                   "Run POST /history/sync to import manifest files.",
        )
    return _to_run_record(row)


@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Rebuild the database from manifest files",
)
def sync_db() -> SyncResponse:
    """Scan artifacts/runs/ and upsert all manifest JSON files into the database.

    Use this endpoint to recover the database after a fresh deployment or when
    runs were executed before the persistence layer was active (e.g. Stage 1/2 runs).

    Returns:
        Number of manifests successfully imported.
    """
    synced = sync_from_manifests()
    return SyncResponse(
        synced=synced,
        message=(
            f"Synced {synced} run manifest(s) into the database."
            if synced > 0
            else "No manifest files found in artifacts/runs/."
        ),
    )
