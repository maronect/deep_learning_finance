"""
Router for GET /health.

Returns application liveness and readiness status.
"""
from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.responses import HealthResponse
from src.pipeline.registry import list_runs
from src.utils.config_loader import get_config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Application health check")
def health_check() -> HealthResponse:
    """Return liveness status and basic operational metrics.

    Always returns HTTP 200. The `runs_available` field shows how many
    completed pipeline runs are stored in the artifact registry.
    """
    api_cfg = get_config("api")
    version: str = api_cfg.get("docs", {}).get("version", "0.1.0")

    df = list_runs()
    n_runs = 0 if df.empty else int((df["status"] == "completed").sum())

    return HealthResponse(
        status="ok",
        version=version,
        runs_available=n_runs,
    )
