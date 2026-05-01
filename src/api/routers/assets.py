"""
Router for GET /assets.

Returns the list of B3 tickers available in the current pipeline configuration
or, if a completed run exists, from the most recent run's selected asset list.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from src.api.deps import latest_completed_run
from src.api.schemas.responses import AssetsResponse
from src.pipeline.registry import load_run_manifest
from src.utils.config_loader import get_config

router = APIRouter(tags=["assets"])


@router.get("/assets", response_model=AssetsResponse, summary="List available assets")
def get_assets(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve selected assets from. Defaults to the latest completed run.",
    ),
) -> AssetsResponse:
    """Return the list of B3 tickers used in the pipeline.

    If a specific `run_id` is provided, returns the assets selected during
    that run. Otherwise, falls back to the latest completed run, and if none
    exists, returns the tickers configured in `config/pipeline.yaml`.
    """
    if run_id is not None:
        manifest = load_run_manifest(run_id)
        assets: list[str] = manifest.get("selected_assets") or []
        return AssetsResponse(
            source="run",
            run_id=run_id,
            assets=assets,
            count=len(assets),
        )

    latest = latest_completed_run()
    if latest is not None:
        resolved_run_id, _ = latest
        manifest = load_run_manifest(resolved_run_id)
        assets = manifest.get("selected_assets") or []
        return AssetsResponse(
            source="latest_run",
            run_id=resolved_run_id,
            assets=assets,
            count=len(assets),
        )

    # No runs available — fall back to config
    pipeline_cfg = get_config("pipeline")
    config_tickers: list[str] = pipeline_cfg.get("data", {}).get("tickers", [])
    return AssetsResponse(
        source="config",
        run_id=None,
        assets=config_tickers,
        count=len(config_tickers),
    )
