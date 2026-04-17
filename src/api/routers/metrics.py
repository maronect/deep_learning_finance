"""
Router for GET /metrics/model and GET /metrics/portfolio.

Returns model evaluation metrics and portfolio performance metrics
from the most recent pipeline run.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.deps import metrics_path, resolve_run
from src.api.schemas.responses import ModelMetricsResponse, PortfolioMetricsResponse
from src.pipeline.registry import load_run_manifest

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get(
    "/portfolio",
    response_model=PortfolioMetricsResponse,
    summary="Retrieve portfolio performance metrics",
)
def get_portfolio_metrics(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve metrics from. Defaults to the latest completed run.",
    ),
) -> PortfolioMetricsResponse:
    """Return portfolio performance metrics for a given run.

    Metrics include Sharpe ratio, annualized return and volatility, and
    cumulative return, all computed on the out-of-sample test period.

    Raises:
        404: If no completed runs exist or the metrics artifact is missing.
    """
    try:
        resolved_id, model = resolve_run(run_id)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = metrics_path(resolved_id, model)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics artifact not found for run_id='{resolved_id}'. "
                   "Ensure the pipeline ran successfully through stage_export_artifacts.",
        )

    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    return PortfolioMetricsResponse(
        run_id=resolved_id,
        model=str(row.get("Model", model)),
        sharpe=float(row.get("Sharpe", 0.0)),
        annualized_return=float(row.get("Annualized_Return", 0.0)),
        annualized_volatility=float(row.get("Annualized_Volatility", 0.0)),
        cumulative_return=float(row.get("Cumulative_Return", 0.0)),
        mean_return=float(row.get("Mean_Return", 0.0)),
        volatility=float(row.get("Volatility", 0.0)),
    )


@router.get(
    "/model",
    response_model=ModelMetricsResponse,
    summary="Retrieve model configuration and training parameters",
)
def get_model_metrics(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve model info from. Defaults to the latest completed run.",
    ),
) -> ModelMetricsResponse:
    """Return ML model configuration recorded in the run manifest.

    The current pipeline does not persist per-asset training metrics (MAE, R²)
    as separate artifacts. This endpoint exposes the model identity and blend
    configuration from the run manifest. Training metrics will be added in a
    future stage when walk-forward diagnostics are persisted.

    Raises:
        404: If no completed runs exist or the run_id is unknown.
    """
    try:
        resolved_id, model = resolve_run(run_id)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manifest = load_run_manifest(resolved_id)
    pipeline_cfg = manifest.get("pipeline_cfg", {})
    models_cfg = pipeline_cfg.get("models", {})
    features_cfg = pipeline_cfg.get("features", {})

    return ModelMetricsResponse(
        run_id=resolved_id,
        model=model,
        blend_alpha=models_cfg.get("blend_alpha"),
        train_ratio=models_cfg.get("train_ratio"),
        lag_window=features_cfg.get("lag_window"),
        note=(
            "Per-asset training metrics (MAE, R²) are not yet persisted by the pipeline. "
            "See ROADMAP Stage 6 for planned test coverage expansion."
        ),
    )
