"""
Router for /portfolio endpoints.

Provides portfolio weights, efficient frontier, equity curve, and cross-model
comparison for a given pipeline run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.deps import (
    equity_curve_path,
    metrics_path,
    model_metrics_path,
    predictions_path,
    resolve_run,
    returns_path,
    weights_path,
)
from src.api.schemas.responses import (
    EquityCurvePoint,
    EquityCurveResponse,
    FrontierPoint,
    FrontierResponse,
    ModelComparisonItem,
    PortfolioCompareResponse,
    PortfolioWeightsResponse,
    WeightItem,
)
from src.optimization.markowitz import (
    portfolio_return,
    portfolio_volatility,
    solve_markowitz,
)
from src.pipeline.registry import load_run_manifest
from src.utils.config_loader import get_config

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

_FRONTIER_POINTS = 50


@router.get(
    "/weights",
    response_model=PortfolioWeightsResponse,
    summary="Retrieve optimized portfolio weights",
)
def get_weights(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve weights from. Defaults to the latest completed run.",
    ),
    model: Optional[str] = Query(
        default=None,
        description="Model name (e.g. 'ridge', 'mlp', 'markowitz'). "
                    "Defaults to the model with the highest Sharpe in the run.",
    ),
) -> PortfolioWeightsResponse:
    """Return the Sharpe-maximizing portfolio weights for a given run and model.

    Each weight is in [0, 1] and all weights sum to 1 (no short selling).

    Raises:
        404: If no completed runs exist or the artifact is missing.
    """
    try:
        resolved_id, resolved_model = resolve_run(run_id, model)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = weights_path(resolved_id, resolved_model)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Weights artifact not found for run_id='{resolved_id}', "
                   f"model='{resolved_model}'. "
                   "Ensure the pipeline ran successfully through stage_export_artifacts.",
        )

    df = pd.read_csv(path)
    items = [
        WeightItem(ticker=row["Ticker"], weight=float(row["Weight"]))
        for _, row in df.iterrows()
    ]

    return PortfolioWeightsResponse(run_id=resolved_id, model=resolved_model, weights=items)


@router.get(
    "/frontier",
    response_model=FrontierResponse,
    summary="Retrieve efficient frontier data",
)
def get_frontier(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to compute frontier from. Defaults to the latest completed run.",
    ),
    model: Optional[str] = Query(
        default=None,
        description="Model name whose predictions are used as mu. "
                    "Defaults to the model with the highest Sharpe.",
    ),
    n_points: int = Query(
        default=_FRONTIER_POINTS,
        ge=10,
        le=200,
        description="Number of frontier points to sample (lambda values from 0 to 1).",
    ),
) -> FrontierResponse:
    """Compute and return the efficient frontier for a given run and model.

    Raises:
        404: If no completed runs exist or required artifacts are missing.
    """
    try:
        resolved_id, resolved_model = resolve_run(run_id, model)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    preds_path = predictions_path(resolved_id, resolved_model)
    ret_path = returns_path(resolved_id)

    for path in (preds_path, ret_path):
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {path}. "
                       "Ensure the pipeline ran through stage_export_artifacts.",
            )

    preds_df = pd.read_csv(preds_path)
    mu = preds_df.set_index("Ticker")["Predicted_Return"].values.astype(float)

    returns_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    cov = returns_df.cov().values.astype(float)

    pipeline_cfg = get_config("pipeline")
    rf_annual: float = pipeline_cfg["optimization"]["risk_free_rate"]
    freq: str = pipeline_cfg["optimization"].get("frequency", "monthly")
    _periods_map = {"daily": 252, "weekly": 52, "monthly": 12, "annually": 1}
    periods: int = _periods_map.get(freq, 12)
    rf_period: float = (1 + rf_annual) ** (1 / periods) - 1

    lambdas = np.linspace(0.0, 1.0, n_points)
    points: list[FrontierPoint] = []

    for lam in lambdas:
        w = solve_markowitz(mu, cov, lamb=float(lam))
        if w is None:
            continue
        ret = float(portfolio_return(w, mu))
        vol = float(portfolio_volatility(w, cov))
        sharpe = (ret - rf_period) / vol if vol > 0 else 0.0
        points.append(
            FrontierPoint(
                lambda_param=round(float(lam), 4),
                volatility=round(vol, 6),
                expected_return=round(ret, 6),
                sharpe=round(sharpe, 4),
            )
        )

    return FrontierResponse(
        run_id=resolved_id,
        risk_free_rate=rf_annual,
        n_points=len(points),
        points=points,
    )


@router.get(
    "/equity-curve",
    response_model=EquityCurveResponse,
    summary="Retrieve portfolio equity curve over the out-of-sample test period",
)
def get_equity_curve(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to retrieve equity curve from. Defaults to the latest completed run.",
    ),
    model: Optional[str] = Query(
        default=None,
        description="Model name. Defaults to the model with the highest Sharpe.",
    ),
) -> EquityCurveResponse:
    """Return the portfolio cumulative return series over the out-of-sample test period.

    Raises:
        404: If no completed runs exist or the equity curve artifact is missing.
    """
    try:
        resolved_id, resolved_model = resolve_run(run_id, model)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    ec_path = equity_curve_path(resolved_id, resolved_model)

    if ec_path.exists():
        df = pd.read_csv(ec_path)
    else:
        ret_path = returns_path(resolved_id)
        w_path = weights_path(resolved_id, resolved_model)
        for p in (ret_path, w_path):
            if not p.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Artifact not found: {p}. Ensure the pipeline ran successfully.",
                )

        returns_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        weights_df = pd.read_csv(w_path)
        manifest = load_run_manifest(resolved_id)
        pcfg = manifest.get("pipeline_cfg", {})
        lag_window: int = pcfg.get("features", {}).get("lag_window", 24)
        train_ratio: float = pcfg.get("models", {}).get("train_ratio", 0.7)

        n = len(returns_df)
        test_start = lag_window + int((n - lag_window) * train_ratio)
        test_returns = returns_df.iloc[test_start:]

        weights_s = weights_df.set_index("Ticker")["Weight"]
        common = weights_s.index.intersection(test_returns.columns)
        portfolio_returns = test_returns[common].dot(weights_s[common].values)

        equity = (1 + portfolio_returns).cumprod()
        start_date = portfolio_returns.index[0] - pd.DateOffset(months=1)
        equity = pd.concat([pd.Series([1.0], index=[start_date]), equity])
        df = pd.DataFrame({"date": equity.index.astype(str), "cumulative_return": equity.values})

    points = [
        EquityCurvePoint(date=str(row["date"]), cumulative_return=float(row["cumulative_return"]))
        for _, row in df.iterrows()
    ]
    return EquityCurveResponse(
        run_id=resolved_id,
        model=resolved_model,
        period_start=str(df["date"].iloc[0]),
        period_end=str(df["date"].iloc[-1]),
        n_points=len(points),
        points=points,
    )


@router.get(
    "/compare",
    response_model=PortfolioCompareResponse,
    summary="Compare weights and metrics across all models in a run",
)
def get_portfolio_compare(
    run_id: Optional[str] = Query(
        default=None,
        description="Run ID to compare models from. Defaults to the latest completed run.",
    ),
) -> PortfolioCompareResponse:
    """Return weights, portfolio metrics, and ML diagnostic metrics for every model in a run.

    The response includes one entry per model (ridge, mlp, markowitz, etc.).
    ML diagnostic fields (ic, icir, hit_rate, spearman_ic) are None for the
    'markowitz' baseline since it involves no prediction model.

    Raises:
        404: If no completed runs exist or the run_id is unknown.
    """
    try:
        resolved_id, _ = resolve_run(run_id)
    except (LookupError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    manifest = load_run_manifest(resolved_id)
    available_models: list[str] = manifest.get("models", [])
    if not available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Run '{resolved_id}' has no multi-model artifacts. "
                   "Re-run the pipeline to generate them.",
        )

    primary_model: str = manifest.get("primary_model", available_models[0])
    comparison: dict[str, ModelComparisonItem] = {}

    for model_name in available_models:
        # Weights
        w_path = weights_path(resolved_id, model_name)
        if not w_path.exists():
            continue
        weights_df = pd.read_csv(w_path)
        weight_items = [
            WeightItem(ticker=row["Ticker"], weight=float(row["Weight"]))
            for _, row in weights_df.iterrows()
        ]

        # Portfolio metrics
        mpath = metrics_path(resolved_id, model_name)
        port_metrics: dict = {}
        if mpath.exists():
            port_metrics = pd.read_csv(mpath).iloc[0].to_dict()

        # ML diagnostic metrics (not available for markowitz)
        diag: dict = {}
        mm_path = model_metrics_path(resolved_id, model_name)
        if mm_path.exists():
            diag = pd.read_csv(mm_path).iloc[0].to_dict()

        comparison[model_name] = ModelComparisonItem(
            model=model_name,
            weights=weight_items,
            sharpe=port_metrics.get("Sharpe"),
            annualized_return=port_metrics.get("Annualized_Return"),
            annualized_volatility=port_metrics.get("Annualized_Volatility"),
            cumulative_return=port_metrics.get("Cumulative_Return"),
            ic=diag.get("ic"),
            icir=diag.get("icir"),
            hit_rate=diag.get("hit_rate"),
            spearman_ic=diag.get("spearman_ic"),
        )

    return PortfolioCompareResponse(
        run_id=resolved_id,
        models=available_models,
        primary_model=primary_model,
        comparison=comparison,
    )
