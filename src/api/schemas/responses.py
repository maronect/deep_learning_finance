"""
Pydantic response schemas for all API endpoints.

Ensures consistent, typed, and documented output structure across all routes.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    runs_available: int


class AssetsResponse(BaseModel):
    """Response for GET /assets."""

    source: str  # "config" | "latest_run"
    run_id: Optional[str]
    assets: list[str]
    count: int


class PredictionItem(BaseModel):
    """Single asset expected-return prediction."""

    ticker: str
    predicted_return: float
    model: str


class PredictionsResponse(BaseModel):
    """Response for GET /predictions."""

    run_id: str
    model: str
    predictions: list[PredictionItem]


class PipelineRunResponse(BaseModel):
    """Response for POST /pipeline/run."""

    run_id: str
    status: str
    message: str


class RunSummary(BaseModel):
    """Summary row for a single pipeline run."""

    run_id: str
    started_at: str
    status: str
    model: str
    n_assets: int
    sharpe: Optional[float]
    annualized_return: Optional[float]


class RunsListResponse(BaseModel):
    """Response for GET /pipeline/runs."""

    runs: list[RunSummary]
    count: int


class RunManifestResponse(BaseModel):
    """Response for GET /pipeline/runs/{run_id}."""

    run_id: str
    started_at: str
    status: str
    error: Optional[str]
    selected_assets: Optional[list[str]]
    artifacts_written: list[str]
    metrics: Optional[dict]


class WeightItem(BaseModel):
    """Portfolio weight for a single asset."""

    ticker: str
    weight: float


class PortfolioWeightsResponse(BaseModel):
    """Response for GET /portfolio/weights."""

    run_id: str
    model: str
    weights: list[WeightItem]


class FrontierPoint(BaseModel):
    """Single point on the efficient frontier."""

    lambda_param: float
    volatility: float
    expected_return: float
    sharpe: float


class FrontierResponse(BaseModel):
    """Response for GET /portfolio/frontier."""

    run_id: str
    risk_free_rate: float
    n_points: int
    points: list[FrontierPoint]


class PortfolioMetricsResponse(BaseModel):
    """Response for GET /metrics/portfolio."""

    run_id: str
    model: str
    sharpe: float
    annualized_return: float
    annualized_volatility: float
    cumulative_return: float
    mean_return: float
    volatility: float


class ModelMetricsResponse(BaseModel):
    """Response for GET /metrics/model.

    Reports model configuration and walk-forward OOS diagnostic metrics.
    Finance-specific metrics (IC, ICIR, hit rate, Spearman IC) are None for
    runs completed before this feature was introduced.
    """

    run_id: str
    model: str
    blend_alpha: Optional[float]
    train_ratio: Optional[float]
    lag_window: Optional[int]
    ic: Optional[float]
    icir: Optional[float]
    hit_rate: Optional[float]
    spearman_ic: Optional[float]
    mae: Optional[float]
    mse: Optional[float]
    r2: Optional[float]


class RunRecord(BaseModel):
    """Full persisted record for a single pipeline run (from SQLite)."""

    run_id: str
    started_at: str
    status: str
    model: str
    blend_alpha: Optional[float]
    train_ratio: Optional[float]
    lag_window: Optional[int]
    frequency: Optional[str]
    risk_free_rate: Optional[float]
    n_assets: int
    selected_assets: list[str]
    sharpe: Optional[float]
    annualized_return: Optional[float]
    annualized_volatility: Optional[float]
    cumulative_return: Optional[float]
    mean_return: Optional[float]
    volatility: Optional[float]
    artifacts_written: list[str]
    error: Optional[str]
    synced_at: Optional[str]


class RunsDbListResponse(BaseModel):
    """Response for GET /history/runs (DB-backed list)."""

    runs: list[RunRecord]
    count: int


class CompareRunsResponse(BaseModel):
    """Response for GET /history/runs/compare."""

    run_ids: list[str]
    found: int
    runs: list[RunRecord]


class SyncResponse(BaseModel):
    """Response for POST /history/sync."""

    synced: int
    message: str


class ModelComparisonItem(BaseModel):
    """Per-model comparison entry for GET /portfolio/compare."""

    model: str
    weights: list[WeightItem]
    sharpe: Optional[float]
    annualized_return: Optional[float]
    annualized_volatility: Optional[float]
    cumulative_return: Optional[float]
    ic: Optional[float]
    icir: Optional[float]
    hit_rate: Optional[float]
    spearman_ic: Optional[float]


class PortfolioCompareResponse(BaseModel):
    """Response for GET /portfolio/compare."""

    run_id: str
    models: list[str]
    primary_model: str
    comparison: dict[str, ModelComparisonItem]


class EquityCurvePoint(BaseModel):
    """Single point on the portfolio equity curve."""

    date: str
    cumulative_return: float


class EquityCurveResponse(BaseModel):
    """Response for GET /portfolio/equity-curve."""

    run_id: str
    model: str
    period_start: str
    period_end: str
    n_points: int
    points: list[EquityCurvePoint]
