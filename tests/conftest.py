"""
Shared pytest fixtures for unit and integration tests.

Provides synthetic price/return DataFrames, mock config objects, and
pre-built PipelineContext instances to avoid duplication across test modules.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from src.pipeline.context import PipelineContext
from src.utils.config_loader import get_config


# ---------------------------------------------------------------------------
# Low-level data builders
# ---------------------------------------------------------------------------

def _make_prices(n_rows: int = 80, n_cols: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    prices = 100 * np.cumprod(
        1 + rng.normal(0.005, 0.03, size=(n_rows, n_cols)), axis=0
    )
    tickers = [f"ASSET{i}" for i in range(n_cols)]
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_returns(n_rows: int = 60, n_cols: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=n_rows, freq="ME")
    data = rng.normal(0.008, 0.04, size=(n_rows, n_cols))
    tickers = [f"ASSET{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_df() -> pd.DataFrame:
    """Synthetic monthly price DataFrame (80 periods, 6 assets)."""
    return _make_prices()


@pytest.fixture
def return_df() -> pd.DataFrame:
    """Synthetic monthly return DataFrame (60 periods, 6 assets)."""
    return _make_returns()


@pytest.fixture
def small_return_df() -> pd.DataFrame:
    """Minimal return DataFrame for fast computations (30 periods, 4 assets)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    data = rng.normal(0.01, 0.05, size=(30, 4))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


@pytest.fixture
def pipeline_cfg() -> dict:
    """Loaded pipeline.yaml config."""
    return get_config("pipeline")


@pytest.fixture
def minimal_context(pipeline_cfg) -> PipelineContext:
    """PipelineContext pre-loaded with synthetic returns and selected assets.

    Skips stage_ingest — suitable for testing stages that start from
    stage_compute_returns onward.
    """
    models_cfg = get_config("models")
    returns = _make_returns(n_rows=60, n_cols=4)
    assets = list(returns.columns)

    ctx = PipelineContext(
        pipeline_cfg=pipeline_cfg,
        models_cfg=models_cfg,
        run_id=datetime.datetime.now().strftime("TEST%Y%m%dT%H%M%S"),
    )
    ctx.prices = _make_prices(n_rows=61, n_cols=4)
    ctx.returns_full = returns
    ctx.selected_assets = assets
    ctx.status = "running"
    return ctx
