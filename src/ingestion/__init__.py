"""
Data ingestion package: download and validate raw market data from external sources.

Public API
----------
run_data_ingestion()
    Unified entry point that runs the full data layer in sequence:
    download prices → compute returns → select assets.

DataLayerResult
    Typed dataclass returned by run_data_ingestion().
"""

import datetime as _dt
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.ingestion.downloader import load_prices
from src.features.returns import compute_returns
from src.features.asset_selection import (
    get_brazilian_stocks_universe,
    select_assets,
)
from src.utils.config_loader import get_config


@dataclass
class DataLayerResult:
    """Typed container for the output of the data ingestion stage.

    Attributes:
        prices: Adjusted closing prices for the full downloaded universe,
            indexed by date (rows) and ticker (columns).
        returns: Percentage returns computed at the configured frequency,
            same shape as prices minus the first row.
        selected_assets: Ordered list of ticker symbols chosen by the
            asset selection strategy.
        config: The full pipeline config dictionary that was used to
            produce this result (for traceability).
    """

    prices: pd.DataFrame
    returns: pd.DataFrame
    selected_assets: list[str]
    config: dict[str, Any]


def run_data_ingestion() -> DataLayerResult:
    """Execute the full data layer pipeline using config/pipeline.yaml.

    Steps executed in sequence:
        1. Load pipeline configuration.
        2. Download prices for the full B3 universe.
        3. Compute returns at the configured frequency.
        4. Select the most decorrelated asset subset (reuses downloaded prices).

    Returns:
        DataLayerResult with prices, returns, selected_assets, and config.

    Raises:
        FileNotFoundError: If config/pipeline.yaml does not exist.
        ValueError: If no assets pass the data quality filters.

    Example:
        >>> result = run_data_ingestion()
        >>> len(result.selected_assets)
        10
    """
    cfg = get_config("pipeline")
    data_cfg = cfg["data"]
    sel_cfg = cfg.get("asset_selection", {})
    features_cfg = cfg.get("features", {})
    models_cfg = cfg.get("models", {})

    end_date: str = data_cfg["end_date"]
    if end_date == "today":
        end_date = _dt.date.today().isoformat()
        cfg["data"]["end_date"] = end_date

    universe = get_brazilian_stocks_universe()

    # Download once, reuse for both return computation and asset selection
    prices = load_prices(
        tickers=universe,
        start=data_cfg["start_date"],
        end=end_date,
        min_data_coverage=sel_cfg.get("min_data_coverage", 0.85),
    )

    returns = compute_returns(prices, freq=data_cfg["frequency"])

    # Compute the training-window cutoff so asset selection cannot see test data.
    # The feature period starts after lag_window periods; train_ratio of that
    # feature period forms the training window. Selection uses only data up to
    # the end of that window.
    _start = pd.Timestamp(data_cfg["start_date"])
    _end = pd.Timestamp(end_date)
    _total_months = (_end.year - _start.year) * 12 + (_end.month - _start.month)
    _lag: int = features_cfg.get("lag_window", 24)
    _ratio: float = models_cfg.get("train_ratio", 0.7)
    _cutoff_months = _lag + int((_total_months - _lag) * _ratio)
    selection_end_date: str = (_start + pd.DateOffset(months=_cutoff_months)).strftime("%Y-%m-%d")

    selected = select_assets(
        start_date=data_cfg["start_date"],
        end_date=end_date,
        method=sel_cfg.get("method", "stable_corr_pairs"),
        n_assets=sel_cfg.get("n_assets", 10),
        return_freq=data_cfg["frequency"],
        min_data_coverage=sel_cfg.get("min_data_coverage", 0.85),
        prices=prices,  # pass pre-downloaded prices — avoids a second network call
        selection_end_date=selection_end_date,
    )

    return DataLayerResult(
        prices=prices,
        returns=returns,
        selected_assets=selected,
        config=cfg,
    )
