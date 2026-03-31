"""
Responsible for downloading raw OHLCV price data from market APIs (yfinance).

Provides a single entry point to fetch historical price series for a list of
B3 tickers over a configurable date range and cache results locally for
subsequent pipeline runs.
"""

import warnings
import pandas as pd

from src.ingestion.validators import (
    drop_empty_rows,
    filter_by_coverage,
    fill_missing_prices,
    validate_not_empty,
)
from src.utils.config_loader import get_config


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    min_data_coverage: float = 0.5,
) -> pd.DataFrame:
    """Download adjusted closing prices for a list of tickers from Yahoo Finance.

    Tickers that do not meet the minimum data coverage threshold are dropped.
    Remaining NaN gaps are filled with forward-fill then backward-fill.

    Args:
        tickers: List of Yahoo Finance ticker symbols (e.g. ["PETR4.SA", "VALE3.SA"]).
        start: Start date in 'YYYY-MM-DD' format (inclusive).
        end: End date in 'YYYY-MM-DD' format (inclusive).
        min_data_coverage: Minimum fraction of non-null data required per ticker
            to be kept in the result. Must be in [0.0, 1.0]. Defaults to 0.5.

    Returns:
        DataFrame of adjusted closing prices indexed by date, one column per
        valid ticker.

    Raises:
        ValueError: If no tickers have sufficient data after filtering.
        Exception: Re-raises any exception from the Yahoo Finance download.

    Example:
        >>> prices = load_prices(["PETR4.SA"], "2023-01-01", "2023-12-31")
        >>> prices.columns.tolist()
        ['PETR4.SA']
    """
    import yfinance as yf

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df = yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=True
            )["Close"]
    except Exception as exc:
        print(f"  Error downloading data: {exc}")
        raise

    # Ensure result is always a DataFrame even for a single ticker
    if isinstance(df, pd.Series):
        df = df.to_frame()
        df.columns = tickers if isinstance(tickers, list) else [tickers]

    df = drop_empty_rows(df)
    validate_not_empty(df, context="downloaded prices")

    df, _ = filter_by_coverage(df, min_coverage=min_data_coverage)

    if len(df.columns) == 0:
        raise ValueError(
            "No tickers passed the minimum data coverage filter. "
            "Try lowering min_data_coverage or expanding the date range."
        )

    df = fill_missing_prices(df)
    df = df.dropna(axis=1, how="all")
    validate_not_empty(df, context="prices after fill")

    return df


def load_prices_from_config() -> pd.DataFrame:
    """Load prices using parameters defined in config/pipeline.yaml.

    Reads the following keys from the 'data' section:
        - tickers: list of ticker symbols to download
        - start_date: start of the historical window
        - end_date: end of the historical window
    And from the 'asset_selection' section:
        - min_data_coverage: minimum non-null fraction per ticker (default 0.5)

    Returns:
        DataFrame of adjusted closing prices as returned by load_prices().

    Raises:
        FileNotFoundError: If config/pipeline.yaml does not exist.
        KeyError: If required config keys are missing.
        ValueError: If no valid tickers are found after filtering.

    Example:
        >>> prices = load_prices_from_config()
        >>> isinstance(prices, pd.DataFrame)
        True
    """
    cfg = get_config("pipeline")
    data_cfg = cfg["data"]
    sel_cfg = cfg.get("asset_selection", {})

    return load_prices(
        tickers=data_cfg["tickers"],
        start=data_cfg["start_date"],
        end=data_cfg["end_date"],
        min_data_coverage=sel_cfg.get("min_data_coverage", 0.5),
    )
