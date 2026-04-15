"""
Responsible for constructing supervised learning feature matrices from return time series.

Generates lag features (t-1 … t-lag_window) for each asset, aligns targets,
and splits data into train/test windows for walk-forward validation.
"""
from __future__ import annotations

import pandas as pd


def build_lag_features(
    returns: pd.DataFrame,
    lag_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a supervised feature matrix from return time series using lag features.

    For each asset, creates lag_window lag features (t-1, t-2, ..., t-lag_window).
    The feature matrix X has columns named "{ticker}_lag_{k}" for k in 1..lag_window.
    The target matrix y contains the contemporaneous return at each period, aligned to X.

    Args:
        returns: DataFrame of asset returns indexed by date. Rows are periods,
            columns are asset tickers.
        lag_window: Number of past periods to use as features. Must be >= 1.

    Returns:
        Tuple of (X, y) where:
            X: Feature DataFrame of shape (T - lag_window, n_assets * lag_window).
            y: Target DataFrame of shape (T - lag_window, n_assets).
            Both share the same DatetimeIndex starting at returns.index[lag_window].

    Raises:
        ValueError: If lag_window < 1 or if returns has fewer than lag_window + 2 rows.
    """
    if lag_window < 1:
        raise ValueError(f"lag_window must be >= 1, got {lag_window}.")
    if len(returns) < lag_window + 2:
        raise ValueError(
            f"returns has {len(returns)} rows but lag_window={lag_window} requires "
            f"at least {lag_window + 2} rows to produce at least one sample."
        )

    lag_frames: list[pd.DataFrame] = []
    for ticker in returns.columns:
        for k in range(1, lag_window + 1):
            col_name = f"{ticker}_lag_{k}"
            lag_frames.append(returns[ticker].shift(k).rename(col_name))

    X_raw = pd.concat(lag_frames, axis=1)
    X = X_raw.dropna()
    y = returns.loc[X.index]

    return X, y


def make_walk_forward_splits(
    n_samples: int,
    train_ratio: float,
    min_train_size: int,
) -> list[dict[str, int]]:
    """Generate expanding-window train/test index splits for walk-forward validation.

    The initial training window uses floor(n_samples * train_ratio) samples or
    min_train_size samples, whichever is larger. Each subsequent split expands
    the training set by one period and tests on the immediately following period.

    Args:
        n_samples: Total number of samples in the feature matrix.
        train_ratio: Fraction of data for the initial training window. Must be in (0, 1).
        min_train_size: Minimum number of training samples required per split.

    Returns:
        List of dicts, each with keys:
            'train_end': Exclusive upper bound of training indices (train uses [0:train_end]).
            'test_idx': Integer positional index of the test observation.

    Raises:
        ValueError: If train_ratio is not in (0, 1) or if no valid splits can be generated.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    initial_train_end = max(min_train_size, int(n_samples * train_ratio))

    if initial_train_end >= n_samples:
        raise ValueError(
            f"initial_train_end={initial_train_end} leaves no test samples "
            f"(n_samples={n_samples}). Reduce min_train_size or train_ratio."
        )

    splits = []
    for test_idx in range(initial_train_end, n_samples):
        splits.append({"train_end": test_idx, "test_idx": test_idx})

    return splits
