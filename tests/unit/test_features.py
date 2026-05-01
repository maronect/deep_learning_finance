"""
Unit tests for src/features/lag_features.py.

Validates that lag feature matrices are constructed without data leakage,
have the correct shape, and respect the walk-forward split boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import build_lag_features, make_walk_forward_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _returns(n_rows: int = 50, n_cols: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_rows, freq="ME")
    data = rng.normal(0.01, 0.04, size=(n_rows, n_cols))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_cols)])


# ---------------------------------------------------------------------------
# build_lag_features
# ---------------------------------------------------------------------------

class TestBuildLagFeatures:
    def test_output_shapes(self) -> None:
        n, n_cols, lag = 50, 3, 6
        ret = _returns(n_rows=n, n_cols=n_cols)
        X, y = build_lag_features(ret, lag_window=lag)
        assert X.shape == (n - lag, n_cols * lag)
        assert y.shape == (n - lag, n_cols)

    def test_X_and_y_share_index(self) -> None:
        ret = _returns()
        X, y = build_lag_features(ret, lag_window=4)
        assert list(X.index) == list(y.index)

    def test_index_starts_after_lag_window(self) -> None:
        ret = _returns(n_rows=40)
        lag = 5
        X, _ = build_lag_features(ret, lag_window=lag)
        assert X.index[0] == ret.index[lag]

    def test_no_nan_in_output(self) -> None:
        ret = _returns()
        X, y = build_lag_features(ret, lag_window=3)
        assert not X.isna().any().any()
        assert not y.isna().any().any()

    def test_column_names_format(self) -> None:
        ret = _returns(n_cols=2)
        X, _ = build_lag_features(ret, lag_window=2)
        assert "A0_lag_1" in X.columns
        assert "A0_lag_2" in X.columns
        assert "A1_lag_1" in X.columns
        assert "A1_lag_2" in X.columns

    def test_no_lookahead_leakage(self) -> None:
        """The lag-1 feature at time t must equal the return at time t-1."""
        ret = _returns(n_rows=20, n_cols=1)
        X, _ = build_lag_features(ret, lag_window=1)
        # X["A0_lag_1"] at position i should equal ret["A0"] at position (lag + i - 1)
        for i, date in enumerate(X.index):
            expected = ret["A0"].iloc[i]  # ret row before the feature row
            actual = X["A0_lag_1"].iloc[i]
            assert abs(actual - expected) < 1e-12, (
                f"Look-ahead leak at {date}: expected {expected}, got {actual}"
            )

    def test_lag_window_one_valid(self) -> None:
        ret = _returns(n_rows=10, n_cols=2)
        X, y = build_lag_features(ret, lag_window=1)
        assert X.shape[0] == 9
        assert y.shape[0] == 9

    def test_lag_window_zero_raises(self) -> None:
        ret = _returns()
        with pytest.raises(ValueError, match="lag_window must be >= 1"):
            build_lag_features(ret, lag_window=0)

    def test_too_few_rows_raises(self) -> None:
        ret = _returns(n_rows=5)
        with pytest.raises(ValueError):
            build_lag_features(ret, lag_window=10)


# ---------------------------------------------------------------------------
# make_walk_forward_splits
# ---------------------------------------------------------------------------

class TestMakeWalkForwardSplits:
    def test_split_count(self) -> None:
        splits = make_walk_forward_splits(n_samples=50, train_ratio=0.7, min_train_size=10)
        # test indices run from initial_train_end to n_samples - 1
        initial_train_end = max(10, int(50 * 0.7))  # max(10, 35) = 35
        expected = 50 - initial_train_end
        assert len(splits) == expected

    def test_each_split_has_required_keys(self) -> None:
        splits = make_walk_forward_splits(n_samples=30, train_ratio=0.6, min_train_size=5)
        for s in splits:
            assert "train_end" in s
            assert "test_idx" in s

    def test_test_idx_equals_train_end(self) -> None:
        splits = make_walk_forward_splits(n_samples=30, train_ratio=0.6, min_train_size=5)
        for s in splits:
            assert s["test_idx"] == s["train_end"]

    def test_no_overlap_between_train_and_test(self) -> None:
        splits = make_walk_forward_splits(n_samples=40, train_ratio=0.6, min_train_size=8)
        for s in splits:
            assert s["test_idx"] >= s["train_end"]

    def test_train_window_expands_by_one(self) -> None:
        splits = make_walk_forward_splits(n_samples=20, train_ratio=0.5, min_train_size=5)
        for i in range(1, len(splits)):
            assert splits[i]["train_end"] == splits[i - 1]["train_end"] + 1

    def test_min_train_size_respected(self) -> None:
        min_size = 15
        splits = make_walk_forward_splits(n_samples=30, train_ratio=0.3, min_train_size=min_size)
        assert splits[0]["train_end"] >= min_size

    def test_invalid_train_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            make_walk_forward_splits(n_samples=50, train_ratio=1.5, min_train_size=5)

    def test_train_ratio_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            make_walk_forward_splits(n_samples=50, train_ratio=0.0, min_train_size=5)

    def test_no_test_samples_raises(self) -> None:
        # If min_train_size >= n_samples, no test samples exist
        with pytest.raises(ValueError):
            make_walk_forward_splits(n_samples=10, train_ratio=0.5, min_train_size=15)

    def test_all_test_indices_in_range(self) -> None:
        n = 60
        splits = make_walk_forward_splits(n_samples=n, train_ratio=0.7, min_train_size=10)
        for s in splits:
            assert 0 <= s["test_idx"] < n
