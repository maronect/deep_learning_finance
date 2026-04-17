"""
Unit tests for src/features/returns.py.

Validates correctness of return calculations at daily, weekly, and monthly
frequencies, including annualization and edge cases (NaN, zero prices).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.returns import ajustar_risk_free, compute_returns, converter_periodo


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------

class TestComputeReturns:
    def _daily_prices(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        return pd.DataFrame({"A": [100.0, 110.0, 99.0, 108.9, 108.9]}, index=dates)

    def test_daily_shape_drops_first_row(self) -> None:
        prices = self._daily_prices()
        result = compute_returns(prices, freq="daily")
        assert result.shape[0] == len(prices) - 1

    def test_daily_first_value(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]},
                               index=pd.date_range("2020-01-01", periods=3))
        result = compute_returns(prices, freq="daily")
        assert abs(result["A"].iloc[0] - 0.1) < 1e-10

    def test_daily_no_nan_output(self) -> None:
        prices = self._daily_prices()
        result = compute_returns(prices, freq="daily")
        assert not result.isna().any().any()

    def test_monthly_shape(self) -> None:
        dates = pd.date_range("2015-01-01", periods=400, freq="B")
        rng = np.random.default_rng(7)
        prices = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 400))},
            index=dates,
        )
        result = compute_returns(prices, freq="monthly")
        assert result.shape[1] == 1
        assert result.shape[0] > 0
        assert not result.isna().any().any()

    def test_weekly_produces_fewer_rows_than_daily(self) -> None:
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        rng = np.random.default_rng(3)
        prices = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 252))},
            index=dates,
        )
        daily = compute_returns(prices, freq="daily")
        weekly = compute_returns(prices, freq="weekly")
        assert len(weekly) < len(daily)

    def test_invalid_freq_raises(self) -> None:
        prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]},
                               index=pd.date_range("2020-01-01", periods=3))
        with pytest.raises(ValueError, match="Invalid frequency"):
            compute_returns(prices, freq="quarterly")  # type: ignore[arg-type]

    def test_multicol_preserves_columns(self) -> None:
        dates = pd.date_range("2020-01-01", periods=10)
        prices = pd.DataFrame(
            {"X": range(100, 110), "Y": range(200, 210)},
            index=dates,
            dtype=float,
        )
        result = compute_returns(prices, freq="daily")
        assert list(result.columns) == ["X", "Y"]

    def test_equal_prices_yield_zero_returns(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"A": [50.0] * 5}, index=dates)
        result = compute_returns(prices, freq="daily")
        assert (result["A"] == 0.0).all()


# ---------------------------------------------------------------------------
# ajustar_risk_free
# ---------------------------------------------------------------------------

class TestAjustarRiskFree:
    def test_daily_round_trip(self) -> None:
        rf = ajustar_risk_free(0.15, freq="daily")
        assert abs((1 + rf) ** 252 - 1.15) < 1e-6

    def test_weekly_round_trip(self) -> None:
        rf = ajustar_risk_free(0.15, freq="weekly")
        assert abs((1 + rf) ** 52 - 1.15) < 1e-6

    def test_monthly_round_trip(self) -> None:
        rf = ajustar_risk_free(0.15, freq="monthly")
        assert abs((1 + rf) ** 12 - 1.15) < 1e-9

    def test_daily_smaller_than_monthly(self) -> None:
        rf_daily = ajustar_risk_free(0.15, freq="daily")
        rf_monthly = ajustar_risk_free(0.15, freq="monthly")
        assert rf_daily < rf_monthly

    def test_zero_rate_yields_zero(self) -> None:
        assert ajustar_risk_free(0.0, freq="daily") == 0.0
        assert ajustar_risk_free(0.0, freq="monthly") == 0.0

    def test_output_is_positive_for_positive_rate(self) -> None:
        assert ajustar_risk_free(0.10, freq="daily") > 0
        assert ajustar_risk_free(0.10, freq="monthly") > 0


# ---------------------------------------------------------------------------
# converter_periodo
# ---------------------------------------------------------------------------

class TestConverterPeriodo:
    def test_vol_annualization(self) -> None:
        period_vol = 0.02
        _, ann_vol = converter_periodo(0.001, period_vol, dias=252)
        assert abs(ann_vol - period_vol * np.sqrt(252)) < 1e-12

    def test_return_compounding(self) -> None:
        period_ret = 0.01
        ann_ret, _ = converter_periodo(period_ret, 0.02, dias=12)
        expected = (1 + period_ret) ** 12 - 1
        assert abs(ann_ret - expected) < 1e-12

    def test_annual_values_exceed_period_values(self) -> None:
        ann_ret, ann_vol = converter_periodo(0.005, 0.015, dias=252)
        assert ann_ret > 0.005
        assert ann_vol > 0.015
