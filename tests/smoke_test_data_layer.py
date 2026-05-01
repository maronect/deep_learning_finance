"""
Smoke tests for the data layer (ingestion + features).

Run with:
    python tests/smoke_test_data_layer.py

All tests use synthetic in-memory data — no network calls required.
Functions that require Yahoo Finance (load_prices, run_data_ingestion)
are marked and skipped here; they belong in integration tests.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n_rows: int = 60, n_cols: int = 10, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic daily price DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_rows, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=(n_rows, n_cols)), axis=0)
    cols = [f"ASSET{i}" for i in range(n_cols)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def _make_return_df(n_rows: int = 60, n_cols: int = 10, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic monthly return DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_rows, freq="ME")
    data = rng.normal(0.01, 0.05, size=(n_rows, n_cols))
    cols = [f"ASSET{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _pass(name: str) -> None:
    print(f"  PASS  {name}")


def _fail(name: str, exc: Exception) -> None:
    print(f"  FAIL  {name}: {exc}")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

def test_drop_empty_rows() -> None:
    from src.ingestion.validators import drop_empty_rows

    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [2.0, None, 4.0]})
    result = drop_empty_rows(df)
    assert len(result) == 2, f"Expected 2 rows, got {len(result)}"
    _pass("drop_empty_rows")


def test_filter_by_coverage_removes_sparse_column() -> None:
    from src.ingestion.validators import filter_by_coverage

    df = pd.DataFrame({
        "GOOD": [1.0, 2.0, 3.0, 4.0, 5.0],
        "BAD":  [None, None, None, None, 1.0],  # 20% coverage
    })
    valid_df, removed = filter_by_coverage(df, min_coverage=0.5)
    assert "GOOD" in valid_df.columns
    assert "BAD" not in valid_df.columns
    assert "BAD" in removed
    _pass("filter_by_coverage — removes sparse column")


def test_filter_by_coverage_keeps_all_valid() -> None:
    from src.ingestion.validators import filter_by_coverage

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    valid_df, removed = filter_by_coverage(df, min_coverage=0.5)
    assert list(valid_df.columns) == ["A", "B"]
    assert removed == []
    _pass("filter_by_coverage — keeps all valid")


def test_fill_missing_prices() -> None:
    from src.ingestion.validators import fill_missing_prices

    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, None]})
    result = fill_missing_prices(df)
    assert not result.isna().any().any(), "NaNs remain after fill"
    _pass("fill_missing_prices")


def test_validate_not_empty_raises_on_empty_df() -> None:
    from src.ingestion.validators import validate_not_empty

    try:
        validate_not_empty(pd.DataFrame(), context="test")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _pass("validate_not_empty — raises on empty DataFrame")


def test_validate_not_empty_passes_on_valid_df() -> None:
    from src.ingestion.validators import validate_not_empty

    validate_not_empty(pd.DataFrame({"A": [1.0, 2.0]}))
    _pass("validate_not_empty — passes on valid DataFrame")


# ---------------------------------------------------------------------------
# features.returns
# ---------------------------------------------------------------------------

def test_compute_returns_daily_shape() -> None:
    from src.features.returns import compute_returns

    prices = _make_price_df(n_rows=20, n_cols=3)
    result = compute_returns(prices, freq="daily")
    assert result.shape == (19, 3), f"Unexpected shape: {result.shape}"
    _pass("compute_returns — daily shape")


def test_compute_returns_daily_values() -> None:
    from src.features.returns import compute_returns

    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
    result = compute_returns(prices, freq="daily")
    assert abs(result["A"].iloc[0] - 0.1) < 1e-9, "First daily return should be 0.1"
    assert abs(result["A"].iloc[1] - 0.1) < 1e-9, "Second daily return should be 0.1"
    _pass("compute_returns — daily values")


def test_compute_returns_invalid_freq() -> None:
    from src.features.returns import compute_returns

    try:
        compute_returns(pd.DataFrame({"A": [1.0, 2.0]}), freq="yearly")  # type: ignore[arg-type]
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _pass("compute_returns — rejects invalid frequency")


def test_ajustar_risk_free_daily() -> None:
    from src.features.returns import ajustar_risk_free

    rf_daily = ajustar_risk_free(0.15, freq="daily")
    assert 0 < rf_daily < 0.01, f"Daily rf out of expected range: {rf_daily}"
    # Verify round-trip: (1 + rf_daily)^252 ≈ 1.15
    assert abs((1 + rf_daily) ** 252 - 1.15) < 1e-6
    _pass("ajustar_risk_free — daily")


def test_ajustar_risk_free_monthly() -> None:
    from src.features.returns import ajustar_risk_free

    rf_monthly = ajustar_risk_free(0.15, freq="monthly")
    assert 0 < rf_monthly < 0.02
    assert abs((1 + rf_monthly) ** 12 - 1.15) < 1e-9
    _pass("ajustar_risk_free — monthly")


def test_converter_periodo() -> None:
    from src.features.returns import converter_periodo

    daily_ret, daily_vol = 0.001, 0.01
    annual_ret, annual_vol = converter_periodo(daily_ret, daily_vol, dias=252)
    assert annual_ret > daily_ret, "Annual return should exceed daily return"
    assert annual_vol > daily_vol, "Annual vol should exceed daily vol"
    assert abs(annual_vol - daily_vol * np.sqrt(252)) < 1e-12
    _pass("converter_periodo")


# ---------------------------------------------------------------------------
# features.asset_selection  (pure logic, no network)
# ---------------------------------------------------------------------------

def test_get_brazilian_stocks_universe() -> None:
    from src.features.asset_selection import get_brazilian_stocks_universe

    universe = get_brazilian_stocks_universe()
    assert len(universe) > 20, f"Universe too small: {len(universe)}"
    assert "PETR4.SA" in universe
    assert "VALE3.SA" in universe
    assert "ITUB4.SA" in universe
    # No duplicates
    assert len(universe) == len(set(universe)), "Universe contains duplicates"
    _pass("get_brazilian_stocks_universe")


def test_select_by_sum_abs_correlation() -> None:
    from src.features.asset_selection import _select_by_sum_abs_correlation

    returns = _make_return_df(n_rows=60, n_cols=10)
    result = _select_by_sum_abs_correlation(returns, n_assets=5)
    assert len(result) == 5
    assert len(set(result)) == 5, "Duplicates in result"
    assert all(t in returns.columns for t in result)
    _pass("_select_by_sum_abs_correlation")


def test_select_by_min_max_correlation() -> None:
    from src.features.asset_selection import _select_by_min_max_correlation

    returns = _make_return_df(n_rows=60, n_cols=10)
    result = _select_by_min_max_correlation(returns, n_assets=4)
    assert len(result) == 4
    assert len(set(result)) == 4
    assert all(t in returns.columns for t in result)
    _pass("_select_by_min_max_correlation")


def test_select_lowest_corr_pairs() -> None:
    from src.features.asset_selection import _select_lowest_corr_pairs

    returns = _make_return_df(n_rows=60, n_cols=10)
    result = _select_lowest_corr_pairs(returns, n_pairs=3)
    assert len(result) == 6, f"Expected 6 assets (3 pairs), got {len(result)}"
    assert len(set(result)) == 6, "Pairs contain duplicate tickers"
    _pass("_select_lowest_corr_pairs")


def test_select_assets_dispatches_correctly() -> None:
    """Test that select_assets calls the right strategy without a network call."""
    from src.features.asset_selection import select_assets

    prices = _make_price_df(n_rows=120, n_cols=10)
    prices.index = pd.date_range("2015-01-01", periods=120, freq="ME")

    for method in ("sum_abs_correlation", "min_max_correlation", "lowest_corr_pairs"):
        result = select_assets(
            start_date="2015-01-01",
            end_date="2024-12-31",
            method=method,
            n_assets=6,
            return_freq="monthly",
            prices=prices,  # pre-supply prices, no download
        )
        assert len(result) == 6, f"{method}: expected 6, got {len(result)}"
    _pass("select_assets — dispatches all strategies correctly")


def test_select_assets_invalid_method_raises() -> None:
    from src.features.asset_selection import select_assets

    prices = _make_price_df(n_rows=60, n_cols=5)
    try:
        select_assets("2020-01-01", "2023-12-31", method="unknown", prices=prices)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    _pass("select_assets — raises on unknown method")


def test_get_correlation_matrix_shape() -> None:
    from src.features.asset_selection import get_correlation_matrix

    returns = _make_return_df(n_rows=50, n_cols=6)
    corr = get_correlation_matrix(returns, selected_assets=["ASSET0", "ASSET1", "ASSET2"])
    assert corr.shape == (3, 3)
    assert all(abs(corr.iloc[i, i] - 1.0) < 1e-10 for i in range(3))
    _pass("get_correlation_matrix — shape and diagonal")


# ---------------------------------------------------------------------------
# ingestion.__init__ — DataLayerResult construction
# ---------------------------------------------------------------------------

def test_data_layer_result_fields() -> None:
    from src.ingestion import DataLayerResult

    prices = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    returns = pd.DataFrame({"A": [0.1], "B": [0.05]})
    result = DataLayerResult(
        prices=prices,
        returns=returns,
        selected_assets=["A", "B"],
        config={"data": {"frequency": "monthly"}},
    )
    assert result.selected_assets == ["A", "B"]
    assert "A" in result.prices.columns
    assert result.config["data"]["frequency"] == "monthly"
    _pass("DataLayerResult — field access")


# ---------------------------------------------------------------------------
# utils.config_loader
# ---------------------------------------------------------------------------

def test_config_loader_pipeline() -> None:
    from src.utils.config_loader import get_config

    cfg = get_config("pipeline")
    assert "data" in cfg
    assert "asset_selection" in cfg
    assert "optimization" in cfg
    assert "artifacts" in cfg
    _pass("get_config — pipeline.yaml loads correctly")


def test_config_loader_missing_file_raises() -> None:
    from src.utils.config_loader import get_config

    try:
        get_config("nonexistent_config_xyz")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    _pass("get_config — raises on missing file")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all() -> None:
    tests = [
        # validators
        test_drop_empty_rows,
        test_filter_by_coverage_removes_sparse_column,
        test_filter_by_coverage_keeps_all_valid,
        test_fill_missing_prices,
        test_validate_not_empty_raises_on_empty_df,
        test_validate_not_empty_passes_on_valid_df,
        # returns
        test_compute_returns_daily_shape,
        test_compute_returns_daily_values,
        test_compute_returns_invalid_freq,
        test_ajustar_risk_free_daily,
        test_ajustar_risk_free_monthly,
        test_converter_periodo,
        # asset selection
        test_get_brazilian_stocks_universe,
        test_select_by_sum_abs_correlation,
        test_select_by_min_max_correlation,
        test_select_lowest_corr_pairs,
        test_select_assets_dispatches_correctly,
        test_select_assets_invalid_method_raises,
        test_get_correlation_matrix_shape,
        # ingestion
        test_data_layer_result_fields,
        # config loader
        test_config_loader_pipeline,
        test_config_loader_missing_file_raises,
    ]

    print(f"\nRunning {len(tests)} smoke tests for the data layer...\n")
    for test_fn in tests:
        try:
            test_fn()
        except SystemExit:
            raise
        except Exception as exc:
            _fail(test_fn.__name__, exc)

    print(f"\nAll {len(tests)} tests passed.\n")


if __name__ == "__main__":
    run_all()
