"""Unit tests for src/models/metrics.compute_model_metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.metrics import compute_model_metrics


def test_ic_one_for_identical_predictions():
    # Cross-sectional Pearson correlation is 1.0 when pred = a * true + b (a > 0).
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal((20, 8))
    y_pred = y_true * 2.0 + 0.01
    result = compute_model_metrics(y_pred, y_true)
    assert result["ic"] == pytest.approx(1.0, abs=1e-9)


def test_ic_near_zero_for_random_predictions():
    # IC should be near 0 when predictions carry no information about realized returns.
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal((200, 10))
    y_pred = rng.standard_normal((200, 10))
    result = compute_model_metrics(y_pred, y_true)
    assert abs(result["ic"]) < 0.15


def test_hit_rate_one_for_correct_directions():
    # Hit Rate = 1.0 when every predicted sign matches the realized sign.
    y_true = np.array([[0.1, -0.2, 0.3, -0.05, 0.15]] * 10, dtype=float)
    y_pred = np.array([[0.05, -0.1, 0.2, -0.03, 0.08]] * 10, dtype=float)
    result = compute_model_metrics(y_pred, y_true)
    assert result["hit_rate"] == pytest.approx(1.0)


def test_hit_rate_near_half_for_random_predictions():
    # A model with no directional skill should hit ~50% of directions.
    rng = np.random.default_rng(7)
    y_true = rng.standard_normal((300, 10))
    y_pred = rng.standard_normal((300, 10))
    result = compute_model_metrics(y_pred, y_true)
    assert 0.40 < result["hit_rate"] < 0.60


def test_icir_exceeds_ic_when_signal_is_consistent():
    # ICIR = mean_IC / std_IC. When IC is consistently high (small std_IC),
    # ICIR >> mean_IC. This holds whenever std_IC < 1, which is guaranteed
    # for a stable positive signal (IC values tightly clustered near 1).
    rng = np.random.default_rng(1)
    n_periods, n_assets = 40, 8
    y_true = rng.standard_normal((n_periods, n_assets))
    # Near-perfect predictions: IC per period close to 1.0 with negligible variance.
    y_pred = y_true + rng.standard_normal((n_periods, n_assets)) * 0.05
    result = compute_model_metrics(y_pred, y_true)
    assert result["ic"] > 0
    assert result["icir"] > result["ic"]


def test_output_keys_are_complete():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((10, 4))
    result = compute_model_metrics(y, y)
    assert set(result.keys()) == {"ic", "icir", "hit_rate", "spearman_ic", "mae", "mse", "r2"}


def test_mae_zero_for_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((10, 4))
    result = compute_model_metrics(y, y)
    assert result["mae"] == pytest.approx(0.0, abs=1e-12)


def test_r2_one_for_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.standard_normal((10, 4))
    result = compute_model_metrics(y, y)
    assert result["r2"] == pytest.approx(1.0, abs=1e-9)


def test_shape_mismatch_raises_value_error():
    y_pred = np.zeros((5, 3))
    y_true = np.zeros((5, 4))
    with pytest.raises(ValueError, match="same shape"):
        compute_model_metrics(y_pred, y_true)


def test_spearman_ic_one_for_identical_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal((20, 8))
    y_pred = y_true * 3.0 - 0.5  # monotone transform preserves ranks
    result = compute_model_metrics(y_pred, y_true)
    assert result["spearman_ic"] == pytest.approx(1.0, abs=1e-9)


def test_single_period_icir_is_nan():
    # ICIR is undefined with only one period (std of a single value is nan).
    rng = np.random.default_rng(0)
    y = rng.standard_normal((1, 5))
    result = compute_model_metrics(y, y)
    assert np.isnan(result["icir"])
