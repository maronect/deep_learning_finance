"""
Unit tests for src/optimization/sharpe.py.

Validates that Sharpe Ratio maximization returns feasible weights, respects
bounds, and produces a higher Sharpe than an equally-weighted portfolio.
"""
from __future__ import annotations

import numpy as np

from src.optimization.sharpe import maximize_sharpe
from src.optimization.markowitz import portfolio_return, portfolio_volatility


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _problem(n: int = 4, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.005, 0.025, n)
    A = rng.standard_normal((n, n))
    cov = A @ A.T + np.eye(n) * 0.01
    return mu, cov


def _sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    ret = portfolio_return(w, mu)
    vol = portfolio_volatility(w, cov)
    return (ret - rf) / vol if vol > 0 else 0.0


# ---------------------------------------------------------------------------
# maximize_sharpe
# ---------------------------------------------------------------------------

class TestMaximizeSharpe:
    def test_returns_array_on_success(self) -> None:
        mu, cov = _problem()
        result = maximize_sharpe(mu, cov, risk_free_rate=0.001)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_weights_sum_to_one(self) -> None:
        mu, cov = _problem()
        w = maximize_sharpe(mu, cov, risk_free_rate=0.001)
        assert w is not None
        assert abs(w.sum() - 1.0) < 1e-4

    def test_no_short_selling(self) -> None:
        mu, cov = _problem()
        w = maximize_sharpe(mu, cov, risk_free_rate=0.001)
        assert w is not None
        assert (w >= -1e-6).all()

    def test_correct_length(self) -> None:
        for n in (2, 5, 8):
            mu, cov = _problem(n=n)
            w = maximize_sharpe(mu, cov)
            assert w is not None
            assert len(w) == n

    def test_higher_sharpe_than_equal_weights(self) -> None:
        mu, cov = _problem(n=5, seed=13)
        rf = 0.001
        w_opt = maximize_sharpe(mu, cov, risk_free_rate=rf)
        assert w_opt is not None
        w_eq = np.full(5, 0.2)
        sharpe_opt = _sharpe(w_opt, mu, cov, rf)
        sharpe_eq = _sharpe(w_eq, mu, cov, rf)
        assert sharpe_opt >= sharpe_eq - 1e-5

    def test_zero_rf_still_feasible(self) -> None:
        mu, cov = _problem()
        w = maximize_sharpe(mu, cov, risk_free_rate=0.0)
        assert w is not None
        assert abs(w.sum() - 1.0) < 1e-4

    def test_single_asset(self) -> None:
        mu = np.array([0.01])
        cov = np.array([[0.0025]])
        w = maximize_sharpe(mu, cov, risk_free_rate=0.0)
        assert w is not None
        assert abs(w[0] - 1.0) < 1e-4

    def test_concentrated_when_one_dominates(self) -> None:
        # Asset 0 has much better Sharpe — optimizer should concentrate weight there
        mu = np.array([0.05, 0.001, 0.001])
        A = np.eye(3) * 0.02
        cov = A @ A.T
        rf = 0.0
        w = maximize_sharpe(mu, cov, risk_free_rate=rf)
        assert w is not None
        assert w[0] > 0.5, f"Expected weight on best asset > 0.5, got {w[0]:.4f}"
