"""
Unit tests for src/optimization/markowitz.py.

Validates portfolio return/volatility calculations, weight constraint enforcement
(no short selling, sum-to-one), and covariance matrix construction.
"""
from __future__ import annotations

import numpy as np

from src.optimization.markowitz import (
    minimize_volatility,
    portfolio_return,
    portfolio_volatility,
    solve_markowitz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_cov(n: int) -> np.ndarray:
    """Return an n×n identity covariance matrix (uncorrelated, unit variance)."""
    return np.eye(n)


def _uniform_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# portfolio_return
# ---------------------------------------------------------------------------

class TestPortfolioReturn:
    def test_equal_weights_equal_assets(self) -> None:
        mu = np.array([0.1, 0.1, 0.1])
        w = _uniform_weights(3)
        assert abs(portfolio_return(w, mu) - 0.1) < 1e-12

    def test_concentrated_portfolio(self) -> None:
        mu = np.array([0.05, 0.20, 0.10])
        w = np.array([0.0, 1.0, 0.0])
        assert abs(portfolio_return(w, mu) - 0.20) < 1e-12

    def test_linearity(self) -> None:
        mu = np.array([0.08, 0.12])
        w = np.array([0.4, 0.6])
        expected = 0.4 * 0.08 + 0.6 * 0.12
        assert abs(portfolio_return(w, mu) - expected) < 1e-12


# ---------------------------------------------------------------------------
# portfolio_volatility
# ---------------------------------------------------------------------------

class TestPortfolioVolatility:
    def test_identity_cov_equal_weights(self) -> None:
        n = 4
        w = _uniform_weights(n)
        cov = _identity_cov(n)
        vol = portfolio_volatility(w, cov)
        # With identity cov and equal weights: vol = sqrt(sum(w_i^2)) = 1/sqrt(n)
        assert abs(vol - 1 / np.sqrt(n)) < 1e-10

    def test_single_asset(self) -> None:
        w = np.array([1.0])
        cov = np.array([[0.04]])
        assert abs(portfolio_volatility(w, cov) - 0.2) < 1e-10

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(5)
        A = rng.standard_normal((5, 5))
        cov = A @ A.T  # guaranteed PSD
        w = _uniform_weights(5)
        assert portfolio_volatility(w, cov) >= 0

    def test_perfect_correlation_increases_vol(self) -> None:
        # Two perfectly correlated assets — portfolio vol = sum(w_i * sigma_i)
        sigma = 0.10
        cov = np.array([[sigma**2, sigma**2], [sigma**2, sigma**2]])
        w = np.array([0.5, 0.5])
        vol = portfolio_volatility(w, cov)
        assert abs(vol - sigma) < 1e-10


# ---------------------------------------------------------------------------
# minimize_volatility
# ---------------------------------------------------------------------------

class TestMinimizeVolatility:
    def _run(self, n: int = 4) -> np.ndarray:
        rng = np.random.default_rng(9)
        mu = rng.uniform(0.005, 0.02, n)
        A = rng.standard_normal((n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        return minimize_volatility(mu, cov)

    def test_returns_array(self) -> None:
        result = self._run()
        assert isinstance(result, np.ndarray)

    def test_weights_sum_to_one(self) -> None:
        result = self._run()
        assert abs(result.sum() - 1.0) < 1e-5

    def test_no_short_selling(self) -> None:
        result = self._run()
        assert (result >= -1e-8).all()

    def test_lower_vol_than_equal_weights(self) -> None:
        n = 4
        rng = np.random.default_rng(11)
        mu = rng.uniform(0.005, 0.02, n)
        A = rng.standard_normal((n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        opt_w = minimize_volatility(mu, cov)
        eq_w = _uniform_weights(n)
        opt_vol = portfolio_volatility(opt_w, cov)
        eq_vol = portfolio_volatility(eq_w, cov)
        assert opt_vol <= eq_vol + 1e-5


# ---------------------------------------------------------------------------
# solve_markowitz
# ---------------------------------------------------------------------------

class TestSolveMarkowitz:
    def _problem(self, n: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.005, 0.025, n)
        A = rng.standard_normal((n, n))
        cov = A @ A.T + np.eye(n) * 0.01
        return mu, cov

    def test_weights_sum_to_one(self) -> None:
        mu, cov = self._problem()
        for lamb in (0.0, 0.3, 0.5, 0.7, 1.0):
            w = solve_markowitz(mu, cov, lamb=lamb)
            assert abs(w.sum() - 1.0) < 1e-4, f"lambda={lamb}: weights sum to {w.sum()}"

    def test_no_short_selling(self) -> None:
        mu, cov = self._problem()
        for lamb in (0.0, 0.5, 1.0):
            w = solve_markowitz(mu, cov, lamb=lamb)
            assert (w >= -1e-6).all(), f"lambda={lamb}: negative weight found"

    def test_lambda_0_favors_return(self) -> None:
        # lambda=0 means full weight on return maximization
        mu, cov = self._problem()
        w_ret = solve_markowitz(mu, cov, lamb=0.0)
        w_eq = _uniform_weights(len(mu))
        assert portfolio_return(w_ret, mu) >= portfolio_return(w_eq, mu) - 1e-5

    def test_lambda_1_favors_low_risk(self) -> None:
        # lambda=1 means full weight on risk minimization
        mu, cov = self._problem()
        w_risk = solve_markowitz(mu, cov, lamb=1.0)
        w_eq = _uniform_weights(len(mu))
        assert portfolio_volatility(w_risk, cov) <= portfolio_volatility(w_eq, cov) + 1e-5

    def test_max_weight_constraint_respected(self) -> None:
        mu, cov = self._problem()
        max_w = 0.4
        w = solve_markowitz(mu, cov, lamb=0.5, max_weight=max_w)
        assert (w <= max_w + 1e-5).all()
        assert abs(w.sum() - 1.0) < 1e-4

    def test_correct_output_length(self) -> None:
        for n in (2, 5, 10):
            mu, cov = self._problem(n=n)
            w = solve_markowitz(mu, cov, lamb=0.5)
            assert len(w) == n
