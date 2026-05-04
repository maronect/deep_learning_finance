"""
Finance-specific model evaluation metrics for return prediction models.

Standard ML metrics (MSE, R²) measure average prediction accuracy but ignore
properties that matter in portfolio construction: cross-sectional ranking skill,
directional accuracy, and signal consistency over time. The metrics here are
standard in quantitative equity research.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_model_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Compute finance-specific and standard ML metrics for return predictions.

    All cross-sectional metrics (IC, Spearman IC) are computed per period across
    assets, then averaged over time. This measures whether the model correctly ranks
    assets within each rebalancing period, which is what matters for portfolio
    construction — not the absolute level of predicted returns.

    Args:
        y_pred: Predicted returns, shape (n_periods, n_assets).
        y_true: Realized returns, shape (n_periods, n_assets).
            Must share shape with y_pred.

    Returns:
        Dict with keys: ic, icir, hit_rate, spearman_ic, mae, mse, r2.

    Raises:
        ValueError: If y_pred and y_true do not share the same shape.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred and y_true must have the same shape. "
            f"Got {y_pred.shape} vs {y_true.shape}."
        )

    n_periods = y_pred.shape[0]

    ic_series = _pearson_ic_series(y_pred, y_true)
    spearman_series = _spearman_ic_series(y_pred, y_true)

    ic_mean = float(np.nanmean(ic_series))

    # ICIR requires at least 2 periods to compute a meaningful std.
    if n_periods > 1:
        ic_std = float(np.nanstd(ic_series, ddof=1))
        icir = ic_mean / ic_std if ic_std > 0.0 else float("nan")
    else:
        icir = float("nan")

    spearman_ic_mean = float(np.nanmean(spearman_series))
    hit_rate = _hit_rate(y_pred, y_true)

    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()

    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = float(r2_score(y_true_flat, y_pred_flat))

    return {
        "ic": ic_mean,
        "icir": icir,
        "hit_rate": hit_rate,
        "spearman_ic": spearman_ic_mean,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }


def _pearson_ic_series(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute cross-sectional Pearson IC for each period.

    IC (Information Coefficient) is the Pearson correlation between predicted and
    realized returns computed across assets for each time period. A positive IC
    indicates that the model correctly ranks assets in that period. In practice,
    mean IC > 0.05 is considered meaningful; IC > 0.10 is strong for equity models.

    Args:
        y_pred: Predicted returns, shape (n_periods, n_assets).
        y_true: Realized returns, shape (n_periods, n_assets).

    Returns:
        Array of shape (n_periods,) with per-period IC values (NaN where undefined).
    """
    n_periods = y_pred.shape[0]
    ic_values = np.full(n_periods, np.nan)
    for t in range(n_periods):
        pred_t = y_pred[t]
        true_t = y_true[t]
        valid = ~(np.isnan(pred_t) | np.isnan(true_t))
        if valid.sum() < 2:
            continue
        # np.corrcoef returns NaN when std of either series is 0 (degenerate period).
        corr_matrix = np.corrcoef(pred_t[valid], true_t[valid])
        ic_values[t] = corr_matrix[0, 1]
    return ic_values


def _spearman_ic_series(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Compute cross-sectional Spearman rank IC for each period.

    Spearman IC ranks both predictions and realizations before computing the
    correlation, making it robust to outliers in return distributions. It measures
    rank ordering skill rather than linear relationship. Values > 0.05 are meaningful
    in quantitative equity research.

    Args:
        y_pred: Predicted returns, shape (n_periods, n_assets).
        y_true: Realized returns, shape (n_periods, n_assets).

    Returns:
        Array of shape (n_periods,) with per-period Spearman IC (NaN where undefined).
    """
    n_periods = y_pred.shape[0]
    ic_values = np.full(n_periods, np.nan)
    for t in range(n_periods):
        pred_t = y_pred[t]
        true_t = y_true[t]
        valid = ~(np.isnan(pred_t) | np.isnan(true_t))
        if valid.sum() < 2:
            continue
        rho, _ = spearmanr(pred_t[valid], true_t[valid])
        ic_values[t] = float(rho)
    return ic_values


def _hit_rate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute directional accuracy (hit rate) averaged across assets.

    Hit rate is the fraction of predictions that correctly identified the sign
    (direction) of the realized return: both positive or both negative. Computed
    per asset over all periods, then averaged across assets.

    A random model achieves ~0.50. Values above 0.52–0.55 are considered meaningful
    for financial return prediction; above 0.60 is excellent.

    Args:
        y_pred: Predicted returns, shape (n_periods, n_assets).
        y_true: Realized returns, shape (n_periods, n_assets).

    Returns:
        Scalar hit rate in [0, 1].
    """
    directions_match = np.sign(y_pred) == np.sign(y_true)
    # Mean over periods per asset, then mean over assets.
    return float(directions_match.mean(axis=0).mean())
