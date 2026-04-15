"""
Responsible for blending ML model predictions with historical mean returns.

Applies the formula: final_mu = alpha * ml_prediction + (1 - alpha) * historical_mean
where alpha is read from config. Prevents extreme predictions from dominating optimization.
"""
from __future__ import annotations

import warnings

import pandas as pd

from src.utils.config_loader import get_config


def blend_predictions(
    ml_predictions: pd.Series,
    historical_means: pd.Series,
    alpha: float,
) -> pd.Series:
    """Blend ML model predictions with historical mean returns.

    Applies conservative shrinkage to prevent extreme ML predictions from
    dominating portfolio optimization:
        blended_mu = alpha * ml_predictions + (1 - alpha) * historical_means

    If any asset has a NaN ML prediction, falls back to the historical mean
    for that asset with a RuntimeWarning.

    Args:
        ml_predictions: Series of model-predicted expected returns indexed by ticker.
        historical_means: Series of historical mean returns indexed by ticker.
            Must share the same index as ml_predictions.
        alpha: Blending weight for the ML component. Must be in [0.0, 1.0].
            alpha=0.0 returns pure historical mean; alpha=1.0 returns pure ML prediction.

    Returns:
        Series of blended expected returns indexed by ticker.

    Raises:
        ValueError: If alpha is not in [0.0, 1.0] or if the Series have mismatched indices.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
    if not ml_predictions.index.equals(historical_means.index):
        raise ValueError(
            "ml_predictions and historical_means must share the same ticker index. "
            f"Got {list(ml_predictions.index)} vs {list(historical_means.index)}."
        )

    nan_mask = ml_predictions.isna()
    if nan_mask.any():
        warnings.warn(
            f"NaN ML predictions detected for assets: {ml_predictions[nan_mask].index.tolist()}. "
            "Falling back to historical mean for those assets.",
            RuntimeWarning,
            stacklevel=2,
        )
        ml_predictions = ml_predictions.fillna(historical_means)

    return alpha * ml_predictions + (1.0 - alpha) * historical_means


def blend_from_config(
    ml_predictions: pd.Series,
    historical_means: pd.Series,
) -> pd.Series:
    """Blend predictions using alpha from config/pipeline.yaml:models:blend_alpha.

    Args:
        ml_predictions: Model-predicted expected returns per ticker.
        historical_means: Historical mean returns per ticker.

    Returns:
        Blended expected returns per ticker.
    """
    cfg = get_config("pipeline")
    alpha: float = cfg["models"].get("blend_alpha", 0.3)
    return blend_predictions(ml_predictions, historical_means, alpha=alpha)
