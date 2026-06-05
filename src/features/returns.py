"""
Responsible for computing asset returns at multiple frequencies from raw price data.

Supports daily, weekly, and monthly return calculation. Also computes
annualization factors and basic statistics (mean, std) per asset.

All returns are logarithmic: r_t = ln(P_t / P_{t-1}).
"""

import numpy as np
import pandas as pd
from typing import Literal

FrequencyLiteral = Literal["daily", "weekly", "monthly", "annually"]

_FREQ_RESAMPLE_MAP: dict[str, str] = {
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "annually": "YE",
}

_FREQ_PERIODS_PER_YEAR: dict[str, int] = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "annual": 1,
}


def compute_returns(
    prices: pd.DataFrame,
    freq: FrequencyLiteral = "daily",
) -> pd.DataFrame:
    """Compute logarithmic returns from adjusted price data at a given frequency.

    r_t = ln(P_t / P_{t-1})

    For non-daily frequencies, prices are resampled to the last observation
    of each period before computing the log difference.

    Args:
        prices: DataFrame of adjusted closing prices indexed by date.
        freq: Return frequency. One of 'daily', 'weekly', 'monthly', 'annually'.

    Returns:
        DataFrame of log returns with the same columns as prices. The first
        row (NaN from shift) is dropped.

    Raises:
        ValueError: If freq is not one of the recognized frequency strings.

    Example:
        >>> import pandas as pd, numpy as np
        >>> prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        >>> abs(compute_returns(prices, freq="daily")["A"].iloc[0] - np.log(1.1)) < 1e-10
        True
    """
    if freq not in _FREQ_RESAMPLE_MAP:
        raise ValueError(
            f"Invalid frequency '{freq}'. "
            f"Use one of: {list(_FREQ_RESAMPLE_MAP.keys())}."
        )

    if freq == "daily":
        return np.log(prices / prices.shift(1)).dropna()

    resampled = prices.resample(_FREQ_RESAMPLE_MAP[freq]).last()

    # Drop the last period if it is incomplete. resample('ME') labels each
    # bucket with the month-end date; if the last actual price is more than
    # 5 days before that label, the period has not closed and its return
    # would be based on partial data, distorting annualized metrics.
    if len(resampled) > 1:
        days_short = (resampled.index[-1] - prices.index[-1]).days
        if days_short > 5:
            resampled = resampled.iloc[:-1]

    return np.log(resampled / resampled.shift(1)).dropna()


def ajustar_risk_free(rf_ano: float, freq: str = "daily") -> float:
    """Convert an annual risk-free rate to the log-return equivalent for a shorter period.

    Uses log scaling consistent with log returns:
        rf_period = ln(1 + rf_annual) / periods_per_year

    Args:
        rf_ano: Annual risk-free rate as a decimal (e.g. 0.15 for 15% p.a.).
        freq: Target frequency. One of 'daily' (252), 'weekly' (52),
            'monthly' (12), 'annual' (1).

    Returns:
        Log risk-free rate for the target period.

    Raises:
        KeyError: If freq is not a recognized period key.

    Example:
        >>> round(ajustar_risk_free(0.15, freq="monthly"), 6)
        0.011647
    """
    return np.log(1 + rf_ano) / _FREQ_PERIODS_PER_YEAR[freq]


def converter_periodo(
    ret: float,
    vol: float,
    dias: int,
) -> tuple[float, float]:
    """Scale a single-period log return and volatility to a longer target horizon.

    Log returns are additive, so scaling is linear for returns and
    square-root-of-time for volatility.

    Args:
        ret: Per-period log return (e.g. daily log return).
        vol: Per-period volatility as a decimal (e.g. daily std dev).
        dias: Number of periods in the target horizon (e.g. 21 for monthly
            from daily, 252 for annual from daily).

    Returns:
        Tuple of (scaled_log_return, scaled_volatility).

    Example:
        >>> ret, vol = converter_periodo(0.001, 0.01, 252)
        >>> ret > 0.001 and vol > 0.01
        True
    """
    ret_conv = ret * dias
    vol_conv = vol * np.sqrt(dias)
    return ret_conv, vol_conv
