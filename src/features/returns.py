"""
Responsible for computing asset returns at multiple frequencies from raw price data.

Supports daily, weekly, and monthly return calculation. Also computes
annualization factors and basic statistics (mean, std) per asset.
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
    """Compute percentage returns from adjusted price data at a given frequency.

    For non-daily frequencies, prices are resampled to the last observation
    of each period before computing percentage change.

    Args:
        prices: DataFrame of adjusted closing prices indexed by date.
        freq: Return frequency. One of 'daily', 'weekly', 'monthly', 'annually'.

    Returns:
        DataFrame of percentage returns with the same columns as prices. The first
        row (NaN from pct_change) is dropped.

    Raises:
        ValueError: If freq is not one of the recognized frequency strings.

    Example:
        >>> import pandas as pd
        >>> prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        >>> compute_returns(prices, freq="daily")["A"].iloc[0]
        0.1
    """
    if freq not in _FREQ_RESAMPLE_MAP:
        raise ValueError(
            f"Invalid frequency '{freq}'. "
            f"Use one of: {list(_FREQ_RESAMPLE_MAP.keys())}."
        )

    if freq == "daily":
        return prices.pct_change().dropna()

    resampled = prices.resample(_FREQ_RESAMPLE_MAP[freq]).last()
    return resampled.pct_change().dropna()


def ajustar_risk_free(rf_ano: float, freq: str = "daily") -> float:
    """Convert an annual risk-free rate to the equivalent rate for a shorter period.

    Uses geometric compounding:
        rf_period = (1 + rf_annual) ^ (1 / periods_per_year) - 1

    Args:
        rf_ano: Annual risk-free rate as a decimal (e.g. 0.15 for 15% p.a.).
        freq: Target frequency. One of 'daily' (252), 'weekly' (52),
            'monthly' (12), 'annual' (1).

    Returns:
        Risk-free rate for the target period as a decimal.

    Raises:
        KeyError: If freq is not a recognized period key.

    Example:
        >>> round(ajustar_risk_free(0.15, freq="monthly"), 6)
        0.011715
    """
    return (1 + rf_ano) ** (1 / _FREQ_PERIODS_PER_YEAR[freq]) - 1


def converter_periodo(
    ret: float,
    vol: float,
    dias: int,
) -> tuple[float, float]:
    """Scale a single-period return and volatility to a longer target horizon.

    Uses compounding for return and square-root-of-time for volatility.

    Args:
        ret: Per-period return as a decimal (e.g. daily return).
        vol: Per-period volatility as a decimal (e.g. daily std dev).
        dias: Number of periods in the target horizon (e.g. 21 for monthly
            from daily, 252 for annual from daily).

    Returns:
        Tuple of (scaled_return, scaled_volatility) as decimals.

    Example:
        >>> ret, vol = converter_periodo(0.001, 0.01, 252)
        >>> ret > 0.001 and vol > 0.01
        True
    """
    ret_conv = (1 + ret) ** dias - 1
    vol_conv = vol * np.sqrt(dias)
    return ret_conv, vol_conv
