"""
Responsible for selecting a subset of uncorrelated assets from the full ticker universe.

Implements four selection strategies: correlation threshold, stable pairs,
clustering, and variance-based filtering. Chosen strategy is read from config.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.features.returns import compute_returns
from src.utils.config_loader import get_config


# 1. Brazilian stock universe

def get_brazilian_stocks_universe() -> list[str]:
    """Return the full universe of B3 tickers used for asset selection.

    The universe spans energy, mining/steel, banks, retail/consumer,
    industrial/tech, financial services, and construction sectors.

    Returns:
        Deduplicated list of Yahoo Finance ticker symbols for Brazilian stocks.

    Example:
        >>> universe = get_brazilian_stocks_universe()
        >>> "PETR4.SA" in universe
        True
    """
    energy = ["PETR4.SA", "PETR3.SA", "ELET3.SA", "ELET6.SA", "EQTL3.SA", "CSAN3.SA", "UGPA3.SA"]
    mining_steel = ["VALE3.SA", "GGBR4.SA", "CSNA3.SA", "USIM5.SA", "CMIG4.SA", "GOAU4.SA"]
    banks = ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "BRSR6.SA", "BPAN4.SA", "ABCB4.SA", "PINE4.SA"]
    retail_consumer = ["ABEV3.SA", "VIVT3.SA", "RENT3.SA", "MGLU3.SA", "PCAR3.SA", "MRVE3.SA"]
    industrial_tech = ["WEGE3.SA", "EMBR3.SA", "RADL3.SA", "TOTS3.SA", "TIMS3.SA", "CYRE3.SA", "KLBN11.SA"]
    financial_services = ["B3SA3.SA", "CAML3.SA", "SUZB3.SA"]
    construction = ["CYRE3.SA", "EZTC3.SA", "JHSF3.SA", "MRVE3.SA"]

    all_stocks = (
        energy + mining_steel + banks
        + retail_consumer + industrial_tech
        + financial_services + construction
    )
    return list(dict.fromkeys(all_stocks))


# 2. Strategy: lowest sum of absolute pairwise correlations

def _select_by_sum_abs_correlation(
    returns: pd.DataFrame,
    n_assets: int,
) -> list[str]:
    """Select assets with the lowest total absolute pairwise correlation.

    Args:
        returns: DataFrame of asset returns (rows = periods, columns = tickers).
        n_assets: Number of assets to select.

    Returns:
        List of n_assets ticker names with the smallest sum of absolute correlations.
    """
    corr_matrix = returns.corr()
    abs_corr_sum = (corr_matrix.abs().sum() - 1.0).sort_values()
    return abs_corr_sum.head(n_assets).index.tolist()


# 3. Strategy: greedy min-max correlation

def _select_by_min_max_correlation(
    returns: pd.DataFrame,
    n_assets: int,
) -> list[str]:
    """Greedily select assets that minimise the maximum pairwise correlation.

    Starts with the asset that has the lowest average absolute correlation,
    then iteratively adds the asset whose maximum correlation with the already-
    selected set is smallest.

    Args:
        returns: DataFrame of asset returns.
        n_assets: Number of assets to select.

    Returns:
        List of n_assets ticker names forming the most decorrelated subset.
    """
    corr = returns.corr().abs()

    ranking = (corr.sum() - 1).sort_values()
    selected: list[str] = [ranking.index[0]]

    for _ in range(n_assets - 1):
        remaining = [a for a in corr.columns if a not in selected]
        best_asset: Optional[str] = None
        best_score = 999.0

        for asset in remaining:
            score = float(corr.loc[selected, asset].max())
            if score < best_score:
                best_score = score
                best_asset = asset

        if best_asset is not None:
            selected.append(best_asset)

    return selected


# 4. Strategy: lowest-correlation pairs

def _select_lowest_corr_pairs(
    returns: pd.DataFrame,
    n_pairs: int = 5,
) -> list[str]:
    """Select assets by picking the n_pairs with the lowest pairwise correlation.

    Each asset can only appear in one selected pair, so the result contains
    exactly 2 * n_pairs unique tickers.

    Args:
        returns: DataFrame of asset returns.
        n_pairs: Number of low-correlation pairs to extract. Defaults to 5.

    Returns:
        List of 2 * n_pairs unique ticker names.
    """
    corr = returns.corr()
    cols = corr.columns

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], float(corr.iloc[i, j])))

    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    selected: list[str] = []
    used: set[str] = set()

    for a, b, _ in pairs_sorted:
        if a not in used and b not in used:
            selected.extend([a, b])
            used.update({a, b})
        if len(selected) >= 2 * n_pairs:
            break

    return selected


# 5. Strategy: temporally stable correlation pairs

def _select_stable_pairs(
    prices: pd.DataFrame,
    start_year: int = 2010,
    end_year: int = 2025,
    n_pairs: int = 5,
    return_freq: str = "monthly",
) -> list[str]:
    """Select asset pairs whose pairwise correlation is low and stable over time.

    For each year in [start_year, end_year], the pairwise correlation matrix is
    computed. Pairs are ranked by a composite score of mean + std + max correlation
    across all years. Lower score = more stable and more decorrelated.

    Pairs are selected greedily so each ticker appears in at most one pair.

    Args:
        prices: DataFrame of adjusted closing prices.
        start_year: First year to include in the stability analysis.
        end_year: Last year to include in the stability analysis.
        n_pairs: Number of stable pairs to select; result has 2 * n_pairs tickers.
        return_freq: Frequency used to compute returns for the correlation analysis.
            One of 'daily', 'weekly', 'monthly'. Defaults to 'monthly'.

    Returns:
        List of 2 * n_pairs unique ticker names.
    """
    returns = compute_returns(prices, freq=return_freq)
    cols = returns.columns

    corr_time: dict[tuple[str, str], list[float]] = {}

    for year in range(start_year, end_year + 1):
        year_slice = returns[returns.index.year == year]
        min_periods = 6 if return_freq == "monthly" else 50
        if len(year_slice) < min_periods:
            continue

        corr_year = year_slice.corr()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pair = (cols[i], cols[j])
                val = float(corr_year.iloc[i, j])
                corr_time.setdefault(pair, []).append(val)

    stats: list[tuple[str, str, float, float, float]] = []
    for (a, b), corr_vals in corr_time.items():
        if len(corr_vals) > 5:
            stats.append((
                a, b,
                float(np.mean(corr_vals)),
                float(np.std(corr_vals)),
                float(np.max(corr_vals)),
            ))

    df_stats = pd.DataFrame(stats, columns=["A", "B", "mean", "std", "max"])
    df_stats["score"] = df_stats["mean"] + df_stats["std"] + df_stats["max"]
    df_sorted = df_stats.sort_values("score")

    selected: list[str] = []
    used: set[str] = set()

    for _, row in df_sorted.iterrows():
        a, b = str(row["A"]), str(row["B"])
        if a not in used and b not in used:
            selected.extend([a, b])
            used.update({a, b})
        if len(selected) >= 2 * n_pairs:
            break

    return selected


# 6. Master function

def select_assets(
    start_date: str,
    end_date: str,
    method: str = "lowest_corr_pairs",
    n_assets: int = 10,
    return_freq: str = "monthly",
    min_data_coverage: float = 0.85,
    prices: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Select a decorrelated subset of assets from the Brazilian stock universe.

    Downloads prices for the full universe (unless pre-downloaded prices are
    supplied), computes returns, and applies the chosen selection strategy.

    Available strategies:
        - ``"sum_abs_correlation"``: assets with lowest total absolute correlation.
        - ``"min_max_correlation"``: greedy selection minimising max pairwise correlation.
        - ``"lowest_corr_pairs"``: selects the n_assets // 2 least-correlated pairs.
        - ``"stable_corr_pairs"``: selects pairs stable across the full time window.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        method: Selection strategy name. Defaults to ``'lowest_corr_pairs'``.
        n_assets: Number of assets to return. For pair-based strategies, must be even.
            Defaults to 10.
        return_freq: Frequency for return computation. Defaults to ``'monthly'``.
        min_data_coverage: Minimum fraction of non-null observations required per
            ticker to be included. Defaults to 0.85.
        prices: Optional pre-downloaded price DataFrame. If provided, the download
            step is skipped (avoids duplicate network calls).

    Returns:
        List of selected ticker symbols.

    Raises:
        ValueError: If method is not one of the four recognised strategies.

    Example:
        >>> tickers = select_assets("2020-01-01", "2023-12-31", method="min_max_correlation", n_assets=6)
        >>> len(tickers)
        6
    """
    if prices is None:
        from src.ingestion.downloader import load_prices
        universe = get_brazilian_stocks_universe()
        prices = load_prices(
            tickers=universe,
            start=start_date,
            end=end_date,
            min_data_coverage=min_data_coverage,
        )

    returns = compute_returns(prices, freq=return_freq)

    if method == "sum_abs_correlation":
        return _select_by_sum_abs_correlation(returns, n_assets)

    if method == "min_max_correlation":
        return _select_by_min_max_correlation(returns, n_assets)

    if method == "lowest_corr_pairs":
        return _select_lowest_corr_pairs(returns, n_pairs=n_assets // 2)

    if method == "stable_corr_pairs":
        return _select_stable_pairs(
            prices,
            start_year=int(start_date[:4]),
            end_year=int(end_date[:4]),
            n_pairs=n_assets // 2,
            return_freq=return_freq,
        )

    raise ValueError(
        f"Unknown method '{method}'. "
        "Use one of: 'sum_abs_correlation', 'min_max_correlation', "
        "'lowest_corr_pairs', 'stable_corr_pairs'."
    )


def select_assets_from_config(
    prices: Optional[pd.DataFrame] = None,
) -> list[str]:
    """Select assets using parameters from config/pipeline.yaml.

    Reads method, n_assets, min_data_coverage, start_date, end_date, and
    frequency from the pipeline config.

    Args:
        prices: Optional pre-downloaded price DataFrame passed to select_assets().

    Returns:
        List of selected ticker symbols.
    """
    cfg = get_config("pipeline")
    data_cfg = cfg["data"]
    sel_cfg = cfg.get("asset_selection", {})

    return select_assets(
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        method=sel_cfg.get("method", "stable_corr_pairs"),
        n_assets=sel_cfg.get("n_assets", 10),
        return_freq=data_cfg["frequency"],
        min_data_coverage=sel_cfg.get("min_data_coverage", 0.85),
        prices=prices,
    )


# 7. Utility: correlation matrix

def get_correlation_matrix(
    returns: pd.DataFrame,
    selected_assets: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute the pairwise Pearson correlation matrix for the given returns.

    Args:
        returns: DataFrame of asset returns (rows = periods, columns = tickers).
        selected_assets: Optional list of tickers to restrict the matrix to.
            If None, all columns in returns are used.

    Returns:
        Square correlation matrix as a DataFrame.

    Example:
        >>> import pandas as pd, numpy as np
        >>> r = pd.DataFrame(np.eye(5), columns=list("ABCDE"))
        >>> get_correlation_matrix(r, selected_assets=["A", "B"]).shape
        (2, 2)
    """
    if selected_assets is not None:
        returns = returns[selected_assets]
    return returns.corr()
