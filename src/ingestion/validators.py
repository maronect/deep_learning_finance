"""
Responsible for validating the integrity and completeness of raw downloaded data.

Checks for missing dates, NaN values, zero-volume days, and minimum history
requirements per ticker before passing data downstream to the feature engineering stage.
"""

import pandas as pd


def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where every column is NaN.

    Args:
        df: Input DataFrame of prices or returns.

    Returns:
        DataFrame with all-NaN rows removed.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [1.0, None], "B": [2.0, None]})
        >>> drop_empty_rows(df).shape[0]
        1
    """
    return df.dropna(how="all")


def filter_by_coverage(
    df: pd.DataFrame,
    min_coverage: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove columns that do not meet a minimum non-null data coverage threshold.

    Args:
        df: Input DataFrame of prices (rows = dates, columns = tickers).
        min_coverage: Minimum fraction of non-null observations required for a
            column to be kept. Must be between 0.0 and 1.0. Defaults to 0.5.

    Returns:
        A tuple of:
            - DataFrame containing only columns that meet the coverage threshold.
            - List of column names that were removed.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [None, None, None]})
        >>> valid_df, removed = filter_by_coverage(df, min_coverage=0.5)
        >>> removed
        ['B']
    """
    total_periods = len(df)
    min_periods = max(1, int(total_periods * min_coverage))

    valid_cols: list[str] = []
    removed_cols: list[str] = []

    for col in df.columns:
        non_null = int(df[col].notna().sum())
        if non_null >= min_periods:
            valid_cols.append(col)
        else:
            removed_cols.append(col)
            print(
                f"  Warning: {col} removed — insufficient data "
                f"({non_null}/{total_periods} periods, minimum {min_periods})"
            )

    return df[valid_cols].copy(), removed_cols


def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN values using forward-fill followed by backward-fill.

    Forward-fill propagates the last known price forward; backward-fill handles
    leading NaNs at the start of a series.

    Args:
        df: DataFrame of prices with possible NaN gaps.

    Returns:
        DataFrame with NaN values filled.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [1.0, None, 3.0]})
        >>> fill_missing_prices(df)["A"].isna().any()
        False
    """
    return df.ffill().bfill()


def validate_not_empty(df: pd.DataFrame, context: str = "data") -> None:
    """Raise ValueError if the DataFrame has no rows or no columns.

    Args:
        df: DataFrame to validate.
        context: Descriptive label used in the error message.

    Raises:
        ValueError: If the DataFrame is empty or has no columns.

    Example:
        >>> import pandas as pd
        >>> validate_not_empty(pd.DataFrame({"A": [1.0]}))  # no error
        >>> validate_not_empty(pd.DataFrame())              # raises ValueError
    """
    if df.empty or len(df.columns) == 0:
        raise ValueError(
            f"No valid {context} after processing. "
            "Check tickers and date range."
        )
