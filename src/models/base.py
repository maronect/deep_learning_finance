"""
Defines the abstract base class and common interface for all return prediction models.

Every model (Ridge, MLP, RNN) must implement this interface so the pipeline
can swap models without changing orchestration logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseReturnModel(ABC):
    """Abstract base class for all expected-return prediction models.

    Every model in src/models/ must subclass this and implement fit() and
    predict(). The pipeline's train and predict stages call only this interface,
    enabling models to be swapped without touching orchestration code.

    Attributes:
        name: Human-readable model identifier used in artifact filenames.
        is_fitted: Set to True after fit() completes successfully.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> None:
        """Train the model on the provided feature and target matrices.

        Args:
            X_train: Feature matrix of shape (n_train_samples, n_features).
                Columns follow the convention "{ticker}_lag_{k}".
            y_train: Target matrix of shape (n_train_samples, n_assets).
                Each column is the contemporaneous return for one asset.
        """

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict returns for each row in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            DataFrame of shape (n_samples, n_assets) with predicted returns,
            indexed the same as X.

        Raises:
            RuntimeError: If predict() is called before fit().
        """

    def predict_expected_returns(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Predict the expected return for each asset using the last row of X.

        This is the primary method called by the pipeline's predict stage.
        Uses only the most recent observation to generate forward-looking mu.

        Args:
            X: Full feature matrix (all available periods). Only the last row is used.

        Returns:
            Series of predicted expected returns indexed by asset ticker.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"Model '{self.name}' must be fitted before calling predict_expected_returns()."
            )
        pred_df = self.predict(X.iloc[[-1]])
        return pred_df.iloc[0]

    def get_params(self) -> dict[str, Any]:
        """Return the model hyperparameters for artifact logging.

        Returns:
            Dict of parameter names to values. Subclasses should override.
        """
        return {}
