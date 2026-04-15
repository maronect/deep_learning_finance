"""
Ridge Regression model for predicting expected asset returns.

Wraps scikit-learn's Ridge with the project's base interface, applying
walk-forward validation and reading hyperparameters from config/models.yaml.
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge as SklearnRidge

from src.models.base import BaseReturnModel
from src.utils.config_loader import get_config


class RidgeReturnModel(BaseReturnModel):
    """Ridge Regression model for predicting expected asset returns.

    Trains one Ridge regressor per asset using the full lag feature matrix as input.
    Hyperparameters are read from config/models.yaml at construction time.

    Attributes:
        alpha: L2 regularization strength from config/models.yaml:ridge:alpha.
    """

    def __init__(self) -> None:
        super().__init__(name="ridge")
        models_cfg = get_config("models")
        self.alpha: float = models_cfg["ridge"]["alpha"]
        self._models: dict[str, SklearnRidge] = {}
        self._target_columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train one Ridge regressor per asset on the training feature matrix.

        Args:
            X_train: Feature matrix of shape (n_train_samples, n_features).
            y_train: Target matrix of shape (n_train_samples, n_assets).
        """
        self._target_columns = list(y_train.columns)
        self._models = {}
        for ticker in self._target_columns:
            model = SklearnRidge(alpha=self.alpha)
            model.fit(X_train.values, y_train[ticker].values)
            self._models[ticker] = model
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict returns for all assets for each row in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            DataFrame of shape (n_samples, n_assets) with predicted returns.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.is_fitted:
            raise RuntimeError("RidgeReturnModel must be fitted before predict().")
        predictions = {
            ticker: self._models[ticker].predict(X.values)
            for ticker in self._target_columns
        }
        return pd.DataFrame(predictions, index=X.index)

    def get_params(self) -> dict:
        """Return Ridge hyperparameters.

        Returns:
            Dict with alpha value.
        """
        return {"alpha": self.alpha}
