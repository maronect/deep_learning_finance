"""
Multi-Layer Perceptron model for predicting expected asset returns.

Wraps scikit-learn's MLPRegressor with the project's base interface, applying
walk-forward validation and reading hyperparameters from config/models.yaml.
"""
from __future__ import annotations

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from src.models.base import BaseReturnModel
from src.utils.config_loader import get_config


class MLPReturnModel(BaseReturnModel):
    """MLP model for predicting expected asset returns.

    Trains one MLPRegressor per asset. Hyperparameters are read from
    config/models.yaml at construction time.

    Attributes:
        hidden_layer_sizes: Tuple of hidden layer sizes.
        activation: Activation function name.
        max_iter: Maximum number of training iterations.
        random_state: Seed for reproducibility.
    """

    def __init__(self) -> None:
        super().__init__(name="mlp")
        models_cfg = get_config("models")
        mlp_cfg = models_cfg["mlp"]
        self.hidden_layer_sizes: tuple[int, ...] = tuple(mlp_cfg["hidden_layer_sizes"])
        self.activation: str = mlp_cfg["activation"]
        self.max_iter: int = mlp_cfg["max_iter"]
        self.random_state: int = mlp_cfg["random_state"]
        self._models: dict[str, MLPRegressor] = {}
        self._target_columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Train one MLPRegressor per asset on the training feature matrix.

        Convergence warnings are suppressed — they are informational and the model
        still produces valid predictions even when max_iter is reached.

        Args:
            X_train: Feature matrix of shape (n_train_samples, n_features).
            y_train: Target matrix of shape (n_train_samples, n_assets).
        """
        self._target_columns = list(y_train.columns)
        self._models = {}
        for ticker in self._target_columns:
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver="adam",
                learning_rate_init=1e-3,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
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
            raise RuntimeError("MLPReturnModel must be fitted before predict().")
        predictions = {
            ticker: self._models[ticker].predict(X.values)
            for ticker in self._target_columns
        }
        return pd.DataFrame(predictions, index=X.index)

    def get_params(self) -> dict:
        """Return MLP hyperparameters.

        Returns:
            Dict with hidden_layer_sizes, activation, max_iter, and random_state.
        """
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
