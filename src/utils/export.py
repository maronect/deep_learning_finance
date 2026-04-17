"""
Functions for persisting pipeline artifacts to disk.

All pipeline outputs (returns, features, model params, predictions, weights,
metrics) are saved here. No ad-hoc CSV writes anywhere else in src/.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def save_portfolio_metrics(
    metrics_dict: dict,
    save_path: str
):
    """
    Salva métricas do portfólio em CSV.
    
    Parâmetros:
    -----------
    metrics_dict : dict
        Dicionário com métricas (Model, Sharpe, Annualized_Return, etc.)
    save_path : str
        Caminho para salvar o CSV
    """
    df = pd.DataFrame([metrics_dict])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Métricas salvas em: {save_path}")


def save_portfolio_weights(
    weights: np.ndarray,
    tickers: list,
    model_name: str,
    save_path: str
):
    """
    Salva pesos do portfólio em CSV.
    
    Parâmetros:
    -----------
    weights : np.ndarray
        Array com pesos
    tickers : list
        Lista de tickers
    model_name : str
        Nome do modelo
    save_path : str
        Caminho para salvar
    """
    df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Model': model_name
    })
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Pesos salvos em: {save_path}")


def save_predicted_returns(
    predictions: pd.Series,
    model_name: str,
    save_path: str
):
    """
    Salva retornos previstos (mu) em CSV.
    
    Parâmetros:
    -----------
    predictions : pd.Series
        Série com retornos previstos
    model_name : str
        Nome do modelo
    save_path : str
        Caminho para salvar
    """
    df = pd.DataFrame({
        'Ticker': predictions.index,
        'Predicted_Return': predictions.values,
        'Model': model_name
    })
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Previsões salvas em: {save_path}")


def save_all_metrics_comparison(
    metrics_list: list,
    save_path: str
):
    """
    Salva comparação de todas as métricas dos modelos.

    Parâmetros:
    -----------
    metrics_list : list
        Lista de dicionários com métricas de cada modelo
    save_path : str
        Caminho para salvar
    """
    df = pd.DataFrame(metrics_list)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Comparação de métricas salva em: {save_path}")


def save_returns(
    returns: pd.DataFrame,
    save_path: str,
) -> None:
    """Persist processed historical returns to CSV.

    Args:
        returns: DataFrame of asset returns indexed by date.
        save_path: Destination file path (including filename).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    returns.to_csv(save_path)
    print(f"Returns saved: {save_path}")


def save_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_path: str,
    y_path: str,
) -> None:
    """Persist feature matrix and target matrix to CSV.

    Args:
        X: Feature DataFrame (lag features), indexed by date.
        y: Target DataFrame (forward returns), indexed by date.
        X_path: Destination path for the feature matrix.
        y_path: Destination path for the target matrix.
    """
    Path(X_path).parent.mkdir(parents=True, exist_ok=True)
    Path(y_path).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(X_path)
    y.to_csv(y_path)
    print(f"Features saved: {X_path}")
    print(f"Targets saved:  {y_path}")


def save_model(
    model: object,
    save_path: str,
) -> None:
    """Persist a trained sklearn model to disk using joblib.

    Args:
        model: Fitted model object (e.g. RidgeReturnModel or MLPReturnModel).
        save_path: Destination file path (should end in .joblib).
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved: {save_path}")


def load_model(load_path: str) -> object:
    """Load a previously saved model from disk.

    Args:
        load_path: Path to the .joblib file written by save_model().

    Returns:
        The deserialized model object.
    """
    return joblib.load(load_path)

