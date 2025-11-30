"""
Módulo para exportar resultados em CSV.
"""
import pandas as pd
import numpy as np
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

