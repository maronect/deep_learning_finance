"""
Utilitários para cálculo e comparação de portfólios.
"""
import numpy as np
import pandas as pd
from src.optimization.markowitz import solve_markowitz, portfolio_return, portfolio_volatility
from src.optimization.sharpe import maximize_sharpe


def get_optimal_portfolio_max_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """
    Obtém portfólio ótimo maximizando Sharpe Ratio.
    
    Parâmetros:
    -----------
    mean_returns : pd.Series
        Retornos esperados
    cov_matrix : pd.DataFrame
        Matriz de covariância
    risk_free_rate : float
        Taxa livre de risco (no período)
    
    Retorna:
    --------
    np.ndarray
        Pesos ótimos
    """
    weights = maximize_sharpe(
        mean_returns.values,
        cov_matrix.values,
        risk_free_rate=risk_free_rate
    )
    
    if weights is None:
        # Fallback: pesos iguais
        n = len(mean_returns)
        weights = np.array([1.0 / n] * n)
    
    return weights


def get_optimal_portfolio_fixed_lambda(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    lamb: float = 0.5,
    max_weight: float = None
) -> np.ndarray:
    """
    Obtém portfólio ótimo com lambda fixo.
    
    Parâmetros:
    -----------
    mean_returns : pd.Series
        Retornos esperados
    cov_matrix : pd.DataFrame
        Matriz de covariância
    lamb : float
        Parâmetro de trade-off (0 = max retorno, 1 = min risco)
    max_weight : float, optional
        Peso máximo por ativo
    
    Retorna:
    --------
    np.ndarray
        Pesos ótimos
    """
    weights = solve_markowitz(
        mean_returns.values,
        cov_matrix.values,
        lamb=lamb,
        max_weight=max_weight
    )
    
    if weights is None:
        # Fallback: pesos iguais
        n = len(mean_returns)
        weights = np.array([1.0 / n] * n)
    
    return weights


def calculate_portfolio_metrics_from_weights(
    weights: np.ndarray,
    returns: pd.DataFrame,
    mean_returns: pd.Series = None,
    cov_matrix: pd.DataFrame = None
) -> dict:
    """
    Calcula métricas do portfólio a partir dos pesos.
    
    Parâmetros:
    -----------
    weights : np.ndarray
        Pesos do portfólio
    returns : pd.DataFrame
        Retornos realizados (para backtest)
    mean_returns : pd.Series, optional
        Retornos esperados (para métricas teóricas)
    cov_matrix : pd.DataFrame, optional
        Matriz de covariância (para métricas teóricas)
    
    Retorna:
    --------
    dict
        Dicionário com métricas
    """
    # Retornos do portfólio
    portfolio_returns = returns.dot(weights)
    
    metrics = {
        'mean_return': portfolio_returns.mean(),
        'volatility': portfolio_returns.std(),
        'cumulative_return': (1 + portfolio_returns).prod(),
        'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0.0
    }
    
    # Métricas teóricas se fornecidas
    if mean_returns is not None and cov_matrix is not None:
        metrics['expected_return'] = portfolio_return(weights, mean_returns.values)
        metrics['expected_volatility'] = portfolio_volatility(weights, cov_matrix.values)
    
    return metrics

