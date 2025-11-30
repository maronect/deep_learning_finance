"""
Módulo de avaliação de portfólios.
Calcula métricas financeiras como Sharpe Ratio, retorno anualizado, volatilidade, etc.
"""
import numpy as np
import pandas as pd
from src.optimization.markowitz import portfolio_return, portfolio_volatility


def calculate_sharpe_ratio(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.15,
    periods_per_year: int = 252
) -> float:
    """
    Calcula o Sharpe Ratio anualizado do portfólio.
    
    Parâmetros:
    -----------
    portfolio_returns : pd.Series
        Retornos do portfólio (diários, mensais, etc.)
    risk_free_rate_annual : float
        Taxa livre de risco anual (ex: 0.15 = 15% ao ano)
    periods_per_year : int
        Número de períodos por ano (252 para diário, 12 para mensal, etc.)
    
    Retorna:
    --------
    float
        Sharpe Ratio anualizado
    """
    if len(portfolio_returns) == 0:
        return 0.0
    
    # Taxa livre de risco no período
    risk_free_rate_period = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    
    # Retorno médio e desvio padrão no período
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    if std_return == 0:
        return 0.0
    
    # Sharpe no período
    sharpe_period = (mean_return - risk_free_rate_period) / std_return
    
    # Anualização: Sharpe anual = Sharpe período * sqrt(períodos por ano)
    sharpe_annual = sharpe_period * np.sqrt(periods_per_year)
    
    return sharpe_annual


def calculate_annualized_return(
    portfolio_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calcula o retorno anualizado do portfólio.
    
    Parâmetros:
    -----------
    portfolio_returns : pd.Series
        Retornos do portfólio
    periods_per_year : int
        Número de períodos por ano
    
    Retorna:
    --------
    float
        Retorno anualizado
    """
    if len(portfolio_returns) == 0:
        return 0.0
    
    mean_return = portfolio_returns.mean()
    annualized_return = (1 + mean_return) ** periods_per_year - 1
    
    return annualized_return


def calculate_annualized_volatility(
    portfolio_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calcula a volatilidade anualizada do portfólio.
    
    Parâmetros:
    -----------
    portfolio_returns : pd.Series
        Retornos do portfólio
    periods_per_year : int
        Número de períodos por ano
    
    Retorna:
    --------
    float
        Volatilidade anualizada
    """
    if len(portfolio_returns) == 0:
        return 0.0
    
    std_return = portfolio_returns.std()
    annualized_vol = std_return * np.sqrt(periods_per_year)
    
    return annualized_vol


def calculate_cumulative_return(portfolio_returns: pd.Series) -> float:
    """
    Calcula o retorno acumulado do portfólio.
    
    Parâmetros:
    -----------
    portfolio_returns : pd.Series
        Retornos do portfólio
    
    Retorna:
    --------
    float
        Retorno acumulado (ex: 2.5 = 150% de retorno total)
    """
    if len(portfolio_returns) == 0:
        return 1.0
    
    cumulative = (1 + portfolio_returns).prod()
    return cumulative


def evaluate_portfolio(
    portfolio_returns: pd.Series,
    risk_free_rate_annual: float = 0.15,
    periods_per_year: int = 252,
    model_name: str = "Model"
) -> pd.Series:
    """
    Avalia um portfólio e retorna todas as métricas principais.
    
    Parâmetros:
    -----------
    portfolio_returns : pd.Series
        Retornos do portfólio
    risk_free_rate_annual : float
        Taxa livre de risco anual
    periods_per_year : int
        Número de períodos por ano
    model_name : str
        Nome do modelo
    
    Retorna:
    --------
    pd.Series
        Série com todas as métricas
    """
    sharpe = calculate_sharpe_ratio(
        portfolio_returns, 
        risk_free_rate_annual, 
        periods_per_year
    )
    ann_return = calculate_annualized_return(portfolio_returns, periods_per_year)
    ann_vol = calculate_annualized_volatility(portfolio_returns, periods_per_year)
    cum_return = calculate_cumulative_return(portfolio_returns)
    
    return pd.Series({
        "Model": model_name,
        "Sharpe": sharpe,
        "Annualized_Return": ann_return,
        "Annualized_Volatility": ann_vol,
        "Cumulative_Return": cum_return,
        "Mean_Return": portfolio_returns.mean(),
        "Volatility": portfolio_returns.std()
    })

