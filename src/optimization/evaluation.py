import pandas as pd
import numpy as np

def evaluate_portfolio(returns: pd.DataFrame, weights: np.ndarray, freq='daily', model_name='baseline'):
    """
    Calcula métricas de desempenho de um portfólio.
    """
    portfolio_returns = returns.dot(weights)
    mean = portfolio_returns.mean()
    vol = portfolio_returns.std()
    sharpe = mean / vol * np.sqrt(252 if freq == 'daily' else 12)

    return {
        "Model": model_name,
        "Mean": mean,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Cumulative_Return": (1 + portfolio_returns).prod() - 1
    }
