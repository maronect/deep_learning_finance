import pandas as pd
import numpy as np

def evaluate_portfolio(returns: pd.DataFrame, weights: np.ndarray, freq='daily', model_name='baseline'):
    """
    Calcula métricas de desempenho de um portfólio.
    não está avaliando o modelo de ML diretament, mas sim o resultado econômico da decisão que ele formulou.
    """
    portfolio_returns = returns.dot(weights)
    mean = portfolio_returns.mean()
    vol = portfolio_returns.std()
    sharpe = mean / vol * np.sqrt(252 if freq == 'daily' else 12) # unidades de retorno do portifolio por unidade de risco, nromalmente entre 1 e 2. acima disso é crazy

    return {
        "Model": model_name,
        "Mean": mean,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Cumulative_Return": (1 + portfolio_returns).prod() - 1
    }
