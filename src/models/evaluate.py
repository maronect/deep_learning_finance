import numpy as np
import pandas as pd
from src.optimization.markowitz import solve_markowitz


def evaluate_model_portfolio(returns, predicted_means, cov, name, lamb=0.5):
    weights = solve_markowitz(predicted_means, cov, lamb)
    portfolio = returns.dot(weights)

    mean = portfolio.mean() * 12
    vol  = portfolio.std() * np.sqrt(12)
    sharpe = mean / vol
    mdd = (portfolio.cumsum().max() - portfolio.cumsum().min())

    return pd.DataFrame([{
        "Modelo": name,
        "Retorno Anual": mean,
        "Vol Anual": vol,
        "Sharpe": sharpe,
        "Max Drawdown (proxy)": mdd
    }])
