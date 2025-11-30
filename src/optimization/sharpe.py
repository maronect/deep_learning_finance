from scipy.optimize import minimize
import numpy as np

from src.optimization.markowitz import portfolio_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - risk_free_rate) / vol  # Negativo porque o objetivo eh maximizar

def maximize_sharpe(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # pesos somam 1
    
    result = minimize(negative_sharpe_ratio, initial_guess,
                      args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x if result.success else None
