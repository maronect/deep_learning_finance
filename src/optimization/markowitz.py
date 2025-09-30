from scipy.optimize import minimize
import numpy as np

def portfolio_return(weights, mean_returns):
    '''
    Retorno esperado do portfólio
    '''
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    '''
    Cálculo do risco (volatilidade) do portfólio (com base na matriz de covariância entre os ativos)
    '''
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# portfolio_volatility -  portfolio_return

def minimize_volatility(mean_returns, cov_matrix, target_return):
    '''
    Busca a combinação de ativos (pesos) que atinge um retorno esperado pré-determinado e minimiza 
    a volatilidade, respeitando as restrições
    '''
    num_assets = len(mean_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, #soma dos pesos = 1
        #{'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target_return} # retorno alvo do portoflio deve ser = ao escolhido
    ]
    bounds = tuple((0, 1) for _ in range(num_assets)) # pesos devem ser valores entre 0 e 1
    initial_guess = num_assets * [1. / num_assets] # a principio mesmo peso para todos
    result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        print("Erro na otimização Sharpe:", result.message)
        return None

