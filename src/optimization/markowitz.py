from scipy.optimize import minimize
import numpy as np

#############################
def portfolio_return(weights, mean_returns):
    '''
    Retorno esperado do portfólio
    '''
    return np.dot(weights, mean_returns)

###########################
def portfolio_volatility(weights, cov_matrix):
    '''
    Cálculo do risco (volatilidade) do portfólio (com base na matriz de covariância entre os ativos)
    '''
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# TODO (1-lamb)*portfolio_volatility -  lamb * portfolio_return, para vários lambs de 0 a 1

#############################
def markowitz_objective(weights, mean_returns, cov_matrix, lamb):
    return lamb* np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) - (1-lamb)*np.dot(weights, mean_returns)

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

#################################
def solve_markowitz(mean_returns, cov_matrix,lamb):
    num_assets = len(mean_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, #soma dos pesos = 1
        #{'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target_return} # retorno alvo do portoflio deve ser = ao escolhido
    ]
    bounds = tuple((0, 1) for _ in range(num_assets)) # pesos devem ser valores entre 0 e 1
    initial_guess = num_assets * [1. / num_assets] # a principio mesmo peso para todos
    result = minimize(markowitz_objective, initial_guess, args=(mean_returns, cov_matrix,lamb),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        print("Erro na otimização Sharpe:", result.message)
        return None

def efficient_frontier_lambda(mean_returns, cov_matrix, lamb_values):
    portfolios = []

    for lamb in lamb_values:
        def objective(weights):
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return (1 - lamb) * vol - lamb * ret

        num_assets = len(mean_returns)
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x
            ret = np.dot(weights, mean_returns)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            portfolios.append((ret, vol, weights))
        else:
            print(f"Erro para lambda={lamb}: {result.message}")

    return portfolios
