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

def markowitz_objective(weights, mean_returns, cov_matrix, lamb):
    '''
    Trade off entre retorno e risco, com um parâmetro lambda
    '''
    risk = portfolio_volatility(weights, cov_matrix)
    ret = portfolio_return(weights, mean_returns)
    return lamb * risk - (1 - lamb) * ret

def minimize_volatility(mean_returns, cov_matrix):
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
        print("Erro na otimização:", result.message)
        return None


def solve_markowitz(mean_returns, cov_matrix, lamb, max_weight=None):
    """
    Resolve o problema de otimização de Markowitz.
    
    Parâmetros:
    -----------
    mean_returns : array-like
        Retornos esperados dos ativos
    cov_matrix : array-like
        Matriz de covariância
    lamb : float
        Parâmetro de trade-off risco-retorno (0 = max retorno, 1 = min risco)
    max_weight : float, optional
        Peso máximo por ativo (para diversificação). Se None, sem limite.
    
    Retorna:
    --------
    array
        Pesos ótimos do portfólio
    """
    num_assets = len(mean_returns)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # soma dos pesos = 1
    ]
    
    # Bounds: se max_weight especificado, limitar peso máximo
    if max_weight is not None:
        bounds = tuple((0, max_weight) for _ in range(num_assets))
    else:
        bounds = tuple((0, 1) for _ in range(num_assets))
    
    initial_guess = num_assets * [1. / num_assets]
    
    result = minimize(
        markowitz_objective, 
        initial_guess, 
        args=(mean_returns, cov_matrix, lamb),
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        print(f"Erro na otimização Markowitz: {result.message}")
        # Retorna pesos iguais como fallback
        return np.array([1. / num_assets] * num_assets)

