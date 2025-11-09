import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.optimization.markowitz import solve_markowitz, portfolio_return, portfolio_volatility
#from src.data.loader import compute_returns

from src.optimization.markowitz import portfolio_return, portfolio_volatility, solve_markowitz

def plot_efficient_frontier(returns: pd.Series, cov_matrix, optimized_weights=None, num_points=10000, add_tradeoff_curve=True):
    """
    Plota a fronteira eficiente e, opcionalmente, o portfólio otimizado.

    Parâmetros:
        returns (pd.Series): retorno médio dos ativos
        cov_matrix (pd.DataFrame): matriz de covariância
        optimized_weights (np.ndarray): pesos do portfólio ótimo (opcional)
        num_points (int): número de portfólios simulados para construir a curva
    """
    num_assets = len(returns)
    means = []
    risks = []
    
    for _ in range(num_points):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        
        mean = np.dot(weights, returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        means.append(mean)
        risks.append(risk)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(risks, means, c=np.array(means)/np.array(risks), marker='o', cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Retorno/Risco (simulado')

    if add_tradeoff_curve:
        lamb_array = np.arange(0.0, 1.05, 0.05)
        ret_list = []
        vol_list = []

        for lamb in lamb_array:
            weights_lambda = solve_markowitz(returns, cov_matrix, lamb=lamb)
            if weights_lambda is not None:
                ret = portfolio_return(weights_lambda, returns)
                vol = portfolio_volatility(weights_lambda, cov_matrix)
                ret_list.append(ret)
                vol_list.append(vol)

        plt.plot(vol_list, ret_list, color='orange', linestyle='--', linewidth=2, label='Fronteira Teórica (λ)')

    '''
    # Benchmarks de renda fixa, convertidos para retorno diário
    benchmarks = {
        'CDI 15% a.a.': 0.15 / 252,
        'SELIC 15% a.a.': 0.15 / 252,
        'Poupança 8% a.a.': 0.08 / 252,
        'Tesouro Prefixado 14% a.a.': 0.14 / 252
    }

    for label, daily_return in benchmarks.items():
        plt.scatter(0.00001, daily_return, label=label, marker='X', s=100)  # risco quase nulo
    '''
    # Portfólio ótimo
    if optimized_weights is not None:
        opt_return = np.dot(optimized_weights, returns)
        opt_risk = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
        plt.scatter(opt_risk, opt_return, marker='*', color='red', s=200, label='Portfólio Ótimo')

    plt.xlabel('Risco (Volatilidade)')
    plt.ylabel('Retorno Esperado')
    plt.title('Fronteira Eficiente - Simulação de Carteiras vs. Renda Fixa')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_time_serie(returns: pd.DataFrame, optimized_weights):
    """
    Plota o crescimento acumulado dos ativos e do portfólio otimizado.

    Parâmetros:
        returns (pd.DataFrame): DataFrame com returns diários dos ativos
        optimized_weights (np.ndarray): pesos calculados para cada ativo
    """
    # Retornos acumulados individuais
    returns_acumulados = (1 + returns).cumprod()

    # Retorno acumulado do portfólio
    retorno_portfolio = (1 + returns.dot(optimized_weights)).cumprod()

    # Adiciona coluna do portfólio
    returns_acumulados['Portfólio'] = retorno_portfolio

    plt.figure(figsize=(12, 6))
    for coluna in returns_acumulados.columns:
        #estilo = '-' if coluna != 'Portfólio' else '--'
        linewidth= 2 if coluna == 'Portfólio' else 1
        color = 'black' if coluna == 'Portfólio' else None
        plt.plot(returns_acumulados.index, returns_acumulados[coluna], label=coluna, linestyle='-', linewidth=linewidth, color=color)

    plt.title("Crescimento Acumulado: Ativos Individuais vs. Portfólio Otimizado")
    plt.xlabel("Data")
    plt.ylabel("Crescimento Acumulado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def markowitz_tradeoff(mean_returns, cov_matrix, interval=0.1):
    lamb_array = np.arange(0.0,1.1,interval)
    ret_list = []
    vol_list = []
    for lamb in lamb_array:
        weights_markowitz = solve_markowitz(mean_returns, cov_matrix,lamb=lamb)
        ret = portfolio_return(weights_markowitz, mean_returns)
        vol = portfolio_volatility(weights_markowitz, cov_matrix)
        ret_list.append(ret)
        vol_list.append(vol)
    return ret_list, vol_list

def compute_benchmark_growth(benchmark_annual_rate, freq, num_periods):
    freq_map = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'annual': 1
    }
    periodic_rate = (1 + benchmark_annual_rate) ** (1 / freq_map[freq]) - 1
    return (1 + periodic_rate) ** np.arange(num_periods)

def plot_frontier_line(vol_list, ret_list):
    plt.plot(vol_list, ret_list, marker='o')
    plt.title("Fronteira Eficiente - Risco vs Retorno")
    plt.xlabel("Risco (Volatilidade)")
    plt.ylabel("Retorno Esperado")
    plt.grid()
    plt.show()

def plot_comparison_time_series(returns_daily, returns_monthly,mean_returns_daily, cov_matrix_daily,mean_returns_monthly, cov_matrix_monthly,label_daily="Portfólio Diário", label_monthly="Portfólio Mensal"):
    """
    Compara as séries temporais acumuladas dos portfólios ótimo diário e mensal.

    Parâmetros:
        returns_daily (pd.DataFrame): retornos diários dos ativos.
        returns_monthly (pd.DataFrame): retornos mensais dos ativos.
        mean_returns_daily, cov_matrix_daily: estatísticas da escala diária.
        mean_returns_monthly, cov_matrix_monthly: estatísticas da escala mensal.
    """
    # Calcula pesos ótimos em cada escala (λ=1 = avesso ao risco)
    weights_daily = solve_markowitz(mean_returns_daily, cov_matrix_daily, lamb=1)
    weights_monthly = solve_markowitz(mean_returns_monthly, cov_matrix_monthly, lamb=1)

    # Série temporal do portfólio diário e mensal
    portfolio_returns_daily = returns_daily.dot(weights_daily)
    portfolio_returns_monthly = returns_monthly.dot(weights_monthly)

    # Converte o portfólio mensal para o mesmo índice de tempo diário (para plot comparável)
    portfolio_cum_daily = (1 + portfolio_returns_daily).cumprod()
    portfolio_cum_monthly = (1 + portfolio_returns_monthly).cumprod()

    # Plot comparativo
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cum_daily.index, portfolio_cum_daily, label=label_daily, color='blue', linewidth=2)
    plt.plot(portfolio_cum_monthly.index, portfolio_cum_monthly, label=label_monthly, color='orange', linewidth=2, linestyle='--')
    plt.title("Comparação de Desempenho: Portfólios Diário vs Mensal")
    plt.xlabel("Data")
    plt.ylabel("Crescimento Acumulado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_lambda_for_target(mean_returns, cov_matrix, target_return, interval=0.01):
    best_lambda = None
    best_diff = float('inf')
    ret_list, vol_list = markowitz_tradeoff(mean_returns, cov_matrix, interval=interval)
    lambdas = np.arange(0, 1 + interval, interval)
    
    for l, r in zip(lambdas, ret_list):
        diff = abs(r - target_return)
        if diff < best_diff:
            best_diff = diff
            best_lambda = l
    return best_lambda
