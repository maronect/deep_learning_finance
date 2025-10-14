import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.optimization.markowitz import efficient_frontier_lambda

def plot_efficient_frontier(returns: pd.Series, cov_matrix, optimized_weights=None, num_points=1000):
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
    plt.colorbar(scatter, label='Sharpe Ratio')

    # Benchmarks de renda fixa, convertidos para retorno diário
    benchmarks = {
        'CDI 15% a.a.': 0.15 / 252,
        'SELIC 15% a.a.': 0.15 / 252,
        'Poupança 8% a.a.': 0.08 / 252,
        'Tesouro Prefixado 14% a.a.': 0.14 / 252
    }

    for label, daily_return in benchmarks.items():
        plt.scatter(0.00001, daily_return, label=label, marker='X', s=100)  # risco quase nulo

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

def plot_lambda_markowitz(mean_returns, cov_matrix):
    lambs = np.linspace(0, 1, 100)
    portfolios = efficient_frontier_lambda(mean_returns, cov_matrix, lambs)

    rets = [p[0] for p in portfolios]
    vols = [p[1] for p in portfolios]
    plt.plot(vols, rets)
    plt.title("Fronteira Eficiente por Combinação Lambda-Risco/Retorno")
    plt.xlabel("Volatilidade")
    plt.ylabel("Retorno Esperado")
    plt.grid(True)
    plt.show()
