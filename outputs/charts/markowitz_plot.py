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
    plt.title('Fronteira Eficiente - Simulação de Carteiras')
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


def compare_pure_mk_with_lr(
    pure_returns: pd.Series,
    predicted_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    num_points=5000,
    add_tradeoff_curve=True
):
    """
    Compara a fronteira eficiente do Markowitz puro com a fronteira usando
    retornos previstos pela Regressão Linear (predicted_returns).

    Parâmetros:
        pure_returns (pd.Series): médias históricas tradicionais dos ativos.
        predicted_returns (pd.Series): retornos previstos pelo modelo (E[R]).
        cov_matrix (pd.DataFrame): matriz de covariância dos ativos.
    """

    num_assets = len(pure_returns)


    pure_means = []
    pure_risks = []

    for _ in range(num_points):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        pure_means.append(np.dot(w, pure_returns))
        pure_risks.append(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))


    pred_means = []
    pred_risks = []

    for _ in range(num_points):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        pred_means.append(np.dot(w, predicted_returns))
        pred_risks.append(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))


    plt.figure(figsize=(12, 7))

    plt.scatter(
        pure_risks, pure_means,
        alpha=0.4, label="Markowitz Puro (Histórico)",
        color="blue"
    )

    plt.scatter(
        pred_risks, pred_means,
        alpha=0.4, label="Markowitz com Regressão (LR)",
        color="orange"
    )


    if add_tradeoff_curve:
        lamb_array = np.arange(0, 1.05, 0.05)

        # --- Puro ---
        ret_pure = []
        vol_pure = []

        for lamb in lamb_array:
            w = solve_markowitz(pure_returns, cov_matrix, lamb=lamb)
            ret_pure.append(portfolio_return(w, pure_returns))
            vol_pure.append(portfolio_volatility(w, cov_matrix))

        plt.plot(
            vol_pure, ret_pure,
            linestyle="--", color="darkblue",
            linewidth=2, label="Fronteira Teórica - Puro"
        )

        # --- LR ---
        ret_pred = []
        vol_pred = []

        for lamb in lamb_array:
            w = solve_markowitz(predicted_returns, cov_matrix, lamb=lamb)
            ret_pred.append(portfolio_return(w, predicted_returns))
            vol_pred.append(portfolio_volatility(w, cov_matrix))

        plt.plot(
            vol_pred, ret_pred,
            linestyle="--", color="darkorange",
            linewidth=2, label="Fronteira Teórica - LR"
        )


    plt.xlabel("Risco (Volatilidade)")
    plt.ylabel("Retorno Esperado")
    plt.title("Comparação das Fronteiras: Markowitz Puro vs. Regressão Linear")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_time_series(
    returns: pd.DataFrame,
    pure_returns: pd.Series,
    predicted_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    lamb=1.0
):
    """
    Compara o crescimento acumulado dos portfólios ótimos:
    - Markowitz Puro (mean historical)
    - Markowitz com Regressão Linear (predicted mean returns)

    Parâmetros:
        returns (pd.DataFrame): retornos diários reais dos ativos
        pure_returns (pd.Series): média histórica dos retornos
        predicted_returns (pd.Series): retornos previstos pelo modelo LR
        cov_matrix (pd.DataFrame): matriz de covariância dos ativos
        lamb (float): parâmetro de aversão a risco (0 = max retorno, 1 = min risco)
    """

    weights_pure = solve_markowitz(pure_returns, cov_matrix, lamb=lamb)
    portfolio_pure = (1 + returns.dot(weights_pure)).cumprod()

    weights_pred = solve_markowitz(predicted_returns, cov_matrix, lamb=lamb)
    portfolio_pred = (1 + returns.dot(weights_pred)).cumprod()

    plt.figure(figsize=(12, 6))

    plt.plot(
        portfolio_pure.index,
        portfolio_pure,
        label="Markowitz Puro",
        linewidth=2,
        color="blue"
    )

    plt.plot(
        portfolio_pred.index,
        portfolio_pred,
        label="Markowitz + Regressão Linear",
        linewidth=2,
        linestyle="--",
        color="orange"
    )

    plt.title("Comparação de Crescimento do Portfólio: Puro vs Regressão Linear")
    plt.xlabel("Data")
    plt.ylabel("Crescimento Acumulado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
