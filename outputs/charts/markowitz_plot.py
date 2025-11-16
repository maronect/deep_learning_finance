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
    weights_daily = solve_markowitz(mean_returns_daily, cov_matrix_daily, lamb=0.5)
    weights_monthly = solve_markowitz(mean_returns_monthly, cov_matrix_monthly, lamb=0.5)

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
    pure_daily_returns: pd.Series,
    predicted_daily_returns: pd.Series,
    cov_daily: pd.DataFrame,
    pure_monthly_returns: pd.Series,
    cov_monthly: pd.DataFrame,
    num_points=4000,
    add_tradeoff_curve=True
):
    """
    Compara 3 fronteiras eficientes na escala mensal:
    1. Markowitz Puro (média diária convertida em mensal)
    2. Markowitz com Regressão Linear (daily predicted convertido em mensal)
    3. Markowitz Puro Mensal REAL (usando retornos mensais)
    Inclui também as linhas das fronteiras (λ tradeoff curves).
    """

    # 1. Converter diário -> mensal
    dias_mes = 21

    pure_monthly_conv = (1 + pure_daily_returns) ** dias_mes - 1
    pred_monthly_conv = (1 + predicted_daily_returns) ** dias_mes - 1

    cov_month_conv = cov_daily * dias_mes

    num_assets = len(pure_daily_returns)

    # 2. Simular carteiras para cada fronteira

    # A) Markowitz Puro (diário convertido -> mensal)
    means_pure, risks_pure = [], []
    for _ in range(num_points):
        w = np.random.random(num_assets)
        w /= np.sum(w)

        ret = np.dot(w, pure_monthly_conv)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_month_conv, w)))

        means_pure.append(ret)
        risks_pure.append(vol)

    # B) Markowitz com Regressão (convertido -> mensal)
    means_pred, risks_pred = [], []
    for _ in range(num_points):
        w = np.random.random(num_assets)
        w /= np.sum(w)

        ret = np.dot(w, pred_monthly_conv)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_month_conv, w)))

        means_pred.append(ret)
        risks_pred.append(vol)

    # C) Markowitz Mensal REAL
    means_real, risks_real = [], []
    for _ in range(num_points):
        w = np.random.random(num_assets)
        w /= np.sum(w)

        ret = np.dot(w, pure_monthly_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_monthly, w)))

        means_real.append(ret)
        risks_real.append(vol)

    # 3. Plotar fronteiras simuladas (scatter)

    plt.figure(figsize=(13, 8))

    plt.scatter(risks_pure, means_pure, alpha=0.35, color="blue",
                label="Puro (diário -> mensal)")
    plt.scatter(risks_pred, means_pred, alpha=0.35, color="orange",
                label="Regressão (diário -> mensal)")
    plt.scatter(risks_real, means_real, alpha=0.35, color="green",
                label="Puro Mensal REAL")

    # 4. Adicionar linhas da fronteira (λ curves)
    if add_tradeoff_curve:

        lamb_arr = np.arange(0, 1.05, 0.05)

        # Linha A: Puro (diário convertido)
        ret_curve_pure = []
        vol_curve_pure = []
        for lamb in lamb_arr:
            w = solve_markowitz(pure_monthly_conv, cov_month_conv, lamb=lamb)
            ret_curve_pure.append(portfolio_return(w, pure_monthly_conv))
            vol_curve_pure.append(portfolio_volatility(w, cov_month_conv))

        plt.plot(
            vol_curve_pure, ret_curve_pure,
            color="navy", linewidth=2.3, linestyle="--",
            label="Fronteira Puro (D->M)"
        )

        # Linha B: Regressão (convertido)
        ret_curve_pred = []
        vol_curve_pred = []
        for lamb in lamb_arr:
            w = solve_markowitz(pred_monthly_conv, cov_month_conv, lamb=lamb)
            ret_curve_pred.append(portfolio_return(w, pred_monthly_conv))
            vol_curve_pred.append(portfolio_volatility(w, cov_month_conv))

        plt.plot(
            vol_curve_pred, ret_curve_pred,
            color="darkorange", linewidth=2.3, linestyle="--",
            label="Fronteira LR (D->M)"
        )

        # Linha C: Mensal REAL
        ret_curve_real = []
        vol_curve_real = []
        for lamb in lamb_arr:
            w = solve_markowitz(pure_monthly_returns, cov_monthly, lamb=lamb)
            ret_curve_real.append(portfolio_return(w, pure_monthly_returns))
            vol_curve_real.append(portfolio_volatility(w, cov_monthly))

        plt.plot(
            vol_curve_real, ret_curve_real,
            color="darkgreen", linewidth=2.3, linestyle="--",
            label="Fronteira Real (Mensal)"
        )

    plt.xlabel("Risco (Volatilidade Mensal)")
    plt.ylabel("Retorno Esperado Mensal")
    plt.title("Comparação de Fronteiras Eficientes na Escala Mensal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_compare_time_series(
    returns_daily: pd.DataFrame,
    returns_monthly: pd.DataFrame,
    pure_daily_returns: pd.Series,
    predicted_daily_returns: pd.Series,
    pure_monthly_returns: pd.Series,
    cov_daily: pd.DataFrame,
    cov_monthly: pd.DataFrame,
    lamb=0.5
):
    """
    Compara o crescimento acumulado MENSAL dos 3 portfólios:

    1. Markowitz Puro (pesos diários convertidos para mensal)
    2. Markowitz com Regressão Linear (pesos diários convertidos para mensal)
    3. Markowitz Puro Mensal REAL (pesos mensais)
    """

    # =============== Pesos na escala diária ===============

    weights_pure_daily = solve_markowitz(pure_daily_returns, cov_daily, lamb=lamb)
    weights_pred_daily = solve_markowitz(predicted_daily_returns, cov_daily, lamb=lamb)

    # =============== Pesos na escala mensal (reais) ===============

    weights_month_real = solve_markowitz(pure_monthly_returns, cov_monthly, lamb=lamb)

    # =============== Retorno diário dos portfólios ===============

    portfolio_pure_daily = returns_daily.dot(weights_pure_daily)
    portfolio_pred_daily = returns_daily.dot(weights_pred_daily)

    # Converter para retornos mensais somando por agregação:

    portfolio_pure_monthly = portfolio_pure_daily.resample("ME").agg(lambda x: (1 + x).prod() - 1)
    portfolio_pred_monthly = portfolio_pred_daily.resample("ME").agg(lambda x: (1 + x).prod() - 1)

    # Portfólio REAL baseado em retornos mensais:

    portfolio_month_real = returns_monthly.dot(weights_month_real)

    # =============== Crescimento acumulado ===============

    pure_acum = (1 + portfolio_pure_monthly).cumprod()
    pred_acum = (1 + portfolio_pred_monthly).cumprod()
    real_acum = (1 + portfolio_month_real).cumprod()

    # =============== Plot ===============

    plt.figure(figsize=(12, 7))

    plt.plot(pure_acum.index, pure_acum, label="Markowitz Puro (diário->mensal)",
             linewidth=2, color="blue")

    plt.plot(pred_acum.index, pred_acum, label="Regressão (diário->mensal)",
             linewidth=2, linestyle="--", color="orange")

    plt.plot(real_acum.index, real_acum, label="Markowitz Mensal REAL",
             linewidth=2, color="green")

    plt.xlabel("Data (Escala Mensal)")
    plt.ylabel("Crescimento Acumulado")
    plt.title("Comparação Temporal dos Portfólios (Escala Mensal)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
