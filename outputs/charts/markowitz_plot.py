import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.optimization.markowitz import solve_markowitz, portfolio_return, portfolio_volatility

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

def find_lambda_for_risk(mean_returns, cov_matrix, target_risk, interval=0.01):
    best_lambda = None
    best_diff = float('inf')
    ret_list, vol_list = markowitz_tradeoff(mean_returns, cov_matrix, interval=interval)
    lambdas = np.arange(0, 1 + interval, interval)
    
    for l, v in zip(lambdas, vol_list):
        diff = abs(v - target_risk)
        if diff < best_diff:
            best_diff = diff
            best_lambda = l
    return best_lambda


def compare_frontiers(
    models: list,
    window: int,
    returns_monthly: pd.DataFrame,
    split_idx: int,
):
    """
    Plota fronteiras eficientes para N modelos na ESCALA MENSAL.

    Cada item de 'models':
    {
        "name": str,
        "mean_returns": pd.Series,
        "cov": pd.DataFrame,
        "is_monthly": bool,     # True = não converter
        "color": str,
        "linestyle": str
    }
    """
    lambdas=np.arange(0, 1.005, 0.005)
    actual_backtest_list = [True, False]
    actual_backtest_list_subplot_idx = [2, 1]


    plt.figure(figsize=(14, 7))

    for cc, actual_backtest in enumerate(actual_backtest_list):
        ret_curve_models = []
        vol_curve_models = []
        w_list_models = []

        for model in models:

            name = model["name"]
            mean = model["mean_returns"].copy()
            cov = model["cov"].copy()



            # ------------ FRONTEIRA λ ------------ Com backteste
            ret_curve, vol_curve, w_list = [], [], []
            for lamb in lambdas:
                w = solve_markowitz(mean, cov, lamb=lamb)
                port = returns_monthly[split_idx:]
                if actual_backtest:
                    ret_curve.append(portfolio_return(w, port.mean()))
                    vol_curve.append(portfolio_volatility(w, port.cov()))
                else:
                    ret_curve.append(portfolio_return(w, mean))
                    vol_curve.append(portfolio_volatility(w, cov))
                w_list.append(w)

                # if lamb == 0.37 or lamb==0.65:
                #     print("Lambda {}, Name: {}, Volatility: {:.4f}, Mean Return: {:.4f}".format(lamb, name,vol_curve[-1], ret_curve[-1]))
            
            ret_curve_models.append(ret_curve)
            vol_curve_models.append(vol_curve)
            w_list_models.append(w_list)

            plt.subplot(1,2,actual_backtest_list_subplot_idx[cc])
            plt.plot(
                vol_curve_models[models.index(model)], ret_curve_models[models.index(model)],
                label=f"Fronteira {name}",
                color=model["color"],
                linestyle=model["linestyle"],
                linewidth=2.5,
                # marker='o', markersize=5
            )
        if actual_backtest:
            plt.title("Comparação de Fronteiras Eficientes no Backtest —30% TESTE FUTURO")
            plt.xlabel("Risco Backtest -30% TESTE FUTURO")
            plt.ylabel("Retorno Backtest-30% TESTE FUTURO")
        else:
            plt.title("Comparação de Fronteiras Eficientes no Modelo")
            plt.xlabel("Risco Modelo")
            plt.ylabel("Retorno Modelo")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

    return ret_curve_models, vol_curve_models, lambdas, w_list_models


def compare_time_series(
    returns_daily: pd.DataFrame,
    models: list,
    target_risk=0.1
):
    """
    Compara o crescimento acumulado mensal de N modelos.

    Cada item de 'models':
    {
        "name": str,
        "mean_returns": pd.Series,
        "cov": pd.DataFrame,
        "color": str,
        "linestyle": str
    }
    """

    plt.figure(figsize=(14, 8))

    for model in models:

        name = model["name"]
        mean = model["mean_returns"]
        cov = model["cov"]
        is_monthly = model["is_monthly"]

         # ------------ CONVERSÃO (diário → mensal) ------------
        if not is_monthly:
            dias = 21
            mean = (1 + mean) ** dias - 1
            cov = cov * dias

        lamb = find_lambda_for_risk(mean, cov, target_risk)

        weights = solve_markowitz(mean, cov, lamb=lamb)

        print(lamb, name, portfolio_volatility(weights, cov), portfolio_return(weights, mean))

        port_monthly = returns_daily.resample("ME").agg(lambda x: (1 + x).prod() - 1)
        port_monthly = port_monthly.dot(weights)

        acum = (1 + port_monthly).cumprod()

        plt.plot(
            acum.index, acum,
            label=name,
            linewidth=2,
            color=model["color"],
            linestyle=model["linestyle"]
        )

    plt.title("Comparação Temporal dos Portfólios (Escala Mensal). Target Risk: {:.2f}".format(target_risk))
    plt.xlabel("Tempo (Mensal)")
    plt.ylabel("Crescimento Acumulado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
