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


# --- Helper: Find Lambda ---
def find_lambda_for_risk(mean_returns, cov_matrix, target_risk):
    best_lambda = 0.5
    best_diff = float('inf')
    for lamb in np.arange(0, 1.05, 0.05):
        w = solve_markowitz(mean_returns, cov_matrix, lamb=lamb)
        if w is None: continue
        risk = portfolio_volatility(w, cov_matrix)
        if abs(risk - target_risk) < best_diff:
            best_diff = abs(risk - target_risk)
            best_lambda = lamb
    return best_lambda

# --- 1. Frontier Plot (Static View) ---
def compare_frontiers(models: list, num_points=2000):
    """
    Plots Efficient Frontiers comparing Average Characteristics.
    """
    plt.figure(figsize=(12, 8))

    for model in models:
        name = model["name"]
        mean = model["mean_returns"].values if isinstance(model["mean_returns"], pd.Series) else model["mean_returns"]
        cov = model["cov"].values
        is_monthly = model.get("is_monthly", False)

        # Convert Daily to Monthly for Plotting Consistency
        if not is_monthly:
            mean = (1 + mean)**21 - 1
            cov = cov * 21
            
        # 1. Simulate Random Portfolios
        w_rand = np.random.dirichlet(np.ones(len(mean)), size=num_points)
        rets = np.dot(w_rand, mean)
        vols = np.sqrt(np.diag(np.dot(w_rand, np.dot(cov, w_rand.T))))
        
        plt.scatter(vols, rets, alpha=0.1, color=model["color"], s=10)

        # 2. Trace Efficient Frontier (Lambda Optimization)
        frontier_ret, frontier_vol = [], []
        for lamb in np.arange(0, 1.01, 0.05):
            w = solve_markowitz(mean, cov, lamb=lamb)
            if w is not None:
                frontier_ret.append(portfolio_return(w, mean))
                frontier_vol.append(portfolio_volatility(w, cov))
        
        plt.plot(frontier_vol, frontier_ret, label=f"{name}", color=model["color"], linewidth=3)

    plt.title("Efficient Frontiers Comparison (Monthly Scale)")
    plt.xlabel("Monthly Volatility (Risk)")
    plt.ylabel("Monthly Expected Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# --- 2. Time Series Plot (Dynamic Backtest) ---
def compare_time_series(returns_daily: pd.DataFrame, models: list, target_risk_annual=0.15):
    """
    Backtests strategies. 
    If a model has 'pred_series', it rebalances monthly using those predictions.
    If not, it calculates static weights based on the provided mean/cov.
    """
    plt.figure(figsize=(14, 7))
    
    # Target risk logic: convert annual target to monthly/daily proxy for lambda finding
    target_risk_monthly = target_risk_annual / np.sqrt(12)

    for model in models:
        name = model["name"]
        color = model["color"]
        
        # A. DYNAMIC MODEL (Has prediction series)
        if "pred_series" in model and model["pred_series"] is not None:
            preds = model["pred_series"]
            # Align dates
            common_idx = preds.index.intersection(returns_daily.index)
            sub_preds = preds.loc[common_idx]
            sub_rets = returns_daily.loc[common_idx]
            
            # Monthly Rebalancing
            # Resample to end of month to find rebalance dates
            rebalance_dates = sub_rets.resample('ME').last().index
            
            portfolio_value = [1.0] # Start at 100%
            current_weights = np.ones(len(sub_rets.columns)) / len(sub_rets.columns)
            
            # Helper to find index location
            date_map = {d: i for i, d in enumerate(sub_rets.index)}
            
            print(f"Backtesting Dynamic: {name}...")
            
            daily_curve = []
            
            for t in range(len(sub_rets)):
                curr_date = sub_rets.index[t]
                
                # Check Rebalance
                if curr_date in rebalance_dates or t == 0:
                    # 1. Get Predicted Return (Next Month approx)
                    # We average the daily predictions for the next 21 days (proxy)
                    # or just use today's prediction vector
                    mu_pred = sub_preds.iloc[t].values 
                    mu_pred_monthly = (1 + mu_pred)**21 - 1
                    
                    # 2. Get Risk (Past 60 days Covariance)
                    past_slice = sub_rets.iloc[max(0, t-60):t]
                    if len(past_slice) > 20:
                        sigma = past_slice.cov().values * 21 # Monthly Cov
                        
                        # Find Lambda that targets our risk
                        lamb = find_lambda_for_risk(mu_pred_monthly, sigma, target_risk_monthly)
                        w = solve_markowitz(mu_pred_monthly, sigma, lamb)
                        if w is not None:
                            current_weights = w

                # Apply Returns
                day_ret = np.dot(current_weights, sub_rets.iloc[t].values)
                if len(daily_curve) == 0: daily_curve.append(1.0 * (1 + day_ret))
                else: daily_curve.append(daily_curve[-1] * (1 + day_ret))
            
            # Plot
            plt.plot(sub_rets.index, daily_curve, label=name, color=color)

        # B. STATIC MODEL (Benchmark / Historical Markowitz)
        else:
            # Calculate ONE set of weights based on provided mean/cov
            mean = model["mean_returns"]
            cov = model["cov"]
            if not model.get("is_monthly", False):
                mean = (1 + mean)**21 - 1
                cov = cov * 21
            
            lamb = find_lambda_for_risk(mean, cov, target_risk_monthly)
            weights = solve_markowitz(mean, cov, lamb)
            
            # Apply to full history
            # Resample daily returns to monthly for smoother plotting or keep daily
            # Let's do cumulative daily
            daily_port_ret = returns_daily.dot(weights)
            cum_ret = (1 + daily_port_ret).cumprod()
            
            plt.plot(cum_ret.index, cum_ret, label=f"{name} (Static)", color=color, linestyle="--")

    plt.title(f"Cumulative Performance (Target Annual Vol ~ {target_risk_annual*100:.0f}%)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()