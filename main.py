from src.data.loader import load_prices, compute_returns
from src.optimization.markowitz import minimize_volatility

tickers = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "B3SA3.SA", "MGLU3.SA", "GGBR4.SA"]
prices = load_prices(tickers, '2020-01-01', '2023-01-01')
returns = compute_returns(prices)

mean_returns = returns.mean()
cov_matrix = returns.cov()

target_return = 0.001
weights = minimize_volatility(mean_returns, cov_matrix, target_return)

if weights is not None:
    print("Pesos ótimos:", weights)
else:
    print("Otimização falhou. Tente um target_return menor ou verifique os dados.")

