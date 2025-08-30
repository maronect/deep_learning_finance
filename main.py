from src.data.loader import load_prices, compute_returns
from src.optimization.markowitz import minimize_volatility

tickers = ['AAPL', 'MSFT', 'GOOGL']
prices = load_prices(tickers, '2020-01-01', '2023-01-01')
returns = compute_returns(prices)

mean_returns = returns.mean()
cov_matrix = returns.cov()

weights = minimize_volatility(mean_returns, cov_matrix, target_return=mean_returns.mean())
print("Pesos ótimos:", weights)
