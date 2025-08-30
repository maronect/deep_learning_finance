import yfinance as yf
import pandas as pd

def load_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end)['Close']
    return df.dropna()

def compute_returns(prices):
    return prices.pct_change().dropna()
