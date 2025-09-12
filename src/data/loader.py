import yfinance as yf
import pandas as pd

def load_prices(tickers, start, end):
    '''
    carrega as cotações de fechamento de cada dia ajustadas
    '''
    df = yf.download(tickers, start=start, end=end)['Close']
    return df.dropna()

def compute_returns(prices): 
    '''
    Variação percentual entre um dia e o anterior.
    dropna remove os valores NaN que aparecem no primeiro dia.
    '''
    return prices.pct_change().dropna() 
