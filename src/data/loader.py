import yfinance as yf
import pandas as pd
import numpy as np

def load_prices(tickers, start, end):
    '''
    carrega as cotações de fechamento de cada dia ajustadas
    '''
    df = yf.download(tickers, start=start, end=end)['Close']
    return df.dropna()

def compute_returns(prices, freq='daily'):
    """
    Calcula os retornos com base na frequencia selecionada.
    
    Parametros:
        prices (pd.DataFrame): preços ajustados dos ativos
        freq (str): 'daily', 'weekly', 'monthly', 'annual'

    Retorna:
        pd.DataFrame com retornos
    """
    freq_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'ME',
        'annually': 'YE'
    }

    if freq not in freq_map:
        raise ValueError("Frequência inválida. Use: 'daily', 'weekly', 'monthly' ou 'annual'.")

    if freq == 'daily':
        return prices.pct_change().dropna()
    else:
        resampled = prices.resample(freq_map[freq]).last()
        return resampled.pct_change().dropna()

def ajustar_risk_free(rf_ano, freq='daily'):
    freq_map = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'annual': 1
    }
    return (1 + rf_ano) ** (1 / freq_map[freq]) - 1

def converter_periodo(ret, vol, dias):
    """
    Converte retorno e volatilidade diária para outro período.
    dias: número médio de dias úteis (21 = mensal, 252 = anual)
    """
    ret_conv = (1 + ret) ** dias - 1
    vol_conv = vol * np.sqrt(dias)
    return ret_conv, vol_conv


