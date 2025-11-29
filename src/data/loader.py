import yfinance as yf
import pandas as pd
import numpy as np

def load_prices(tickers, start, end, min_data_coverage=0.5):
    """
    Carrega as cotações de fechamento de cada dia ajustadas.
    
    Parâmetros:
    -----------
    tickers : list
        Lista de tickers para download
    start : str
        Data inicial (formato 'YYYY-MM-DD')
    end : str
        Data final (formato 'YYYY-MM-DD')
    min_data_coverage : float
        Proporção mínima de dados não-nulos requerida por ativo (0.0 a 1.0)
        Ativos com cobertura menor serão removidos
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com preços, contendo apenas ativos com dados suficientes
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)['Close']
    except Exception as e:
        print(f"  Erro ao baixar dados: {e}")
        raise
    
    # Se apenas um ticker, garantir que seja DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
        df.columns = tickers if isinstance(tickers, list) else [tickers]
    
    # Remover linhas onde todos os valores são NaN (antes de filtrar colunas)
    df = df.dropna(how='all')
    
    # Se DataFrame está vazio após remover linhas vazias, erro
    if len(df) == 0:
        raise ValueError(
            f"Nenhum dado foi baixado. Verifique os tickers e o periodo de analise."
        )
    
    # Filtrar ativos com dados insuficientes
    total_periods = len(df)
    min_periods = max(1, int(total_periods * min_data_coverage))
    
    valid_columns = []
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        if non_null_count >= min_periods:
            valid_columns.append(col)
        else:
            print(f"  Aviso: {col} removido - dados insuficientes ({non_null_count}/{total_periods} periodos)")
    
    if not valid_columns:
        raise ValueError(
            f"Nenhum ativo tem dados suficientes. "
            f"Verifique os tickers e o periodo de analise."
        )
    
    # Filtrar para ativos válidos
    df_valid = df[valid_columns].copy()
    
    # Preencher NaNs restantes com forward fill e backward fill
    df_valid = df_valid.ffill().bfill()
    
    # Verificação final: garantir que não há colunas completamente vazias
    df_valid = df_valid.dropna(axis=1, how='all')
    
    if df_valid.empty or len(df_valid.columns) == 0:
        raise ValueError(
            f"Nenhum ativo valido apos processamento. "
            f"Verifique os tickers e o periodo de analise."
        )
    
    return df_valid

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


