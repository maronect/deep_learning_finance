import numpy as np
import pandas as pd

def create_monthly_target(returns: pd.DataFrame, horizon=21) -> pd.DataFrame:
    """
    Para cada ativo:
    y[t] = retorno acumulado nos próximos 21 dias (shift negativo)
    """
    future_ret = (1 + returns).rolling(horizon).apply(np.prod, raw=True) - 1
    future_ret = future_ret.shift(-horizon)
    return future_ret.dropna()
