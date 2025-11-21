import pandas as pd
import numpy as np

def create_features(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Features com poder preditivo real:
    - momentum (21, 63, 126 dias)
    - volatilidade (21, 63, 126)
    - retornos acumulados
    - lags
    - z-score
    """
    feats = pd.DataFrame(index=returns.index)

    windows = [21, 63, 126]

    for col in returns.columns:
        for w in windows:
            feats[f"{col}_mom_{w}"] = returns[col].rolling(w).sum()
            feats[f"{col}_vol_{w}"] = returns[col].rolling(w).std()
            feats[f"{col}_z_{w}"]   = (returns[col] - returns[col].rolling(w).mean()) / returns[col].rolling(w).std()

        # lag
        feats[f"{col}_lag1"] = returns[col].shift(1)
        feats[f"{col}_lag5"] = returns[col].shift(5)
    
    return feats.dropna()
