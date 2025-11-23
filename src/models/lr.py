import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.data.loader import compute_returns

def create_features(returns: pd.DataFrame, windows=[5, 21]) -> pd.DataFrame:
    """Creates lagged features and rolling stats."""
    features = pd.DataFrame(index=returns.index)
    for col in returns.columns:
        # 1. Lagged Returns (Yesterday's return)
        features[f"{col}_lag1"] = returns[col].shift(1)
        
        # 2. Rolling Momentum & Volatility
        for w in windows:
            features[f"{col}_mean_{w}"] = returns[col].rolling(w).mean().shift(1)
            features[f"{col}_std_{w}"] = returns[col].rolling(w).std().shift(1)
            
    return features.dropna()

def predict_daily_series_lr(prices: pd.DataFrame, window_features=[5, 21], training_window=252, refit_step=21) -> pd.DataFrame:
    """
    Walk-Forward Linear Regression.
    - training_window: How many past days to use for training (e.g., 1 year).
    - refit_step: Train a new model every N days (e.g., every month) to save time.
    """
    returns = compute_returns(prices, freq="daily")
    features = create_features(returns, window_features)
    
    # Align indices
    common_idx = features.index.intersection(returns.index)
    features = features.loc[common_idx]
    returns = returns.loc[common_idx]
    
    tickers = returns.columns
    preds_df = pd.DataFrame(index=features.index, columns=tickers)
    
    print(f"Starting LR Walk-Forward (Train Window: {training_window}, Refit: {refit_step})...")
    
    # Pre-initialize models and scaler
    models = {t: Ridge(alpha=1.0) for t in tickers}
    scaler = StandardScaler()
    
    # Start loop after enough data exists
    start_index = training_window
    
    for t in range(start_index, len(features)):
        curr_date = features.index[t]
        
        # 1. Re-Train Model (periodically)
        if (t - start_index) % refit_step == 0:
            # Training Data: Window [t-training_window : t]
            X_train = features.iloc[t-training_window : t].values
            
            # FIT SCALER HERE (Prevents Leakage)
            X_train_scaled = scaler.fit_transform(X_train)
            
            for ticker in tickers:
                y_train = returns[ticker].iloc[t-training_window : t].values
                models[ticker].fit(X_train_scaled, y_train)
        
        # 2. Predict Today
        # Transform today's features using the scaler fitted on the past
        X_test = features.iloc[t].values.reshape(1, -1)
        X_test_scaled = scaler.transform(X_test)
        
        for ticker in tickers:
            preds_df.loc[curr_date, ticker] = models[ticker].predict(X_test_scaled)[0]
            
    return preds_df.astype(float).dropna()

def expected_return_from_predictions(pred_series: pd.DataFrame):
    """Calculates average daily and monthly expected returns."""
    mu_daily = pred_series.mean()
    mu_monthly = (1 + mu_daily)**21 - 1
    return mu_daily, mu_monthly

def evaluate_prediction_series(real_returns: pd.DataFrame, pred_series: pd.DataFrame):
    results = []
    common_idx = real_returns.index.intersection(pred_series.index)
    
    for col in real_returns.columns:
        y_true = real_returns.loc[common_idx, col]
        y_pred = pred_series.loc[common_idx, col]
        
        mse = ((y_true - y_pred)**2).mean()
        # Out-of-Sample R2
        r2 = 1 - (mse / y_true.var())
        corr = y_true.corr(y_pred)
        
        results.append({"Ticker": col, "MSE": mse, "R2": r2, "Corr": corr})
        
    return pd.DataFrame(results)