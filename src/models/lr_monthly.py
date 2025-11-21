import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.models.features import create_features
from src.models.targets import create_monthly_target

def predict_monthly_lr(prices, horizon=21, alpha_blend=0.35, clip_value=0.02):
    returns = prices.pct_change().dropna()

    X = create_features(returns)
    y_future = create_monthly_target(returns, horizon=horizon)

    X, y_future = X.align(y_future, join="inner", axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    predicted_means = {}
    historical_mean = y_future.mean()

    for col in returns.columns:
        y = y_future[col].values

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        # previsão usando a última janela
        X_last = X_scaled[-1].reshape(1, -1)
        y_pred = model.predict(X_last)[0]

        y_pred = np.clip(y_pred, -clip_value, clip_value)

        y_final = alpha_blend * y_pred + (1 - alpha_blend) * historical_mean[col]
        predicted_means[col] = y_final

    return pd.Series(predicted_means, name="LR_Monthly_Prediction")
