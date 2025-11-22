# src/predictors/expected_returns.py
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.loader import compute_returns
from sklearn.metrics import mean_squared_error, r2_score


def create_features(returns: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Cria features simples baseadas nos retornos passados.
    - média móvel
    - desvio padrão móvel
    - retorno defasado (lag)

    Retorna um DataFrame com as features para cada ativo.
    """
    features = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        features[f"{col}_lag1"] = returns[col].shift(1)
        features[f"{col}_mean_{window}"] = returns[col].rolling(window).mean()
        features[f"{col}_std_{window}"] = returns[col].rolling(window).std()

    return features.dropna()  # remove NaNs gerados pelos shifts e rolling


# NOVO — PREVISÃO TEMPORAL (ROLLING WALK-FORWARD)
def predict_daily_series_lr(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Gera uma SÉRIE TEMPORAL de previsões diárias usando walk-forward.

    PARA CADA ATIVO:
        Para cada dia t:
            - Treina o modelo com dados até t-1
            - Prevê o retorno de t
        Gera uma série (mesmo tamanho de returns)

    Isso corrige o erro anterior (que usava APENAS o último dia).
    """

    returns = compute_returns(prices, freq="daily")
    features = create_features(returns, window)
    features = features.dropna()

    tickers = returns.columns
    preds = pd.DataFrame(index=features.index, columns=tickers)

    X_raw = features.values

    # Padronização fixa (mas NÃO treinamos o modelo nos dados futuros)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # -----------------------------
    # Um loop temporal por ativo
    # -----------------------------
    for col in tickers:
        y = returns[col].loc[features.index].values
        y_pred_list = []

        min_train = 30

        for t in range(min_train, len(X_scaled)):
            X_train = X_scaled[:t]
            y_train = y[:t]

            X_test = X_scaled[t].reshape(1, -1)

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            y_pred_t = model.predict(X_test)[0]
            y_pred_list.append(y_pred_t)

        # CORREÇÃO: substitui chained assignment pelo método correto
        preds.loc[features.index[min_train:], col] = y_pred_list
    return preds


# Retorno esperado (μ) a partir da PREVISÃO temporal
def expected_return_from_predictions(pred_series: pd.DataFrame):
    """
    Converte a série de previsões diárias (walk-forward)
    em um retorno esperado estimado:

    μ_diário = média das previsões
    μ_mensal = (1 + μ_diário)^21 - 1
    """
    mu_daily = pred_series.mean()
    mu_monthly = (1 + mu_daily) ** 21 - 1
    return mu_daily, mu_monthly


# Avaliação temporal REAL da previsão (walk-forward)
def evaluate_prediction_series(real: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """
    Avalia previsão temporal diária (walk-forward) com MSE, R² e correlação.
    """
    results = []

    for col in real.columns:
        r = real[col].loc[pred.index].dropna()
        p = pred[col].loc[pred.index].dropna()

        # alinhar tamanho
        idx = r.index.intersection(p.index)
        r = r.loc[idx]
        p = p.loc[idx]

        mse = ((r - p) ** 2).mean()
        corr = r.corr(p)
        r2 = 1 - (r - p).var() / r.var()

        results.append({
            "Ticker": col,
            "MSE": mse,
            "R²": r2,
            "Correlação": corr
        })

    return pd.DataFrame(results)


# (Mantido) — Inspeção de coeficientes (mesma lógica)
def inspect_coefficients(prices: pd.DataFrame, window: int = 5):
    """
    Mostra os coeficientes da regressão linear para cada ativo
    usando COMO TREINO o dataset inteiro (sem previsão).
    """
    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window)
    X = features.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    coefs = pd.DataFrame(index=X.columns)

    for col in returns.columns:
        y = returns[col].loc[X.index].values
        model = Ridge(alpha=1.0).fit(X_scaled, y)
        coefs[col] = model.coef_

    return coefs
