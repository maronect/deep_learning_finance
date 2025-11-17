import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.models.lr import create_features
from src.data.loader import compute_returns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
import math


def plot_parity_return_prediction(prices: pd.DataFrame, window: int = 5):
    """
    Gráfico de paridade (Real vs Previsto) para a Regressão Linear.
    Mostra visualmente o alinhamento entre os valores reais e previstos.
    
    Eixo X = previsto pelo modelo
    Eixo Y = retorno real
    """

    # ------------- Preparar dados -------------
    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)

    X = features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    plt.figure(figsize=(10, 8))

    # Para armazenar todos os pontos
    y_real_all = []
    y_pred_all = []

    # ------------- Treinar e prever -------------
    for col in returns.columns:
        y = returns[col].loc[X.index]

        # Usar o mesmo modelo qualificado
        model = RidgeCV(alphas=np.logspace(-4, 2, 20))
        model.fit(X_scaled, y.values)

        # previsões para TODA a série temporal
        y_pred = model.predict(X_scaled)

        y_real_all.extend(y.values)
        y_pred_all.extend(y_pred)

        # scatter para cada ativo
        plt.scatter(y_pred, y.values, alpha=0.25, label=col)

    # ------------- Linha de paridade -------------
    lim_min = min(min(y_pred_all), min(y_real_all))
    lim_max = max(max(y_pred_all), max(y_real_all))

    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=2, label="Linha 45°")

    # ------------- Cálculo da correlação global -------------
    corr = np.corrcoef(y_real_all, y_pred_all)[0, 1]

    plt.xlabel("Retorno Previsto")
    plt.ylabel("Retorno Real")
    plt.title(f"Gráfico de Paridade (Real vs Previsto) — Correlação Global = {corr:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def plot_parity_by_all_assets(prices, window=5):
    """
    Cria um painel de subplots (3x3, 2x3 etc dependendo do número de ativos)
    mostrando os gráficos de paridade (Real vs Previsto) para cada ativo individualmente.
    """

    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)
    X = features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tickers = returns.columns
    n = len(tickers)

    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, ticker in enumerate(tickers):

        y_real = returns[ticker].loc[X.index].values

        model = LinearRegression()
        model.fit(X_scaled, y_real)

        y_pred = model.predict(X_scaled)

        corr = np.corrcoef(y_real, y_pred)[0, 1]

        plt.subplot(rows, cols, i + 1)
        plt.scatter(y_pred, y_real, alpha=0.4, s=12)
        lim_min = min(min(y_pred), min(y_real))
        lim_max = max(max(y_pred), max(y_real))
        plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1)

        plt.title(f"{ticker} (corr={corr:.2f})")
        plt.xlabel("Previsto")
        plt.ylabel("Real")
        plt.grid(True)

    plt.tight_layout()
    plt.show()

