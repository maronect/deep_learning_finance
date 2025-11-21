import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.models.lr import create_features
from src.data.loader import compute_returns
from sklearn.preprocessing import StandardScaler
import math
from sklearn.linear_model import Ridge

from src.optimization.markowitz import solve_markowitz #, LinearRegression, RidgeCV


def plot_LR_parity_return_prediction(prices: pd.DataFrame, window: int = 5):

    returns = compute_returns(prices, freq='daily')
    features = create_features(returns, window=window)

    X = features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_real_all = []
    y_pred_all = []

    plt.figure(figsize=(10, 8))

    for col in returns.columns:
        y_real = returns[col].loc[X.index].values

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_real)

        y_pred = model.predict(X_scaled)

        y_real_all.extend(y_real)
        y_pred_all.extend(y_pred)

        plt.scatter(y_pred, y_real, alpha=0.25, label=col)

    lim_min = min(min(y_pred_all), min(y_real_all))
    lim_max = max(max(y_pred_all), max(y_real_all))

    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=2)

    corr = np.corrcoef(y_real_all, y_pred_all)[0, 1]

    plt.xlabel("Retorno Previsto")
    plt.ylabel("Retorno Real")
    plt.title(f"Gráfico Real vs Previsto — Correlação Global = {corr:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_LR_parity_by_all_assets(prices, window=5):

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

        model = Ridge(alpha=1.0)
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


def compare_frontiers(
    pure_monthly_returns,
    lr_pred_monthly,
    rnn_pred_monthly,
    ens_pred_monthly,
    cov_monthly,
    num_points=3000,
):
    """
    Compara 4 fronteiras eficientes:
    1. Markowitz puro mensal
    2. Regressão Linear
    3. RNN
    4. Ensemble
    """

    models = {
        "Puro": pure_monthly_returns,
        "LR": lr_pred_monthly,
        "RNN": rnn_pred_monthly,
        "Ensemble": ens_pred_monthly
    }

    plt.figure(figsize=(14, 9))

    for label, means in models.items():
        risks, rets = [], []
        for _ in range(num_points):
            w = np.random.random(len(means))
            w /= np.sum(w)

            ret = np.dot(w, means)
            vol = np.sqrt(np.dot(w.T, np.dot(cov_monthly, w)))

            risks.append(vol)
            rets.append(ret)

        plt.scatter(risks, rets, s=12, alpha=0.35, label=label)

    plt.xlabel("Risco (vol. mensal)")
    plt.ylabel("Retorno esperado mensal")
    plt.title("Comparação das Fronteiras — Puro vs LR vs RNN vs Ensemble")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_time_series(
    returns_monthly,
    pure,
    lr_pred,
    rnn_pred,
    ens_pred,
    cov_monthly,
    lamb=0.5
):
    # Pesos ótimos
    w_pure = solve_markowitz(pure, cov_monthly, lamb)
    w_lr = solve_markowitz(lr_pred, cov_monthly, lamb)
    w_rnn = solve_markowitz(rnn_pred, cov_monthly, lamb)
    w_ens = solve_markowitz(ens_pred, cov_monthly, lamb)

    p_pure = (1 + returns_monthly.dot(w_pure)).cumprod()
    p_lr   = (1 + returns_monthly.dot(w_lr)).cumprod()
    p_rnn  = (1 + returns_monthly.dot(w_rnn)).cumprod()
    p_ens  = (1 + returns_monthly.dot(w_ens)).cumprod()

    plt.figure(figsize=(14, 8))
    plt.plot(p_pure, label="Markowitz Puro", linewidth=2)
    plt.plot(p_lr, label="LR", linestyle="--", linewidth=2)
    plt.plot(p_rnn, label="RNN", linestyle="--", linewidth=2)
    plt.plot(p_ens, label="Ensemble", linewidth=3)

    plt.title("Comparação das Séries Temporais — Puro, LR, RNN e Ensemble")
    plt.xlabel("Data (mensal)")
    plt.ylabel("Crescimento acumulado")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
