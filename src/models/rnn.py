# =============================================================
# RNN RÁPIDA WALK-FORWARD (TREINO ÚNICO + AUTO-REGRESSÃO)
# =============================================================

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.loader import compute_returns


# =============================================================
# (A) Criar janelas univariadas
# =============================================================
def create_sequences(y, seq_len):
    X, y_out = [], []
    for i in range(seq_len, len(y)):
        X.append(y[i-seq_len:i].reshape(seq_len, 1))
        y_out.append(y[i])
    return np.array(X), np.array(y_out)


# =============================================================
# (B) Modelo LSTM simples
# =============================================================
class RNNRegressor(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)


# =============================================================
# (C) Treinar modelo
# =============================================================
def train_rnn(X_train, y_train, hidden_dim, num_layers, epochs, batch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.from_numpy(X_train).float().to(device)
    y_t = torch.from_numpy(y_train).float().to(device)

    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)

    model = RNNRegressor(hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return model


# =============================================================
# (D) WALK-FORWARD RÁPIDO (Treina 1 vez + prevê T dias)
# =============================================================
def predict_daily_series_rnn(
    prices,
    seq_len=20,
    hidden_dim=32,
    num_layers=1,
    epochs=40,
    batch_size=32,
    train_ratio=0.8
):
    """
    Para cada ativo:
        1. divide série em treino e teste (temporal)
        2. treina UMA RNN com a parte de treino
        3. previsão walk-forward auto-regressiva no teste
    """
    returns = compute_returns(prices)
    tickers = returns.columns
    preds_df = pd.DataFrame(index=returns.index, columns=tickers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for col in tickers:

        y = returns[col].dropna().values
        idx = returns[col].dropna().index

        if len(y) < seq_len + 40:
            continue

        # divisão temporal
        split = int(len(y) * train_ratio)
        y_train = y[:split]
        y_test  = y[split:]

        # criar janelas só com dados de treino
        X_train, y_train_seq = create_sequences(y_train, seq_len)

        # treinar modelo 1 vez
        model = train_rnn(
            X_train, y_train_seq,
            hidden_dim, num_layers,
            epochs, batch_size
        )

        # previsão walk-forward REAL
        preds = [np.nan] * len(y)

        # janela inicial (última parte do treino)
        janela = y[split-seq_len:split].copy()

        model.eval()
        for t in range(split, len(y)):
            x = torch.from_numpy(janela.reshape(1, seq_len, 1)).float().to(device)

            with torch.no_grad():
                pred = model(x).item()

            preds[t] = pred

            # shift da janela
            janela = np.roll(janela, -1)
            janela[-1] = pred   # usa previsão (auto-regressão)

        preds_df[col].loc[idx] = preds

    return preds_df


# =============================================================
# (E) Retorno esperado (μ previsto)
# =============================================================
def expected_return_from_predictions(pred_series):
    mu_daily = pred_series.mean()
    mu_monthly = (1 + mu_daily)**21 - 1
    return mu_daily, mu_monthly


# =============================================================
# (F) Avaliação temporal
# =============================================================
def evaluate_rnn_model(prices, pred_series):
    returns = compute_returns(prices)
    results = []

    for col in returns.columns:
        r = returns[col].loc[pred_series.index].dropna()
        p = pred_series[col].loc[pred_series.index].dropna()

        idx = r.index.intersection(p.index)
        r = r.loc[idx]
        p = p.loc[idx]

        mse = ((r - p)**2).mean()
        corr = r.corr(p)
        r2  = 1 - (r - p).var() / r.var()

        results.append({
            "Ticker": col,
            "MSE": mse,
            "R²": r2,
            "Correlação": corr
        })

    return pd.DataFrame(results)
