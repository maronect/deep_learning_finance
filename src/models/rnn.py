import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, r2_score
from src.data.loader import compute_returns


# =============================================================
# (A) Criar janelas univariadas
# =============================================================
def create_univariate_sequences(y: np.ndarray, seq_len=20):
    X, y_out = [], []
    for i in range(seq_len, len(y)):
        janela = y[i-seq_len:i]
        alvo   = y[i]
        X.append(janela.reshape(seq_len, 1))
        y_out.append(alvo)
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
        h = out[:, -1, :]        # último passo
        return self.fc(h).squeeze(-1)


# =============================================================
# (C) Treinar modelo em um único bloco de dados
# =============================================================
def _train_model(X_train, y_train, hidden_dim, num_layers, epochs, batch_size):
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
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return model


# =============================================================
# (D) PREVISÃO DIÁRIA WALK-FORWARD (sem leakage)
# =============================================================
def predict_daily_series_rnn(
    prices,
    seq_len=20,
    hidden_dim=32,
    num_layers=1,
    epochs=40,
    batch_size=32
):
    """
    Para cada ativo:
        Para cada dia t:
            - Treina modelo com dados até t-1
            - Preve retorno em t
        -> série temporal de previsão real
    """

    returns = compute_returns(prices)
    tickers = returns.columns

    preds_df = pd.DataFrame(index=returns.index, columns=tickers)

    for col in tickers:
        y = returns[col].dropna().values
        idx = returns[col].dropna().index

        if len(y) < seq_len + 30:
            continue

        pred_series = [np.nan] * len(y)

        # walk-forward
        for t in range(seq_len+30, len(y)):
            y_train = y[:t]

            X_train, y_train_seq = create_univariate_sequences(y_train, seq_len)

            if len(X_train) < 10:
                continue

            model = _train_model(
                X_train, y_train_seq,
                hidden_dim, num_layers, epochs, batch_size
            )

            # previsão do retorno em t
            x_last = y[t-seq_len:t].reshape(1, seq_len, 1)
            x_last_t = torch.from_numpy(x_last).float()

            model.eval()
            with torch.no_grad():
                pred = model(x_last_t).item()

            pred_series[t] = pred

        preds_df[col].loc[idx] = pred_series

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
