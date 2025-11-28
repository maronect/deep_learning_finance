# src/models/rnn.py

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.data.loader import compute_returns
from src.models.lr import create_features  # ajusta se seu caminho for outro


# -----------------------
# Helper: cria sequências (janela deslizante)
# -----------------------
def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    """
    Transforma uma série tabular em sequências para RNN/LSTM.

    X: array 2D (tempo x features)
    y: array 1D (retornos)
    seq_len: tamanho da janela temporal

    Retorna:
        X_seq: (num_sequencias, seq_len, num_features)
        y_seq: (num_sequencias,)
    """
    seq_X = []
    seq_y = []
    for i in range(seq_len, len(X)):
        seq_X.append(X[i - seq_len:i])
        seq_y.append(y[i])
    return np.array(seq_X), np.array(seq_y)


# -----------------------
# Modelo RNN simples com LSTM
# -----------------------
class RNNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_dim)
        last_out = out[:, -1, :]       # pega apenas o último passo de tempo
        out = self.fc(last_out)        # (batch, 1)
        return out.squeeze(-1)         # (batch,)


# -----------------------
# Treinar um modelo RNN em PyTorch
# -----------------------
def _train_rnn_model(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> RNNRegressor:
    """
    Treina um modelo RNN simples (LSTM) para regressão.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.from_numpy(X_seq).float().to(device)
    y_tensor = torch.from_numpy(y_seq).float().to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_seq.shape[2]
    model = RNNRegressor(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        # Se quiser ver o treino:
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(dataset):.6f}")

    return model


# -----------------------
# (1) Prever retorno médio por ativo com RNN
# -----------------------
def predict_mean_returns_rnn(
    prices: pd.DataFrame,
    window: int = 5,
    seq_len: int = 20,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 40,
    batch_size: int = 32,
    alpha_blend: float = 0.35,
    clip_value: float = 0.005,
) -> pd.Series:
    """
    Usa uma RNN (LSTM) para prever o retorno esperado (médio) de cada ativo,
    baseado em janelas temporais das features.

    Mantém a mesma ideia do modelo linear:
    - Treina um modelo por ativo
    - Usa toda a série para treino
    - Preve o próximo retorno (t+1)
    - Aplica clipping + shrink (blend com média histórica) para não explodir.
    """
    returns = compute_returns(prices, freq="daily")
    features = create_features(returns, window=window)

    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tickers = returns.columns
    historical_mean = returns.mean()

    predicted_means = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for col in tickers:
        # Série de retornos desse ativo alinhada às features
        y = returns[col].loc[features.index].values

        # Cria sequências para RNN
        X_seq, y_seq = create_sequences(X_scaled, y, seq_len=seq_len)

        # Treina o modelo RNN nesse ativo
        model = _train_rnn_model(
            X_seq,
            y_seq,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Previsão do próximo retorno: usa a última janela conhecida
        X_last_seq = X_scaled[-seq_len:].reshape(1, seq_len, X_scaled.shape[1])
        X_last_tensor = torch.from_numpy(X_last_seq).float().to(device)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_last_tensor).cpu().numpy()[0]

        # Clipping para não explodir
        y_pred = np.clip(y_pred, -clip_value, clip_value)

        # Shrink (blend com média histórica)
        y_pred_final = alpha_blend * y_pred + (1 - alpha_blend) * historical_mean[col]

        predicted_means[col] = y_pred_final

    return pd.Series(predicted_means, name="Predicted_RNN_Mean_Returns")


# -----------------------
# (2) Avaliar qualidade preditiva da RNN (MSE, R², Correlação)
# -----------------------
def evaluate_rnn_model(
    prices: pd.DataFrame,
    window: int = 5,
    seq_len: int = 20,
    hidden_dim: int = 32,
    num_layers: int = 1,
    epochs: int = 40,
    batch_size: int = 32,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """
    Avalia a RNN para cada ativo usando divisão treino/teste sequencial.

    Retorna um DataFrame com:
        - MSE
        - R²
        - Correlação (real vs previsto)
    """
    returns = compute_returns(prices, freq="daily")
    features = create_features(returns, window=window)

    X_raw = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tickers = returns.columns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for col in tickers:
        y = returns[col].loc[features.index].values

        X_seq, y_seq = create_sequences(X_scaled, y, seq_len=seq_len)

        if len(X_seq) < 10:
            # Muito pouco dado para esse ativo nessa configuração
            continue

        split_idx = int(len(X_seq) * (1 - test_size))

        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Treina modelo na parte de treino
        model = _train_rnn_model(
            X_train,
            y_train,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
        )

        # Previsão na parte de teste
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(device)
            y_pred_tensor = model(X_test_tensor)
            y_pred = y_pred_tensor.cpu().numpy()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        corr = np.corrcoef(y_test, y_pred)[0, 1]

        results.append({
            "Ticker": col,
            "MSE": mse,
            "R²": r2,
            "Correlação": corr,
        })

    return pd.DataFrame(results)
