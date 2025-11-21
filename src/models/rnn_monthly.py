import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.features import create_features
from src.models.targets import create_monthly_target

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

def train_rnn(X, y, seq_len=60, hidden_dim=64, epochs=80):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]

    X_seq, y_seq = create_sequences(X, y, seq_len)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(device)

    model = RNNModel(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor).squeeze()
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= 12:  # Early stopping
                break

    model.load_state_dict(best_state)
    return model

def predict_monthly_rnn(prices, horizon=21, seq_len=60, alpha_blend=0.20, clip_value=0.02):
    returns = prices.pct_change().dropna()
    X = create_features(returns)
    y_future = create_monthly_target(returns, horizon=horizon)

    X, y_future = X.align(y_future, join="inner", axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tickers = returns.columns
    historical_mean = y_future.mean()

    predicted_means = {}

    for col in tickers:
        y = y_future[col].values

        model = train_rnn(X_scaled, y, seq_len=seq_len)

        X_last = torch.tensor(X_scaled[-seq_len:].reshape(1, seq_len, X_scaled.shape[1]), dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_last).item()

        y_pred = np.clip(y_pred, -clip_value, clip_value)
        y_final = alpha_blend * y_pred + (1 - alpha_blend) * historical_mean[col]

        predicted_means[col] = y_final

    return pd.Series(predicted_means, name="RNN_Monthly_Prediction")
