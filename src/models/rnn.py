import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.data.loader import compute_returns

# --- Model Definition ---
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Use last time step

# --- Helper: Create Sequences ---
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- Main Prediction Function ---
def predict_daily_series_rnn(prices: pd.DataFrame, seq_len=20, epochs=20, hidden_dim=32) -> pd.DataFrame:
    """
    Generates predictions using LSTM. 
    Uses a single Train/Test split approach for speed, 
    but applies the model in a walk-forward manner on the test set.
    """
    returns = compute_returns(prices, freq="daily").dropna()
    tickers = returns.columns
    preds_df = pd.DataFrame(index=returns.index, columns=tickers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LSTM on {device}...")

    for col in tickers:
        # Prepare Data
        data_raw = returns[col].values.reshape(-1, 1)
        
        # Split: Train on first 70%, Predict on last 30%
        split = int(len(data_raw) * 0.7)
        train_raw = data_raw[:split]
        test_raw = data_raw[split - seq_len:] # Need overlap for first prediction
        
        # Scale (Fit on TRAIN only)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        
        # Create Train Sequences
        X_train, y_train = create_sequences(train_scaled, seq_len)
        
        # Train Model
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
            batch_size=32, shuffle=True
        )
        
        model = LSTMRegressor(input_size=1, hidden_size=hidden_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        
        # Predict on Test (Walk-Forward using Actual History)
        model.eval()
        X_test, _ = create_sequences(test_scaled, seq_len)
        
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).float().to(device)
            preds_scaled = model(X_test_t).cpu().numpy()
            
        preds_actual = scaler.inverse_transform(preds_scaled).flatten()
        
        # Align Dates
        # The predictions start at index 'split'
        test_dates = returns.index[split:]
        
        # Safety clip length
        n_points = min(len(test_dates), len(preds_actual))
        preds_df.loc[test_dates[:n_points], col] = preds_actual[:n_points]
        
    return preds_df.astype(float).dropna()