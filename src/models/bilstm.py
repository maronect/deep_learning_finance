import torch.nn as nn

class BILSTMmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.bilstm = nn.LSTM( input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out[:, -1, :])
