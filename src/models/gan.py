'''
import torch.nn as nn

class ganmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gan = nn.gan(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gan(x)
        return self.fc(out[:, -1, :])
'''