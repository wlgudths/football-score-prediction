import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list=[128, 64], output_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    