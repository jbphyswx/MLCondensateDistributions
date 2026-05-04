import torch
import torch.nn as nn
import torch.nn.functional as F

class CondensateMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64, 32]):
        super(CondensateMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        # Softplus to ensure non-negative condensate
        return F.softplus(x)
