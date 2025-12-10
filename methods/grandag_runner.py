# methods/grandag_runner.py

import numpy as np
import torch
from torch import nn

# Minimal neural SEM for demonstration
class MLPSEM(nn.Module):
    def __init__(self, p, hidden=16):
        super().__init__()
        self.p = p
        self.W = nn.Parameter(torch.zeros(p, p))
        self.mlp = nn.Sequential(
            nn.Linear(p, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p)
        )

    def forward(self, X):
        X = self.mlp(X)
        return X @ self.W

def run_grandag(X, lr=0.01, steps=2000):
    X = torch.tensor(X, dtype=torch.float32)
    p = X.shape[1]

    model = MLPSEM(p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        opt.zero_grad()
        X_hat = model(X)
        loss = torch.mean((X - X_hat)**2)
        loss.backward()
        opt.step()

    return model.W.detach().numpy()
