"""Baseline classifier: a small MLP over per-cell features."""
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, n_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)
