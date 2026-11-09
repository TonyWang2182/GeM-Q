"""Attention pooling and regression for cached frame features."""

import torch
from torch import nn


class Head(nn.Module):
    """Map features of shape (batch, frames, dim) to one score per clip."""

    def __init__(self, dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, x):
        a = self.attention(x).squeeze(-1).softmax(-1)
        return self.mlp(torch.bmm(a.unsqueeze(1), x).squeeze(1)).squeeze(-1)
