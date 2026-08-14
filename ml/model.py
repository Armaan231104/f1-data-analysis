from __future__ import annotations

import torch
from torch import nn


class PointsPredictor(nn.Module):
    """Small feedforward regressor: predicts a driver's next-race points from rolling stats."""

    def __init__(self, n_features: int, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
