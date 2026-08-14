"""Trains the points-prediction model. Holds out the most recent season for validation.

Usage:
    python -m ml.train
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from app.core.config import settings
from app.core.db import get_engine
from ml.dataset import build_dataset
from ml.model import PointsPredictor

logger = logging.getLogger(__name__)

TRAIN_SEASONS = [2021, 2022, 2023]
VALIDATION_SEASON = 2024
ROLLING_WINDOW = 5
EPOCHS = 300
LEARNING_RATE = 0.01


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def train() -> dict:
    engine = get_engine()

    X_train, y_train = build_dataset(engine, TRAIN_SEASONS, n=ROLLING_WINDOW)
    X_val, y_val = build_dataset(engine, [VALIDATION_SEASON], n=ROLLING_WINDOW)
    logger.info("train rows: %d, validation rows: %d", len(X_train), len(X_val))

    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    X_train_norm = _normalize(X_train, mean, std)
    X_val_norm = _normalize(X_val, mean, std)

    model = PointsPredictor(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        preds = model(X_train_norm)
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch == EPOCHS:
            logger.info("epoch %d/%d train_mse=%.4f", epoch, EPOCHS, loss.item())

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_norm)
        mae = (val_preds - y_val).abs().mean().item()
        rmse = torch.sqrt(((val_preds - y_val) ** 2).mean()).item()

    print(f"Validation ({VALIDATION_SEASON} holdout, n={len(y_val)}): MAE={mae:.3f}  RMSE={rmse:.3f}")

    return {
        "state_dict": model.state_dict(),
        "n_features": X_train.shape[1],
        "mean": mean,
        "std": std,
        "rolling_window": ROLLING_WINDOW,
        "train_seasons": TRAIN_SEASONS,
        "validation_season": VALIDATION_SEASON,
        "val_mae": mae,
        "val_rmse": rmse,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = train()

    path = Path(settings.model_artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to {path}")


if __name__ == "__main__":
    main()
