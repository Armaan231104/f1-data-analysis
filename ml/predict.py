"""Loads the trained model and predicts a driver's next-race points from rolling stats."""

from __future__ import annotations

from pathlib import Path

import torch

from app.core.config import settings
from ml.model import PointsPredictor


def load_checkpoint(path: str | None = None) -> dict:
    checkpoint_path = Path(path or settings.model_artifact_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"no trained model at {checkpoint_path}, run `python -m ml.train` first"
        )
    return torch.load(checkpoint_path, weights_only=False)


def load_model(checkpoint: dict) -> PointsPredictor:
    model = PointsPredictor(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_points(checkpoint: dict, model: PointsPredictor, feature_values: list[float]) -> float:
    """feature_values must be ordered per app.services.features.FEATURE_NAMES."""
    x = torch.tensor([feature_values], dtype=torch.float32)
    x_norm = (x - checkpoint["mean"]) / checkpoint["std"].clamp_min(1e-6)
    with torch.no_grad():
        prediction = model(x_norm)
    return max(0.0, float(prediction.item()))
