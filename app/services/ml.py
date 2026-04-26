from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.analytics import AnalyticsService

MODEL_DIR = Path("data/processed/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionResult:
    framework: str
    driver_code: str
    season: int
    next_round: int
    predicted_points: float
    model_path: str


class MLService:
    """Framework-backed point prediction service for PyTorch or TensorFlow."""

    def __init__(self) -> None:
        self.analytics = AnalyticsService()

    @staticmethod
    def _ensure_framework(framework: str) -> None:
        module_name = "torch" if framework == "pytorch" else "tensorflow"
        if importlib.util.find_spec(module_name) is None:
            raise ModuleNotFoundError(
                f"{framework} is not installed. Install project extras with `pip install -e '.[ml]'`."
            )

    @staticmethod
    def _prepare_xy(rounds: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        if len(rounds) < 3:
            raise ValueError("Need at least three rounds to train a prediction model")

        x = rounds.astype(np.float32).reshape(-1, 1)
        y = points.astype(np.float32).reshape(-1, 1)
        next_round = int(rounds.max() + 1)
        return x, y, next_round

    def _train_pytorch(self, x: np.ndarray, y: np.ndarray, artifact: Path) -> float:
        import torch
        from torch import nn

        x_t = torch.tensor(x)
        y_t = torch.tensor(y)

        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

        for _ in range(300):
            optimizer.zero_grad()
            pred = model(x_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), artifact)
        next_round_tensor = torch.tensor([[float(x.max() + 1)]])
        predicted = float(model(next_round_tensor).item())
        return predicted

    def _train_tensorflow(self, x: np.ndarray, y: np.ndarray, artifact: Path) -> float:
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.03), loss="mse")
        model.fit(x, y, epochs=150, verbose=0)
        model.save(artifact, overwrite=True)

        prediction = model.predict(np.array([[float(x.max() + 1)]], dtype=np.float32), verbose=0)
        return float(prediction[0][0])

    def predict_next_race_points(
        self,
        framework: str,
        season: int,
        driver_code: str,
        rounds: int = 10,
    ) -> PredictionResult:
        if framework not in {"pytorch", "tensorflow"}:
            raise ValueError("framework must be 'pytorch' or 'tensorflow'")

        self._ensure_framework(framework)
        trend = self.analytics.driver_points_trend(
            season=season,
            driver_code=driver_code,
            rounds=rounds,
        )

        rounds_arr = trend["round"].to_numpy(dtype=float)
        points_arr = trend["points"].to_numpy(dtype=float)
        x, y, next_round = self._prepare_xy(rounds_arr, points_arr)

        if framework == "pytorch":
            model_path = MODEL_DIR / f"{framework}_{season}_{driver_code.lower()}.pt"
            prediction = self._train_pytorch(x, y, model_path)
        else:
            model_path = MODEL_DIR / f"{framework}_{season}_{driver_code.lower()}.keras"
            prediction = self._train_tensorflow(x, y, model_path)

        return PredictionResult(
            framework=framework,
            driver_code=driver_code.upper(),
            season=season,
            next_round=next_round,
            predicted_points=round(max(0.0, prediction), 2),
            model_path=str(model_path),
        )
