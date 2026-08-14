"""Shared fixtures: a tmp SQLite DB seeded with tiny fixture data, and a tiny untrained model.

Everything here is hand-built and deterministic: no network calls, no real FastF1 data, no
real training. Fast enough to run on every commit.
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.core import config as config_module
from app.core.db import metadata, results
from app.services import keys_repo

# Two constructors, four drivers, three rounds across two seasons, two DNFs (one per driver on
# TeamY), enough to exercise trends, head-to-head, reliability, and leakage-safe ML features.
FIXTURE_COLUMNS = [
    "season", "round", "event_name", "driver_code", "driver_name", "constructor",
    "grid_position", "finish_position", "points", "status",
    "avg_lap_time", "fastest_lap", "pit_stops",
]
FIXTURE_ROWS = [
    (2023, 1, "Race A", "AAA", "Driver A", "TeamX", 1, 1, 25.0, "Finished", 90.0, 88.0, 2),
    (2023, 1, "Race A", "BBB", "Driver B", "TeamX", 2, 2, 18.0, "Finished", 90.5, 88.5, 2),
    (2023, 1, "Race A", "CCC", "Driver C", "TeamY", 3, 3, 15.0, "Finished", 91.0, 89.0, 1),
    (2023, 1, "Race A", "DDD", "Driver D", "TeamY", 4, None, 0.0, "Retired", None, None, 0),
    (2023, 2, "Race B", "AAA", "Driver A", "TeamX", 2, 1, 25.0, "Finished", 90.1, 88.1, 2),
    (2023, 2, "Race B", "BBB", "Driver B", "TeamX", 1, 2, 18.0, "Finished", 90.2, 88.2, 2),
    (2023, 2, "Race B", "CCC", "Driver C", "TeamY", 3, None, 0.0, "Retired", None, None, 1),
    (2023, 2, "Race B", "DDD", "Driver D", "TeamY", 4, 3, 15.0, "Finished", 91.5, 89.5, 1),
    (2024, 1, "Race C", "AAA", "Driver A", "TeamX", 1, 2, 18.0, "Finished", 90.3, 88.3, 2),
    (2024, 1, "Race C", "BBB", "Driver B", "TeamX", 3, 1, 25.0, "Finished", 90.0, 88.0, 2),
    (2024, 1, "Race C", "CCC", "Driver C", "TeamY", 2, 3, 15.0, "Finished", 91.2, 89.2, 1),
    (2024, 1, "Race C", "DDD", "Driver D", "TeamY", 4, 4, 12.0, "Finished", 91.8, 89.8, 1),
]


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """A throwaway SQLite file seeded with FIXTURE_ROWS; settings.database_url is monkeypatched
    so every get_engine() call in the app (routers, services, ml) resolves to it."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setattr(config_module.settings, "database_url", db_url)

    engine = create_engine(db_url)
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(results.insert(), [dict(zip(FIXTURE_COLUMNS, row)) for row in FIXTURE_ROWS])
    return engine


@pytest.fixture
def tiny_model_checkpoint(tmp_path, monkeypatch):
    """A randomly-initialized (untrained) model saved through the real torch.save path, so
    predict.py's load/forward/serialization plumbing is exercised without a real trained model."""
    from app.routers.predictions import _get_model
    from app.services.features import FEATURE_NAMES
    from ml.model import PointsPredictor

    model_path = tmp_path / "model.pt"
    monkeypatch.setattr(config_module.settings, "model_artifact_path", str(model_path))

    n_features = len(FEATURE_NAMES)
    checkpoint = {
        "state_dict": PointsPredictor(n_features=n_features).state_dict(),
        "n_features": n_features,
        "mean": torch.zeros(n_features),
        "std": torch.ones(n_features),
        "rolling_window": 5,
        "train_seasons": [2023],
        "validation_season": 2024,
        "val_mae": 1.23,
        "val_rmse": 2.34,
    }
    torch.save(checkpoint, model_path)

    _get_model.cache_clear()  # predictions router caches the loaded model; use the tiny one
    yield model_path
    _get_model.cache_clear()


@pytest.fixture
def authed_client(tmp_db, tiny_model_checkpoint):
    api_key = keys_repo.add_key("test")
    from app.main import app

    with TestClient(app) as client:
        client.headers.update({"X-API-Key": api_key})
        yield client
