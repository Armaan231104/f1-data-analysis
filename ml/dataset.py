"""Builds PyTorch training tensors from the tidy `results` table using leakage-safe features."""

from __future__ import annotations

import torch
from sqlalchemy import Engine, select

from app.core.db import results
from app.services.features import FEATURE_NAMES, assemble_feature_vector, build_rolling_features


def build_dataset(engine: Engine, seasons: list[int], n: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (X, y): X shape (n_samples, len(FEATURE_NAMES)), y shape (n_samples, 1) = points scored."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                results.c.season,
                results.c.round,
                results.c.driver_code,
                results.c.constructor,
                results.c.grid_position,
                results.c.points,
            )
            .where(results.c.season.in_(seasons))
            .order_by(results.c.season, results.c.round)
        ).fetchall()

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        if row.grid_position is None:
            continue  # no known starting position (e.g. DNS) -> can't build this feature
        rolling = build_rolling_features(engine, row.driver_code, row.constructor, row.season, row.round, n=n)
        feature_rows.append(assemble_feature_vector(rolling, row.grid_position))
        targets.append(row.points)

    if not feature_rows:
        return torch.empty((0, len(FEATURE_NAMES))), torch.empty((0, 1))

    X = torch.tensor(feature_rows, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return X, y
