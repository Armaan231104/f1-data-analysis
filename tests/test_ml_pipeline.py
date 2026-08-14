"""Tests for the leakage-safe feature builder and the ML dataset/model/predict plumbing.

Uses tests/conftest.py's tmp_db fixture (never a real season) and a tiny randomly-initialized
model. No network, no real training.
"""

from __future__ import annotations

import torch

from app.services.features import (
    FEATURE_NAMES,
    assemble_feature_vector,
    build_rolling_features,
)
from ml.dataset import build_dataset
from ml.model import PointsPredictor
from ml.predict import load_checkpoint, load_model, predict_points


def test_rolling_features_no_leakage_on_first_round(tmp_db) -> None:
    # AAA's first-ever round on file: no prior history should exist yet.
    rolling = build_rolling_features(tmp_db, "AAA", "TeamX", season=2023, round_number=1, n=5)
    assert rolling["avg_finish_position"] == 21.0  # DNF-imputed default when no history exists
    assert rolling["avg_points"] == 0.0
    assert rolling["season_to_date_points"] == 0.0


def test_rolling_features_use_only_strictly_prior_rounds(tmp_db) -> None:
    # Going into round 2, AAA's only prior result is round 1: finish=1, points=25.0.
    rolling = build_rolling_features(tmp_db, "AAA", "TeamX", season=2023, round_number=2, n=5)
    assert rolling["avg_finish_position"] == 1.0
    assert rolling["avg_points"] == 25.0
    assert rolling["season_to_date_points"] == 25.0


def test_rolling_features_impute_dnf_in_average(tmp_db) -> None:
    # Going into round 2, CCC has one prior round with a real finish (round 1: finish=3).
    # DDD's prior round (round 1) was a DNF (finish_position None) -> imputed to 21.0.
    ddd_rolling = build_rolling_features(tmp_db, "DDD", "TeamY", season=2023, round_number=2, n=5)
    assert ddd_rolling["avg_finish_position"] == 21.0


def test_assemble_feature_vector_matches_feature_names_order() -> None:
    rolling = {
        "avg_finish_position": 3.0,
        "avg_points": 10.0,
        "constructor_avg_points": 8.0,
        "season_to_date_points": 20.0,
    }
    vector = assemble_feature_vector(rolling, grid_position=2)
    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("grid_position")] == 2.0
    assert vector[FEATURE_NAMES.index("avg_points")] == 10.0


def test_build_dataset_shape(tmp_db) -> None:
    X, y = build_dataset(tmp_db, seasons=[2023], n=5)
    # 2023 fixture has 4 drivers x 2 rounds = 8 rows, all with a known grid_position.
    assert X.shape == (8, len(FEATURE_NAMES))
    assert y.shape == (8, 1)
    assert (y >= 0).all()


def test_build_dataset_skips_rows_with_no_grid_position(tmp_db) -> None:
    from app.core.db import results

    with tmp_db.begin() as conn:
        conn.execute(
            results.insert(),
            {
                "season": 2023, "round": 3, "event_name": "Race D", "driver_code": "EEE",
                "driver_name": "Driver E", "constructor": "TeamX", "grid_position": None,
                "finish_position": None, "points": 0.0, "status": "DNS",
                "avg_lap_time": None, "fastest_lap": None, "pit_stops": None,
            },
        )

    X, _ = build_dataset(tmp_db, seasons=[2023], n=5)
    assert X.shape[0] == 8  # the extra no-grid row is excluded, not just the original 8


def test_points_predictor_forward_pass_shape() -> None:
    model = PointsPredictor(n_features=len(FEATURE_NAMES))
    x = torch.randn(3, len(FEATURE_NAMES))
    out = model(x)
    assert out.shape == (3, 1)


def test_predict_points_round_trip(tiny_model_checkpoint) -> None:
    checkpoint = load_checkpoint(str(tiny_model_checkpoint))
    model = load_model(checkpoint)

    prediction = predict_points(checkpoint, model, [3.0, 10.0, 2.0, 8.0, 20.0])
    assert isinstance(prediction, float)
    assert prediction >= 0.0  # predict_points floors negative outputs at 0


def test_load_checkpoint_missing_file_raises(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.pt"
    try:
        load_checkpoint(str(missing_path))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
