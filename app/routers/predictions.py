"""Next-race points prediction, backed by the trained PyTorch model.

The model predicts *expected points based on historical trend*, not a guaranteed outcome. See
the README's Prediction Model section for the honest scope/limitations framing.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_api_key
from app.core.db import get_engine
from app.schemas import PredictionResponse
from app.services import results_repo
from app.services.features import assemble_feature_vector, build_rolling_features
from ml.predict import load_checkpoint, load_model, predict_points

router = APIRouter(prefix="/predictions", tags=["predictions"], dependencies=[Depends(require_api_key)])


@lru_cache(maxsize=1)
def _get_model():
    checkpoint = load_checkpoint()
    return checkpoint, load_model(checkpoint)


@router.get("/{driver_code}/next-race", response_model=PredictionResponse)
def predict_next_race(
    driver_code: str,
    grid_position: int
    | None = Query(
        default=None,
        ge=1,
        le=24,
        description=(
            "Hypothetical/expected grid slot for the next race (actual grid position isn't known "
            "until qualifying). Defaults to the driver's most recent race grid position."
        ),
    ),
) -> PredictionResponse:
    driver_code = driver_code.upper()
    engine = get_engine()

    latest = results_repo.latest_driver_entry(engine, driver_code)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"no historical results found for driver {driver_code}")

    try:
        checkpoint, model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    next_season, next_round = latest.season, latest.round + 1

    grid_source = "provided"
    if grid_position is None:
        grid_position = latest.grid_position
        grid_source = "estimated_from_last_race"
        if grid_position is None:
            grid_position = 20
            grid_source = "estimated_default"

    rolling = build_rolling_features(
        engine, driver_code, latest.constructor, next_season, next_round, n=checkpoint["rolling_window"]
    )
    vector = assemble_feature_vector(rolling, grid_position)
    predicted = predict_points(checkpoint, model, vector)

    return PredictionResponse(
        driver_code=driver_code,
        season=next_season,
        round=next_round,
        constructor=latest.constructor,
        grid_position=grid_position,
        grid_position_source=grid_source,
        predicted_points=predicted,
        rolling_window=checkpoint["rolling_window"],
        model_validation_mae=checkpoint["val_mae"],
        model_validation_rmse=checkpoint["val_rmse"],
    )
