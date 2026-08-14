"""Constructor season trend and reliability/DNF-rate endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_api_key
from app.core.db import get_engine
from app.schemas import (
    ConstructorReliabilityResponse,
    ConstructorTrendPoint,
    ConstructorTrendResponse,
)
from app.services import results_repo

router = APIRouter(prefix="/constructors", tags=["constructors"], dependencies=[Depends(require_api_key)])


@router.get("/{constructor}/trend", response_model=ConstructorTrendResponse)
def constructor_trend(
    constructor: str, season: int = Query(default=2024, ge=2018)
) -> ConstructorTrendResponse:
    engine = get_engine()
    df = results_repo.constructor_trend(engine, constructor, season)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"no results for constructor '{constructor}' in season {season}"
        )

    grouped = (
        df.groupby("round", as_index=False)
        .agg(
            event_name=("event_name", "first"),
            points=("points", "sum"),
            avg_finish_position=("finish_position", "mean"),
        )
        .sort_values("round")
    )
    grouped["cumulative_points"] = grouped["points"].cumsum()

    rounds = [
        ConstructorTrendPoint(
            round=int(row["round"]),
            event_name=row["event_name"],
            points=float(row["points"]),
            cumulative_points=float(row["cumulative_points"]),
            avg_finish_position=(
                None if pd.isna(row["avg_finish_position"]) else float(row["avg_finish_position"])
            ),
        )
        for _, row in grouped.iterrows()
    ]
    return ConstructorTrendResponse(constructor=constructor, season=season, rounds=rounds)


@router.get("/{constructor}/reliability", response_model=ConstructorReliabilityResponse)
def constructor_reliability(
    constructor: str, season: int | None = Query(default=None, ge=2018)
) -> ConstructorReliabilityResponse:
    engine = get_engine()
    stats = results_repo.constructor_reliability(engine, constructor, season)
    if stats["total_entries"] == 0:
        suffix = f" in season {season}" if season is not None else ""
        raise HTTPException(status_code=404, detail=f"no results for constructor '{constructor}'{suffix}")
    return ConstructorReliabilityResponse(constructor=constructor, season=season, **stats)
