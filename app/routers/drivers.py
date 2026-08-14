"""Driver season trend, rolling form, and head-to-head vs teammate endpoints."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import require_api_key
from app.core.db import get_engine
from app.schemas import DriverTrendPoint, DriverTrendResponse, HeadToHeadResponse
from app.services import results_repo

router = APIRouter(prefix="/drivers", tags=["drivers"], dependencies=[Depends(require_api_key)])

ROLLING_FORM_WINDOW = 5


def _optional_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


@router.get("/{driver_code}/trend", response_model=DriverTrendResponse)
def driver_trend(driver_code: str, season: int = Query(default=2024, ge=2018)) -> DriverTrendResponse:
    driver_code = driver_code.upper()
    engine = get_engine()
    df = results_repo.driver_trend(engine, driver_code, season)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"no results for driver {driver_code} in season {season}"
        )

    df = df.sort_values("round").reset_index(drop=True)
    df["cumulative_points"] = df["points"].cumsum()
    df["rolling_avg_points"] = df["points"].rolling(window=ROLLING_FORM_WINDOW, min_periods=1).mean()

    rounds = [
        DriverTrendPoint(
            round=int(row["round"]),
            event_name=row["event_name"],
            points=float(row["points"]),
            cumulative_points=float(row["cumulative_points"]),
            rolling_avg_points=float(row["rolling_avg_points"]),
            finish_position=_optional_int(row["finish_position"]),
            grid_position=_optional_int(row["grid_position"]),
        )
        for _, row in df.iterrows()
    ]
    return DriverTrendResponse(driver_code=driver_code, season=season, rounds=rounds)


@router.get("/{driver_code}/head-to-head/{teammate_code}", response_model=HeadToHeadResponse)
def head_to_head(
    driver_code: str, teammate_code: str, season: int = Query(default=2024, ge=2018)
) -> HeadToHeadResponse:
    driver_code, teammate_code = driver_code.upper(), teammate_code.upper()
    engine = get_engine()
    df = results_repo.head_to_head_rounds(engine, driver_code, teammate_code, season)
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"no rounds found where {driver_code} and {teammate_code} were teammates "
                f"in season {season}"
            ),
        )

    def _outcome(row) -> str:
        a_dnf, b_dnf = pd.isna(row["finish_position_a"]), pd.isna(row["finish_position_b"])
        if a_dnf and b_dnf:
            return "tie"
        if a_dnf:
            return "teammate"
        if b_dnf:
            return "driver"
        if row["finish_position_a"] < row["finish_position_b"]:
            return "driver"
        if row["finish_position_b"] < row["finish_position_a"]:
            return "teammate"
        return "tie"

    outcomes = df.apply(_outcome, axis=1)
    return HeadToHeadResponse(
        driver_code=driver_code,
        teammate_code=teammate_code,
        season=season,
        rounds_compared=len(df),
        driver_wins=int((outcomes == "driver").sum()),
        teammate_wins=int((outcomes == "teammate").sum()),
        ties=int((outcomes == "tie").sum()),
    )
