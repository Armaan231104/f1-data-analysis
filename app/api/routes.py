from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.core.security import auth_dependency
from app.models.schemas import (
    ConstructorTrendPoint,
    ConstructorTrendResponse,
    DriverTrendPoint,
    DriverTrendResponse,
    HealthResponse,
    InsightItem,
    InsightsResponse,
    MLPredictionResponse,
)
from app.services.analytics import AnalyticsService
from app.services.ml import MLService

router = APIRouter(prefix=f"/api/{settings.api_version}", dependencies=[Depends(auth_dependency)])
analytics_service = AnalyticsService()
ml_service = MLService()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)


@router.get("/drivers/{driver_code}/trend", response_model=DriverTrendResponse)
def driver_trend(
    driver_code: str,
    season: int = Query(default=2024, ge=2018),
    rounds: int = Query(default=8, ge=1, le=24),
) -> DriverTrendResponse:
    df = analytics_service.driver_points_trend(season=season, driver_code=driver_code, rounds=rounds)
    points = [DriverTrendPoint(**item) for item in df.to_dict(orient="records")]
    return DriverTrendResponse(driver_code=driver_code.upper(), season=season, points_by_round=points)


@router.get("/constructors/trend", response_model=ConstructorTrendResponse)
def constructor_trend(
    season: int = Query(default=2024, ge=2018),
    rounds: int = Query(default=6, ge=1, le=24),
) -> ConstructorTrendResponse:
    df = analytics_service.constructor_points(season=season, rounds=rounds)
    constructors = [
        ConstructorTrendPoint(round=int(row["round"]), constructor=row["TeamName"], points=float(row["Points"]))
        for _, row in df.iterrows()
    ]
    return ConstructorTrendResponse(season=season, constructors=constructors)


@router.get("/insights", response_model=InsightsResponse)
def insights(season: int = Query(default=2024, ge=2018)) -> InsightsResponse:
    entries = [InsightItem(**row) for row in analytics_service.top_insights(season=season)]
    return InsightsResponse(season=season, insights=entries)


@router.get("/ml/next-race-points", response_model=MLPredictionResponse)
def next_race_points_prediction(
    framework: str = Query(default="pytorch", pattern="^(pytorch|tensorflow)$"),
    season: int = Query(default=2024, ge=2018),
    driver_code: str = Query(default="VER", min_length=3, max_length=3),
    rounds: int = Query(default=10, ge=2, le=24),
) -> MLPredictionResponse:
    try:
        prediction = ml_service.predict_next_race_points(
            framework=framework,
            season=season,
            driver_code=driver_code,
            rounds=rounds,
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MLPredictionResponse(**prediction.__dict__)
