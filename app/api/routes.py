from fastapi import APIRouter, Depends, Query

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
)
from app.services.analytics import AnalyticsService

router = APIRouter(prefix=f"/api/{settings.api_version}", dependencies=[Depends(auth_dependency)])
analytics_service = AnalyticsService()


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
