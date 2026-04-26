from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str


class DriverTrendPoint(BaseModel):
    round: int
    race_name: str
    points: float
    cumulative_points: float


class DriverTrendResponse(BaseModel):
    driver_code: str
    season: int
    points_by_round: list[DriverTrendPoint]


class ConstructorTrendPoint(BaseModel):
    round: int
    constructor: str
    points: float


class ConstructorTrendResponse(BaseModel):
    season: int
    constructors: list[ConstructorTrendPoint]


class InsightItem(BaseModel):
    title: str
    value: str
    context: str


class InsightsResponse(BaseModel):
    season: int
    insights: list[InsightItem]
