from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str


class DriverTrendPoint(BaseModel):
    round: int
    event_name: str
    points: float
    cumulative_points: float
    rolling_avg_points: float
    finish_position: int | None
    grid_position: int | None


class DriverTrendResponse(BaseModel):
    driver_code: str
    season: int
    rounds: list[DriverTrendPoint]


class HeadToHeadResponse(BaseModel):
    driver_code: str
    teammate_code: str
    season: int
    rounds_compared: int
    driver_wins: int
    teammate_wins: int
    ties: int


class ConstructorTrendPoint(BaseModel):
    round: int
    event_name: str
    points: float
    cumulative_points: float
    avg_finish_position: float | None


class ConstructorTrendResponse(BaseModel):
    constructor: str
    season: int
    rounds: list[ConstructorTrendPoint]


class ConstructorReliabilityResponse(BaseModel):
    constructor: str
    season: int | None
    total_entries: int
    dnf_count: int
    reliability_rate: float


class PredictionResponse(BaseModel):
    driver_code: str
    season: int
    round: int
    constructor: str
    grid_position: int
    grid_position_source: str
    predicted_points: float
    rolling_window: int
    model_validation_mae: float
    model_validation_rmse: float
