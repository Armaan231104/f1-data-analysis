from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.db import init_db
from app.schemas import HealthResponse
from app.services.keys_repo import seed_from_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_from_env()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Formula 1 analytics and ML API with FastF1 ETL pipelines and PyTorch predictions.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "F1 Data Analysis Platform is running",
        "docs": "/docs",
        "api": f"/api/{settings.api_version}/health",
    }


@app.get(f"/api/{settings.api_version}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", app=settings.app_name)
