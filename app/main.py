from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Formula 1 analytics API with FastF1 ingestion and Pandas-based metrics.",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "F1 Data Analysis Platform is running",
        "docs": "/docs",
        "api": f"/api/{settings.api_version}/health",
    }
