# F1 Data Analysis Platform

A resume-ready Formula 1 analytics project using **Python, Pandas, FastAPI, Docker, PostgreSQL, FastF1**, and optional **PyTorch/TensorFlow** forecasting.

## Features

- FastAPI REST API with versioned routes (`/api/v1/...`)
- API key authentication (`X-API-Key` header)
- FastF1-powered data ingestion for race results
- Pandas-based analytics endpoints
- Optional ML prediction endpoint that trains a lightweight PyTorch or TensorFlow model and saves artifacts
- Dockerized local development stack with PostgreSQL
- Test suite with `pytest`

## Project Structure

```text
app/
  api/routes.py            # FastAPI endpoints
  core/config.py           # Settings and env config
  core/security.py         # API key auth dependency
  models/schemas.py        # Response models
  services/analytics.py    # FastF1 + Pandas analytics logic
  services/ml.py           # Optional ML prediction service
scripts/
  ingest_fastf1.py         # Data pipeline script
tests/
  test_app.py              # Basic API tests
docker-compose.yml         # API + Postgres stack
```

## Quickstart (Local)

1. Copy env file:

```bash
cp .env.example .env
```

2. Install dependencies:

```bash
pip install -e .[dev]
```

3. (Optional) Install ML extras:

```bash
pip install -e .[ml]
```

4. Start API:

```bash
uvicorn app.main:app --reload
```

5. Open docs:

- Swagger: `http://localhost:8000/docs`

## Run with Docker

```bash
docker compose up --build
```

## Ingest FastF1 Data

```bash
python scripts/ingest_fastf1.py
```

Outputs:
- `data/processed/race_results_2024.parquet`
- `data/processed/race_results_2024.csv`

## Example Endpoints

> Include `X-API-Key: dev-api-key`

- `GET /api/v1/health`
- `GET /api/v1/drivers/VER/trend?season=2024&rounds=8`
- `GET /api/v1/constructors/trend?season=2024&rounds=6`
- `GET /api/v1/insights?season=2024`
- `GET /api/v1/ml/next-race-points?framework=pytorch&season=2024&driver_code=VER&rounds=10` (trains model and returns saved artifact path)

## Resume Talking Points

- Built end-to-end F1 analytics platform with data ingestion, transformation, and API delivery.
- Used FastF1 and Pandas to process and aggregate multi-race performance data.
- Implemented secure versioned REST API via FastAPI and API-key auth.
- Added optional ML forecasting endpoint using PyTorch/TensorFlow backend selection.
- Containerized services and orchestrated app/database stack with Docker Compose.
- Structured for cloud deployment readiness (ECS/Fargate-compatible architecture).
