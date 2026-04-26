# F1 Data Analysis Platform

A resume-ready Formula 1 analytics project using **Python, Pandas, FastAPI, Docker, PostgreSQL, and FastF1**.

## Features

- FastAPI REST API with versioned routes (`/api/v1/...`)
- API key authentication (`X-API-Key` header)
- FastF1-powered data ingestion for race results
- Pandas-based analytics endpoints
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

3. Start API:

```bash
uvicorn app.main:app --reload
```

4. Open docs:

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
