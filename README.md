# F1 Data Analysis Platform

A FastAPI backend that pulls Formula 1 race results from FastF1, flattens them into a SQLite
table, and serves driver/constructor stats plus a small PyTorch model that predicts a driver's
next-race points from their recent form.

Everything here runs against real data. The ETL pipeline has actually been run against the
2021–2024 seasons, and the model has actually been trained and validated on a held-out season.
Nothing here is a stub.

## What's in here

```text
etl/            extract -> transform -> load pipeline (FastF1 -> tidy CSV + SQLite)
ml/             feature engineering, model, training loop, inference
app/            FastAPI app: routers, auth, config, SQLite access
tests/          pytest suite, no network calls
data/           FastF1 cache + processed output (gitignored, regenerate it yourself)
```

Request flow is boring on purpose: routers read from SQLite (`app/services/results_repo.py`),
never from FastF1 directly. All the FastF1 traffic happens once, offline, in the ETL step. The
API itself never touches the network.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env
```

`torch` is a normal dependency now. If you want the CPU-only wheel instead of pulling in CUDA
packages you don't need, install with:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Swagger UI is at `http://localhost:8000/docs`. Every route under `/api/v1` except `/health`
requires an `X-API-Key` header. See [Authentication](#authentication) below.

## Running the ETL pipeline

```bash
python -m etl.pipeline --seasons 2021 2022 2023 2024
```

This pulls every race session for the given seasons, cleans it into one row per driver per race,
writes `data/processed/results_<seasons>.csv`, and upserts into `data/f1.db`.

One thing worth knowing before you run this cold: FastF1's public schedule/session API caps out
at 500 calls/hour, and a fresh, uncached pull across several seasons blows past that in a few
minutes. The pipeline backs off and retries automatically when it hits the limit, so it won't
crash. It'll just sit there retrying for a while if you hit the cap early. Once a race is cached
on disk, though, re-running the pipeline doesn't cost any of that budget again, so a second run
(or picking back up after a rate-limit wait) is fast. Budget real time for the first full run;
after that, it's quick.

Rounds that fail to load (cancelled sessions, no classified results, etc.) get logged and
skipped rather than aborting the whole run.

## Training the prediction model

```bash
python -m ml.train
```

Trains a small feedforward network on 2021–2023, holds out 2024 for validation, and saves the
checkpoint to `ml/artifacts/model.pt`. It prints MAE/RMSE when it's done. On the actual
2021–2024 data pulled by this pipeline, that came out to:

```
Validation (2024 holdout, n=479): MAE=2.629  RMSE=4.307
```

For reference, just predicting the training-set mean for every row gets MAE=5.903 on the same
holdout. So the model is picking up real signal, not just guessing around the average.

Features are five rolling stats per driver going into a race: average finish position and
average points over their last 5 races, season-to-date points, their constructor's rolling
average points, and grid position for the race in question. All of them are computed with a
strict "only rounds before this one" filter, in `app/services/features.py`, and that same
function is shared between training and the live prediction endpoint. So there's no way for the
two to drift apart, and no way for the model to accidentally train on future results.

## Running with Docker

```bash
docker compose up --build
```

One container, no separate database. SQLite lives on a mounted volume (`./data`) along with the
FastF1 cache and processed CSVs, so none of it gets wiped on a rebuild. The trained model
checkpoint is mounted too (`./ml/artifacts`), so you don't have to retrain inside the container.
You still need to run the ETL pipeline and `ml/train.py` at least once, though. The image
doesn't ship with any data or a trained model baked in.

## Authentication

API keys are stored hashed in a SQLite table, not as a single string in an env var. On first
boot, whatever's in `API_KEYS_SEED` (comma-separated, in `.env`) gets hashed and inserted if the
table is empty. That's what lets `dev-api-key-1` / `dev-api-key-2` work out of the box locally.
Beyond that, `app/services/keys_repo.py` has `add_key()` / `revoke_key()` / `verify_key()` for
managing keys for real. There's no HTTP endpoint for this on purpose; it's not something this API
needs to expose.

## API reference

All requests below assume:

```bash
export API_KEY=dev-api-key-1
export BASE=http://localhost:8000/api/v1
```

**Health check** (no key required)

```bash
curl "$BASE/health"
```

**Driver trend**: points per round, cumulative points, 5-race rolling form

```bash
curl -H "X-API-Key: $API_KEY" "$BASE/drivers/VER/trend?season=2023"
```

**Head-to-head vs. teammate**: tallied per round they actually shared a constructor, so it holds
up across mid-season driver swaps

```bash
curl -H "X-API-Key: $API_KEY" "$BASE/drivers/VER/head-to-head/PER?season=2023"
```

**Constructor trend**: team points and average finish per round

```bash
curl -H "X-API-Key: $API_KEY" -G "$BASE/constructors/Red%20Bull%20Racing/trend" --data-urlencode "season=2023"
```

**Constructor reliability**: DNF rate, optionally scoped to a season

```bash
curl -H "X-API-Key: $API_KEY" -G "$BASE/constructors/Red%20Bull%20Racing/reliability" --data-urlencode "season=2023"
```

**Next-race prediction**

```bash
curl -H "X-API-Key: $API_KEY" "$BASE/predictions/VER/next-race"
curl -H "X-API-Key: $API_KEY" "$BASE/predictions/VER/next-race?grid_position=3"
```

## Prediction model: what it actually does and doesn't do

This predicts *expected points based on recent form*, not a real outcome. A few things worth
being upfront about:

- **It doesn't know the future grid position.** A driver's actual starting position for a race
  they haven't qualified for yet obviously isn't known ahead of time. The `/next-race` endpoint
  either takes a hypothetical grid position as a query param or falls back to the driver's most
  recent race's grid slot as a stand-in. Treat that fallback as a rough estimate, not a forecast.
- **It only sees race sessions.** The ETL pipeline pulls the "R" session only, so sprint race
  points aren't in the data. Season point totals computed from this dataset will run a bit below
  the real championship standings in any season with sprints. That's expected, not a bug.
- **The reliability metric is stricter than it sounds.** "Reliability" here is defined as
  `finish_position IS NOT NULL`, meaning the driver was classified. F1 actually classifies most
  retirees who completed enough of the race distance, so a "Retired" status in the data usually
  still comes with a real finishing position. In practice this means the reliability numbers this
  API reports skew close to 100% for most constructors. It's really only catching the rare
  did-not-classify/did-not-start cases, not general mechanical failures, so don't read too much
  into a constructor's reliability score on its own.
- **It doesn't know about driver or car changes.** The model assumes a driver stays with their
  most recently recorded constructor. A mid-season seat change wouldn't be reflected until
  there's actual race data for the new team.

If you want a next-race number that's meant to be a serious forecast, this isn't it. It's a
form-based estimate built from a handful of rolling stats, nothing more.

## Structured for cloud deployment

This isn't deployed anywhere, but it's built so that it could be without much rework. Config is
entirely environment-based (`.env` / `app/core/config.py`, no hardcoded values), the API
container is stateless and holds no state that can't be rebuilt from `data/`, and SQLite and the
FastF1 cache are the only things standing between this and a managed Postgres instance plus a
shared cache layer. Swapping those in wouldn't touch the API or ML code at all.

## Tests

```bash
pytest
```

Everything runs against a throwaway SQLite fixture and a tiny untrained model saved through the
real serialization path. No network calls, no real training, and the whole suite runs in a
couple of seconds.
