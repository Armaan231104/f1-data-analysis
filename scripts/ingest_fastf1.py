"""Extract race result data from FastF1 and persist as parquet/csv for downstream API use."""

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("data/raw/fastf1_cache")


def ingest_season_results(season: int, max_rounds: int = 24) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    races = schedule[schedule["EventFormat"] != "testing"].head(max_rounds)
    frames: list[pd.DataFrame] = []

    for _, event in races.iterrows():
        race_name = event["EventName"]
        session = fastf1.get_session(season, race_name, "R")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results.copy()
        results["Season"] = season
        results["RoundNumber"] = int(event["RoundNumber"])
        results["RaceName"] = race_name
        frames.append(results)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    season = 2024
    df = ingest_season_results(season=season)
    parquet_path = OUTPUT_DIR / f"race_results_{season}.parquet"
    csv_path = OUTPUT_DIR / f"race_results_{season}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {parquet_path} and {csv_path}")


if __name__ == "__main__":
    main()
