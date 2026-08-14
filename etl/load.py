"""Persists tidy results: upserts into SQLite for the API, and writes processed CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, delete

from app.core.db import results as results_table

PROCESSED_DIR = Path("data/processed")


def upsert_round_results(engine: Engine, df: pd.DataFrame) -> None:
    """Replace any existing rows for this (season, round) and insert the given tidy rows."""
    if df.empty:
        return
    season = int(df.iloc[0]["season"])
    round_number = int(df.iloc[0]["round"])
    with engine.begin() as conn:
        conn.execute(
            delete(results_table).where(
                results_table.c.season == season, results_table.c.round == round_number
            )
        )
        conn.execute(results_table.insert(), df.to_dict(orient="records"))


def write_processed_csv(df: pd.DataFrame, seasons: list[int]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    span = f"{min(seasons)}_{max(seasons)}" if len(seasons) > 1 else str(seasons[0])
    path = PROCESSED_DIR / f"results_{span}.csv"
    df.to_csv(path, index=False)
    return path
