"""Reads the tidy `results` table from SQLite for the API routers. No live FastF1 calls."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, select

from app.core.db import results


def driver_trend(engine: Engine, driver_code: str, season: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                results.c.round,
                results.c.event_name,
                results.c.points,
                results.c.finish_position,
                results.c.grid_position,
            )
            .where(results.c.driver_code == driver_code, results.c.season == season)
            .order_by(results.c.round)
        ).fetchall()
    return pd.DataFrame(
        rows, columns=["round", "event_name", "points", "finish_position", "grid_position"]
    )


def head_to_head_rounds(engine: Engine, driver_a: str, driver_b: str, season: int) -> pd.DataFrame:
    """Rounds where both drivers raced for the same constructor (i.e. were actual teammates)."""
    with engine.connect() as conn:
        a_rows = conn.execute(
            select(results.c.round, results.c.constructor, results.c.finish_position)
            .where(results.c.driver_code == driver_a, results.c.season == season)
        ).fetchall()
        b_rows = conn.execute(
            select(results.c.round, results.c.constructor, results.c.finish_position)
            .where(results.c.driver_code == driver_b, results.c.season == season)
        ).fetchall()

    a_df = pd.DataFrame(a_rows, columns=["round", "constructor", "finish_position"]).set_index("round")
    b_df = pd.DataFrame(b_rows, columns=["round", "constructor", "finish_position"]).set_index("round")
    joined = a_df.join(b_df, lsuffix="_a", rsuffix="_b", how="inner")
    joined = joined[joined["constructor_a"] == joined["constructor_b"]]
    return joined.reset_index()


def constructor_trend(engine: Engine, constructor: str, season: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                results.c.round, results.c.event_name, results.c.points, results.c.finish_position
            )
            .where(results.c.constructor == constructor, results.c.season == season)
            .order_by(results.c.round)
        ).fetchall()
    return pd.DataFrame(rows, columns=["round", "event_name", "points", "finish_position"])


def constructor_reliability(engine: Engine, constructor: str, season: int | None = None) -> dict:
    stmt = select(results.c.finish_position).where(results.c.constructor == constructor)
    if season is not None:
        stmt = stmt.where(results.c.season == season)
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()

    total = len(rows)
    dnf = sum(1 for r in rows if r.finish_position is None)
    reliability_rate = (1 - dnf / total) if total else 0.0
    return {"total_entries": total, "dnf_count": dnf, "reliability_rate": reliability_rate}


def latest_driver_entry(engine: Engine, driver_code: str):
    """Most recent (season, round, constructor, grid_position) on file for a driver, or None."""
    with engine.connect() as conn:
        return conn.execute(
            select(results.c.season, results.c.round, results.c.constructor, results.c.grid_position)
            .where(results.c.driver_code == driver_code)
            .order_by(results.c.season.desc(), results.c.round.desc())
            .limit(1)
        ).first()
