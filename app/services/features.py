"""Leakage-safe rolling-stat features, shared by ml/dataset.py (training) and the predictions
router (serving) so the two can't drift apart.

Every rolling stat only looks at rows strictly before the target (season, round), which is what
makes it safe to use both to build historical training labels and to score a real "next race."
"""

from __future__ import annotations

from sqlalchemy import Engine, func, select

from app.core.db import results

# A finishing position worse than any real F1 grid (max 20 cars), used to impute DNF/DNS/DSQ
# rows (finish_position IS NULL) in rolling averages, rather than silently dropping them and
# rewarding DNF-heavy drivers with an artificially good rolling average.
DNF_IMPUTED_FINISH_POSITION = 21.0

FEATURE_NAMES = [
    "avg_finish_position",
    "avg_points",
    "grid_position",
    "constructor_avg_points",
    "season_to_date_points",
]


def _before(season: int, round_number: int):
    return (results.c.season < season) | (
        (results.c.season == season) & (results.c.round < round_number)
    )


def _rolling_driver_stats(
    engine: Engine, driver_code: str, season: int, round_number: int, n: int
) -> tuple[float, float]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(results.c.finish_position, results.c.points)
            .where(results.c.driver_code == driver_code, _before(season, round_number))
            .order_by(results.c.season.desc(), results.c.round.desc())
            .limit(n)
        ).fetchall()

    if not rows:
        return DNF_IMPUTED_FINISH_POSITION, 0.0

    finishes = [DNF_IMPUTED_FINISH_POSITION if r.finish_position is None else r.finish_position for r in rows]
    points = [r.points for r in rows]
    return sum(finishes) / len(finishes), sum(points) / len(points)


def _constructor_avg_points(
    engine: Engine, constructor: str, season: int, round_number: int, n: int
) -> float:
    with engine.connect() as conn:
        rows = conn.execute(
            select(results.c.points)
            .where(results.c.constructor == constructor, _before(season, round_number))
            .order_by(results.c.season.desc(), results.c.round.desc())
            .limit(n * 2)  # two cars per round
        ).fetchall()
    if not rows:
        return 0.0
    return sum(r.points for r in rows) / len(rows)


def _season_to_date_points(engine: Engine, driver_code: str, season: int, round_number: int) -> float:
    with engine.connect() as conn:
        total = conn.execute(
            select(func.coalesce(func.sum(results.c.points), 0.0)).where(
                results.c.driver_code == driver_code,
                results.c.season == season,
                results.c.round < round_number,
            )
        ).scalar()
    return float(total)


def build_rolling_features(
    engine: Engine, driver_code: str, constructor: str, season: int, round_number: int, n: int = 5
) -> dict:
    """Rolling features for a driver entering (season, round_number).

    grid_position is deliberately NOT included here: it's the one feature only known once
    qualifying for the target race has happened. Training reads it directly off the historical
    row; serving supplies an estimate (see app/routers/predictions.py).
    """
    avg_finish, avg_points = _rolling_driver_stats(engine, driver_code, season, round_number, n)
    return {
        "avg_finish_position": avg_finish,
        "avg_points": avg_points,
        "constructor_avg_points": _constructor_avg_points(engine, constructor, season, round_number, n),
        "season_to_date_points": _season_to_date_points(engine, driver_code, season, round_number),
    }


def assemble_feature_vector(rolling: dict, grid_position: float) -> list[float]:
    """Combine rolling features + grid_position into the fixed FEATURE_NAMES order."""
    return [
        rolling["avg_finish_position"],
        rolling["avg_points"],
        float(grid_position),
        rolling["constructor_avg_points"],
        rolling["season_to_date_points"],
    ]
