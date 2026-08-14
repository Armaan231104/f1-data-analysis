from __future__ import annotations

from pathlib import Path

from sqlalchemy import (
    Column,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
)

from app.core.config import settings

metadata = MetaData()

results = Table(
    "results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("season", Integer, nullable=False),
    Column("round", Integer, nullable=False),
    Column("event_name", String, nullable=False),
    Column("driver_code", String, nullable=False),
    Column("driver_name", String, nullable=False),
    Column("constructor", String, nullable=False),
    Column("grid_position", Integer, nullable=True),
    Column("finish_position", Integer, nullable=True),
    Column("points", Float, nullable=False),
    Column("status", String, nullable=False),
    Column("avg_lap_time", Float, nullable=True),
    Column("fastest_lap", Float, nullable=True),
    Column("pit_stops", Integer, nullable=True),
    UniqueConstraint("season", "round", "driver_code", name="uq_results_season_round_driver"),
)
Index("ix_results_season_round", results.c.season, results.c.round)
Index("ix_results_driver_season", results.c.driver_code, results.c.season)

api_keys = Table(
    "api_keys",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("key_hash", String, nullable=False, unique=True),
    Column("label", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("revoked_at", String, nullable=True),
)


def _ensure_sqlite_dir(database_url: str) -> None:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        Path(database_url[len(prefix) :]).parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> Engine:
    _ensure_sqlite_dir(settings.database_url)
    return create_engine(settings.database_url)


def init_db() -> None:
    metadata.create_all(get_engine())
