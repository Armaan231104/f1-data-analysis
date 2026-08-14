"""Pulls race session data from FastF1, with local caching so repeat runs don't re-download."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# FastF1's hosted schedule/session-info APIs cap at 500 calls/hour (a shared, sliding window).
# A cold-cache multi-season pull makes several calls per round and can exceed that mid-run;
# once already-fetched requests are on disk they're served from cache and don't count again,
# so backing off and retrying resumes cheaply rather than needing a wider rewrite.
_RATE_LIMIT_BACKOFF_SECONDS = 65
_RATE_LIMIT_MAX_RETRIES = 60


def _with_rate_limit_retry(func: Callable[[], T]) -> T:
    for attempt in range(1, _RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return func()
        except RateLimitExceededError as exc:
            logger.warning(
                "FastF1 rate limit hit (%s), backing off %ss (attempt %d/%d)",
                exc,
                _RATE_LIMIT_BACKOFF_SECONDS,
                attempt,
                _RATE_LIMIT_MAX_RETRIES,
            )
            time.sleep(_RATE_LIMIT_BACKOFF_SECONDS)
    return func()


def enable_cache() -> None:
    Path(settings.fastf1_cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(settings.fastf1_cache_dir)


def iter_season_rounds(season: int) -> Iterator[tuple[int, str]]:
    """Yield (round_number, event_name) for each non-testing event on a season's calendar."""
    schedule = _with_rate_limit_retry(
        lambda: fastf1.get_event_schedule(season, include_testing=False)
    )
    schedule = schedule[schedule["EventFormat"] != "testing"]
    for _, event in schedule.iterrows():
        yield int(event["RoundNumber"]), event["EventName"]


def extract_round(season: int, round_number: int, event_name: str) -> dict:
    """Load one race session's results + laps from FastF1.

    Raises on a missing/cancelled session or empty results. The pipeline decides whether to
    skip and continue rather than aborting the whole multi-season run.
    """

    def _load_session():
        session = fastf1.get_session(season, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)
        return session

    session = _with_rate_limit_retry(_load_session)

    results: pd.DataFrame = session.results.copy()
    if results.empty:
        raise ValueError(f"no classified results for season {season} round {round_number}")

    return {
        "season": season,
        "round_number": round_number,
        "event_name": event_name,
        "results": results,
        "laps": session.laps.copy(),
    }
