"""Orchestrates extract -> transform -> load across one or more seasons.

Usage:
    python -m etl.pipeline --seasons 2021 2022 2023 2024
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from app.core.db import get_engine, init_db
from etl.extract import enable_cache, extract_round, iter_season_rounds
from etl.load import upsert_round_results, write_processed_csv
from etl.transform import transform_round

logger = logging.getLogger(__name__)


def run_pipeline(seasons: list[int]) -> pd.DataFrame:
    enable_cache()
    init_db()
    engine = get_engine()

    frames: list[pd.DataFrame] = []
    try:
        for season in seasons:
            try:
                rounds = list(iter_season_rounds(season))
            except Exception as exc:  # noqa: BLE001 -- FastF1's exception surface is unstable across versions
                logger.warning("skipping season %s entirely: %s", season, exc)
                continue

            for round_number, event_name in rounds:
                try:
                    raw = extract_round(season, round_number, event_name)
                    tidy = transform_round(**raw)
                except Exception as exc:  # noqa: BLE001 -- covers missing/cancelled sessions & empty results
                    logger.warning(
                        "skipping season %s round %s (%s): %s",
                        season,
                        round_number,
                        event_name,
                        exc,
                    )
                    continue

                upsert_round_results(engine, tidy)
                frames.append(tidy)
                logger.info(
                    "season %s round %s (%s): %d rows", season, round_number, event_name, len(tidy)
                )
    finally:
        # Always flush whatever was collected to CSV, even if a season/round loop above raised
        # something unexpected. Already-upserted SQLite rows are never at risk, since each round
        # commits its own transaction, but the combined CSV is only written here.
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            path = write_processed_csv(combined, seasons)
            logger.info("wrote %d total rows to %s", len(combined), path)
        else:
            logger.warning("no rows extracted for seasons %s", seasons)

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FastF1 ETL pipeline for one or more seasons.")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="e.g. --seasons 2021 2022 2023 2024")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(args.seasons)


if __name__ == "__main__":
    main()
