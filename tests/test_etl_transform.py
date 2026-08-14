"""Tests for etl/transform.py: raw-shaped FastF1 results/laps -> tidy schema. No network and no
real FastF1 session objects, just hand-built DataFrames shaped like FastF1's real column names.
"""

from __future__ import annotations

import pandas as pd
import pytest

from etl.transform import transform_round

RAW_RESULTS = pd.DataFrame(
    [
        {
            "Abbreviation": "AAA",
            "FullName": "Driver Aaa",
            "BroadcastName": "A AAA",
            "TeamName": "TeamX",
            "GridPosition": 1.0,
            "Position": 1.0,
            "Points": 25.0,
            "Status": "Finished",
        },
        {
            "Abbreviation": "BBB",
            "FullName": None,
            "BroadcastName": "B BBB",
            "TeamName": "TeamX",
            "GridPosition": 2.0,
            "Position": float("nan"),  # unclassified -> DNF signal
            "Points": float("nan"),
            "Status": "Retired",
        },
    ]
)

RAW_LAPS = pd.DataFrame(
    [
        {"Driver": "AAA", "LapTime": pd.to_timedelta(90.0, unit="s"), "PitInTime": pd.NaT},
        {"Driver": "AAA", "LapTime": pd.to_timedelta(91.0, unit="s"), "PitInTime": pd.Timestamp("2023-01-01")},
        {"Driver": "AAA", "LapTime": pd.to_timedelta(89.0, unit="s"), "PitInTime": pd.NaT},
        # BBB has no laps at all -> exercises the "no lap data for this driver" branch
    ]
)


def test_transform_round_schema_and_shape() -> None:
    df = transform_round(2023, 1, "Race A", RAW_RESULTS, RAW_LAPS)

    assert list(df.columns) == [
        "season", "round", "event_name", "driver_code", "driver_name", "constructor",
        "grid_position", "finish_position", "points", "status",
        "avg_lap_time", "fastest_lap", "pit_stops",
    ]
    assert len(df) == 2
    assert (df["season"] == 2023).all()
    assert (df["round"] == 1).all()
    assert (df["event_name"] == "Race A").all()


def test_transform_round_classified_driver() -> None:
    df = transform_round(2023, 1, "Race A", RAW_RESULTS, RAW_LAPS)
    aaa = df[df["driver_code"] == "AAA"].iloc[0]

    assert aaa["driver_name"] == "Driver Aaa"  # prefers FullName over BroadcastName
    assert aaa["constructor"] == "TeamX"
    assert aaa["grid_position"] == 1
    assert aaa["finish_position"] == 1
    assert aaa["points"] == 25.0
    assert aaa["status"] == "Finished"
    assert aaa["pit_stops"] == 1
    assert aaa["fastest_lap"] == pytest.approx(89.0)
    assert aaa["avg_lap_time"] == pytest.approx(90.0)


def test_transform_round_unclassified_driver() -> None:
    df = transform_round(2023, 1, "Race A", RAW_RESULTS, RAW_LAPS)
    bbb = df[df["driver_code"] == "BBB"].iloc[0]

    assert bbb["driver_name"] == "B BBB"  # falls back to BroadcastName when FullName is missing
    assert pd.isna(bbb["finish_position"])  # unclassified -> null, not a real position
    assert bbb["points"] == 0.0  # missing points default to 0.0, not null
    assert pd.isna(bbb["avg_lap_time"])  # no laps recorded for this driver
    assert pd.isna(bbb["fastest_lap"])
    assert pd.isna(bbb["pit_stops"])


def test_transform_round_empty_laps_does_not_crash() -> None:
    empty_laps = pd.DataFrame(columns=["Driver", "LapTime", "PitInTime"])
    df = transform_round(2023, 1, "Race A", RAW_RESULTS, empty_laps)

    assert len(df) == 2
    assert df["avg_lap_time"].isna().all()
