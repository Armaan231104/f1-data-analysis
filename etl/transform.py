"""Reshapes raw FastF1 session results/laps into the tidy one-row-per-driver-per-race schema."""

from __future__ import annotations

import pandas as pd


def _to_nullable_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _driver_name(row: pd.Series) -> str:
    for col in ("FullName", "BroadcastName"):
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value
    return str(row["Abbreviation"])


def _lap_stats(laps: pd.DataFrame, driver_code: str) -> dict:
    if laps.empty or "Driver" not in laps.columns:
        return {"avg_lap_time": None, "fastest_lap": None, "pit_stops": None}

    driver_laps = laps[laps["Driver"] == driver_code]
    if driver_laps.empty:
        return {"avg_lap_time": None, "fastest_lap": None, "pit_stops": None}

    lap_times = driver_laps["LapTime"].dropna()
    seconds = lap_times.dt.total_seconds() if not lap_times.empty else lap_times
    avg_lap_time = float(seconds.mean()) if not seconds.empty else None
    fastest_lap = float(seconds.min()) if not seconds.empty else None
    pit_stops = int(driver_laps["PitInTime"].notna().sum()) if "PitInTime" in driver_laps else None

    return {"avg_lap_time": avg_lap_time, "fastest_lap": fastest_lap, "pit_stops": pit_stops}


def transform_round(
    season: int,
    round_number: int,
    event_name: str,
    results: pd.DataFrame,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """One race's raw FastF1 results/laps -> tidy rows, one per driver."""
    rows = []
    for _, r in results.iterrows():
        driver_code = str(r["Abbreviation"])
        rows.append(
            {
                "season": season,
                "round": round_number,
                "event_name": event_name,
                "driver_code": driver_code,
                "driver_name": _driver_name(r),
                "constructor": str(r["TeamName"]),
                "grid_position": _to_nullable_int(r.get("GridPosition")),
                "finish_position": _to_nullable_int(r.get("Position")),
                "points": float(r["Points"]) if pd.notna(r.get("Points")) else 0.0,
                "status": str(r.get("Status", "")),
                **_lap_stats(laps, driver_code),
            }
        )
    return pd.DataFrame(rows)
