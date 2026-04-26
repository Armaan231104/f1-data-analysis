from __future__ import annotations

from functools import lru_cache

import fastf1
import pandas as pd


class AnalyticsService:
    """Analytics queries powered by FastF1 race result tables."""

    def __init__(self, cache_dir: str = "data/raw/fastf1_cache") -> None:
        fastf1.Cache.enable_cache(cache_dir)

    @staticmethod
    @lru_cache(maxsize=128)
    def _race_results(season: int, gp_name: str) -> pd.DataFrame:
        session = fastf1.get_session(season, gp_name, "R")
        session.load(telemetry=False, weather=False, messages=False)
        results = session.results.copy()
        if "Points" not in results.columns:
            results["Points"] = 0.0
        results["EventName"] = gp_name
        return results

    def driver_points_trend(self, season: int, driver_code: str, rounds: int = 8) -> pd.DataFrame:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        races = schedule[schedule["EventFormat"] != "testing"].head(rounds)
        rows: list[dict] = []

        for i, event in races.iterrows():
            race_name = event["EventName"]
            results = self._race_results(season, race_name)
            target = results[results["Abbreviation"] == driver_code.upper()]
            points = float(target["Points"].iloc[0]) if not target.empty else 0.0
            rows.append(
                {
                    "round": int(i + 1),
                    "race_name": race_name,
                    "points": points,
                }
            )

        df = pd.DataFrame(rows)
        df["cumulative_points"] = df["points"].cumsum()
        return df

    def constructor_points(self, season: int, rounds: int = 6) -> pd.DataFrame:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        races = schedule[schedule["EventFormat"] != "testing"].head(rounds)
        frame: list[pd.DataFrame] = []

        for _, event in races.iterrows():
            race_name = event["EventName"]
            results = self._race_results(season, race_name)
            by_team = results.groupby("TeamName", as_index=False)["Points"].sum()
            by_team["round"] = int(event["RoundNumber"])
            frame.append(by_team)

        return pd.concat(frame, ignore_index=True).sort_values(["round", "TeamName"])

    def top_insights(self, season: int) -> list[dict[str, str]]:
        # Use first race for a lightweight, deterministic starter insight endpoint.
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        first_race = schedule[schedule["EventFormat"] != "testing"].iloc[0]["EventName"]
        results = self._race_results(season, first_race)

        winner = results.sort_values("Position").iloc[0]
        fastest_lap = results.sort_values("FastestLapTime").iloc[0]
        most_points_team = (
            results.groupby("TeamName", as_index=False)["Points"].sum().sort_values("Points", ascending=False).iloc[0]
        )

        return [
            {
                "title": "First race winner",
                "value": str(winner["BroadcastName"]),
                "context": f"{first_race} {season}",
            },
            {
                "title": "Fastest lap benchmark",
                "value": str(fastest_lap["BroadcastName"]),
                "context": f"{first_race} lap pace",
            },
            {
                "title": "Highest scoring constructor",
                "value": str(most_points_team["TeamName"]),
                "context": f"{first_race} points total",
            },
        ]
