"""API tests: auth enforcement, response shapes, and correctness against tests/conftest.py's
fixture data (see FIXTURE_ROWS there for the exact rounds/points/DNFs these assertions rely on).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_root() -> None:
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_is_unauthenticated() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_route_requires_api_key(tmp_db) -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/drivers/AAA/trend?season=2023")
    assert response.status_code == 401


def test_protected_route_rejects_invalid_api_key(tmp_db) -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/drivers/AAA/trend?season=2023", headers={"X-API-Key": "not-a-real-key"}
        )
    assert response.status_code == 401


def test_driver_trend(authed_client) -> None:
    response = authed_client.get("/api/v1/drivers/AAA/trend?season=2023")
    assert response.status_code == 200
    body = response.json()
    assert body["driver_code"] == "AAA"
    assert len(body["rounds"]) == 2
    assert body["rounds"][0]["points"] == 25.0
    assert body["rounds"][0]["finish_position"] == 1
    assert body["rounds"][1]["cumulative_points"] == 50.0


def test_driver_trend_unknown_driver_404(authed_client) -> None:
    response = authed_client.get("/api/v1/drivers/ZZZ/trend?season=2023")
    assert response.status_code == 404


def test_head_to_head(authed_client) -> None:
    response = authed_client.get("/api/v1/drivers/AAA/head-to-head/BBB?season=2023")
    assert response.status_code == 200
    body = response.json()
    assert body["rounds_compared"] == 2
    assert body["driver_wins"] == 2
    assert body["teammate_wins"] == 0
    assert body["ties"] == 0


def test_head_to_head_no_shared_rounds_404(authed_client) -> None:
    # AAA and CCC never share a constructor -> never teammates
    response = authed_client.get("/api/v1/drivers/AAA/head-to-head/CCC?season=2023")
    assert response.status_code == 404


def test_constructor_trend(authed_client) -> None:
    response = authed_client.get("/api/v1/constructors/TeamX/trend?season=2023")
    assert response.status_code == 200
    body = response.json()
    assert body["constructor"] == "TeamX"
    assert len(body["rounds"]) == 2
    assert body["rounds"][0]["points"] == 43.0  # AAA 25 + BBB 18
    assert body["rounds"][0]["avg_finish_position"] == pytest.approx(1.5)


def test_constructor_reliability(authed_client) -> None:
    response = authed_client.get("/api/v1/constructors/TeamY/reliability?season=2023")
    assert response.status_code == 200
    body = response.json()
    assert body["total_entries"] == 4
    assert body["dnf_count"] == 2
    assert body["reliability_rate"] == pytest.approx(0.5)


def test_constructor_reliability_unknown_constructor_404(authed_client) -> None:
    response = authed_client.get("/api/v1/constructors/NoSuchTeam/reliability")
    assert response.status_code == 404


def test_prediction_next_race(authed_client) -> None:
    response = authed_client.get("/api/v1/predictions/AAA/next-race")
    assert response.status_code == 200
    body = response.json()
    assert body["driver_code"] == "AAA"
    assert body["predicted_points"] >= 0.0
    assert body["grid_position_source"] == "estimated_from_last_race"


def test_prediction_with_explicit_grid_position(authed_client) -> None:
    response = authed_client.get("/api/v1/predictions/AAA/next-race?grid_position=3")
    assert response.status_code == 200
    body = response.json()
    assert body["grid_position"] == 3
    assert body["grid_position_source"] == "provided"


def test_prediction_unknown_driver_404(authed_client) -> None:
    response = authed_client.get("/api/v1/predictions/ZZZ/next-race")
    assert response.status_code == 404
