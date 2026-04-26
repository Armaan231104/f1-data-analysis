from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_requires_api_key() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 401


def test_health_with_api_key() -> None:
    response = client.get("/api/v1/health", headers={"X-API-Key": "dev-api-key"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
