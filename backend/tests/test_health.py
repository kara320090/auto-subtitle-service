from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert data["message"] == "Auto Subtitle Service API is running"


def test_health_check():
    response = client.get("/health/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "auto-subtitle-backend"
    assert "checks" in data
    assert "ffmpeg_available" in data["checks"]