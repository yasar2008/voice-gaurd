"""
Tests for the FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app


@pytest.fixture
def client():
    """Create a test client with the FastAPI app."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Test GET /health."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data
        assert "device" in data

    def test_health_models_loaded(self, client):
        response = client.get("/health")
        data = response.json()
        models = data["models_loaded"]
        assert models["spoof_detector"] is True
        assert models["prosody_analyzer"] is True
        assert models["risk_scorer"] is True


class TestConfigEndpoint:
    """Test GET/PUT /config."""

    def test_get_config(self, client):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "fusion_weights" in data
        assert "alert_threshold" in data
        assert abs(sum(data["fusion_weights"].values()) - 1.0) < 0.01

    def test_update_threshold(self, client):
        response = client.put("/config", json={"alert_threshold": 75.0})
        assert response.status_code == 200
        data = response.json()
        assert data["alert_threshold"] == 75.0

    def test_update_invalid_weights(self, client):
        response = client.put(
            "/config",
            json={
                "fusion_weights": {
                    "spoof_detection": 0.5,
                    "speaker_similarity": 0.5,
                    "prosody_naturalness": 0.5,
                }
            },
        )
        assert response.status_code == 400


class TestEnrollEndpoint:
    """Test POST/DELETE /enroll."""

    def test_enroll_no_file_returns_422(self, client):
        response = client.post("/enroll")
        assert response.status_code == 422  # Missing file

    def test_clear_enrollment(self, client):
        response = client.delete("/enroll")
        assert response.status_code == 200
        assert response.json()["status"] == "enrollment_cleared"


class TestAnalyzeEndpoint:
    """Test POST /analyze."""

    def test_analyze_no_file_returns_422(self, client):
        response = client.post("/analyze")
        assert response.status_code == 422
