from fastapi.testclient import TestClient

from src.api.main import app


def test_health_check_endpoint():
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["panchayats_indexed"] == 234


def test_live_inference_endpoint():
    client = TestClient(app)
    response = client.post("/api/v1/infer", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_panchayats_mapped"] == 234
    assert len(data["top_wettest_panchayats"]) == 5
    assert len(data["driest_panchayats"]) == 5
    assert data["mass_conservation_error_pct"] < 0.1
