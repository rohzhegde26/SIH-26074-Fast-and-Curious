import time
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_forecast_response():
    t0 = time.time()
    response = client.get("/api/forecast/215504")
    elapsed_ms = (time.time() - t0) * 1000

    assert response.status_code == 200
    assert elapsed_ms < 100, f"Response time {elapsed_ms:.1f}ms exceeds 100ms threshold!"

    body = response.json()
    assert body["lgd_code"] == "215504"
    assert body["district"] == "MANDYA"
    assert body["rainfall_mm"]["likely_min"] <= body["rainfall_mm"]["likely_max"]
    assert "calibrated" in body["rainfall_mm"]["empirical_coverage"]
    assert body["advisory"]["ragi"]["action_kn"]
    assert body["advisory"]["paddy"]["action_en"]
    assert body["advisory"]["sugarcane"]["action_kn"]
    assert body["advisory"]["sugarcane"]["stage"] == "Grand_Growth"


def test_bulk_forecasts_returns_all_mandya_panchayats():
    response = client.get("/api/forecasts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 234, f"Expected 234 Mandya panchayats, got {len(body)}"


def test_multiple_panchayats_queryable():
    # Query different panchayats across Mandya
    test_codes = ["215504", "215524", "219204"]
    for code in test_codes:
        res = client.get(f"/api/forecast/{code}")
        assert res.status_code == 200
        data = res.json()
        assert data["lgd_code"] == code
        assert data["rainfall_mm"]["expected"] >= 0.0


def test_unknown_panchayat_is_not_found():
    assert client.get("/api/forecast/99999999").status_code == 404
