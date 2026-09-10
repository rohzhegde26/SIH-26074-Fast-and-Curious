"""
tests/api/test_multiday_nwp.py

Unit and integration tests for multi-day NWP lead times (Days 1 to 5),
multivariate payload serialization, and backward compatibility.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.inference_service import run_live_inference, get_default_coarse_grid


@pytest.fixture
def client():
    return TestClient(app)


def test_single_day_backward_compatibility():
    """Verify that calling run_live_inference with single day maintains full backward compatibility."""
    resp = run_live_inference()
    assert resp.status == "success"
    assert resp.input_shape == [1, 1, 16, 16]
    assert resp.output_shape == [1, 1, 80, 80]
    assert resp.total_panchayats_mapped == 234
    assert len(resp.top_wettest_panchayats) == 5
    assert len(resp.driest_panchayats) == 5


def test_multiday_batch_inference():
    """Verify multi-day NWP batch processing for 5 lead days."""
    # 5 days of 16x16 grids
    base = get_default_coarse_grid()
    coarse_5d = [base * (1.0 + 0.1 * d) for d in range(5)]
    coarse_5d_list = [g.tolist() for g in coarse_5d]

    resp = run_live_inference(coarse_grids=coarse_5d_list, lead_days=5)
    assert resp.status == "success"
    assert resp.lead_days == 5
    assert resp.input_shape == [5, 1, 16, 16]
    assert resp.output_shape == [5, 1, 80, 80]

    # Check top wettest panchayats have multivariate attributes
    first_gp = resp.top_wettest_panchayats[0]
    assert hasattr(first_gp, "rainfall_expected_mm")
    assert hasattr(first_gp, "tmax_c")
    assert hasattr(first_gp, "tmin_c")
    assert hasattr(first_gp, "rh_pct")
    assert hasattr(first_gp, "wind_kph")
    assert hasattr(first_gp, "heat_stress_level")

    # Validate physical bounds
    assert first_gp.tmin_c <= first_gp.tmax_c
    assert 10.0 <= first_gp.rh_pct <= 100.0
    assert 0.5 <= first_gp.wind_kph <= 120.0


def test_api_inference_endpoint_multiday(client):
    """Verify POST /api/inference supports multi-day NWP payload."""
    base = get_default_coarse_grid().tolist()
    payload = {
        "coarse_grids": [base, base, base],
        "lead_days": 3,
    }
    response = client.post("/api/inference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["lead_days"] == 3
    assert data["input_shape"] == [3, 1, 16, 16]
    assert data["output_shape"] == [3, 1, 80, 80]


def test_api_forecast_panchayat_multivariate(client):
    """Verify GET /api/forecast/{panchayat_id} includes multivariate fields."""
    response = client.get("/api/forecast/215504")  # Banavasi
    assert response.status_code == 200
    data = response.json()
    assert "agromet_context" in data
    agromet = data["agromet_context"]
    assert "temp_c" in agromet
    assert "rh_pct" in agromet
    assert "wind_kph" in agromet
    assert "tmax_c" in agromet
    assert "tmin_c" in agromet
    assert "heat_stress_level" in agromet

    # Verify 7-day items have multivariate fields
    multi_day = data["multi_day_forecast"]
    assert len(multi_day) == 7
    first_day = multi_day[0]
    assert "tmax_c" in first_day
    assert "tmin_c" in first_day
    assert "rh_pct" in first_day
    assert "wind_kph" in first_day
    assert "heat_stress_level" in first_day
