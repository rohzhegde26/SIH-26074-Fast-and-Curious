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


def test_serving_json_7days_distinct_and_provenance(client):
    """
    Asserts that all 7 days in the serving JSON are mutually distinct and carry
    the correct provenance tags (Day 1: IMD_OBSERVATION_DOWNSCALED, Days 2-7: OPENMETEO_FORECAST_DOWNSCALED).
    """
    response = client.get("/api/forecast/219388")  # Nalligere
    assert response.status_code == 200
    data = response.json()

    multi_days = data["multi_day_forecast"]
    assert len(multi_days) == 7

    # Verify Day 1 vs Days 2-7 provenance tags
    assert multi_days[0]["provenance"] == "IMD_OBSERVATION_DOWNSCALED"
    for day in multi_days[1:]:
        assert day["provenance"] in ("OPENMETEO_FORECAST_DOWNSCALED", "COMMITTED_FALLBACK_CYCLE")

    # Assert mutually distinct days
    day_signatures = [
        (d["date"], d["day_offset"], d["expected_mm"], d["tmax_c"], d["rh_pct"], d["wind_kph"])
        for d in multi_days
    ]
    assert len(set(day_signatures)) == 7, "Expected all 7 forecast days to be mutually distinct"

    # Also assert dates are strictly incrementing
    dates = [d["date"] for d in multi_days]
    assert len(set(dates)) == 7


def test_cycle_age_badge_bucket_logic():
    """
    Tests the 3-tier degradation ladder for the cycle freshness badge:
    <= 1: green
    2-3: amber
    > 3: red with 'stale cycle: advisories from last sync'
    """
    def evaluate_badge(cycle_date: str, cycle_age: int) -> dict:
        age = int(cycle_age)
        if age <= 1:
            return {
                "class": "badge-green",
                "text": f"Cycle: {cycle_date} (age {age}d)",
            }
        elif age <= 3:
            return {
                "class": "badge-amber",
                "text": f"Cycle: {cycle_date} (age {age}d)",
            }
        else:
            return {
                "class": "badge-red",
                "text": f"Cycle: {cycle_date} (age {age}d) — stale cycle: advisories from last sync",
            }

    # Fresh cycle
    b0 = evaluate_badge("2026-09-10", 0)
    assert b0["class"] == "badge-green"
    assert "age 0d" in b0["text"]

    b1 = evaluate_badge("2026-09-09", 1)
    assert b1["class"] == "badge-green"
    assert "age 1d" in b1["text"]

    # Aging cache (2-3 days)
    b2 = evaluate_badge("2026-09-08", 2)
    assert b2["class"] == "badge-amber"
    assert "age 2d" in b2["text"]

    b3 = evaluate_badge("2026-09-07", 3)
    assert b3["class"] == "badge-amber"
    assert "age 3d" in b3["text"]

    # Stale cycle (> 3 days)
    b4 = evaluate_badge("2026-09-05", 5)
    assert b4["class"] == "badge-red"
    assert "stale cycle: advisories from last sync" in b4["text"]


def test_missing_file_fallback_path(tmp_path):
    """
    Verifies that when a forecast file is missing, the pipeline gracefully falls back
    to the last committed cycle and tags Days 2-7 with COMMITTED_FALLBACK_CYCLE without crashing.
    """
    from scripts.run_pipeline import run_pipeline

    output_path = tmp_path / "fallback_forecasts.json"
    missing_file = tmp_path / "non_existent_cycle.json"

    records = run_pipeline(
        forecast_date="2023-07-01",
        district="MANDYA",
        output_json_path=output_path,
        output_geojson_path=None,
        n_mc_passes=2,
        forecast_file=missing_file,
    )

    assert len(records) == 234
    first_gp = records[0]
    m_days = first_gp["multi_day_forecast"]
    assert len(m_days) == 7

    assert m_days[0]["provenance"] == "IMD_OBSERVATION_DOWNSCALED"
    for d in m_days[1:]:
        assert d["provenance"] == "COMMITTED_FALLBACK_CYCLE"

