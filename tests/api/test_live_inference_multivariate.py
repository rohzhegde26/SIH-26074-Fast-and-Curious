"""
tests/api/test_live_inference_multivariate.py

Targeted regression and integration tests verifying that live inference:
1. Loads and ingests real coarse NWP forecast meteorological inputs (Tmax, Tmin, RH, Wind).
2. Replaces climatological fallbacks with real coarse NWP context.
3. Successfully falls back to climatology defaults when coarse forecast artifacts are absent.
4. Correctly validates and prioritizes explicit caller-supplied meteorological grids.
5. Emits unambiguous provenance metadata distinguishing NWP vs Climatology.
6. Preserves full backward compatibility with existing request formats and schemas.
"""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.inference_service import run_live_inference, get_default_coarse_grid
from src.data.forecast_loader import load_latest_coarse_forecast, extract_coarse_meteorological_tensors
from src.models.multivariate import CLIMATOLOGY_DEFAULTS, PROVENANCE_TAG

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def client():
    return TestClient(app)


def test_1_real_forecast_path_loads_nwp_context():
    """
    Test 1: Given a valid coarse forecast containing T/RH/wind:
    /api/v1/infer loads those fields and does NOT use raw climatology defaults.
    """
    resp = run_live_inference()
    assert resp.status == "success"
    assert resp.total_panchayats_mapped == 234

    # Check top wettest GP
    top_gp = resp.top_wettest_panchayats[0]

    # Operational coarse NWP in Mandya for Sept 10 has Tmax ~28.0°C, RH ~87%, Wind ~20 km/h
    # Climatology defaults are 31.5, 68.0, 8.5.
    # Verify that live inference reflects real NWP values rather than raw defaults
    assert abs(top_gp.tmax_c - CLIMATOLOGY_DEFAULTS["tmax"]) > 1.0 or \
           abs(top_gp.rh_pct - CLIMATOLOGY_DEFAULTS["rh"]) > 5.0 or \
           abs(top_gp.wind_kph - CLIMATOLOGY_DEFAULTS["wind"]) > 3.0

    # Provenance must indicate NWP or committed cycle rather than synthetic climatology
    assert "OPENMETEO_FORECAST_DOWNSCALED" in resp.provenance or "COMMITTED_FALLBACK_CYCLE" in resp.provenance
    assert PROVENANCE_TAG not in resp.provenance


def test_2_missing_forecast_falls_back_to_climatology(monkeypatch, tmp_path):
    """
    Test 2: Given no valid forecast artifact:
    Inference still succeeds, existing climatology fallback is used, and provenance indicates fallback.
    """
    # Point forecast loader to an empty directory
    empty_dir = tmp_path / "empty_forecasts"
    empty_dir.mkdir()

    import src.api.inference_service as infer_mod
    monkeypatch.setattr(
        infer_mod,
        "load_latest_coarse_forecast",
        lambda *args, **kwargs: (None, True),
    )

    resp = infer_mod.run_live_inference()
    assert resp.status == "success"
    assert resp.total_panchayats_mapped == 234

    # Check that provenance explicitly flags climatological fallback
    assert PROVENANCE_TAG in resp.provenance


def test_3_explicit_request_inputs_precedence():
    """
    Test 3: If optional explicit grids are supplied, they take precedence over artifact loading.
    """
    # Create explicit non-default coarse grids
    explicit_tmax = np.full((16, 16), 37.0, dtype=np.float32).tolist()
    explicit_tmin = np.full((16, 16), 26.0, dtype=np.float32).tolist()
    explicit_rh = np.full((16, 16), 45.0, dtype=np.float32).tolist()
    explicit_wind = np.full((16, 16), 18.5, dtype=np.float32).tolist()

    resp = run_live_inference(
        coarse_tmax_grid=explicit_tmax,
        coarse_tmin_grid=explicit_tmin,
        coarse_rh_grid=explicit_rh,
        coarse_wind_grid=explicit_wind,
    )
    assert resp.status == "success"
    assert "EXPLICIT_COARSE_NWP" in resp.provenance

    # Output GPs should reflect the high 37°C base temperature (adjusted by elevation)
    for gp in resp.top_wettest_panchayats:
        assert gp.tmax_c >= 34.0, f"Expected Tmax >= 34.0°C from 37.0°C base, got {gp.tmax_c}"
        assert gp.rh_pct < 60.0, f"Expected low RH (< 60%) from 45% base, got {gp.rh_pct}"


def test_4_shape_validation_rejects_malformed_grids(client):
    """
    Test 4: Malformed T/RH/wind grids (e.g. 10x10 instead of 16x16 or non-finite) fail clearly with HTTP 400.
    """
    # Wrong row count
    bad_grid = [[25.0] * 16 for _ in range(10)]
    resp = client.post("/api/v1/infer", json={"coarse_tmax_grid": bad_grid})
    assert resp.status_code == 400

    # Wrong column count
    bad_cols = [[25.0] * 8 for _ in range(16)]
    resp2 = client.post("/api/v1/infer", json={"coarse_rh_grid": bad_cols})
    assert resp2.status_code == 400


def test_5_existing_api_compatibility_empty_json(client):
    """
    Test 5: The existing request format {} must continue to work without error.
    """
    response = client.post("/api/v1/infer", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_panchayats_mapped"] == 234
    assert len(data["top_wettest_panchayats"]) == 5
    assert len(data["driest_panchayats"]) == 5
    assert "provenance" in data


def test_6_forecasts_response_schema_preserved(client):
    """
    Test 6: Confirm /api/forecasts response format remains completely unchanged.
    """
    response = client.get("/api/forecasts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 234
    sample = data[0]
    for key in ["lgd_code", "panchayat_name", "rainfall_mm", "advisory", "agromet_context", "multi_day_forecast"]:
        assert key in sample, f"Missing key '{key}' in /api/forecasts response"

    agromet = sample["agromet_context"]
    assert "tmax_c" in agromet
    assert "tmin_c" in agromet
    assert "rh_pct" in agromet
    assert "wind_kph" in agromet


def test_7_regression_deliberate_non_climatological_nwp():
    """
    REGRESSION TEST AGAINST CURRENT BUG:
    Supply coarse NWP inputs deliberately different from climatology defaults:
      coarse Tmax = 35.0
      coarse Tmin = 24.0
      coarse RH   = 52.0
      coarse wind = 15.0
    Verify that the physics module receives and downscales those values rather than
    climatological defaults (31.5, 21.0, 68.0, 8.5).
    """
    tmax_35 = np.full((16, 16), 35.0, dtype=np.float32).tolist()
    tmin_24 = np.full((16, 16), 24.0, dtype=np.float32).tolist()
    rh_52 = np.full((16, 16), 52.0, dtype=np.float32).tolist()
    wind_15 = np.full((16, 16), 15.0, dtype=np.float32).tolist()

    resp = run_live_inference(
        coarse_tmax_grid=tmax_35,
        coarse_tmin_grid=tmin_24,
        coarse_rh_grid=rh_52,
        coarse_wind_grid=wind_15,
    )

    first_gp = resp.top_wettest_panchayats[0]

    # Verify that Tmax is close to 35°C base (not 31.5°C)
    assert abs(first_gp.tmax_c - 31.5) > 1.5, f"Bug detected: Tmax ({first_gp.tmax_c}) fell back to 31.5°C default!"
    assert 33.0 <= first_gp.tmax_c <= 37.0, f"Expected Tmax in [33, 37]°C, got {first_gp.tmax_c}"

    # Verify that Tmin is close to 24°C base (not 21.0°C)
    assert abs(first_gp.tmin_c - 21.0) > 1.5, f"Bug detected: Tmin ({first_gp.tmin_c}) fell back to 21.0°C default!"
    assert 22.0 <= first_gp.tmin_c <= 26.0, f"Expected Tmin in [22, 26]°C, got {first_gp.tmin_c}"

    # Verify that RH is close to 52% base (not 68.0%)
    assert abs(first_gp.rh_pct - 68.0) > 8.0, f"Bug detected: RH ({first_gp.rh_pct}) fell back to 68% default!"
    assert 45.0 <= first_gp.rh_pct <= 60.0, f"Expected RH in [45, 60]%, got {first_gp.rh_pct}"

    # Verify that Wind is close to 15 km/h base with topographic acceleration (not 8.5 km/h)
    assert abs(first_gp.wind_kph - 8.5) > 3.0, f"Bug detected: Wind ({first_gp.wind_kph}) fell back to 8.5 km/h default!"
    assert first_gp.wind_kph >= 14.0, f"Expected Wind >= 14.0 km/h, got {first_gp.wind_kph}"
