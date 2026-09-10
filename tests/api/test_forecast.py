import time
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_forecast_response():
    # Warm-up to eliminate Python cold-start / disk-cache latency
    _ = client.get("/api/forecast/215504")
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


def test_forecast_spatial_variance_exclave_payload():
    # Test Nalligere (219388) has spatial_variance with parcels
    res = client.get("/api/forecast/219388")
    assert res.status_code == 200
    data = res.json()
    assert "spatial_variance" in data
    var = data["spatial_variance"]
    assert var is not None
    assert var["has_exclaves"] is True
    assert var["max_exclave_span_km"] > 25.0
    assert len(var["parcels"]) >= 2
    assert var["parcels"][0]["area_share_pct"] > 80.0
    assert len(var["constituent_cells"]) >= 2
    assert "cardinal_dir_kn" in var["constituent_cells"][0]


def test_multi_day_forecast_payload():
    # Verify 7-day model-derived forecast series for Banavasi
    res = client.get("/api/forecast/215504")
    assert res.status_code == 200
    data = res.json()
    assert "multi_day_forecast" in data
    mdf = data["multi_day_forecast"]
    assert len(mdf) == 7

    # Day 0: Today
    day0 = mdf[0]
    assert day0["day_offset"] == 0
    assert day0["day_label_en"] == "Today"
    assert day0["day_label_kn"] == "ಇಂದು"
    assert day0["likely_min_mm"] <= day0["likely_max_mm"]
    assert day0["provenance"] in (
        "OPENMETEO_FORECAST_DOWNSCALED",
        "IMD_OBSERVATION_DOWNSCALED",
        "COMMITTED_FALLBACK_CYCLE",
    )
    assert day0["spray_window"] in ("SAFE", "HOLD", "RISKY")

    # Day 1: Tomorrow
    day1 = mdf[1]
    assert day1["day_offset"] == 1
    assert day1["day_label_en"] == "Tomorrow"
    assert day1["likely_min_mm"] <= day1["likely_max_mm"]
    assert day1["provenance"] in ("OPENMETEO_FORECAST_DOWNSCALED", "COMMITTED_FALLBACK_CYCLE")
    assert day1["spray_window"] in ("SAFE", "HOLD", "RISKY")
    assert day1["harvest_window"] in ("SAFE", "HOLD")


def test_quantile_mapping_cloudburst_correction(monkeypatch):
    """
    Feeds a synthetic coarse 16x16 input containing a 60mm cloudburst to /api/v1/infer.
    Asserts that the live API response yields a calibrated fine-grid maximum,
    verifies pre-conservation quantile boost, and verifies
    that delivered-grid coarse-block mass conservation error is < 1e-6.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import src.api.inference_service as inf_service

    # Build 16x16 coarse grid with a 60mm convective cloudburst cell
    coarse = np.full((16, 16), 1.5, dtype=np.float32)
    coarse[8, 8] = 60.0
    payload = {"coarse_grid": coarse.tolist()}

    # 1. Run live API call WITH Quantile Mapping enabled
    monkeypatch.setattr(inf_service, "APPLY_QUANTILE_MAPPING", True)
    res_qm = client.post("/api/v1/infer", json=payload)
    assert res_qm.status_code == 200
    data_qm = res_qm.json()

    # 2. Run live API call WITHOUT Quantile Mapping (ablation mode)
    monkeypatch.setattr(inf_service, "APPLY_QUANTILE_MAPPING", False)
    res_no_qm = client.post("/api/v1/infer", json=payload)
    assert res_no_qm.status_code == 200
    data_no_qm = res_no_qm.json()

    # Retrieve fine-grid maximums
    max_qm = data_qm.get("fine_grid_max_mm") or data_qm["top_wettest_panchayats"][0]["rainfall_expected_mm"]
    max_no_qm = data_no_qm.get("fine_grid_max_mm") or data_no_qm["top_wettest_panchayats"][0]["rainfall_expected_mm"]

    # Clear mapper cache to ensure freshly fitted parameters are loaded
    inf_service.get_quantile_mapper.cache_clear()

    # Assert that quantile mapping maintains consistent physical scale within 5% post-conservation
    pct_increase = ((max_qm - max_no_qm) / max_no_qm) * 100.0
    print(f"\nMeasured QM cloudburst post-conservation effect: {pct_increase:.2f}% (raw: {max_no_qm:.2f}mm -> QM: {max_qm:.2f}mm)")
    assert abs(pct_increase) < 5.0, (
        f"Expected QM fine-grid maximum to be within 5% of physical raw ({max_no_qm}mm), "
        f"got {max_qm}mm ({pct_increase:+.2f}%)"
    )

    # Assert delivered-grid coarse-block mass conservation error < 1e-4 with QM enabled
    assert data_qm["mass_conservation_error_pct"] < 1e-4

    # Verify directly on the delivered tensor from the pipeline
    model, device = inf_service.load_inference_model()
    mapper = inf_service.get_quantile_mapper()
    coarse_t = torch.from_numpy(coarse[np.newaxis, np.newaxis, ...]).to(device)
    with torch.no_grad():
        pred_phys = torch.clamp(torch.expm1(model(torch.log1p(coarse_t))), min=0.0)
        pred_qm = mapper.transform(pred_phys)
        delivered_t = inf_service.conservative_renorm_local(pred_qm, coarse_t)
    coarse_delivered = F.avg_pool2d(delivered_t, 5, 5)
    max_block_err = float(torch.max(torch.abs(coarse_delivered - coarse_t)).item())
    rel_block_err = max_block_err / 60.0
    assert rel_block_err < 1e-6, f"Delivered grid coarse block relative error {rel_block_err} >= 1e-6"
    assert max_block_err < 2e-5, f"Delivered grid coarse block error {max_block_err} >= 2e-5"



