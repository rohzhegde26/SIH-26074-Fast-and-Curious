import asyncio
import json
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from src.advisory.engine import build_advisory
from src.advisory.rules import compute_financial_risk
import src.api.main as api_main
from src.api.main import app

client = TestClient(app)


def test_virtual_arg_endpoint():
    response = client.get("/api/v1/virtual-arg/215504")
    assert response.status_code == 200
    data = response.json()

    # Verify IMD AWS/ARG standard schema compliance
    assert data["station_id"] == "VARG_KA_MAN_215504"
    assert "Virtual ARG" in data["station_name"]
    assert data["lgd_code"] == "215504"
    assert data["district"] == "MANDYA"
    assert data["state"] == "KARNATAKA"
    assert 12.0 <= data["latitude"] <= 13.5
    assert 76.0 <= data["longitude"] <= 77.5
    assert data["elevation_m"] > 0
    assert "08:30:00 IST" in data["observation_datetime_ist"]
    assert data["rainfall_24h_mm"] >= 0
    assert data["uncertainty_range_90pct"]["lower_bound_mm"] <= data["uncertainty_range_90pct"]["upper_bound_mm"]
    assert data["qc_status"] == "VALIDATED_MASS_CONSERVED"

    # Mandatory Guardrail: Never present synthetic data as physical hardware
    assert data["data_type"] == "SYNTHETIC_DOWNSCALED_FEATURE_STREAM"
    assert data["provenance"] == "SIH26074_vARG_Unet5x_GLO30"


def test_virtual_arg_bulk():
    response = client.get("/api/v1/virtual-arg")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 234, f"Expected 234 Virtual ARGs for Mandya district, got {len(data)}"


def test_nandini_validation_submission_and_stats():
    # 1. Post validation from Dairy Secretary
    payload = {
        "lgd_code": "215504",
        "panchayat_name": "BANAVASI",
        "rained_bool": False,
        "observer_role": "DAIRY_SECRETARY",
        "milk_center_id": "KMF_MAN_2155",
        "observation_period": "LAST_12_HOURS",
    }
    post_res = client.post("/api/v1/validation/nandini", json=payload)
    assert post_res.status_code == 200
    post_data = post_res.json()
    assert post_data["status"] == "success"
    assert post_data["lgd_code"] == "215504"
    assert post_data["validation_id"]

    # 2. Get aggregate stats
    stats_res = client.get("/api/v1/validation/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_validations"] >= 1
    assert 0.0 <= stats["model_agreement_rate_pct"] <= 100.0


def test_phenology_financial_cost_of_error():
    # 1. Vegetative stage with >5mm rain -> Fertilizer leaching risk (~₹1,800/acre)
    veg_risk = compute_financial_risk("ragi", "vegetative", expected_mm=8.0, likely_max_mm=14.0)
    assert veg_risk.risk_level == "MODERATE_WARNING"
    assert veg_risk.cost_estimate_inr == 1800
    assert "Fertilizer Leaching" in veg_risk.impact_title_en
    assert "ರಸಗೊಬ್ಬರ ಕೊಚ್ಚಿಹೋಗುವ" in veg_risk.impact_title_kn

    # 2. Harvest stage with >5mm rain -> Crop spoilage & rotting risk (~₹6,500/acre)
    harvest_risk = compute_financial_risk("ragi", "harvest", expected_mm=10.0, likely_max_mm=18.0)
    assert harvest_risk.risk_level == "HIGH_FINANCIAL_LOSS"
    assert harvest_risk.cost_estimate_inr == 6500
    assert "Crop Spoilage" in harvest_risk.impact_title_en
    assert "ಧಾನ್ಯ ಕೊಳೆಯುವಿಕೆ" in harvest_risk.impact_title_kn

    # 3. Dry harvest window -> Safe (₹0 loss risk)
    dry_harvest = compute_financial_risk("ragi", "harvest", expected_mm=0.2, likely_max_mm=1.5)
    assert dry_harvest.risk_level == "LOW"
    assert dry_harvest.cost_estimate_inr == 0

    # 4. Engine integration
    adv = build_advisory("ragi", "vegetative", expected_mm=8.0, likely_max_mm=14.0)
    assert adv.financial_risk is not None
    assert adv.financial_risk.cost_estimate_inr == 1800


def test_concurrent_nandini_writes(tmp_path, monkeypatch):
    """
    Test 20 concurrent requests to /api/v1/validation/nandini using asyncio.gather.
    Asserts all 20 successfully append without 500 errors or JSON corruption.
    """
    temp_feedback_file = tmp_path / "nandini_feedback_test.json"
    monkeypatch.setattr(api_main, "FEEDBACK_PATH", temp_feedback_file)

    async def _fire_concurrent_requests():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            tasks = [
                async_client.post(
                    "/api/v1/validation/nandini",
                    json={
                        "lgd_code": "215504",
                        "panchayat_name": f"PANCHAYAT_{i:02d}",
                        "rained_bool": (i % 2 == 0),
                        "observer_role": "DAIRY_SECRETARY",
                        "milk_center_id": f"KMF_KA_MAN_{i:04d}",
                        "observation_period": "LAST_12_HOURS",
                    },
                )
                for i in range(20)
            ]
            return await asyncio.gather(*tasks)

    responses = asyncio.run(_fire_concurrent_requests())

    # 1. Assert all 20 responses returned HTTP 200 without throwing 500 errors
    assert len(responses) == 20
    for idx, resp in enumerate(responses):
        assert resp.status_code == 200, f"Request {idx} failed with status {resp.status_code}: {resp.text}"
        payload = resp.json()
        assert payload["status"] == "success"
        assert payload["validation_id"]

    # 2. Assert the feedback JSON exists and is valid, uncorrupted JSON
    assert temp_feedback_file.exists(), "Target feedback JSON file should exist"
    with open(temp_feedback_file, encoding="utf-8") as f:
        data = json.load(f)

    # 3. Assert all 20 writes were successfully appended and recorded
    assert isinstance(data, list)
    assert len(data) == 20, f"Expected 20 appended records, found {len(data)}"

    validation_ids = [entry["validation_id"] for entry in data]
    assert len(set(validation_ids)) == 20, "All validation IDs must be distinct"

    panchayat_names = {entry["panchayat_name"] for entry in data}
    expected_names = {f"PANCHAYAT_{i:02d}" for i in range(20)}
    assert panchayat_names == expected_names

