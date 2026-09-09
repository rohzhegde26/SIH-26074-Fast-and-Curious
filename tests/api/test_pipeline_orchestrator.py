import time
from pathlib import Path
from scripts.run_pipeline import run_pipeline
from src.api.payload_validation import map_gpcodes, validate_records


def test_pipeline_orchestrator_execution_and_contract(tmp_path):
    output_json = tmp_path / "test_forecasts.json"
    t0 = time.time()
    records = run_pipeline(
        forecast_date="2023-07-01",
        district="MANDYA",
        output_json_path=output_json,
        output_geojson_path=None,
        n_mc_passes=5,
    )
    elapsed = time.time() - t0

    # 1. Timing requirement (< 35 seconds on CPU / < 5s on GPU)
    assert elapsed < 35.0, f"Pipeline took {elapsed:.2f}s, exceeding 35.0s limit!"

    # 2. Complete Mandya coverage (234 panchayats)
    assert len(records) == 234, f"Expected 234 records, got {len(records)}"

    # 3. Payload validation against simplified map
    valid_gpcodes = map_gpcodes(Path("frontend/mandya_simplified.topojson"))
    validate_records(records, valid_gpcodes)

    # 4. Consistency checks
    for r in records:
        assert r["district"] == "MANDYA"
        assert r["expected_mm"] >= 0.0
        assert r["likely_min_mm"] <= r["likely_max_mm"]
        assert r["timestamp_utc"].endswith("Z")
        assert r["ragi_stage"] in ("sowing", "vegetative", "flowering", "harvest")
        assert r["paddy_stage"] in ("sowing", "vegetative", "flowering", "harvest")
