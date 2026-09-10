"""
tests/test_credibility_hardening.py

Unit tests for SIH credibility hardening:
1. 7-day monsoon decay plausibility (monotonic decay, bounded orographic modifier)
2. SQLite concurrent write integrity
3. Stats endpoint real vs seed counts
4. INGEST_MODE=live raising NotImplementedError
5. crop_economics.json loading and schema verification
6. Stage-2 integration adapter stubs (NCUM, SMAP)
"""

import asyncio
import os
from pathlib import Path
import pytest

from src.advisory.economics import load_crop_economics, get_crop_risk_inr, get_crop_economics_provenance
from src.advisory.engine import build_7day_forecast, compute_orographic_mod
from src.api.feedback_store import (
    init_db,
    insert_feedback,
    get_feedback_stats,
    list_feedbacks,
)
from scripts.run_pipeline import SyntheticIngestionHarness
from src.integrations.ncum_adapter import NCUMAdapter
from src.integrations.smap_adapter import SMAPAdapter


def test_7day_monsoon_decay_plausibility():
    """Verify physical 7-day monsoon decay: monotonic decrease over lead days and bounded modifier."""
    # Test orographic modifier bounds across extreme elevations
    for elev in [0.0, 300.0, 670.0, 850.0, 1200.0, 2500.0]:
        mod = compute_orographic_mod("test_lgd", elev)
        assert 0.9 <= mod <= 1.15, f"Modifier {mod} out of bounds [0.9, 1.15] for elev {elev}"

    # Verify higher elevation provides higher or equal retention modifier
    assert compute_orographic_mod("lgd_high", 850.0) >= compute_orographic_mod("lgd_low", 600.0)

    # Test forecast generation on an un-mocked generic GP
    record = {
        "lgd_code": "999999",
        "panchayat_name": "TEST_GP",
        "forecast_date": "2023-07-01",
        "expected_mm": 18.0,
        "likely_min_mm": 10.0,
        "likely_max_mm": 28.0,
        "elevation_m": 720.0,
    }
    forecast = build_7day_forecast(record)
    assert len(forecast) == 7, f"Expected 7-day forecast series, got {len(forecast)}"

    # Day 0 is baseline expected_mm
    assert forecast[0]["expected_mm"] == 18.0

    # Days 1 to 6 should exhibit monotonic decay
    prev_rain = forecast[1]["expected_mm"]
    for day_i in range(2, 7):
        cur_rain = forecast[day_i]["expected_mm"]
        assert cur_rain <= prev_rain, f"Monsoon decay violation at day {day_i}: {cur_rain} > {prev_rain}"
        prev_rain = cur_rain


def test_sqlite_concurrent_write_integrity(tmp_path):
    """Verify SQLite thread-safe concurrent append of 20 entries in WAL mode."""
    test_db = tmp_path / "concurrent_test.db"
    init_db(test_db)

    def write_entry(idx: int):
        entry = {
            "validation_id": f"val_{idx:03d}",
            "lgd_code": "215504",
            "panchayat_name": "BANAVASI",
            "rained_bool": (idx % 2 == 0),
            "observer_role": "DAIRY_SECRETARY",
            "milk_center_id": f"KMF_KA_MAN_{idx:04d}",
            "observation_period": "LAST_12_HOURS",
            "source": "field_submission",
        }
        insert_feedback(entry, test_db)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(write_entry, i) for i in range(20)]
        concurrent.futures.wait(futures)
        for f in futures:
            assert f.exception() is None

    feedbacks = list_feedbacks(test_db)
    assert len(feedbacks) == 20
    ids = {f["validation_id"] for f in feedbacks}
    assert len(ids) == 20


def test_nandini_stats_real_vs_seed_counts(tmp_path):
    """Verify stats endpoint reports real field submissions and seed fixtures separately."""
    test_db = tmp_path / "stats_test.db"
    init_db(test_db)

    # Insert 3 seed fixtures
    for i in range(3):
        insert_feedback(
            {
                "validation_id": f"seed_{i}",
                "lgd_code": "215504",
                "rained_bool": True,
                "source": "seed_fixture",
            },
            test_db,
        )

    # Insert 2 real field submissions
    for i in range(2):
        insert_feedback(
            {
                "validation_id": f"real_{i}",
                "lgd_code": "215504",
                "rained_bool": False,
                "source": "field_submission",
            },
            test_db,
        )

    stats = get_feedback_stats(test_db)
    assert stats["total_validations"] == 5
    assert stats["seed_count"] == 3
    assert stats["real_count"] == 2
    assert stats["rain_reported_count"] == 3
    assert stats["no_rain_reported_count"] == 2


def test_ingest_mode_live_raises_not_implemented(monkeypatch):
    """Verify INGEST_MODE=live invokes IMDLiveAdapter and raises descriptive NotImplementedError."""
    monkeypatch.setenv("INGEST_MODE", "live")
    harness = SyntheticIngestionHarness()

    with pytest.raises(NotImplementedError) as exc_info:
        harness.ingest("2023-07-01")
    assert "ministry IP-whitelisted credentials" in str(exc_info.value)


def test_crop_economics_loading_and_citations():
    """Verify crop_economics.json loads cleanly and provides UAS Bangalore citations."""
    economics = load_crop_economics()
    assert len(economics) >= 8, f"Expected at least 8 crop economics entries, got {len(economics)}"

    crops = {e["crop"] for e in economics}
    assert "ragi" in crops
    assert "paddy" in crops
    assert "sugarcane" in crops

    # Check schema completeness
    for item in economics:
        assert "crop" in item
        assert "stage" in item
        assert "risk_inr_per_acre" in item
        assert item["risk_inr_per_acre"] > 0
        assert "source" in item
        assert "Bangalore" in item["source"] or "ICAR" in item["source"]
        assert "assumption" in item

    # Verify key phenological stage values match UAS Bangalore benchmarks
    assert get_crop_risk_inr("ragi", "harvest") == 6500
    assert get_crop_risk_inr("ragi", "vegetative") == 1800
    assert get_crop_risk_inr("ragi", "flowering") == 1400
    assert get_crop_risk_inr("ragi", "sowing") == 2500

    # Verify provenance lookup
    prov = get_crop_economics_provenance("ragi", "harvest")
    assert prov is not None
    assert "UAS" in prov["source"] or "University" in prov["source"]


def test_adapter_stubs_raise_not_implemented():
    """Verify Stage-2 NCUM and SMAP typed adapter stubs define contracts and raise descriptive errors."""
    ncum = NCUMAdapter()
    with pytest.raises(NotImplementedError) as exc_info:
        ncum.fetch_3hourly_tensors("2023-07-01")
    assert "NCMRWF/NSIDC sandbox credentials" in str(exc_info.value)

    smap = SMAPAdapter()
    with pytest.raises(NotImplementedError) as exc_info:
        smap.fetch_soil_moisture_raster("2023-07-01", (12.0, 76.0, 13.0, 77.0))
    assert "NCMRWF/NSIDC sandbox credentials" in str(exc_info.value)
