"""
Unit and integration tests for multi-region subcontinent deployment.
Validates:
1. /api/districts endpoint metadata
2. /api/forecasts district querying & backwards compatibility
3. TopoJSON size budget (<400 KB for PWA)
4. Cross-district forecast retrieval
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.api.main import app

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[2]


def test_api_districts_endpoint():
    """Verify /api/districts lists all 3 pilot domains across India."""
    response = client.get("/api/districts")
    assert response.status_code == 200
    data = response.json()
    assert "districts" in data
    districts = data["districts"]
    ids = [d["id"] for d in districts]
    assert "mandya" in ids, "Mandya (South) must be registered"
    assert "baghpat" in ids, "Baghpat (North) must be registered"
    assert "barpeta" in ids, "Barpeta (East/Northeast) must be registered"

    for d in districts:
        assert "center" in d and len(d["center"]) == 2
        assert "bounds" in d and len(d["bounds"]) == 2
        assert "gp_count" in d and d["gp_count"] > 0
        assert "zone" in d and len(d["zone"]) > 0


def test_api_forecasts_backward_compatibility():
    """Ensure default /api/forecasts (no param) still serves Mandya."""
    response = client.get("/api/forecasts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 80, "Default forecast call must return Mandya panchayats"
    assert data[0]["district"].upper() == "MANDYA"


def test_api_forecasts_multi_district():
    """Ensure /api/forecasts?district= serves respective districts."""
    # 1. Mandya
    res_man = client.get("/api/forecasts?district=mandya")
    assert res_man.status_code == 200
    data_man = res_man.json()
    assert len(data_man) >= 80
    assert all(r["district"].upper() == "MANDYA" for r in data_man)

    # 2. Baghpat
    res_bag = client.get("/api/forecasts?district=baghpat")
    assert res_bag.status_code == 200
    data_bag = res_bag.json()
    assert len(data_bag) >= 80, f"Expected >= 80 GPs for Baghpat, got {len(data_bag)}"
    assert all(r["district"].upper() == "BAGHPAT" for r in data_bag)

    # 3. Barpeta
    res_bar = client.get("/api/forecasts?district=barpeta")
    assert res_bar.status_code == 200
    data_bar = res_bar.json()
    assert len(data_bar) >= 80, f"Expected >= 80 GPs for Barpeta, got {len(data_bar)}"
    assert all(r["district"].upper() == "BARPETA" for r in data_bar)


def test_cross_district_panchayat_lookup():
    """Verify /api/forecast/{lgd_code} finds panchayats across all 3 districts."""
    # Mandya GP (Banavasi)
    r1 = client.get("/api/forecast/215504")
    assert r1.status_code == 200
    assert r1.json()["district"].upper() == "MANDYA"

    # Baghpat sample GP
    bag_records = client.get("/api/forecasts?district=baghpat").json()
    sample_bag_code = bag_records[0]["lgd_code"]
    r2 = client.get(f"/api/forecast/{sample_bag_code}")
    assert r2.status_code == 200
    assert r2.json()["district"].upper() == "BAGHPAT"

    # Barpeta sample GP
    bar_records = client.get("/api/forecasts?district=barpeta").json()
    sample_bar_code = bar_records[0]["lgd_code"]
    r3 = client.get(f"/api/forecast/{sample_bar_code}")
    assert r3.status_code == 200
    assert r3.json()["district"].upper() == "BARPETA"


def test_topojson_sizes_under_pwa_budget():
    """Ensure all simplified TopoJSON map files are < 400 KB for rapid PWA load."""
    for slug in ["mandya", "baghpat", "barpeta"]:
        p = ROOT / "frontend" / f"{slug}_simplified.topojson"
        assert p.exists(), f"TopoJSON missing: {p}"
        size_kb = p.stat().st_size / 1024
        assert size_kb < 400.0, f"{slug}_simplified.topojson exceeds 400 KB: {size_kb:.1f} KB"
