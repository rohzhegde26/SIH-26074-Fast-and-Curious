"""
test_gis.py
Unit tests for GIS validation, Mandya panchayats, and dual export.
"""

import json
from pathlib import Path
import geopandas as gpd
import pytest


def test_pinned_district_config():
    pinned_path = Path("src/data/pinned_district.json")
    assert pinned_path.exists(), "src/data/pinned_district.json must exist"
    with open(pinned_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert config["primary"] == "MANDYA"
    assert config["backup"] == "MYSURU"
    assert "BANGALORE URBAN" in config["forbidden"]
    assert "BBMP" in config["forbidden"]
    assert 80 <= config["gp_count"] <= 300, f"GP count {config['gp_count']} not in [80, 300]"
    assert config["valid_fraction"] >= 0.98, f"Valid fraction {config['valid_fraction']} < 0.98"


def test_mandya_full_geojson():
    geojson_path = Path("data/processed/mandya_full.geojson")
    assert geojson_path.exists(), "data/processed/mandya_full.geojson must exist"
    gdf = gpd.read_file(geojson_path)

    # Assert GP count is within official bounds
    assert 80 <= len(gdf) <= 300, f"Mandya GP count {len(gdf)} out of bounds [80, 300]"
    # Assert geometry validity
    assert all(gdf.geometry.is_valid), "All Mandya GP geometries in full GeoJSON must be valid"
    # Assert CRS is WGS84 EPSG:4326
    assert gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {gdf.crs}"
    # Assert key attributes present
    for col in ["gpcode", "gpname", "dtname"]:
        assert col in gdf.columns, f"Missing attribute column {col}"


def test_mandya_simplified_topojson():
    topo_path = Path("data/processed/mandya_simplified.topojson")
    assert topo_path.exists(), "data/processed/mandya_simplified.topojson must exist"
    size_kb = topo_path.stat().st_size / 1024
    assert size_kb < 400.0, f"Simplified TopoJSON {size_kb:.1f} KB exceeds 400 KB budget"

    with open(topo_path, "r", encoding="utf-8") as f:
        topo = json.load(f)
    assert topo["type"] == "Topology"
    assert "objects" in topo
    assert "panchayats" in topo["objects"]


def test_holdout_buffer():
    holdout_path = Path("data/processed/mandya_holdout_buffer.geojson")
    assert holdout_path.exists(), "Holdout buffer GeoJSON must exist"
    gdf_holdout = gpd.read_file(holdout_path)
    assert len(gdf_holdout) == 1
    assert gdf_holdout.geometry.iloc[0].is_valid

    # District full geometry must be completely contained in holdout buffer
    gdf_full = gpd.read_file("data/processed/mandya_full.geojson")
    district_geom = gdf_full.union_all() if hasattr(gdf_full, "union_all") else gdf_full.unary_union
    holdout_geom = gdf_holdout.geometry.iloc[0]
    assert holdout_geom.contains(district_geom), "Holdout buffer must contain the entire district"
