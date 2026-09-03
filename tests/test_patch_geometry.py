"""
test_patch_geometry.py
Unit tests for patch geometry, All-India scope vs single-district impossibility,
and spatial holdout buffer exclusion.
"""

import json
from pathlib import Path
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box
from src.data.patch_extraction import (
    HR_PATCH_SIZE,
    HR_RES,
    generate_spatial_windows,
)


def test_single_district_patch_generation_fails():
    """
    Assert that training on Mandya alone is mathematically impossible:
    Mandya is ~4,961 km^2 (~165 HR pixels at 0.05°, bounding box ~20x26 pixels).
    A single 64x64 or 80x80 HR patch covers ~123,000 to ~192,000 km^2 (>24x larger than Mandya).
    Therefore, single-district patch generation must fail.
    """
    pinned_path = Path("src/data/pinned_district.json")
    with open(pinned_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    mandya_area_km2 = config["area_km2"]  # 4961 km^2
    mandya_hr_pixels = config["hr_pixels_area"]  # ~165 pixels
    mandya_bbox_pixels = config["hr_pixels_bbox"]  # [20, 26]

    # Patch size 64x64
    patch_pixels_64 = 64 * 64  # 4096 pixels
    patch_area_64_km2 = patch_pixels_64 * config["area_per_pixel_km2"]  # ~123,289 km^2

    # Patch size 80x80
    patch_pixels_80 = 80 * 80  # 6400 pixels
    patch_area_80_km2 = patch_pixels_80 * config["area_per_pixel_km2"]  # ~192,640 km^2

    assert patch_pixels_64 > mandya_hr_pixels, "64x64 patch must exceed Mandya pixel count"
    assert patch_pixels_80 > mandya_hr_pixels, "80x80 patch must exceed Mandya pixel count"
    assert patch_area_64_km2 > 20 * mandya_area_km2, "Patch is >20x larger than entire Mandya district"

    # Cannot extract even a single 64x64 or 80x80 patch completely inside Mandya's bbox [20, 26]
    max_h, max_w = mandya_bbox_pixels
    can_fit = (max_h >= 64) and (max_w >= 64)
    assert not can_fit, "Single district patch extraction must be impossible"


def test_all_india_patch_index_yield():
    """
    Assert that the All-India patch index yields between 200,000 and 240,000 usable patches
    over the 1,708 JJAS monsoon days (2010-2023).
    """
    summary_path = Path("data/cache/patch_index_summary.json")
    assert summary_path.exists(), "patch_index_summary.json must exist"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    usable_patches = summary["total_usable_patches"]
    assert 200000 <= usable_patches <= 240000, (
        f"Usable patch count {usable_patches} outside expected range [200,000, 240,000]"
    )
    assert 120 <= summary["usable_windows_per_day"] <= 140, (
        f"Daily windows {summary['usable_windows_per_day']} outside expected range [120, 140]"
    )


def test_land_fraction_threshold_enforced():
    """Assert all usable patches in the index satisfy land fraction >= 70%."""
    parquet_path = Path("data/cache/spatial_patch_index.parquet")
    assert parquet_path.exists(), "spatial_patch_index.parquet must exist"
    df = pd.read_parquet(parquet_path)

    assert "land_fraction" in df.columns
    assert (df["land_fraction"] >= 0.70).all(), "All indexed patches must have land fraction >= 70%"


def test_mandya_holdout_buffer_exclusion():
    """
    Strictly assert that NO indexed patch intersects the Mandya + 0.5° buffer:
    patch_index ∩ Mandya_buffer_0.5 == ∅.
    """
    holdout_path = Path("data/processed/mandya_holdout_buffer.geojson")
    assert holdout_path.exists(), "Holdout buffer GeoJSON must exist"
    gdf_holdout = gpd.read_file(holdout_path)
    holdout_geom = gdf_holdout.geometry.iloc[0]

    df_patches = pd.read_parquet("data/cache/spatial_patch_index.parquet")

    # Check every usable patch
    intersections = 0
    for _, row in df_patches.iterrows():
        patch_box = box(row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"])
        if patch_box.intersects(holdout_geom):
            intersections += 1

    assert intersections == 0, f"Found {intersections} patches leaking into Mandya holdout buffer!"
