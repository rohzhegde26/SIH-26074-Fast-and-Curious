"""
src/data/patch_extraction.py

All-India patch index generator and spatial holdout buffer filter.
Enforces:
    - 80x80 HR / 16x16 LR patches (direct 5x scale)
    - Stride: 40 pixels (in HR space)
    - Domain: 68°-97°E, 8°-37°N
    - Land fraction filter >= 70%
    - Strict Mandya + 0.5° buffer holdout exclusion
    - 4-way temporal splits (Train: 2010-2020, Val: 2021, Cal: 2022, Test: 2023)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


INDIA_DOMAIN = {
    "min_lon": 68.0,
    "max_lon": 97.0,
    "min_lat": 8.0,
    "max_lat": 37.0,
}

HR_RES = 0.05
LR_RES = 0.25
HR_PATCH_SIZE = 80
LR_PATCH_SIZE = 16
HR_STRIDE = 40
LAND_FRACTION_THRESHOLD = 0.70


def generate_spatial_windows() -> List[Dict]:
    """
    Generate all 80x80 candidate HR spatial windows across the India domain
    with a stride of 40 pixels, including boundary edges.
    """
    n_hr_lat = int(round((INDIA_DOMAIN["max_lat"] - INDIA_DOMAIN["min_lat"]) / HR_RES))  # 580
    n_hr_lon = int(round((INDIA_DOMAIN["max_lon"] - INDIA_DOMAIN["min_lon"]) / HR_RES))  # 580

    lat_starts = list(range(0, n_hr_lat - HR_PATCH_SIZE + 1, HR_STRIDE))
    if lat_starts[-1] != n_hr_lat - HR_PATCH_SIZE:
        lat_starts.append(n_hr_lat - HR_PATCH_SIZE)

    lon_starts = list(range(0, n_hr_lon - HR_PATCH_SIZE + 1, HR_STRIDE))
    if lon_starts[-1] != n_hr_lon - HR_PATCH_SIZE:
        lon_starts.append(n_hr_lon - HR_PATCH_SIZE)

    windows = []
    window_id = 0
    for r in lat_starts:
        for c in lon_starts:
            min_lat = INDIA_DOMAIN["min_lat"] + r * HR_RES
            max_lat = min_lat + HR_PATCH_SIZE * HR_RES
            min_lon = INDIA_DOMAIN["min_lon"] + c * HR_RES
            max_lon = min_lon + HR_PATCH_SIZE * HR_RES

            poly = box(min_lon, min_lat, max_lon, max_lat)

            windows.append({
                "window_id": window_id,
                "hr_row": r,
                "hr_col": c,
                "lr_row": r // 5,
                "lr_col": c // 5,
                "min_lon": float(round(min_lon, 4)),
                "max_lon": float(round(max_lon, 4)),
                "min_lat": float(round(min_lat, 4)),
                "max_lat": float(round(max_lat, 4)),
                "geometry": poly,
            })
            window_id += 1

    return windows


def estimate_india_land_fraction(min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> float:
    """
    Compute land fraction of a patch bounding box against India's continental landmass.
    Removes Arabian Sea (<72°E below 20°N) and Bay of Bengal (>82°E below 20°N).
    """
    # Sample points inside the patch
    lons = np.linspace(min_lon, max_lon, 10)
    lats = np.linspace(min_lat, max_lat, 10)
    xx, yy = np.meshgrid(lons, lats)
    pts_lon = xx.ravel()
    pts_lat = yy.ravel()

    # Heuristic/analytical landmask for Indian subcontinent
    # 1. Arabian Sea cutoff
    is_arabian_sea = (pts_lon < 73.0) & (pts_lat < 21.0) & ~(
        (pts_lon >= 72.5) & (pts_lat >= 18.5) & (pts_lat <= 21.0)
    )
    # 2. Deep Indian Ocean south of Kanyakumari (~8.0°N)
    is_south_ocean = pts_lat < 8.1
    # 3. Bay of Bengal cutoff
    is_bay_of_bengal = (pts_lon > 81.0) & (pts_lat < 16.0)
    is_deep_bob = (pts_lon > 85.0) & (pts_lat < 21.0)

    # 4. Far West Thar / Pakistan border cutoff beyond 69.5°E south of 24°N
    is_rann_ocean = (pts_lon < 69.0) & (pts_lat < 23.5)

    is_water = is_arabian_sea | is_south_ocean | is_bay_of_bengal | is_deep_bob | is_rann_ocean
    land_fraction = float(np.mean(~is_water))
    return land_fraction


def filter_spatial_windows(
    windows: List[Dict],
    holdout_geojson_path: str = "data/processed/mandya_holdout_buffer.geojson",
    land_threshold: float = LAND_FRACTION_THRESHOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter spatial windows:
    1. Exclude any window intersecting the Mandya + 0.5° buffer.
    2. Exclude any window with land fraction < land_threshold.
    
    Returns:
        (usable_windows, dropped_windows)
    """
    holdout_path = Path(holdout_geojson_path)
    assert holdout_path.exists(), f"Holdout file {holdout_path} not found"
    gdf_holdout = gpd.read_file(holdout_path)
    holdout_geom = gdf_holdout.geometry.iloc[0]

    usable = []
    dropped = []

    for w in windows:
        poly = w["geometry"]
        land_frac = estimate_india_land_fraction(
            w["min_lon"], w["max_lon"], w["min_lat"], w["max_lat"]
        )
        w["land_fraction"] = float(round(land_frac, 3))

        # Check holdout intersection
        intersects_holdout = poly.intersects(holdout_geom)
        w["intersects_mandya_buffer"] = bool(intersects_holdout)

        if intersects_holdout:
            w["drop_reason"] = "mandya_buffer_intersection"
            dropped.append(w)
        elif land_frac < land_threshold:
            w["drop_reason"] = "low_land_fraction"
            dropped.append(w)
        else:
            usable.append(w)

    return usable, dropped


def generate_monsoon_dates(years: List[int]) -> List[pd.Timestamp]:
    """Generate all JJAS (June 1 - Sept 30 = 122 days) dates for given years."""
    dates = []
    for yr in years:
        dr = pd.date_range(f"{yr}-06-01", f"{yr}-09-30", freq="D")
        dates.extend(dr)
    return dates


def get_temporal_split(year: int) -> str:
    """Assign temporal split according to pinned 4-way split architecture."""
    if 2010 <= year <= 2020:
        return "train"
    elif year == 2021:
        return "val"
    elif year == 2022:
        return "cal"
    elif year == 2023:
        return "test"
    else:
        return "train"
