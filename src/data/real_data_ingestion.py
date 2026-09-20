"""
src/data/real_data_ingestion.py

Ingestion & Packaging Pipeline for Multi-Task Weather Downscaling.
Produces data/cache/multitask_real.npz containing 1,220 authentic daily samples
across 10 monsoon seasons (2014-2023) and the canonical 4°x4° Peninsular domain:
    1. Rain target & coarse input: Supervised strictly by UCSB CHIRPS v2.0
       (coarse: cogs/p25/ [16x16], target: cogs/p05/ [80x80]).
    2. Thermodynamic targets & coarse inputs: Authentic ECMWF ERA5 & ERA5-Land
       (coarse: ERA5 0.25° [16x16], target: ERA5-Land 0.1° regridded to 0.05° [80x80])
       aggregated strictly on the 03:00-02:00 UTC canonical 24-hour meteorological day.
    3. Terrain prior: Authentic Copernicus GLO-30 DSM elevation, Horn 3x3 slope,
       continuous circular aspect, curvature, and 250° monsoon windward lift via build_terrain_tensor_5ch.
    4. Coarse NWP input: 0.25° (16x16) atmospheric fields with operational forecast bias augmentation
       applied during training.

Partitions:
    - Train: 8 seasons (2014-2021, 976 samples, 80.0%)
    - Val: 1 season (2022, 122 samples, 10.0%)
    - Test: 1 season (2023, 122 samples, 10.0%)
    Total: exactly 1,220 spatiotemporal daily samples.
Zero synthetic fallback; hard failure if required raw datasets are missing.
"""

from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.agera5_loader import apply_nwp_forecast_bias_augmentation
from src.data.terrain_features import build_terrain_tensor_5ch

CACHE_DIR = ROOT / "data" / "cache"
REAL_CACHE_FILE = CACHE_DIR / "multitask_real.npz"
DEM_FILE = ROOT / "data" / "raw" / "dem" / "glo30_mandya_terrain.nc"
CHIRPS_CACHE_DIR = ROOT / "data" / "raw" / "chirps" / "cache"
CHIRPS_NC_FILE = ROOT / "data" / "raw" / "chirps" / "chirps_daily.nc"
ERA5_NC_FILE = ROOT / "data" / "raw" / "era5_land" / "era5_land_daily.nc"


def build_and_cache_real_multitask_dataset(
    cache_path: Path = REAL_CACHE_FILE,
    years: Optional[List[int]] = None,
    force_rebuild: bool = False,
) -> Path:
    """
    Constructs and serializes the authentic 1,220 spatiotemporal sample dataset.
    Strictly raises FileNotFoundError if required raw datasets are missing.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_rebuild:
        print(f"[*] Found cached real multi-task dataset: {cache_path} ({cache_path.stat().st_size / (1024*1024):.1f} MB)")
        return cache_path

    if years is None:
        years = list(range(2014, 2024))  # 10 seasons: 2014-2023

    # 1. Enforce presence of authentic DEM
    if not DEM_FILE.exists():
        raise FileNotFoundError(
            f"Required authentic Copernicus GLO-30 DEM file not found at: {DEM_FILE}. "
            f"Run python src/data/glo30_terrain.py to generate it."
        )

    # 2. Enforce presence of ERA5/ERA5-Land reanalysis
    if not ERA5_NC_FILE.exists():
        raise FileNotFoundError(
            f"Required authentic ERA5/ERA5-Land NetCDF file not found at: {ERA5_NC_FILE}. "
            f"Run python src/data/openmeteo_era5_ingestion.py to generate it."
        )

    print(f"[*] Materializing authentic multi-task dataset ({len(years) * 122} daily samples)...")

    # Load authentic Copernicus GLO-30 DSM terrain features
    with xr.open_dataset(DEM_FILE) as dem_ds:
        elev_arr = dem_ds["elevation"].values[:80, :80].astype(np.float32)
        slope_arr = dem_ds["slope"].values[:80, :80].astype(np.float32)
        aspect_arr = dem_ds["aspect"].values[:80, :80].astype(np.float32)

    # Standardize via build_terrain_tensor_5ch: [5, 80, 80]
    canonical_terrain_tensor = build_terrain_tensor_5ch(
        elev_arr, slope_arr, aspect_arr, center_lat_deg=13.0, month=7
    ).cpu().numpy()

    # Load authentic CHIRPS v2.0 precipitation
    p05_list = []
    p25_list = []
    all_dates = []

    if CHIRPS_NC_FILE.exists():
        with xr.open_dataset(CHIRPS_NC_FILE) as ch_ds:
            chirps_p05 = ch_ds["precip"].values.astype(np.float32)       # [1220, 80, 80]
            chirps_p25 = ch_ds["coarse_precip"].values.astype(np.float32) # [1220, 16, 16]
            chirps_dates = [str(t) for t in ch_ds["time"].values]
    else:
        # Load from per-season caches
        for yr in years:
            ch_cache = CHIRPS_CACHE_DIR / f"chirps_{yr}.npz"
            if not ch_cache.exists():
                raise FileNotFoundError(
                    f"Required CHIRPS cache for season {yr} not found at {ch_cache}. "
                    f"Run python src/data/chirps_ingestion.py."
                )
            cdata = np.load(ch_cache)
            p05_list.append(cdata["p05"])
            p25_list.append(cdata["p25"])
            all_dates.extend(cdata["dates"].tolist())

        chirps_p05 = np.concatenate(p05_list, axis=0)  # [1220, 80, 80]
        chirps_p25 = np.concatenate(p25_list, axis=0)  # [1220, 16, 16]
        chirps_dates = all_dates

    # Load authentic ERA5 & ERA5-Land reanalysis
    with xr.open_dataset(ERA5_NC_FILE) as era5_ds:
        era5_tmax = era5_ds["tmax"].values.astype(np.float32)
        era5_tmin = era5_ds["tmin"].values.astype(np.float32)
        era5_rh = era5_ds["rh"].values.astype(np.float32)
        era5_wind = era5_ds["wind"].values.astype(np.float32)

        coarse_tmax = era5_ds["coarse_tmax"].values.astype(np.float32)
        coarse_tmin = era5_ds["coarse_tmin"].values.astype(np.float32)
        coarse_rh = era5_ds["coarse_rh"].values.astype(np.float32)
        coarse_wind = era5_ds["coarse_wind"].values.astype(np.float32)

    total_samples = len(years) * 122
    assert len(chirps_p05) == total_samples, f"CHIRPS p05 samples {len(chirps_p05)} != {total_samples}"
    assert len(era5_tmax) == total_samples, f"ERA5 samples {len(era5_tmax)} != {total_samples}"

    coarse_list = []
    terrain_list = []
    target_list = []
    split_list = []
    year_list = []
    day_list = []

    seed_counter = 42

    for y_idx, year in enumerate(years):
        if year <= 2021:
            split_tag = 0  # train (8 years = 976 samples)
        elif year == 2022:
            split_tag = 1  # val (1 year = 122 samples)
        else:
            split_tag = 2  # test (1 year = 122 samples)

        for d in range(122):
            idx = y_idx * 122 + d
            rng = np.random.default_rng(seed_counter)

            # Fine Terrain Prior [5, 80, 80]
            fine_terrain = canonical_terrain_tensor.copy()

            # Fine Targets [5, 80, 80]
            fine_targets = np.stack([
                chirps_p05[idx],
                era5_tmax[idx],
                era5_tmin[idx],
                era5_rh[idx],
                era5_wind[idx],
            ], axis=0).astype(np.float32)

            # Coarse NWP Inputs [5, 16, 16]
            coarse_nwp = np.stack([
                chirps_p25[idx],
                coarse_tmax[idx],
                coarse_tmin[idx],
                coarse_rh[idx],
                coarse_wind[idx],
            ], axis=0).astype(np.float32)

            # Operational forecast bias augmentation during training split
            if split_tag == 0:
                coarse_nwp = apply_nwp_forecast_bias_augmentation(coarse_nwp, rng=rng)

            coarse_list.append(coarse_nwp)
            terrain_list.append(fine_terrain)
            target_list.append(fine_targets)
            split_list.append(split_tag)
            year_list.append(year)
            day_list.append(d)
            seed_counter += 1

    coarse_arr = np.stack(coarse_list, axis=0)    # [1220, 5, 16, 16]
    terrain_arr = np.stack(terrain_list, axis=0)  # [1220, 5, 80, 80]
    target_arr = np.stack(target_list, axis=0)    # [1220, 5, 80, 80]
    split_arr = np.array(split_list, dtype=np.int32)
    year_arr = np.array(year_list, dtype=np.int32)
    day_arr = np.array(day_list, dtype=np.int32)
    dates_arr = np.array(chirps_dates)

    tmp_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
    np.savez_compressed(
        tmp_path,
        coarse_nwp=coarse_arr,
        fine_terrain=terrain_arr,
        fine_targets=target_arr,
        splits=split_arr,
        years=year_arr,
        day_indices=day_arr,
        dates=dates_arr,
    )
    if tmp_path.exists():
        if cache_path.exists():
            try:
                cache_path.unlink()
            except Exception:
                pass
        shutil.move(str(tmp_path), str(cache_path))

    print(f"[+] Serialized authentic multi-task dataset to: {cache_path} ({cache_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"    Train samples (2014-2021): {np.sum(split_arr == 0)}")
    print(f"    Val samples   (2022):      {np.sum(split_arr == 1)}")
    print(f"    Test samples  (2023):      {np.sum(split_arr == 2)}")
    return cache_path


if __name__ == "__main__":
    build_and_cache_real_multitask_dataset(force_rebuild=True)
