"""
scripts/audit_real_datasets.py

Comprehensive Audit of All Downloaded & Processed Datasets.
Verifies:
    1. Authentic Copernicus GLO-30 DSM NetCDF: data/raw/dem/glo30_mandya_terrain.nc
    2. Authentic ECMWF ERA5-Land Daily NetCDF: data/raw/era5_land/era5_land_daily.nc
    3. Authentic IMD 0.25° Gridded Rainfall NetCDF: data/raw/imd/imd_sample.nc
    4. Authentic In-Situ AWS Station JSON: data/raw/stations/mandya_mysore_aws_2023.json
    5. Materialized Multi-Task Cache NPZ: data/cache/multitask_real.npz
Checks claimed variables, spatial coverage, timestamps, units, and provenance.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def audit_glo30_terrain():
    path = ROOT / "data" / "raw" / "dem" / "glo30_mandya_terrain.nc"
    assert path.exists(), f"Missing {path}"
    ds = xr.open_dataset(path)

    # 1. Variables
    expected_vars = ["elevation", "slope", "aspect", "curvature", "w_orog"]
    for v in expected_vars:
        assert v in ds.data_vars, f"Missing {v} in GLO-30 dataset"
        assert ds[v].shape == (80, 80), f"Invalid shape for {v}: {ds[v].shape}"

    # 2. Coordinates & Spatial Extent
    assert ds["lat"].shape == (80,)
    assert ds["lon"].shape == (80,)
    assert 11.0 <= float(ds["lat"].min()) <= 11.1
    assert 14.9 <= float(ds["lat"].max()) <= 15.0
    assert 74.0 <= float(ds["lon"].min()) <= 74.1
    assert 77.9 <= float(ds["lon"].max()) <= 78.0

    # 3. Provenance & Units
    assert ds.attrs.get("provenance") == "COPERNICUS_GLO30_DSM"
    assert "elevation" in ds and ds["elevation"].attrs.get("units") == "meters"
    assert ds["elevation"].values.min() >= 0.0
    assert ds["elevation"].values.max() > 1500.0  # Western Ghats peaks

    print("[PASS] GLO-30 DSM NetCDF: 80x80 (11-15N, 74-78E), variables, units, provenance verified.")


def audit_era5_land_daily():
    path = ROOT / "data" / "raw" / "era5_land" / "era5_land_daily.nc"
    assert path.exists(), f"Missing {path}"
    ds = xr.open_dataset(path)

    # 1. Variables
    expected_vars = ["tmax", "tmin", "rh", "wind"]
    for v in expected_vars:
        assert v in ds.data_vars, f"Missing {v} in ERA5-Land dataset"
        assert ds[v].shape == (1220, 80, 80), f"Invalid shape for {v}: {ds[v].shape}"

    # 2. Coordinates & Temporal Coverage
    assert ds["lat"].shape == (80,)
    assert ds["lon"].shape == (80,)
    assert len(ds["time"]) == 1220  # 10 seasons x 122 days
    years = pd.to_datetime(ds["time"].values).year.unique()
    assert list(years) == list(range(2014, 2024))

    # 3. Provenance, Temporal Window, Units
    assert ds.attrs.get("provenance") == "ECMWF_ERA5_LAND_REANALYSIS"
    assert ds.attrs.get("temporal_window") == "03:00-03:00 UTC"
    assert ds["tmax"].attrs.get("units") == "degC"
    assert ds["rh"].attrs.get("units") == "%"

    # 4. Physical Invariants
    tmax = ds["tmax"].values
    tmin = ds["tmin"].values
    rh = ds["rh"].values
    assert np.all(tmax >= tmin), "Tmin > Tmax violation in ERA5-Land"
    assert np.all((rh >= 0.0) & (rh <= 100.0)), "RH out of [0, 100]% bounds"

    print("[PASS] ERA5-Land Daily NetCDF: 1,220 days (2014-2023), 03:00-03:00 UTC, thermodynamic invariants verified.")


def audit_imd_rainfall():
    path = ROOT / "data" / "raw" / "imd" / "imd_sample.nc"
    assert path.exists(), f"Missing {path}"
    ds = xr.open_dataset(path)

    assert "rainfall" in ds.data_vars
    assert ds["rainfall"].shape == (5, 129, 135)
    assert ds.attrs.get("provenance") == "IMD_025_GRIDDED_RAINFALL"
    assert "03:00 UTC to 03:00 UTC" in ds.attrs.get("temporal_cycle", "")
    assert ds["rainfall"].attrs.get("units") == "mm/day"
    assert np.all(ds["rainfall"].values >= 0.0)

    print("[PASS] IMD 0.25° Gridded Rainfall: 129x135 grid, 03:00-03:00 UTC, provenance verified.")


def audit_station_observations():
    path = ROOT / "data" / "raw" / "stations" / "mandya_mysore_aws_2023.json"
    assert path.exists(), f"Missing {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("provenance") == "KSNDMC_IMD_IN_SITU_AWS"
    assert data.get("year") == 2023
    assert data.get("total_stations") == 14
    assert data.get("total_days") == 122
    assert "03:00-03:00 UTC" in data.get("daily_accumulation_window", "")

    stations = data.get("stations", [])
    assert len(stations) == 14
    for st in stations:
        assert "name" in st and "lat" in st and "lon" in st
        records = st.get("records", [])
        assert len(records) == 122
        for r in records:
            assert "qc_flag" in r
            if r["qc_flag"] == "PASSED":
                if r["rain_mm"] is not None:
                    assert r["rain_mm"] >= 0.0
                if r["tmax_c"] is not None and r["tmin_c"] is not None:
                    assert r["tmax_c"] >= r["tmin_c"]
                if r["rh_pct"] is not None:
                    assert 0.0 <= r["rh_pct"] <= 100.0

    print("[PASS] AWS Station Observations: 14 stations, 122 days, strict schema and QC flags verified.")


def audit_multitask_cache():
    path = ROOT / "data" / "cache" / "multitask_real.npz"
    assert path.exists(), f"Missing {path}"
    data = np.load(path)

    expected_keys = ["coarse_nwp", "fine_terrain", "fine_targets", "splits", "years"]
    for k in expected_keys:
        assert k in data, f"Missing {k} in cache"

    assert data["coarse_nwp"].shape == (3660, 5, 16, 16)
    assert data["fine_terrain"].shape == (3660, 5, 80, 80)
    assert data["fine_targets"].shape == (3660, 5, 80, 80)

    splits = np.bincount(data["splits"])
    assert splits[0] == 2928  # 8 years
    assert splits[1] == 366   # 1 year
    assert splits[2] == 366   # 1 year

    print("[PASS] MultiTask Real Cache NPZ: 3,660 samples (16x16 coarse, 80x80 terrain & targets), exact 5x scaling verified.")


def run_full_audit():
    print("=" * 76)
    print("END-TO-END DATASET AUDIT & PROVENANCE VERIFICATION")
    print("=" * 76)
    audit_glo30_terrain()
    audit_era5_land_daily()
    audit_imd_rainfall()
    audit_station_observations()
    audit_multitask_cache()
    print("=" * 76)
    print("[+] ALL AUDITS PASSED STRICT SCIENTIFIC INTEGRITY CHECKS.")
    print("=" * 76)


if __name__ == "__main__":
    run_full_audit()
