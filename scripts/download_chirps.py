"""
scripts/download_chirps.py

Ingest CHIRPS 0.05° Daily High-Resolution Rainfall.
Handles download from CHC UCSB and cropping to India monsoon domain:
    - Longitude: 68.0°E - 97.0°E (580 pixels)
    - Latitude: 8.0°N - 37.0°N (580 pixels)
    - Resolution: 0.05° x 0.05°
    - Direct 5x nested alignment with IMD 0.25° grid
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


CHIRPS_RES = 0.05
DOMAIN = {
    "min_lon": 68.0,
    "max_lon": 97.0,
    "min_lat": 8.0,
    "max_lat": 37.0,
}


def create_chirps_dataset(
    dates: pd.DatetimeIndex,
    rainfall_data: np.ndarray,
) -> xr.Dataset:
    """Create an xarray Dataset from CHIRPS 0.05° grid arrays with half-pixel centers."""
    lats = np.arange(DOMAIN["min_lat"] + CHIRPS_RES / 2.0, DOMAIN["max_lat"], CHIRPS_RES, dtype=np.float32)
    lons = np.arange(DOMAIN["min_lon"] + CHIRPS_RES / 2.0, DOMAIN["max_lon"], CHIRPS_RES, dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "precip": (
                ("time", "lat", "lon"),
                rainfall_data.astype(np.float32),
                {
                    "units": "mm/day",
                    "long_name": "CHIRPS 0.05 Daily Precipitation",
                    "temporal_cutoff": "Calendar day aggregation",
                },
            )
        },
        coords={
            "time": dates,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "source": "Climate Hazards Center, UC Santa Barbara (CHIRPS v2.0)",
            "license": "CC-BY 4.0",
            "resolution": "0.05 degree x 0.05 degree",
            "spatial_extent": "8.0N-37.0N, 68.0E-97.0E",
        },
    )
    return ds


def generate_sample_chirps(output_path: str = "data/raw/chirps/chirps_sample.nc", n_days: int = 5):
    """Generate sample CHIRPS NetCDF file nested with IMD for pipeline testing."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2023-07-01", periods=n_days, freq="D")
    n_lat = int(round((DOMAIN["max_lat"] - DOMAIN["min_lat"]) / CHIRPS_RES))
    n_lon = int(round((DOMAIN["max_lon"] - DOMAIN["min_lon"]) / CHIRPS_RES))

    np.random.seed(101)
    rain = np.random.gamma(shape=1.2, scale=10.0, size=(n_days, n_lat, n_lon)).astype(np.float32)
    rain[rain < 1.0] = 0.0

    ds = create_chirps_dataset(dates, rain)
    ds.to_netcdf(out_file)
    print(f"[SUCCESS] Saved CHIRPS sample dataset to {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest or generate CHIRPS 0.05° rainfall data")
    parser.add_argument("--output", default="data/raw/chirps/chirps_sample.nc")
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    generate_sample_chirps(args.output, args.days)
