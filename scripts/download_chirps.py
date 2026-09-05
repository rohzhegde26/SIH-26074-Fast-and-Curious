"""
scripts/download_chirps.py

Ingest CHIRPS 0.05° Daily High-Resolution Rainfall with network resilience.

Resilience Capabilities:
    1. HTTP chunk streaming (8 KB chunks).
    2. Urllib3 Retry adapter: 5 retries, exponential backoff (1s), status forcelist [429, 500, 502, 503, 504].
    3. Range-header partial download resumption: checks .part file and issues 'Range: bytes={existing}-'.
    4. Integrity audit: compares Content-Length against received bytes.
    5. Atomic commit: writes to .part and renames to .nc only upon clean completion.
    6. Mock sample generator fallback for offline / unit test environments.
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.util import Retry
import xarray as xr


CHIRPS_RES = 0.05
DOMAIN = {
    "min_lon": 68.0,
    "max_lon": 97.0,
    "min_lat": 8.0,
    "max_lat": 37.0,
}

CHUNK_SIZE = 8192  # 8 KB streaming chunks


def get_resilient_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests Session equipped with urllib3 exponential backoff retries."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file_resilient(
    url: str,
    output_path: Path,
    chunk_size: int = CHUNK_SIZE,
    timeout: int = 30,
) -> bool:
    """
    Download a remote file with chunk streaming, Range-header resumption, and atomic renaming.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[SKIP] Target file already exists: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
        return True

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    session = get_resilient_session()

    # Determine remote content size via HEAD request
    try:
        head_resp = session.head(url, timeout=timeout, allow_redirects=True)
        head_resp.raise_for_status()
        total_size = int(head_resp.headers.get("content-length", 0))
    except Exception as e:
        print(f"[WARNING] HEAD request failed ({e}), falling back to direct GET stream.")
        total_size = 0

    # Resume from existing .part file if present
    headers = {}
    mode = "wb"
    existing_bytes = 0
    if part_path.exists():
        existing_bytes = part_path.stat().st_size
        if total_size > 0 and existing_bytes >= total_size:
            print(f"[RECOVERY] .part file is already complete. Atomically committing...")
            part_path.replace(output_path)
            return True
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"[RESUME] Resuming from byte {existing_bytes:,} / {total_size:,}...")

    try:
        with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
            # Check range support
            if resp.status_code == 416:  # Range Not Satisfiable
                print("[WARNING] Range not satisfiable, re-downloading from byte 0...")
                headers.pop("Range", None)
                mode = "wb"
                existing_bytes = 0
                resp = session.get(url, stream=True, timeout=timeout)

            resp.raise_for_status()

            # Handle non-206 partial content fallback
            if mode == "ab" and resp.status_code != 206:
                print("[WARNING] Server did not acknowledge 206 Partial Content, restarting download...")
                mode = "wb"
                existing_bytes = 0

            downloaded = existing_bytes
            with open(part_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

        # Integrity verification
        if total_size > 0 and downloaded < total_size:
            raise IOError(f"Incomplete download: received {downloaded} bytes, expected {total_size} bytes.")

        # Atomic commit
        part_path.replace(output_path)
        print(f"[SUCCESS] Download completed and committed: {output_path} ({downloaded / (1024*1024):.2f} MB)")
        return True

    except Exception as e:
        print(f"[ERROR] Download interrupted for {url}: {e}")
        print(f"        Partial download preserved at: {part_path}")
        return False


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
    parser = argparse.ArgumentParser(description="Ingest or generate CHIRPS 0.05° rainfall data with network resilience")
    parser.add_argument("--url", type=str, default=None, help="Remote URL to download with resume support")
    parser.add_argument("--output", default="data/raw/chirps/chirps_sample.nc")
    parser.add_argument("--sample", action="store_true", help="Generate synthetic sample netCDF array")
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.url:
        download_file_resilient(args.url, out_path)
    else:
        generate_sample_chirps(args.output, args.days)
