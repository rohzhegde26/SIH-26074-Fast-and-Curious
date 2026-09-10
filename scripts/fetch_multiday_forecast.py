"""
scripts/fetch_multiday_forecast.py

Offline-ready fetcher for operational multi-day NWP coarse grids from Open-Meteo.
Queries the 16x16 Mandya regional bounding domain (11.0N-14.75N, 75.0E-78.75E)
at 0.25 degree spacing (256 locations) in batches of 50 to respect rate limits.

Outputs:
    data/raw/forecast/multiday_coarse_<YYYYMMDD>.json
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "raw" / "forecast"

# 16x16 Coarse Grid coordinates matching the trained UNet5x 0.25° domain
LATS = [round(11.0 + i * 0.25, 2) for i in range(16)]
LONS = [round(75.0 + j * 0.25, 2) for j in range(16)]
TOTAL_POINTS = len(LATS) * len(LONS)  # 256


def get_resilient_session(retries: int = 5, backoff_factor: float = 1.5) -> requests.Session:
    """Creates a requests session equipped with urllib3 exponential backoff."""
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


def fetch_multiday_forecast(
    output_dir: Path = OUTPUT_DIR,
    forecast_days: int = 7,
    batch_size: int = 50,
) -> Path:
    """
    Fetches 7-day operational weather forecasts from Open-Meteo across the 16x16 coarse grid.
    Batch size defaults to 50 locations per request with 1s sleep between chunks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session = get_resilient_session()

    # Create ordered list of (lat, lon) coordinates
    coords: List[Dict[str, float]] = []
    coord_indices: List[tuple[int, int]] = []
    for r, lat in enumerate(LATS):
        for c, lon in enumerate(LONS):
            coords.append({"lat": lat, "lon": lon})
            coord_indices.append((r, c))

    print(f"[*] Fetching operational multi-day NWP grids for {TOTAL_POINTS} grid points ({batch_size} per chunk)...")

    # Arrays for 7 days across 16x16: [7, 16, 16]
    precip_grids = np.zeros((forecast_days, 16, 16), dtype=np.float32)
    tmax_grids = np.zeros((forecast_days, 16, 16), dtype=np.float32)
    tmin_grids = np.zeros((forecast_days, 16, 16), dtype=np.float32)
    rh_grids = np.zeros((forecast_days, 16, 16), dtype=np.float32)
    wind_grids = np.zeros((forecast_days, 16, 16), dtype=np.float32)

    cycle_dates: List[str] = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    num_batches = (TOTAL_POINTS + batch_size - 1) // batch_size
    endpoint = "https://api.open-meteo.com/v1/forecast"

    for b_idx in range(num_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, TOTAL_POINTS)
        batch_coords = coords[start_i:end_i]
        batch_indices = coord_indices[start_i:end_i]

        batch_lats = [pt["lat"] for pt in batch_coords]
        batch_lons = [pt["lon"] for pt in batch_coords]

        params = {
            "latitude": batch_lats,
            "longitude": batch_lons,
            "daily": [
                "precipitation_sum",
                "temperature_2m_max",
                "temperature_2m_min",
                "relative_humidity_2m_mean",
                "wind_speed_10m_max",
            ],
            "timezone": "Asia/Kolkata",
            "forecast_days": forecast_days,
        }

        print(f"    [Chunk {b_idx + 1}/{num_batches}] Fetching {len(batch_coords)} points (index {start_i}..{end_i - 1})...")

        resp = session.get(endpoint, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Open-Meteo API returned error {resp.status_code}: {resp.text}")

        data = resp.json()
        items = data if isinstance(data, list) else [data]

        for item_idx, item in enumerate(items):
            r, c = batch_indices[item_idx]
            daily = item.get("daily", {})
            if not cycle_dates and "time" in daily:
                cycle_dates = daily["time"][:forecast_days]

            precip_vals = daily.get("precipitation_sum", [0.0] * forecast_days)
            tmax_vals = daily.get("temperature_2m_max", [31.5] * forecast_days)
            tmin_vals = daily.get("temperature_2m_min", [21.0] * forecast_days)
            rh_vals = daily.get("relative_humidity_2m_mean", [68.0] * forecast_days)
            wind_vals = daily.get("wind_speed_10m_max", [8.5] * forecast_days)

            for d in range(min(forecast_days, len(precip_vals))):
                precip_grids[d, r, c] = float(precip_vals[d] or 0.0)
                tmax_grids[d, r, c] = float(tmax_vals[d] or 31.5)
                tmin_grids[d, r, c] = float(tmin_vals[d] or 21.0)
                rh_grids[d, r, c] = float(rh_vals[d] or 68.0)
                wind_grids[d, r, c] = float(wind_vals[d] or 8.5)

        if b_idx < num_batches - 1:
            time.sleep(1.0)

    cycle_date_str = cycle_dates[0] if cycle_dates else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    clean_date_tag = cycle_date_str.replace("-", "")
    target_file = output_dir / f"multiday_coarse_{clean_date_tag}.json"

    payload: Dict[str, Any] = {
        "source": "Open-Meteo (ECMWF/GFS blend)",
        "cycle_date": cycle_date_str,
        "fetched_at_utc": now_utc,
        "lats": LATS,
        "lons": LONS,
        "lead_days": forecast_days,
        "dates": cycle_dates,
        "grids": {
            "precipitation_sum": precip_grids.tolist(),
            "temperature_2m_max": tmax_grids.tolist(),
            "temperature_2m_min": tmin_grids.tolist(),
            "relative_humidity_2m_mean": rh_grids.tolist(),
            "wind_speed_10m_max": wind_grids.tolist(),
        },
    }

    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[SUCCESS] Operational multi-day coarse forecast saved to: {target_file}")
    return target_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch operational multi-day coarse weather grids.")
    parser.add_argument("--days", type=int, default=7, help="Forecast lead days (default: 7)")
    parser.add_argument("--batch-size", type=int, default=50, help="Chunk batch size (default: 50)")
    args = parser.parse_args()

    fetch_multiday_forecast(forecast_days=args.days, batch_size=args.batch_size)
