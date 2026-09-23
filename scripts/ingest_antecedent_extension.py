"""
scripts/ingest_antecedent_extension.py

Ingests authentic May 18-27 antecedent observations for the 9 operational seasons (2015-2023)
from ECMWF ERA5 reanalysis via Open-Meteo Historical Weather API using requests with strict timeouts.
Retrieves all 6 coarse meteorological variables:
  1. coarse_precip: sum of hourly precipitation [mm/day]
  2. coarse_tmax: max of 24h temperature_2m [deg C]
  3. coarse_tmin: min of 24h temperature_2m [deg C]
  4. coarse_rh: mean of August-Roche-Magnus relative humidity [%]
  5. coarse_wind_u: mean of wind_u_component_10m [m/s]
  6. coarse_wind_v: mean of wind_v_component_10m [m/s]

Outputs:
  data/raw/antecedent_extension_2015_2023.npz
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.openmeteo_era5_ingestion import august_roche_magnus_rh

OUTPUT_FILE = ROOT / "data" / "raw" / "antecedent_extension_2015_2023.npz"
CACHE_DIR = ROOT / "data" / "raw" / "antecedent_cache"

COARSE_LATS = np.linspace(14.875, 11.125, 16, dtype=np.float32)
COARSE_LONS = np.linspace(74.125, 77.875, 16, dtype=np.float32)

OPERATIONAL_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
DAYS_PER_SEASON = 10  # May 18 through May 27
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_batch_resilient(
    session: requests.Session,
    lats: List[float],
    lons: List[float],
    start_date: str,
    end_date: str,
    max_retries: int = 5,
) -> List[Dict]:
    """Fetches hourly ERA5 variables for a batch of points with requests timeout."""
    lat_str = ",".join(f"{lat:.4f}" for lat in lats)
    lon_str = ",".join(f"{lon:.4f}" for lon in lons)
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,dew_point_2m,precipitation,wind_u_component_10m,wind_v_component_10m",
        "models": "era5",
        "cell_selection": "nearest",
        "timezone": "GMT",
    }
    headers = {"User-Agent": "SIH-26074-Weather-Downscaler/2.0 (Resilient-Session)"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(OPENMETEO_ARCHIVE_URL, params=params, headers=headers, timeout=(6.0, 20.0))
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    data = [data]
                return data
            elif resp.status_code == 429:
                sleep_s = 20.0 * attempt
                print(f"      [!] 429 Rate limit. Sleeping {sleep_s:.0f}s (attempt {attempt}/{max_retries})...", flush=True)
                time.sleep(sleep_s)
            else:
                sleep_s = 3.0 * attempt
                print(f"      [!] HTTP {resp.status_code}. Sleeping {sleep_s:.0f}s (attempt {attempt}/{max_retries})...", flush=True)
                time.sleep(sleep_s)
        except Exception as e:
            sleep_s = 2.0 * attempt
            print(f"      [!] Request exception: {e}. Sleeping {sleep_s:.0f}s (attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch batch after {max_retries} attempts.")


def aggregate_antecedent_day(
    hourly_dict: Dict[str, List[float]],
    num_days: int = 10,
) -> Dict[str, np.ndarray]:
    """Aggregates 24-hour meteorological day from D 03:00 UTC to (D+1) 02:00 UTC."""
    times = hourly_dict["time"]
    t2m = np.asarray(hourly_dict["temperature_2m"], dtype=np.float32)
    td2m = np.asarray(hourly_dict["dew_point_2m"], dtype=np.float32)
    precip = np.asarray(hourly_dict["precipitation"], dtype=np.float32)
    wind_u = np.asarray(hourly_dict["wind_u_component_10m"], dtype=np.float32)
    wind_v = np.asarray(hourly_dict["wind_v_component_10m"], dtype=np.float32)

    precip_arr = np.zeros(num_days, dtype=np.float32)
    tmax_arr = np.zeros(num_days, dtype=np.float32)
    tmin_arr = np.zeros(num_days, dtype=np.float32)
    rh_arr = np.zeros(num_days, dtype=np.float32)
    u_arr = np.zeros(num_days, dtype=np.float32)
    v_arr = np.zeros(num_days, dtype=np.float32)

    for d in range(num_days):
        start_idx = d * 24 + 3
        end_idx = start_idx + 24
        t_slice = t2m[start_idx:end_idx]
        td_slice = td2m[start_idx:end_idx]
        p_slice = precip[start_idx:end_idx]
        u_slice = wind_u[start_idx:end_idx]
        v_slice = wind_v[start_idx:end_idx]

        tmax_arr[d] = float(np.max(t_slice))
        tmin_arr[d] = float(np.min(t_slice))
        rh_hourly = august_roche_magnus_rh(t_slice, td_slice)
        rh_arr[d] = float(np.clip(np.mean(rh_hourly), 0.0, 100.0))
        precip_arr[d] = float(np.clip(np.sum(p_slice), 0.0, 500.0))
        u_arr[d] = float(np.mean(u_slice))
        v_arr[d] = float(np.mean(v_slice))

    return {
        "precip": precip_arr,
        "tmax": tmax_arr,
        "tmin": tmin_arr,
        "rh": rh_arr,
        "wind_u": u_arr,
        "wind_v": v_arr,
    }


def ingest_extension(years: List[int] = OPERATIONAL_YEARS) -> Path:
    """Ingests May 18-27 for the specified operational years."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_dates = []
    for yr in years:
        dr = pd.date_range(f"{yr}-05-18", f"{yr}-05-27", freq="D")
        all_dates.extend([d.strftime("%Y-%m-%d") for d in dr])

    total_days = len(all_dates)
    print(f"[*] Ingesting antecedent extension for {len(years)} seasons ({total_days} daily dates)...", flush=True)

    lat_mesh, lon_mesh = np.meshgrid(COARSE_LATS, COARSE_LONS, indexing="ij")
    flat_lats = lat_mesh.ravel().tolist()
    flat_lons = lon_mesh.ravel().tolist()
    coarse_batch_size = 32
    n_batches = int(np.ceil(len(flat_lats) / coarse_batch_size))

    coarse_precip = np.zeros((total_days, 16, 16), dtype=np.float32)
    coarse_tmax = np.zeros((total_days, 16, 16), dtype=np.float32)
    coarse_tmin = np.zeros((total_days, 16, 16), dtype=np.float32)
    coarse_rh = np.zeros((total_days, 16, 16), dtype=np.float32)
    coarse_wind_u = np.zeros((total_days, 16, 16), dtype=np.float32)
    coarse_wind_v = np.zeros((total_days, 16, 16), dtype=np.float32)

    session = requests.Session()

    for y_idx, yr in enumerate(years):
        s_date = f"{yr}-05-18"
        e_date = f"{yr}-05-28"  # Next day 02:00 UTC coverage
        d_start = y_idx * DAYS_PER_SEASON
        d_end = d_start + DAYS_PER_SEASON
        season_cache = CACHE_DIR / f"antecedent_{yr}.npz"

        if season_cache.exists():
            print(f"  [{y_idx + 1}/{len(years)}] Loading cached May 18-27 for season {yr}...", flush=True)
            c = np.load(season_cache)
            coarse_precip[d_start:d_end] = c["precip"]
            coarse_tmax[d_start:d_end] = c["tmax"]
            coarse_tmin[d_start:d_end] = c["tmin"]
            coarse_rh[d_start:d_end] = c["rh"]
            coarse_wind_u[d_start:d_end] = c["wind_u"]
            coarse_wind_v[d_start:d_end] = c["wind_v"]
            continue

        print(f"  [{y_idx + 1}/{len(years)}] Fetching May 18-27 for season {yr} ({n_batches} batches of {coarse_batch_size} pts)...", flush=True)

        yr_precip = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)
        yr_tmax = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)
        yr_tmin = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)
        yr_rh = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)
        yr_u = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)
        yr_v = np.zeros((DAYS_PER_SEASON, 16, 16), dtype=np.float32)

        for b in range(n_batches):
            b_start = b * coarse_batch_size
            b_end = min((b + 1) * coarse_batch_size, len(flat_lats))
            records = fetch_batch_resilient(
                session,
                flat_lats[b_start:b_end],
                flat_lons[b_start:b_end],
                s_date,
                e_date,
            )
            for i, pt in enumerate(records):
                pt_idx = b_start + i
                r = pt_idx // 16
                c = pt_idx % 16
                daily = aggregate_antecedent_day(pt["hourly"], num_days=DAYS_PER_SEASON)
                yr_precip[:, r, c] = daily["precip"]
                yr_tmax[:, r, c] = daily["tmax"]
                yr_tmin[:, r, c] = daily["tmin"]
                yr_rh[:, r, c] = daily["rh"]
                yr_u[:, r, c] = daily["wind_u"]
                yr_v[:, r, c] = daily["wind_v"]

            print(f"    - Season {yr} batch {b + 1}/{n_batches} completed", flush=True)
            time.sleep(0.4)

        np.savez_compressed(
            season_cache,
            precip=yr_precip,
            tmax=yr_tmax,
            tmin=yr_tmin,
            rh=yr_rh,
            wind_u=yr_u,
            wind_v=yr_v,
        )

        coarse_precip[d_start:d_end] = yr_precip
        coarse_tmax[d_start:d_end] = yr_tmax
        coarse_tmin[d_start:d_end] = yr_tmin
        coarse_rh[d_start:d_end] = yr_rh
        coarse_wind_u[d_start:d_end] = yr_u
        coarse_wind_v[d_start:d_end] = yr_v

        print(
            f"  [+] Season {yr} done: Precip mean={yr_precip.mean():.2f} mm, "
            f"Tmax mean={yr_tmax.mean():.2f} C, RH mean={yr_rh.mean():.1f}%, "
            f"Wind U mean={yr_u.mean():.2f} m/s",
            flush=True,
        )

    # Verification: zero NaNs across all channels
    for name, arr in [
        ("precip", coarse_precip),
        ("tmax", coarse_tmax),
        ("tmin", coarse_tmin),
        ("rh", coarse_rh),
        ("wind_u", coarse_wind_u),
        ("wind_v", coarse_wind_v),
    ]:
        nan_count = int(np.isnan(arr).sum())
        if nan_count > 0:
            raise ValueError(f"Found {nan_count} NaNs in {name} array!")

    np.savez_compressed(
        OUTPUT_FILE,
        dates=np.array(all_dates),
        coarse_precip=coarse_precip,
        coarse_tmax=coarse_tmax,
        coarse_tmin=coarse_tmin,
        coarse_rh=coarse_rh,
        coarse_wind_u=coarse_wind_u,
        coarse_wind_v=coarse_wind_v,
    )
    print(f"[+] Successfully serialized authentic antecedent extension to: {OUTPUT_FILE}", flush=True)
    return OUTPUT_FILE


if __name__ == "__main__":
    ingest_extension()
