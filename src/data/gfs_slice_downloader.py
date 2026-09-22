"""
src/data/gfs_slice_downloader.py

Regional GFS Forecast Slicer & Multi-Step Accumulation Index Parser.
Features:
  1. GFSMessageKey: Disambiguates repeated variables/levels across forecast steps and accumulation buckets.
  2. parse_gfs_idx_multi_step: Computes exact byte ranges for every individual message in the GFS index.
  3. Regional Slicing: Uses HTTP Range requests on AWS Open Data to fetch only target forecast buckets.
"""

from dataclasses import dataclass
from datetime import date, datetime
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GFS_CACHE_DIR = ROOT / "data" / "raw" / "forecast" / "gfs"
AWS_GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


@dataclass(frozen=True)
class GFSMessageKey:
    """
    Structured key uniquely identifying a GFS GRIB2 message within a forecast run.
    Prevents collision between repeated variables across different accumulation steps.
    """
    var_name: str         # e.g., "APCP", "TMP", "RH", "UGRD", "VGRD"
    level: str            # e.g., "surface", "2 m above ground", "10 m above ground"
    forecast_step: str    # e.g., "0-6 hour acc fcst", "18-24 hour acc fcst", "24 hour fcst"
    byte_start: int
    byte_end: Optional[int]

    @property
    def lookup_key(self) -> str:
        return f"{self.var_name}:{self.level}:{self.forecast_step}"


def parse_gfs_idx_multi_step(idx_text: str) -> Dict[str, GFSMessageKey]:
    """
    Parses a NOAA GFS index file (.idx) into a dictionary of GFSMessageKey objects.
    Keyed by `f"{var_name}:{level}:{forecast_step}"`.
    Calculates exact byte start and end boundaries for each message.
    """
    lines = [line.strip() for line in idx_text.strip().splitlines() if line.strip()]
    raw_entries = []

    for line in lines:
        parts = line.split(":")
        if len(parts) >= 6:
            entry_id = int(parts[0])
            byte_start = int(parts[1])
            var_name = parts[3].strip()
            level = parts[4].strip()
            forecast_step = parts[5].strip()
            raw_entries.append({
                "id": entry_id,
                "byte_start": byte_start,
                "var_name": var_name,
                "level": level,
                "forecast_step": forecast_step,
            })

    # Calculate byte spans
    message_map: Dict[str, GFSMessageKey] = {}
    total_entries = len(raw_entries)

    for i, e in enumerate(raw_entries):
        start = e["byte_start"]
        end = raw_entries[i + 1]["byte_start"] - 1 if i + 1 < total_entries else None
        key_obj = GFSMessageKey(
            var_name=e["var_name"],
            level=e["level"],
            forecast_step=e["forecast_step"],
            byte_start=start,
            byte_end=end,
        )
        message_map[key_obj.lookup_key] = key_obj

    return message_map


def format_gfs_s3_key_for_run(
    init_date: date,
    cycle_hour: int = 0,
    forecast_hour: int = 24,
) -> str:
    """
    Constructs the canonical S3 key for NOAA GFS 0.25° on AWS Open Data.
    """
    date_str = init_date.strftime("%Y%m%d")
    cycle_str = f"{cycle_hour:02d}"
    lead_str = f"{forecast_hour:03d}"

    if init_date >= date(2021, 3, 22):
        return f"gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{lead_str}"
    else:
        return f"gfs.{date_str}/{cycle_str}/gfs.t{cycle_str}z.pgrb2.0p25.f{lead_str}"


def fetch_gfs_index(
    init_date: date,
    cycle_hour: int = 0,
    forecast_hour: int = 24,
    timeout: int = 15,
) -> Dict[str, GFSMessageKey]:
    """
    Retrieves and parses the GFS .idx file for a specific initialization and lead time.
    """
    key = format_gfs_s3_key_for_run(init_date, cycle_hour, forecast_hour)
    idx_url = f"{AWS_GFS_BASE_URL}/{key}.idx"

    resp = requests.get(idx_url, timeout=timeout)
    if resp.status_code != 200:
        raise FileNotFoundError(f"Failed to fetch GFS index from {idx_url} (HTTP {resp.status_code})")

    return parse_gfs_idx_multi_step(resp.text)


def download_gfs_variable_slice(
    init_date: date,
    msg_key: GFSMessageKey,
    cycle_hour: int = 0,
    forecast_hour: int = 24,
    timeout: int = 20,
) -> bytes:
    """
    Downloads a single variable message using HTTP Range headers.
    """
    key = format_gfs_s3_key_for_run(init_date, cycle_hour, forecast_hour)
    grib_url = f"{AWS_GFS_BASE_URL}/{key}"

    range_header = f"bytes={msg_key.byte_start}-"
    if msg_key.byte_end is not None:
        range_header = f"bytes={msg_key.byte_start}-{msg_key.byte_end}"

    resp = requests.get(grib_url, headers={"Range": range_header}, timeout=timeout)
    if resp.status_code not in (200, 206):
        raise RuntimeError(
            f"Failed to fetch byte range {range_header} from {grib_url} (HTTP {resp.status_code})"
        )

    content = resp.content
    if content[:4] != b"GRIB":
        raise ValueError(f"Downloaded slice does not start with b'GRIB' magic header: {content[:4]}")

    return content


DEFAULT_COARSE_LATS = np.linspace(14.875, 11.125, 16, dtype=np.float32)
DEFAULT_COARSE_LONS = np.linspace(74.125, 77.875, 16, dtype=np.float32)


def sample_gfs_grib_to_grid(
    raw_bytes: bytes,
    target_lats: Optional[np.ndarray] = None,
    target_lons: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Decodes in-memory GRIB2 bytes using rasterio and bilinearly interpolates
    to the target coarse lat/lon grid (16x16).
    """
    import rasterio
    from scipy.interpolate import RegularGridInterpolator

    lats = target_lats if target_lats is not None else DEFAULT_COARSE_LATS
    lons = target_lons if target_lons is not None else DEFAULT_COARSE_LONS

    with rasterio.open(rasterio.MemoryFile(raw_bytes)) as src:
        data = src.read(1)
        # GFS global grid: 721 latitudes (90 down to -90), 1440 longitudes (0 to 359.75)
        gfs_lats = np.linspace(90.0, -90.0, 721)
        gfs_lons = np.linspace(0.0, 359.75, 1440)
        interp = RegularGridInterpolator(
            (gfs_lats[::-1], gfs_lons),
            data[::-1, :],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")
        sampled = interp((mesh_lat, mesh_lon)).astype(np.float32)
        return sampled


def extract_7day_gfs_forecast(
    init_date: date,
    cycle_hour: int = 0,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> np.ndarray:
    """
    Extracts 7-day forecast lead tensor [7, 6, 16, 16] for 00Z cycle on init_date.
    Channels: [precipitation, tmax, tmin, rh, wind_u, wind_v].
    Caches extracted array to disk in NPZ format.
    """
    cdir = cache_dir or DEFAULT_GFS_CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    date_str = init_date.strftime("%Y%m%d")
    cycle_str = f"{cycle_hour:02d}"
    cache_file = cdir / f"gfs_{date_str}_{cycle_str}z_16x16.npz"

    if cache_file.exists() and not force_refresh:
        loaded = np.load(cache_file)
        return loaded["forecast"].astype(np.float32)

    leads_hours = [24, 48, 72, 96, 120, 144, 168]
    lead_arrays = []
    prev_cum_p = 0.0

    for k, lead_h in enumerate(leads_hours):
        idx_map = fetch_gfs_index(init_date, cycle_hour=cycle_hour, forecast_hour=lead_h)

        # 1. Precipitation (APCP)
        apcp_keys = [key for key in idx_map if "APCP:surface" in key]
        if not apcp_keys:
            raise KeyError(f"No APCP surface key found in GFS index for lead {lead_h}h")
        # Prefer cumulative day or largest interval key
        day_cum_keys = [key for key in apcp_keys if f"0-{k+1} day" in key]
        selected_apcp_key = day_cum_keys[0] if day_cum_keys else apcp_keys[0]
        raw_apcp = download_gfs_variable_slice(init_date, idx_map[selected_apcp_key], cycle_hour, lead_h)
        sampled_p = sample_gfs_grib_to_grid(raw_apcp)

        if "day acc" in selected_apcp_key:
            daily_precip = np.maximum(0.0, sampled_p - prev_cum_p)
            prev_cum_p = sampled_p.copy()
        else:
            daily_precip = np.maximum(0.0, sampled_p)

        # 2. Tmax & 3. Tmin
        tmax_keys = [key for key in idx_map if "TMAX:2 m above ground" in key]
        tmin_keys = [key for key in idx_map if "TMIN:2 m above ground" in key]
        tmp_keys = [key for key in idx_map if "TMP:2 m above ground" in key]

        if tmax_keys and tmin_keys:
            raw_tmax = download_gfs_variable_slice(init_date, idx_map[tmax_keys[0]], cycle_hour, lead_h)
            raw_tmin = download_gfs_variable_slice(init_date, idx_map[tmin_keys[0]], cycle_hour, lead_h)
            sampled_tmax = sample_gfs_grib_to_grid(raw_tmax)
            sampled_tmin = sample_gfs_grib_to_grid(raw_tmin)
        elif tmp_keys:
            raw_tmp = download_gfs_variable_slice(init_date, idx_map[tmp_keys[0]], cycle_hour, lead_h)
            sampled_tmp = sample_gfs_grib_to_grid(raw_tmp)
            sampled_tmax = sampled_tmp + 4.0
            sampled_tmin = sampled_tmp - 4.0
        else:
            raise KeyError(f"No temperature keys found in GFS index for lead {lead_h}h")

        # Convert Kelvin to Celsius if necessary (GDAL returns Celsius if < 150)
        if np.nanmean(sampled_tmax) > 150.0:
            sampled_tmax = sampled_tmax - 273.15
        if np.nanmean(sampled_tmin) > 150.0:
            sampled_tmin = sampled_tmin - 273.15
        # Enforce physical invariant Tmax >= Tmin
        sampled_tmax = np.maximum(sampled_tmax, sampled_tmin + 0.1)

        # 4. Relative Humidity (RH)
        rh_keys = [key for key in idx_map if "RH:2 m above ground" in key]
        if not rh_keys:
            raise KeyError(f"No RH key found in GFS index for lead {lead_h}h")
        raw_rh = download_gfs_variable_slice(init_date, idx_map[rh_keys[0]], cycle_hour, lead_h)
        sampled_rh = np.clip(sample_gfs_grib_to_grid(raw_rh), 0.0, 100.0)

        # 5. U Wind & 6. V Wind
        u_keys = [key for key in idx_map if "UGRD:10 m above ground" in key]
        v_keys = [key for key in idx_map if "VGRD:10 m above ground" in key]
        if not u_keys or not v_keys:
            raise KeyError(f"No U/V wind keys found in GFS index for lead {lead_h}h")
        raw_u = download_gfs_variable_slice(init_date, idx_map[u_keys[0]], cycle_hour, lead_h)
        raw_v = download_gfs_variable_slice(init_date, idx_map[v_keys[0]], cycle_hour, lead_h)
        sampled_u = sample_gfs_grib_to_grid(raw_u)
        sampled_v = sample_gfs_grib_to_grid(raw_v)

        # Stack into [6, 16, 16]
        lead_arr = np.stack(
            [daily_precip, sampled_tmax, sampled_tmin, sampled_rh, sampled_u, sampled_v],
            axis=0,
        ).astype(np.float32)
        lead_arrays.append(lead_arr)

    # Stack 7 leads into [7, 6, 16, 16]
    forecast_tensor = np.stack(lead_arrays, axis=0).astype(np.float32)

    # Save to disk cache
    np.savez_compressed(
        cache_file,
        forecast=forecast_tensor,
        init_date=date_str,
        cycle_hour=cycle_hour,
        leads_hours=leads_hours,
        channels=["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"],
    )
    return forecast_tensor

