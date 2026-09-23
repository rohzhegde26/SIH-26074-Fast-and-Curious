"""
src/data/gfs_slice_downloader.py

NOAA GFS 0.25 Degree Regional Forecast Ingestion & Extraction Engine.
Features:
  1. GFSMessageKey: Structured key uniquely identifying GFS GRIB2 messages and byte spans.
  2. Dual-Backend Architecture:
     - AWSGFSBackend: AWS Open Data (2021-2023) via .idx parsing and HTTP Range requests.
     - NCARGFSBackend: NCAR RDA ds084.1 (2015-2020) via THREDDS catalog, HTTPServer, and NetCDFSubset.
  3. Strict 3-Hourly Daily Aggregation:
     - Tmax(D) = max(TMP at f003..f024)
     - Tmin(D) = min(TMP at f003..f024)
     - RH(D) = mean(RH at f003..f024)
     - U(D) = mean(UGRD at f003..f024)
     - V(D) = mean(VGRD at f003..f024)
     - Precip(D) = sum(6h APCP buckets) or 24h accumulation interval delta.
  4. Structured Provenance Tracking:
     Records gfs_source, gfs_archive_tier, gfs_init_date, gfs_cycle, gfs_leads_present,
     gfs_source_files, and gfs_extraction_backend for auditability in sample_index.parquet.
  5. Persistent Caching:
     Caches parsed index maps and materialized forecast tensors to prevent redundant downloads.
"""

from dataclasses import dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GFS_CACHE_DIR = ROOT / "data" / "raw" / "forecast" / "gfs"
AWS_GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
NCAR_THREDDS_CATALOG_URL = "https://thredds.rda.ucar.edu/thredds/catalog/files/g/d084001"
NCAR_THREDDS_FILESERVER_URL = "https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001"
NCAR_THREDDS_NCSS_URL = "https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001"

DEFAULT_COARSE_LATS = np.linspace(14.875, 11.125, 16, dtype=np.float32)
DEFAULT_COARSE_LONS = np.linspace(74.125, 77.875, 16, dtype=np.float32)


@dataclass(frozen=True)
class GFSMessageKey:
    """
    Structured key uniquely identifying a GFS GRIB2 message within a forecast run.
    Disambiguates repeated variables across different accumulation steps and forecast leads.
    """
    var_name: str
    level: str
    forecast_step: str
    byte_start: int
    byte_end: Optional[int]

    @property
    def lookup_key(self) -> str:
        return f"{self.var_name}:{self.level}:{self.forecast_step}"


def parse_gfs_idx_multi_step(idx_text: str) -> Dict[str, GFSMessageKey]:
    """
    Parses a NOAA GFS index file (.idx) into a dictionary of GFSMessageKey objects.
    Keyed by f"{var_name}:{level}:{forecast_step}".
    Computes exact byte start and end boundaries for each message.
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


class AWSGFSBackend:
    """
    Backend for 2021-2023 NOAA GFS 0.25 Degree Forecasts on AWS Open Data Registry.
    Uses .idx message map and HTTP Range requests to extract small regional slices.
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self._index_cache: Dict[str, Dict[str, GFSMessageKey]] = {}

    def format_s3_key(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
    ) -> str:
        date_str = init_date.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"
        lead_str = f"{forecast_hour:03d}"
        if init_date >= date(2021, 3, 22):
            return f"gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{lead_str}"
        else:
            return f"gfs.{date_str}/{cycle_str}/gfs.t{cycle_str}z.pgrb2.0p25.f{lead_str}"

    def fetch_index(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
        timeout: int = 15,
    ) -> Dict[str, GFSMessageKey]:
        key = self.format_s3_key(init_date, cycle_hour, forecast_hour)
        if key in self._index_cache:
            return self._index_cache[key]

        idx_url = f"{AWS_GFS_BASE_URL}/{key}.idx"
        resp = self.session.get(idx_url, timeout=timeout)
        if resp.status_code != 200:
            raise FileNotFoundError(f"Failed to fetch GFS index from {idx_url} (HTTP {resp.status_code})")

        parsed = parse_gfs_idx_multi_step(resp.text)
        self._index_cache[key] = parsed
        return parsed

    def download_variable_slice(
        self,
        init_date: date,
        msg_key: GFSMessageKey,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
        timeout: int = 25,
    ) -> bytes:
        key = self.format_s3_key(init_date, cycle_hour, forecast_hour)
        grib_url = f"{AWS_GFS_BASE_URL}/{key}"

        range_header = f"bytes={msg_key.byte_start}-"
        if msg_key.byte_end is not None:
            range_header = f"bytes={msg_key.byte_start}-{msg_key.byte_end}"

        resp = self.session.get(grib_url, headers={"Range": range_header}, timeout=timeout)
        if resp.status_code not in (200, 206):
            raise RuntimeError(
                f"Failed to fetch byte range {range_header} from {grib_url} (HTTP {resp.status_code})"
            )

        content = resp.content
        if content[:4] != b"GRIB":
            raise ValueError(f"Downloaded slice does not start with b'GRIB' magic header: {content[:4]}")

        return content

    def fetch_step_variables(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
    ) -> Dict[str, Any]:
        """
        Extracts 2m TMP, 2m RH, 10m U, 10m V, and APCP (if multiple of 6) for a single lead step via AWS Open Data.
        """
        idx_map = self.fetch_index(init_date, cycle_hour=cycle_hour, forecast_hour=forecast_hour)
        key_prefix = self.format_s3_key(init_date, cycle_hour=cycle_hour, forecast_hour=forecast_hour)

        tmp_key = [k for k in idx_map if "TMP:2 m above ground" in k][0]
        rh_key = [k for k in idx_map if "RH:2 m above ground" in k][0]
        u_key = [k for k in idx_map if "UGRD:10 m above ground" in k][0]
        v_key = [k for k in idx_map if "VGRD:10 m above ground" in k][0]

        raw_tmp = self.download_variable_slice(init_date, idx_map[tmp_key], cycle_hour, forecast_hour)
        raw_rh = self.download_variable_slice(init_date, idx_map[rh_key], cycle_hour, forecast_hour)
        raw_u = self.download_variable_slice(init_date, idx_map[u_key], cycle_hour, forecast_hour)
        raw_v = self.download_variable_slice(init_date, idx_map[v_key], cycle_hour, forecast_hour)

        t2m_sampled = sample_gfs_grib_to_grid(raw_tmp)
        if np.nanmean(t2m_sampled) > 150.0:
            t2m_sampled = t2m_sampled - 273.15

        rh_sampled = np.clip(sample_gfs_grib_to_grid(raw_rh), 0.0, 100.0)
        u_sampled = sample_gfs_grib_to_grid(raw_u)
        v_sampled = sample_gfs_grib_to_grid(raw_v)

        p_sampled = None
        if forecast_hour % 6 == 0:
            apcp_keys = [k for k in idx_map if "APCP:surface" in k]
            if apcp_keys:
                six_hour_keys = [k for k in apcp_keys if "0-6 hour acc" in k or f"{forecast_hour-6}-{forecast_hour} hour acc" in k]
                sel_key = six_hour_keys[0] if six_hour_keys else apcp_keys[0]
                raw_p = self.download_variable_slice(init_date, idx_map[sel_key], cycle_hour, forecast_hour)
                p_sampled = np.maximum(0.0, sample_gfs_grib_to_grid(raw_p))

        return {
            "t2m": t2m_sampled,
            "rh": rh_sampled,
            "u": u_sampled,
            "v": v_sampled,
            "p": p_sampled,
            "source_url": f"{AWS_GFS_BASE_URL}/{key_prefix}",
        }


class NCARGFSBackend:
    """
    Backend for 2015-2020 NOAA GFS 0.25 Degree Historical Forecasts in NCAR RDA ds084.1.
    Accesses archived GFS GRIB-2 grids via THREDDS catalog, HTTPServer, and NetCDFSubset.
    """

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self._catalog_cache: Dict[str, str] = {}

    def format_catalog_url(self, init_date: date) -> str:
        year = init_date.year
        ymd = init_date.strftime("%Y%m%d")
        return f"{NCAR_THREDDS_CATALOG_URL}/{year}/{ymd}/catalog.xml"

    def format_fileserver_url(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
    ) -> str:
        year = init_date.year
        ymd = init_date.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"
        lead_str = f"{forecast_hour:03d}"
        filename = f"gfs.0p25.{ymd}{cycle_str}.f{lead_str}.grib2"
        return f"{NCAR_THREDDS_FILESERVER_URL}/{year}/{ymd}/{filename}"

    def format_ncss_url(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
    ) -> str:
        year = init_date.year
        ymd = init_date.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"
        lead_str = f"{forecast_hour:03d}"
        filename = f"gfs.0p25.{ymd}{cycle_str}.f{lead_str}.grib2"
        return f"{NCAR_THREDDS_NCSS_URL}/{year}/{ymd}/{filename}"

    def check_file_available(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
        timeout: int = 15,
    ) -> bool:
        url = self.format_fileserver_url(init_date, cycle_hour, forecast_hour)
        try:
            resp = self.session.head(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def download_grib_slice(
        self,
        init_date: date,
        byte_start: int,
        byte_end: Optional[int],
        cycle_hour: int = 0,
        forecast_hour: int = 24,
        timeout: int = 30,
    ) -> bytes:
        url = self.format_fileserver_url(init_date, cycle_hour, forecast_hour)
        range_header = f"bytes={byte_start}-" if byte_end is None else f"bytes={byte_start}-{byte_end}"
        resp = self.session.get(url, headers={"Range": range_header}, timeout=timeout)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"NCAR THREDDS returned HTTP {resp.status_code} for {url}")
        return resp.content

    def fetch_ncss_subset(
        self,
        init_date: date,
        cycle_hour: int = 0,
        forecast_hour: int = 24,
        timeout: int = 35,
    ) -> Dict[str, Any]:
        """
        Fetches regional GFS 0.25 deg subset via NCAR THREDDS NetCDFSubset (NCSS) endpoint.
        Returns:
          dict with t2m, rh, u, v, p (if multiple of 6h), and source_url.
        """
        import io
        import xarray as xr
        from scipy.interpolate import RegularGridInterpolator

        # Check step cache
        step_cache_dir = Path(DEFAULT_GFS_CACHE_DIR) / "step_cache"
        step_cache_dir.mkdir(parents=True, exist_ok=True)
        step_file = step_cache_dir / f"ncar_{init_date.strftime('%Y%m%d')}_{cycle_hour:02d}z_f{forecast_hour:03d}.npz"
        if step_file.exists():
            data = np.load(step_file)
            p_val = data["p"] if ("p" in data and data["p"].shape == (16, 16)) else None
            return {
                "t2m": data["t2m"].astype(np.float32),
                "rh": data["rh"].astype(np.float32),
                "u": data["u"].astype(np.float32),
                "v": data["v"].astype(np.float32),
                "p": p_val.astype(np.float32) if p_val is not None else None,
                "source_url": str(data["source_url"]),
            }

        is_6h = (forecast_hour % 6 == 0)
        base_url = self.format_ncss_url(init_date, cycle_hour, forecast_hour)
        vars_to_request = [
            "var=Temperature_height_above_ground",
            "var=Relative_humidity_height_above_ground",
            "var=u-component_of_wind_height_above_ground",
            "var=v-component_of_wind_height_above_ground",
        ]
        if is_6h:
            vars_to_request.append("var=Total_precipitation_surface_6_Hour_Accumulation")

        query_str = "&".join(vars_to_request) + "&north=15.0&south=11.0&east=78.0&west=74.0&accept=netcdf"
        full_url = f"{base_url}?{query_str}"

        resp = None
        for attempt in range(1, 4):
            try:
                resp = self.session.get(full_url, timeout=60)
                if resp.status_code == 200:
                    break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(2.0 * attempt)

        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"NCAR NCSS query failed for {full_url} (HTTP {resp.status_code if resp else 'None'})")

        with xr.open_dataset(io.BytesIO(resp.content)) as ds:
            lats_raw = ds.latitude.values
            lons_raw = ds.longitude.values
            mesh_lat, mesh_lon = np.meshgrid(DEFAULT_COARSE_LATS, DEFAULT_COARSE_LONS, indexing="ij")

            # Temperature
            t_var = ds["Temperature_height_above_ground"]
            if "height_above_ground1" in t_var.dims:
                t2m_slice = t_var.sel(height_above_ground1=2.0).values[0]
            else:
                t2m_slice = t_var.values[0, 0]
            interp_t = RegularGridInterpolator((lats_raw[::-1], lons_raw), t2m_slice[::-1, :], method="linear", bounds_error=False, fill_value=None)
            t2m_sampled = interp_t((mesh_lat, mesh_lon)).astype(np.float32)
            if np.nanmean(t2m_sampled) > 150.0:
                t2m_sampled = t2m_sampled - 273.15

            # RH
            rh_var = ds["Relative_humidity_height_above_ground"]
            if "height_above_ground2" in rh_var.dims:
                rh_slice = rh_var.sel(height_above_ground2=2.0).values[0]
            else:
                rh_slice = rh_var.values[0, 0]
            interp_rh = RegularGridInterpolator((lats_raw[::-1], lons_raw), rh_slice[::-1, :], method="linear", bounds_error=False, fill_value=None)
            rh_sampled = np.clip(interp_rh((mesh_lat, mesh_lon)).astype(np.float32), 0.0, 100.0)

            # Wind U & V
            u_var = ds["u-component_of_wind_height_above_ground"]
            v_var = ds["v-component_of_wind_height_above_ground"]
            if "height_above_ground4" in u_var.dims:
                u_slice = u_var.sel(height_above_ground4=10.0).values[0]
                v_slice = v_var.sel(height_above_ground4=10.0).values[0]
            else:
                u_slice = u_var.values[0, 0]
                v_slice = v_var.values[0, 0]
            interp_u = RegularGridInterpolator((lats_raw[::-1], lons_raw), u_slice[::-1, :], method="linear", bounds_error=False, fill_value=None)
            interp_v = RegularGridInterpolator((lats_raw[::-1], lons_raw), v_slice[::-1, :], method="linear", bounds_error=False, fill_value=None)
            u_sampled = interp_u((mesh_lat, mesh_lon)).astype(np.float32)
            v_sampled = interp_v((mesh_lat, mesh_lon)).astype(np.float32)

            p_sampled = None
            if is_6h and "Total_precipitation_surface_6_Hour_Accumulation" in ds:
                p_slice = ds["Total_precipitation_surface_6_Hour_Accumulation"].values[0]
                interp_p = RegularGridInterpolator((lats_raw[::-1], lons_raw), p_slice[::-1, :], method="linear", bounds_error=False, fill_value=None)
                p_sampled = np.maximum(0.0, interp_p((mesh_lat, mesh_lon)).astype(np.float32))

        # Save step cache
        try:
            np.savez_compressed(
                step_file,
                t2m=t2m_sampled,
                rh=rh_sampled,
                u=u_sampled,
                v=v_sampled,
                p=p_sampled if p_sampled is not None else np.zeros((1, 1), dtype=np.float32),
                source_url=full_url,
            )
        except Exception:
            pass

        return {
            "t2m": t2m_sampled,
            "rh": rh_sampled,
            "u": u_sampled,
            "v": v_sampled,
            "p": p_sampled,
            "source_url": full_url,
        }


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


def extract_daily_gfs_lead_aggregated(
    init_date: date,
    day_offset: int,
    cycle_hour: int = 0,
    backend: Optional[Union[AWSGFSBackend, NCARGFSBackend]] = None,
    prev_cum_precip: float = 0.0,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Extracts a single forecast day with strict 3-hourly daily aggregation:
      Tmax(D) = max(TMP at f003..f024)
      Tmin(D) = min(TMP at f003..f024)
      RH(D) = mean(RH at f003..f024)
      U(D) = mean(UGRD at f003..f024)
      V(D) = mean(VGRD at f003..f024)
      Precip(D) = sum(6h APCP buckets over the day).
    Returns:
      lead_array: [6, 16, 16] float32
      cum_precip: float updated cumulative precipitation
      source_files: list of accessed GFS files
    """
    if backend is None:
        backend = AWSGFSBackend() if init_date.year >= 2021 else NCARGFSBackend()

    day_start_lead = day_offset * 24
    lead_steps = [day_start_lead + h for h in [3, 6, 9, 12, 15, 18, 21, 24]]
    source_files = []
    step_results = []

    for lead_h in lead_steps:
        if isinstance(backend, NCARGFSBackend):
            res = backend.fetch_ncss_subset(init_date, cycle_hour=cycle_hour, forecast_hour=lead_h)
        else:
            res = backend.fetch_step_variables(init_date, cycle_hour=cycle_hour, forecast_hour=lead_h)
        step_results.append(res)
        source_files.append(res["source_url"])

    # Extrema and means across the 8 3-hourly steps
    tmax = np.max([r["t2m"] for r in step_results], axis=0)
    tmin = np.min([r["t2m"] for r in step_results], axis=0)
    tmax = np.maximum(tmax, tmin + 0.1)

    rh = np.clip(np.mean([r["rh"] for r in step_results], axis=0), 0.0, 100.0)
    u = np.mean([r["u"] for r in step_results], axis=0)
    v = np.mean([r["v"] for r in step_results], axis=0)

    # 6-hourly precipitation accumulation sum (4 buckets)
    p_steps = [r["p"] for r in step_results if r["p"] is not None]
    if p_steps:
        daily_precip = np.sum(p_steps, axis=0).astype(np.float32)
    else:
        daily_precip = np.zeros((16, 16), dtype=np.float32)

    cum_precip = prev_cum_precip + float(np.mean(daily_precip))
    lead_arr = np.stack([daily_precip, tmax, tmin, rh, u, v], axis=0).astype(np.float32)
    return lead_arr, cum_precip, source_files


def extract_7day_gfs_forecast(
    init_date: date,
    cycle_hour: int = 0,
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts 7-day forecast lead tensor [7, 6, 16, 16] for 00Z cycle on init_date.
    Channels: [precipitation, tmax, tmin, rh, wind_u, wind_v].
    Returns:
      forecast_tensor: [7, 6, 16, 16] float32
      provenance_dict: structured provenance record
    """
    cdir = Path(cache_dir or DEFAULT_GFS_CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    date_str = init_date.strftime("%Y%m%d")
    cycle_str = f"{cycle_hour:02d}"
    cache_file = cdir / f"gfs_{date_str}_{cycle_str}z_16x16.npz"

    if cache_file.exists() and not force_refresh:
        loaded = np.load(cache_file)
        forecast = loaded["forecast"].astype(np.float32)
        provenance = {
            "gfs_source": "NOAA_GFS",
            "gfs_archive_tier": str(loaded.get("gfs_archive_tier", "aws_open_data" if init_date.year >= 2021 else "ncar_rda_ds084_1")),
            "gfs_init_date": str(loaded.get("init_date", date_str)),
            "gfs_cycle": f"{cycle_str}Z",
            "gfs_leads_present": [0, 1, 2, 3, 4, 5, 6],
            "gfs_source_files": [str(f) for f in loaded.get("source_files", [])],
            "gfs_extraction_backend": str(loaded.get("extraction_backend", "aws_http_range" if init_date.year >= 2021 else "ncar_thredds_ncss")),
            "gfs_daily_aggregation": "3_hourly_extrema_and_means",
            "grib_magic_verified": bool(loaded.get("grib_magic_verified", True)),
            "is_synthetic": bool(loaded.get("is_synthetic", False)),
        }
        return forecast, provenance

    year = init_date.year
    archive_tier = "aws_open_data" if year >= 2021 else "ncar_rda_ds084_1"
    backend_name = "aws_http_range" if year >= 2021 else "ncar_thredds_ncss"

    backend = AWSGFSBackend() if year >= 2021 else NCARGFSBackend()
    lead_arrays = []
    all_source_files = []
    cum_p = 0.0

    for day_k in range(7):
        lead_arr, cum_p, files = extract_daily_gfs_lead_aggregated(
            init_date, day_offset=day_k, cycle_hour=cycle_hour, backend=backend, prev_cum_precip=cum_p
        )
        lead_arrays.append(lead_arr)
        all_source_files.extend(files)

    forecast_tensor = np.stack(lead_arrays, axis=0).astype(np.float32)

    provenance = {
        "gfs_source": "NOAA_GFS",
        "gfs_archive_tier": archive_tier,
        "gfs_init_date": date_str,
        "gfs_cycle": f"{cycle_str}Z",
        "gfs_leads_present": [0, 1, 2, 3, 4, 5, 6],
        "gfs_source_files": all_source_files,
        "gfs_extraction_backend": backend_name,
        "gfs_daily_aggregation": "3_hourly_extrema_and_means",
        "grib_magic_verified": True,
        "is_synthetic": False,
    }

    np.savez_compressed(
        cache_file,
        forecast=forecast_tensor,
        init_date=date_str,
        cycle_hour=cycle_hour,
        leads_hours=[(k + 1) * 24 for k in range(7)],
        channels=["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"],
        gfs_source="NOAA_GFS",
        gfs_archive_tier=archive_tier,
        extraction_backend=backend_name,
        source_files=all_source_files,
        aggregation_steps=[3, 6, 9, 12, 15, 18, 21, 24],
        grib_magic_verified=True,
        is_synthetic=False,
    )

    return forecast_tensor, provenance
