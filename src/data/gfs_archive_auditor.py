"""
src/data/gfs_archive_auditor.py

Auditing engine for NOAA Global Forecast System (GFS) 0.25° historical forecast archive
hosted on the AWS Open Data Registry (s3://noaa-gfs-bdp-pds).

Validates:
  - Public anonymous HTTPS access to historical 00Z forecast cycles.
  - S3 key structure across 2015-2023.
  - Byte-range indexing (.idx files) enabling selective variable extraction.
  - The 2015 archive start boundary (confirming absence of 2014 in this bucket).
"""

from datetime import datetime, date
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
import requests

logger = logging.getLogger(__name__)

AWS_GFS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def format_gfs_s3_key(
    init_date: date,
    cycle_hour: int = 0,
    forecast_hour: int = 24,
) -> str:
    """
    Constructs the canonical S3 key for NOAA GFS 0.25° pgrb2 files.
    Format since ~2021: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
    Legacy format (2015-2020): gfs.YYYYMMDDHH/gfs.tHHz.pgrb2.0p25.fFFF
    """
    ymd = init_date.strftime("%Y%m%d")
    hh = f"{cycle_hour:02d}"
    fff = f"{forecast_hour:03d}"
    
    # Modern directory structure
    if init_date.year >= 2021:
        key = f"gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{fff}"
    else:
        # Pre-2021 structure on AWS
        key = f"gfs.{ymd}/{hh}/gfs.t{hh}z.pgrb2.0p25.f{fff}"
        
    return key


def audit_gfs_forecast_file(
    init_date: date,
    cycle_hour: int = 0,
    forecast_hour: int = 24,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Probes NOAA GFS 0.25° GRIB2 and index files via public HTTPS endpoint.
    """
    key = format_gfs_s3_key(init_date, cycle_hour, forecast_hour)
    url_grib = f"{AWS_GFS_BASE_URL}/{key}"
    url_idx = f"{url_grib}.idx"
    
    t0 = datetime.now()
    try:
        # Check index file first (small, fast ~15 KB)
        resp_idx = requests.head(url_idx, timeout=timeout)
        elapsed_sec = (datetime.now() - t0).total_seconds()
        
        idx_available = (resp_idx.status_code == 200)
        idx_size = int(resp_idx.headers.get("content-length", 0)) if idx_available else 0
        
        # If modern key failed on older date, try legacy path
        if not idx_available and init_date.year < 2021:
            legacy_key = f"gfs.{init_date.strftime('%Y%m%d%H')}/gfs.t{cycle_hour:02d}z.pgrb2.0p25.f{forecast_hour:03d}"
            url_idx_legacy = f"{AWS_GFS_BASE_URL}/{legacy_key}.idx"
            resp_idx_legacy = requests.head(url_idx_legacy, timeout=timeout)
            if resp_idx_legacy.status_code == 200:
                return {
                    "init_date": init_date.isoformat(),
                    "cycle": f"{cycle_hour:02d}Z",
                    "lead_hour": forecast_hour,
                    "s3_key": legacy_key,
                    "available": True,
                    "index_size_bytes": int(resp_idx_legacy.headers.get("content-length", 0)),
                    "latency_sec": elapsed_sec,
                }

        return {
            "init_date": init_date.isoformat(),
            "cycle": f"{cycle_hour:02d}Z",
            "lead_hour": forecast_hour,
            "s3_key": key,
            "available": idx_available,
            "index_size_bytes": idx_size,
            "latency_sec": elapsed_sec,
        }
    except Exception as e:
        return {
            "init_date": init_date.isoformat(),
            "cycle": f"{cycle_hour:02d}Z",
            "lead_hour": forecast_hour,
            "s3_key": key,
            "available": False,
            "error": str(e),
        }


def parse_gfs_idx_byte_ranges(idx_text: str) -> Dict[str, Tuple[int, Optional[int]]]:
    """
    Parses a GFS index file (.idx) to extract byte ranges for required variables:
      - APCP (surface precipitation)
      - TMP / TMAX / TMIN (2m temperature)
      - RH / SPFH (2m relative / specific humidity)
      - UGRD / VGRD (10m wind components)
      
    Enables HTTP Range requests (Range: bytes=start-end) to download only required ~2 MB slices
    instead of the full ~500 MB global GRIB2 file.
    """
    lines = idx_text.strip().splitlines()
    entries = []
    for line in lines:
        parts = line.split(":")
        if len(parts) >= 6:
            entry_id = int(parts[0])
            byte_start = int(parts[1])
            var_name = parts[3]
            level = parts[4]
            entries.append({
                "id": entry_id,
                "byte_start": byte_start,
                "var_name": var_name,
                "level": level,
            })
            
    # Calculate byte offsets
    var_ranges = {}
    for i, e in enumerate(entries):
        start = e["byte_start"]
        end = entries[i + 1]["byte_start"] - 1 if i + 1 < len(entries) else None
        key = f"{e['var_name']}:{e['level']}"
        var_ranges[key] = (start, end)
        
    return var_ranges
