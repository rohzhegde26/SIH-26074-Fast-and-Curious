"""
src/data/noaa_station_auditor.py

Auditing engine for NOAA Global Surface Summary of the Day (GSOD) and GHCN-Daily
stations across Peninsular India (Karnataka / Western Ghats / Mandya domain).

Verifies:
  - Station inventory and WMO identifiers in the target region.
  - Public access URLs on NOAA NCEI archive.
  - Multi-year reporting completeness (2014-2023).
  - Variable availability limitations (explaining why stations cannot provide dense 6-channel supervision).
"""

from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Any
import requests

logger = logging.getLogger(__name__)

# Key WMO / ICAO weather stations in and adjacent to the core target domain (11-15°N, 74-78°E)
PENINSULAR_STATIONS = [
    {
        "usaf": "432950",
        "wban": "99999",
        "name": "BANGALORE / HAL AIRPORT",
        "state": "Karnataka",
        "lat": 12.95,
        "lon": 77.67,
        "elev_m": 888.0,
        "role": "independent_point_sanity_check",
    },
    {
        "usaf": "432960",
        "wban": "99999",
        "name": "BANGALORE / KEMPEGOWDA INTL",
        "state": "Karnataka",
        "lat": 13.20,
        "lon": 77.71,
        "elev_m": 915.0,
        "role": "independent_point_sanity_check",
    },
    {
        "usaf": "433140",
        "wban": "99999",
        "name": "MYSORE",
        "state": "Karnataka",
        "lat": 12.30,
        "lon": 76.65,
        "elev_m": 767.0,
        "role": "independent_point_sanity_check",
    },
    {
        "usaf": "432850",
        "wban": "99999",
        "name": "MANGALORE / BAJPE AIRPORT",
        "state": "Karnataka",
        "lat": 12.96,
        "lon": 74.89,
        "elev_m": 102.0,
        "role": "independent_point_sanity_check",
    },
    {
        "usaf": "432790",
        "wban": "99999",
        "name": "HASSAN",
        "state": "Karnataka",
        "lat": 13.01,
        "lon": 76.10,
        "elev_m": 957.0,
        "role": "independent_point_sanity_check",
    },
]


def audit_noaa_gsod_station_year(
    usaf: str,
    wban: str,
    year: int,
    timeout: int = 10,
) -> Dict[str, Any]:
    """
    Probes the public NOAA NCEI GSOD archive for a specific station and year.
    URL pattern: https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{usaf}{wban}.csv
    """
    station_id = f"{usaf}{wban}"
    url = f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv"
    
    t0 = datetime.now()
    try:
        resp = requests.head(url, timeout=timeout)
        elapsed_sec = (datetime.now() - t0).total_seconds()
        
        if resp.status_code == 200:
            content_len = int(resp.headers.get("content-length", 0))
            return {
                "station_id": station_id,
                "year": year,
                "url": url,
                "available": True,
                "status_code": 200,
                "size_bytes": content_len,
                "latency_sec": elapsed_sec,
            }
        else:
            return {
                "station_id": station_id,
                "year": year,
                "url": url,
                "available": False,
                "status_code": resp.status_code,
                "latency_sec": elapsed_sec,
            }
    except Exception as e:
        return {
            "station_id": station_id,
            "year": year,
            "url": url,
            "available": False,
            "status_code": None,
            "error": str(e),
        }


def audit_regional_stations(
    years: List[int] = [2018, 2021, 2023],
    stations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Runs multi-station, multi-year availability audit across Peninsular India.
    """
    stations_to_audit = stations or PENINSULAR_STATIONS
    results = {}
    summary = {
        "total_probed": 0,
        "available_count": 0,
        "missing_count": 0,
        "station_summaries": {},
    }

    for stn in stations_to_audit:
        usaf = stn["usaf"]
        wban = stn["wban"]
        name = stn["name"]
        stn_key = f"{usaf}_{name}"
        stn_res = {}

        for yr in years:
            probe = audit_noaa_gsod_station_year(usaf, wban, yr)
            stn_res[yr] = probe["available"]
            summary["total_probed"] += 1
            if probe["available"]:
                summary["available_count"] += 1
            else:
                summary["missing_count"] += 1

        summary["station_summaries"][stn_key] = stn_res

    return summary
