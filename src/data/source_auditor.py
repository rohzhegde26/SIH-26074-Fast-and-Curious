"""
src/data/source_auditor.py

Automated Data Source Auditing & Integrity Engine for Sprint 1.
Provides:
  - Strict anti-leakage temporal boundary verification for 00Z forecast cycles.
  - Mathematical derivation and unit conversion for GFS forecast fields.
  - Validation of forecast archive coverage boundaries (2015-2023 vs 2014).
  - Live probe methods for CHIRPS, Open-Meteo, and S3 GFS archives.
"""

from datetime import datetime, date, time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import requests
import yaml

logger = logging.getLogger(__name__)


class DataLeakageError(ValueError):
    """Raised when data from after the forecast initialization boundary is detected in history inputs."""
    pass


def verify_temporal_leakage_boundary(
    forecast_init_time: datetime,
    history_end_time: datetime,
) -> bool:
    """
    Enforces strict anti-leakage temporal boundary.
    For a forecast initialized on day D at 00:00 UTC, history context must terminate
    strictly at or before D 00:00 UTC (i.e. completed calendar day D-1).

    If history_end_time > forecast_init_time, raises DataLeakageError.
    """
    if history_end_time > forecast_init_time:
        raise DataLeakageError(
            f"Temporal data leakage detected: history ends at {history_end_time.isoformat()}, "
            f"which is after forecast initialization at {forecast_init_time.isoformat()}! "
            f"Day D completed observations are not available at 00Z Day D."
        )
    return True


def derive_gfs_daily_precipitation(
    apcp_values: np.ndarray,
    is_bucket: bool = True,
) -> float:
    """
    Derives 24-hour total precipitation from GFS APCP forecasts.
    
    Args:
        apcp_values: Array of APCP forecast values (e.g. 4 6-hour values for a 24-hour day).
        is_bucket: If True, values represent 6-hour interval buckets (sum them).
                   If False, values are cumulative from t0 (de-accumulate by differencing).
                   
    Returns:
        float daily accumulated precipitation in mm (kg/m^2).
    """
    if is_bucket:
        daily_p = float(np.sum(apcp_values))
    else:
        # De-accumulate cumulative steps
        diffs = np.diff(np.insert(apcp_values, 0, 0.0))
        daily_p = float(np.sum(np.clip(diffs, a_min=0.0, a_max=None)))
        
    return max(0.0, daily_p)


def derive_gfs_daily_temperatures(
    tmp_sequence_kelvin: np.ndarray,
) -> Tuple[float, float]:
    """
    Derives daily Tmax and Tmin in Celsius from a 3-hourly forecast temperature sequence in Kelvin.
    
    Args:
        tmp_sequence_kelvin: Array of forecast 2m temperatures across the 24-hour valid day.
        
    Returns:
        (tmax_degc, tmin_degc): Daily maximum and minimum temperatures in Celsius.
    """
    if len(tmp_sequence_kelvin) == 0:
        raise ValueError("Temperature sequence cannot be empty")
        
    celsius_vals = tmp_sequence_kelvin - 273.15
    tmax = float(np.max(celsius_vals))
    tmin = float(np.min(celsius_vals))
    
    if tmax < tmin:
        raise ValueError(f"Physical invariant violated: Tmax ({tmax}) < Tmin ({tmin})")
        
    return tmax, tmin


def derive_gfs_daily_wind(
    u_vals: np.ndarray,
    v_vals: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Derives daily mean U, V, and scalar wind speed from forecast wind component sequences.
    
    Returns:
        (mean_u, mean_v, scalar_speed) in m/s.
    """
    mean_u = float(np.mean(u_vals))
    mean_v = float(np.mean(v_vals))
    scalar_speeds = np.sqrt(u_vals**2 + v_vals**2)
    mean_speed = float(np.mean(scalar_speeds))
    return mean_u, mean_v, mean_speed


def validate_forecast_archive_year(year: int, mode: str = "forecast_conditioned") -> bool:
    """
    Enforces the archive boundary between 2014 and 2015-2023.
    
    - 'forecast_conditioned': Requires 2015-2023 (matching NOAA GFS on AWS Open Data).
    - 'history_only' / 'target_only': Supports full 2014-2023 period.
    """
    if mode == "forecast_conditioned":
        if year < 2015 or year > 2023:
            raise ValueError(
                f"NOAA GFS AWS archive is only available for 2015-2023. "
                f"Year {year} cannot be used for forecast-conditioned experiments."
            )
        return True
    elif mode in ("history_only", "target_only"):
        if year < 2014 or year > 2023:
            raise ValueError(f"Year {year} outside valid historical period 2014-2023.")
        return True
    else:
        raise ValueError(f"Unknown mode '{mode}'. Must be 'forecast_conditioned', 'history_only', or 'target_only'.")


# ---------------------------------------------------------------------------
# Live Source Auditing Probes
# ---------------------------------------------------------------------------

def audit_chirps_http(year: int = 2023, month: int = 7, day: int = 15) -> Dict[str, Any]:
    """Probes UCSB Climate Hazards Center CHIRPS v2.0 p05 daily COG endpoint."""
    filename = f"chirps-v2.0.{year}.{month:02d}.{day:02d}.cog"
    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/{year}/{filename}"
    t0 = datetime.now()
    try:
        resp = requests.head(url, timeout=10)
        elapsed_sec = (datetime.now() - t0).total_seconds()
        return {
            "source": "CHIRPS_v2_p05",
            "url": url,
            "status_code": resp.status_code,
            "success": resp.status_code == 200,
            "content_length": int(resp.headers.get("content-length", 0)),
            "content_type": resp.headers.get("content-type", ""),
            "latency_sec": elapsed_sec,
            "provenance_class": "satellite_gauge_product",
        }
    except Exception as e:
        return {
            "source": "CHIRPS_v2_p05",
            "url": url,
            "status_code": None,
            "success": False,
            "error": str(e),
            "provenance_class": "satellite_gauge_product",
        }


def audit_openmeteo_era5(
    lat: float = 13.0,
    lon: float = 76.0,
    start_date: str = "2023-07-01",
    end_date: str = "2023-07-03",
) -> Dict[str, Any]:
    """Probes Open-Meteo Historical Archive API for ERA5-Land hourly fields."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
        f"&models=era5_land&elevation=nan&timezone=GMT"
    )
    t0 = datetime.now()
    try:
        resp = requests.get(url, timeout=15)
        elapsed_sec = (datetime.now() - t0).total_seconds()
        headers = dict(resp.headers)
        rate_limit_headers = {k: v for k, v in headers.items() if "ratelimit" in k.lower() or "retry" in k.lower()}
        data = resp.json() if resp.status_code == 200 else {}
        hourly = data.get("hourly", {})
        has_vars = all(k in hourly for k in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"])
        return {
            "source": "Open-Meteo_ERA5-Land",
            "status_code": resp.status_code,
            "success": resp.status_code == 200 and has_vars,
            "hours_returned": len(hourly.get("time", [])),
            "latency_sec": elapsed_sec,
            "rate_limit_headers": rate_limit_headers,
            "provenance_class": "reanalysis",
        }
    except Exception as e:
        return {
            "source": "Open-Meteo_ERA5-Land",
            "status_code": None,
            "success": False,
            "error": str(e),
            "provenance_class": "reanalysis",
        }
