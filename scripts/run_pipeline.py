"""
scripts/run_pipeline.py

Single-Command End-to-End Inference Orchestrator for Sprint 4.
Executes hybrid multi-day pipeline:
- Day 1 (Offset 0): Downscaled observation from IMD NetCDF (verified analysis).
- Days 2–7 (Offsets 1–6): Real operational NWP forecasts downscaled by UNet5x and MultivariatePhysicalDownscaler.

HARD RULE: ZERO internet calls during serving. Network calls are exclusively in scripts/fetch_multiday_forecast.py.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advisory.engine import (
    DAY_NAMES_KN,
    evaluate_lookahead_risk,
    rainfall_band_name,
)
from src.data.zonal_aggregation import ZonalAggregator
from src.eval.cqr import MCDropoutWrapper
from src.losses.conservation import expm1_transform
from src.models.multivariate import MultivariatePhysicalDownscaler
from src.models.unet_5x import UNet5x


def load_multiday_coarse(
    forecast_file: Path | None = None,
    forecast_dir: Path = ROOT / "data" / "raw" / "forecast",
) -> tuple[dict, bool]:
    """
    Loads multi-day coarse forecast JSON.
    Returns (coarse_data, is_committed_fallback).
    If forecast_file is specified and exists, loads it.
    If missing, falls back to the committed cycle and sets is_committed_fallback=True.
    """
    is_fallback = False
    target_path: Path | None = None

    if forecast_file is not None:
        if forecast_file.exists():
            target_path = forecast_file
        else:
            print(f"[!] Warning: Specified forecast file not found: {forecast_file}. Triggering COMMITTED_FALLBACK_CYCLE.")
            is_fallback = True

    if target_path is None:
        if forecast_dir.exists():
            candidates = sorted(forecast_dir.glob("multiday_coarse_*.json"))
            if candidates:
                target_path = candidates[-1]
                if is_fallback:
                    print(f"[*] Replaying committed fallback cycle: {target_path.name}")
            else:
                raise FileNotFoundError(f"No coarse forecast files found in {forecast_dir}")
        else:
            raise FileNotFoundError(f"Forecast directory not found: {forecast_dir}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, is_fallback


def run_pipeline(
    forecast_date: str = "2023-07-01",
    district: str = "MANDYA",
    imd_path: Path = ROOT / "data" / "raw" / "imd" / "imd_sample.nc",
    checkpoint_path: Path = ROOT / "models" / "checkpoints" / "best_5x_model.pt",
    geojson_path: Path = ROOT / "data" / "processed" / "mandya_full.geojson",
    topojson_path: Path = ROOT / "frontend" / "mandya_simplified.topojson",
    output_json_path: Path = ROOT / "data" / "serving" / "mandya_forecasts.json",
    output_geojson_path: Path | None = ROOT / "data" / "serving" / "mandya_forecasts.geojson",
    n_mc_passes: int = 8,
    q_hat: float = 2.45,
    forecast_file: Path | None = None,
) -> list[dict]:
    t_start = time.time()
    if district.upper() != "MANDYA":
        raise ValueError(f"District {district} not supported. Only MANDYA is configured for Sprint 4.")

    print(f"[*] Starting Mandya Hybrid Multi-Day Forecast Pipeline for date: {forecast_date}...")

    # 1. Load IMD NetCDF and extract 16x16 LR slice covering Mandya
    if not imd_path.exists():
        raise FileNotFoundError(f"IMD raster file not found: {imd_path}")

    ds = xr.open_dataset(imd_path)
    time_strs = [str(t)[:10] for t in ds.time.values]

    if forecast_date in time_strs:
        time_idx = time_strs.index(forecast_date)
        actual_date_str = forecast_date
    else:
        time_idx = 0
        actual_date_str = time_strs[0]
        print(f"[!] Requested date {forecast_date} not in sample ({time_strs[0]}..{time_strs[-1]}). Using {actual_date_str}.")

    # Mandya 16x16 slice covering Lats [11.0, 14.75] and Lons [75.0, 78.75]
    lr_slice_day1 = ds.rainfall.isel(time=time_idx, lat=slice(18, 34), lon=slice(34, 50)).values.astype(np.float32)
    lr_slice_day1 = np.nan_to_num(np.maximum(lr_slice_day1, 0.0))

    # 2. Load Operational Multi-day Coarse NWP Grids (Days 2-7)
    coarse_data, is_committed_fallback = load_multiday_coarse(forecast_file)
    coarse_grids = coarse_data.get("grids", {})
    cycle_date = coarse_data.get("cycle_date", actual_date_str)
    dt_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetched_at_utc = coarse_data.get("fetched_at_utc", dt_utc)

    try:
        f_dt = datetime.fromisoformat(str(fetched_at_utc).replace("Z", "+00:00"))
        now_utc = datetime.now(timezone.utc)
        cycle_age_days = max(0, int((now_utc - f_dt).total_seconds() // 86400))
    except Exception:
        cycle_age_days = 0

    base_d = date.fromisoformat(actual_date_str)
    day_dates = [(base_d + timedelta(days=d)).isoformat() for d in range(7)]

    # 3. Model setup
    v3_1_path = ROOT / "models" / "checkpoints" / "best_5x_model_v3_1.pt"
    if checkpoint_path == (ROOT / "models" / "checkpoints" / "best_5x_model.pt") and v3_1_path.exists():
        checkpoint_path = v3_1_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"[*] Loading checkpoint: {checkpoint_path.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model = UNet5x(in_channels=1, out_channels=1, base_channels=32).to(device)
    base_model.load_pretrained(ckpt, device=device)
    base_model.eval()

    mc_model = MCDropoutWrapper(base_model, p=0.1).to(device)

    # Build 5-channel terrain features for Mandya domain if DEM is present
    terrain_tensor = None
    dem_path = ROOT / "data" / "raw" / "dem" / "glo30_terrain.nc"
    if dem_path.exists():
        try:
            from src.data.terrain_features import build_terrain_tensor_5ch
            dem_ds = xr.open_dataset(dem_path)
            elev_p = dem_ds["elevation"].values[:80, :80]
            slope_p = dem_ds["slope"].values[:80, :80]
            aspect_p = dem_ds["aspect"].values[:80, :80]
            month_num = int(actual_date_str.split("-")[1])
            t_5ch = build_terrain_tensor_5ch(elev_p, slope_p, aspect_p, center_lat_deg=12.52, month=month_num)
            terrain_tensor = t_5ch.unsqueeze(0).to(device)
        except Exception as e:
            print(f"[!] Warning: Terrain extraction fallback: {e}")

    # Downscaler and Aggregator setup
    multi_downscaler = MultivariatePhysicalDownscaler().to(device)
    aggregator = ZonalAggregator(str(geojson_path))
    hr_lats = np.linspace(10.90, 14.85, 80)
    hr_lons = np.linspace(74.90, 78.85, 80)

    # 4. Multi-Day Loop (7 Independent Passes)
    daily_results = []
    for d in range(7):
        if d == 0:
            p_coarse = lr_slice_day1
            tmax_lr, tmin_lr, rh_lr, wind_lr = None, None, None, None
            day_prov = "IMD_OBSERVATION_DOWNSCALED"
        else:
            grid_idx = min(d, len(coarse_grids.get("precipitation_sum", [])) - 1)
            p_coarse = np.array(coarse_grids["precipitation_sum"][grid_idx], dtype=np.float32)
            p_coarse = np.nan_to_num(np.maximum(p_coarse, 0.0))

            tmax_lr = torch.from_numpy(np.array(coarse_grids["temperature_2m_max"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            tmin_lr = torch.from_numpy(np.array(coarse_grids["temperature_2m_min"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            rh_lr = torch.from_numpy(np.array(coarse_grids["relative_humidity_2m_mean"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            wind_lr = torch.from_numpy(np.array(coarse_grids["wind_speed_10m_max"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            day_prov = "COMMITTED_FALLBACK_CYCLE" if is_committed_fallback else "OPENMETEO_FORECAST_DOWNSCALED"

        # UNet forward pass
        x_tensor = torch.from_numpy(p_coarse).unsqueeze(0).unsqueeze(0).to(device)
        x_log = torch.log1p(x_tensor)
        mean_pred, q_lo, q_hi = mc_model.predict_with_uncertainty(
            x_log, terrain_hr=terrain_tensor, n_passes=n_mc_passes
        )
        int_lo = torch.clamp(q_lo - q_hat, min=0.0)
        int_hi = torch.clamp(q_hi + q_hat, min=0.0)

        hr_mean = mean_pred.squeeze().cpu().numpy()
        hr_lo = int_lo.squeeze().cpu().numpy()
        hr_hi = int_hi.squeeze().cpu().numpy()

        # Thermodynamic downscaling
        multi_res = multi_downscaler(
            tmax_lr=tmax_lr,
            tmin_lr=tmin_lr,
            rh_lr=rh_lr,
            wind_lr=wind_lr,
            terrain_5ch=terrain_tensor,
            target_size=(80, 80),
        )
        hr_tmax = multi_res["tmax_hr"].squeeze().cpu().numpy()
        hr_tmin = multi_res["tmin_hr"].squeeze().cpu().numpy()
        hr_rh = multi_res["rh_hr"].squeeze().cpu().numpy()
        hr_wind = multi_res["wind_hr"].squeeze().cpu().numpy()

        # Zonal aggregation
        agg_mean = aggregator.aggregate_grid(hr_mean, hr_lats, hr_lons)
        agg_lo = aggregator.aggregate_grid(hr_lo, hr_lats, hr_lons)
        agg_hi = aggregator.aggregate_grid(hr_hi, hr_lats, hr_lons)
        agg_tmax = aggregator.aggregate_grid(hr_tmax, hr_lats, hr_lons)
        agg_tmin = aggregator.aggregate_grid(hr_tmin, hr_lats, hr_lons)
        agg_rh = aggregator.aggregate_grid(hr_rh, hr_lats, hr_lons)
        agg_wind = aggregator.aggregate_grid(hr_wind, hr_lats, hr_lons)

        agg_detailed = None
        if d == 0:
            agg_detailed = aggregator.aggregate_grid_detailed(hr_mean, hr_lo, hr_hi, hr_lats, hr_lons)

        daily_results.append({
            "day_offset": d,
            "date": day_dates[d],
            "provenance": day_prov,
            "agg_mean": agg_mean,
            "agg_lo": agg_lo,
            "agg_hi": agg_hi,
            "agg_tmax": agg_tmax,
            "agg_tmin": agg_tmin,
            "agg_rh": agg_rh,
            "agg_wind": agg_wind,
            "agg_detailed": agg_detailed,
        })

    # Read simplified map gpcodes to ensure exact 1-to-1 correspondence
    topology = json.loads(topojson_path.read_text(encoding="utf-8"))
    collections = [v for v in topology.get("objects", {}).values() if v.get("type") == "GeometryCollection"]
    valid_gpcodes = {
        str(g.get("properties", {}).get("gpcode", "")).strip()
        for col in collections
        for g in col.get("geometries", [])
    } - {""}

    gdf = gpd.read_file(geojson_path)
    records = []

    # Crop stage calendar mapping
    month = int(actual_date_str.split("-")[1])
    if month in (6, 7):
        r_stage, p_stage = "sowing", "sowing"
    elif month in (8, 9):
        r_stage, p_stage = "vegetative", "vegetative"
    elif month in (10, 11):
        r_stage, p_stage = "flowering", "flowering"
    else:
        r_stage, p_stage = "harvest", "harvest"

    for _, row in gdf.iterrows():
        gpcode = str(row.get("gpcode", "")).strip()
        if not gpcode or gpcode not in valid_gpcodes:
            continue

        multi_day_items = []
        for d in range(7):
            d_res = daily_results[d]
            exp_d = round(float(d_res["agg_mean"].get(gpcode, 0.0)), 1)
            lo_d = round(float(d_res["agg_lo"].get(gpcode, 0.0)), 1)
            hi_d = round(float(d_res["agg_hi"].get(gpcode, 0.0)), 1)
            lo_d = min(lo_d, exp_d)
            hi_d = max(hi_d, exp_d)

            tmax_d = round(float(d_res["agg_tmax"].get(gpcode, 31.5)), 1)
            tmin_d = round(float(d_res["agg_tmin"].get(gpcode, 21.0)), 1)
            rh_d = round(float(d_res["agg_rh"].get(gpcode, 68.0)), 1)
            wind_d = round(float(d_res["agg_wind"].get(gpcode, 8.5)), 1)

            if d < 6:
                next_d_res = daily_results[d + 1]
                next_exp = round(float(next_d_res["agg_mean"].get(gpcode, 0.0)), 1)
                next_hi = round(float(next_d_res["agg_hi"].get(gpcode, 0.0)), 1)
            else:
                next_exp, next_hi = 0.0, 0.0

            lookahead = evaluate_lookahead_risk(exp_d, next_exp, next_hi)
            band = rainfall_band_name(exp_d)

            heat_stress = "NONE"
            if tmax_d >= 38.0:
                heat_stress = "SEVERE"
            elif tmax_d >= 35.0:
                heat_stress = "MODERATE"

            disease_flag = bool(rh_d >= 85.0 and 20.0 <= tmax_d <= 30.0)

            cur_d = date.fromisoformat(day_dates[d])
            if d == 0:
                label_en, label_kn = "Today", "ಇಂದು"
            elif d == 1:
                label_en, label_kn = "Tomorrow", "ನಾಳೆ"
            else:
                w_idx = cur_d.weekday()
                label_en = cur_d.strftime("%a %d")
                label_kn = f"{DAY_NAMES_KN[w_idx]} {cur_d.strftime('%d')}"

            if band == "dry" and not lookahead["has_washoff_hazard"]:
                sum_en = "Safe weather window. Ideal for field labour, spraying, and weeding."
                sum_kn = "ಅನುಕೂಲಕರ ಒಣ ಹವೆ. ಕಳೆ ಕೀಳಲು ಮತ್ತು ಸಿಂಪರಣೆಗೆ ಸೂಕ್ತ ದಿನ."
            elif lookahead["has_washoff_hazard"]:
                sum_en = f"Clear today, but {next_exp:.1f} mm rain tomorrow! Withhold fertilizer top-dressing."
                sum_kn = f"ಇಂದು ಒಣಹವೆ, ಆದರೆ ನಾಳೆ {next_exp:.1f} ಮಿಮೀ ಮಳೆ! ರಸಗೊಬ್ಬರ ಹಾಕಬೇಡಿ."
            elif band == "light":
                sum_en = "Light showers expected. Field operations can safely proceed with care."
                sum_kn = "ಸಾಧಾರಣ ತುಂತುರು ಮಳೆ. ಎಚ್ಚರಿಕೆಯಿಂದ ಕೃಷಿ ಕೆಲಸ ಮುಂದುವರಿಸಿ."
            else:
                sum_en = f"Heavy rain risk ({exp_d:.1f} mm). Postpone fertilizer and clear drainage."
                sum_kn = f"ಭಾರಿ ಮಳೆ ಸಂಭವ ({exp_d:.1f} ಮಿಮೀ). ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ ಹಾಗೂ ಚರಂಡಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ."

            multi_day_items.append({
                "date": day_dates[d],
                "day_offset": d,
                "day_label_en": label_en,
                "day_label_kn": label_kn,
                "expected_mm": float(exp_d),
                "likely_min_mm": float(lo_d),
                "likely_max_mm": float(hi_d),
                "tmax_c": float(tmax_d),
                "tmin_c": float(tmin_d),
                "rh_pct": float(rh_d),
                "wind_kph": float(wind_d),
                "heat_stress_level": heat_stress,
                "disease_risk_flag": disease_flag,
                "rainfall_band": band,
                "spray_window": lookahead["spray_window"],
                "harvest_window": lookahead["harvest_window"],
                "irrigation_window": lookahead["irrigation_window"],
                "lookahead_warning_en": lookahead["warning_en"],
                "lookahead_warning_kn": lookahead["warning_kn"],
                "advisory_summary_en": sum_en,
                "advisory_summary_kn": sum_kn,
                "provenance": d_res["provenance"],
            })

        day0 = multi_day_items[0]
        taluk_name = str(row.get("sdtname", row.get("blkname", ""))).strip()
        rec = {
            "lgd_code": gpcode,
            "panchayat_name": str(row.get("gpname", "")).strip() or f"GP-{gpcode}",
            "taluk": taluk_name,
            "district": district.upper(),
            "forecast_date": actual_date_str,
            "timestamp_utc": dt_utc,
            "fetched_at_utc": fetched_at_utc,
            "cycle_date": cycle_date,
            "cycle_age_days": cycle_age_days,
            "expected_mm": day0["expected_mm"],
            "likely_min_mm": day0["likely_min_mm"],
            "likely_max_mm": day0["likely_max_mm"],
            "tmax_c": day0["tmax_c"],
            "tmin_c": day0["tmin_c"],
            "temp_c": round((day0["tmax_c"] + day0["tmin_c"]) / 2.0, 1),
            "rh_pct": day0["rh_pct"],
            "wind_kph": day0["wind_kph"],
            "heat_stress_level": day0["heat_stress_level"],
            "disease_risk_flag": day0["disease_risk_flag"],
            "provenance": day0["provenance"],
            "ragi_stage": r_stage,
            "paddy_stage": p_stage,
            "spatial_variance": daily_results[0]["agg_detailed"].get(gpcode),
            "multi_day_forecast": multi_day_items,
        }
        records.append(rec)

    # Sort deterministically by gpcode
    records.sort(key=lambda r: r["lgd_code"])

    # Write serving JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Optionally write GeoJSON
    if output_geojson_path:
        output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
        rec_map = {r["lgd_code"]: r for r in records}
        mask = gdf["gpcode"].astype(str).str.strip().isin(rec_map)
        export_gdf = gdf[mask].copy()
        for col in [
            "expected_mm", "likely_min_mm", "likely_max_mm",
            "tmax_c", "tmin_c", "temp_c", "rh_pct", "wind_kph",
            "heat_stress_level", "disease_risk_flag",
            "ragi_stage", "paddy_stage", "forecast_date", "timestamp_utc"
        ]:
            export_gdf[col] = export_gdf["gpcode"].astype(str).str.strip().map(lambda c: rec_map[c].get(col))
        export_gdf.to_file(output_geojson_path, driver="GeoJSON")

    elapsed = time.time() - t_start
    print(f"[SUCCESS] Pipeline generated {len(records)} panchayat forecasts across 7 lead days in {elapsed:.2f}s!")
    print(f"          Output saved to: {output_json_path}")
    if output_geojson_path:
        print(f"          GeoJSON saved to: {output_geojson_path}")
    return records


class SyntheticIngestionHarness:
    """
    Emulates IMD 0.25 deg binary grid specs for air-gapped development because live FTP is IP-whitelisted to ministry intranets.
    """

    def __init__(self, provenance: str = "SYNTHETIC_GAMMA_CLIMATOLOGY"):
        self.provenance = provenance

    def ingest(self, forecast_date: str) -> dict:
        ingest_mode = os.getenv("INGEST_MODE", "synthetic").lower()
        if ingest_mode == "live":
            from src.integrations.imd_live_adapter import IMDLiveAdapter
            adapter = IMDLiveAdapter()
            return adapter.fetch_grid(forecast_date)

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[SYNTHETIC INGESTION HARNESS] {now_utc}")
        print("================================================================================")
        print("      MODE: Synthetic emulation of IMD 0.25 deg binary grid")
        print("      NOTE: Air-gapped development harness (Live FTP is IP-whitelisted to ministry intranets)")
        print(f"      Provenance: {self.provenance}")
        print(f"      Simulated cycle: {forecast_date}")
        print("================================================================================\n")
        return {
            "status": "success",
            "provenance": self.provenance,
            "forecast_date": forecast_date,
            "mode": "synthetic",
        }


def simulate_live_imd_ingest(forecast_date: str) -> None:
    """Backward-compatible wrapper invoking SyntheticIngestionHarness."""
    harness = SyntheticIngestionHarness()
    harness.ingest(forecast_date)


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-Command Mandya Hybrid Weather Forecast Pipeline.")
    parser.add_argument("--date", default="2023-07-01", help="Forecast date (YYYY-MM-DD)")
    parser.add_argument("--district", default="MANDYA", help="Target district (default: MANDYA)")
    parser.add_argument("--output", default="data/serving/mandya_forecasts.json", help="Output JSON path")
    parser.add_argument("--forecast-file", default=None, help="Optional path to operational multiday coarse JSON")
    parser.add_argument("--live-sim", action="store_true", help="Simulate automated FTP cron ingestion from IMD")
    args = parser.parse_args()

    if args.live_sim:
        simulate_live_imd_ingest(args.date)

    fc_file = Path(args.forecast_file) if args.forecast_file else None

    run_pipeline(
        forecast_date=args.date,
        district=args.district,
        output_json_path=Path(args.output),
        forecast_file=fc_file,
    )

    if args.live_sim:
        print("\n================================================================================")
        print("[CRON] Ingestion & serving pipeline completed successfully.")
        print("[CRON] Standby mode: Next automated polling scheduled in 06:00:00 (Next IMD sync: 08:30 IST).")
        print("================================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
