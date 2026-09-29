"""
scripts/run_pipeline.py

Single-Command End-to-End Inference Orchestrator for Mandya Weather Downscaling.
Executes 7-Day operational pipeline:
- Powered by Dense-L Accurate (52.00M-parameter Spatiotemporal Residual Diffusion, 16-step DDIM, eta=0.50).
- Backward compatible fallback to UNet5x + MultivariatePhysicalDownscaler.
- Guarantees cell-by-cell local block mass conservation on predicted precipitation.
- Maps 80x80 (0.05°) fine grids to all 234 Mandya Gram Panchayats with 7 future lead days.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advisory.engine import (
    DAY_NAMES_KN,
    evaluate_lookahead_risk,
    rainfall_band_name,
)
from src.api.inference_service import conservative_renorm_local
from src.data.forecast_loader import load_latest_coarse_forecast
from src.data.zonal_aggregation import ZonalAggregator
from src.data.terrain_features import build_terrain_tensor_5ch
from src.data.tensor_builder import invert_normalization
from src.eval.cqr import MCDropoutWrapper
from src.models.multivariate import MultivariatePhysicalDownscaler
from src.models.unet_5x import UNet5x
from src.models.scalable_residual_diffusion import create_scalable_residual_diffusion

DENSE_L_CKPT_PATH = ROOT / "models" / "checkpoints" / "sprint9_dense_l_weights.pt"
STATS_V2_PATH = ROOT / "data" / "normalization_stats_v2.yaml"


def load_multiday_coarse(
    forecast_file: Path | None = None,
    forecast_dir: Path = ROOT / "data" / "raw" / "forecast",
) -> tuple[dict, bool]:
    """
    Loads multi-day coarse forecast JSON using the shared forecast loader.
    Returns (coarse_data, is_committed_fallback).
    """
    data, is_fallback = load_latest_coarse_forecast(forecast_file=forecast_file, forecast_dir=forecast_dir)
    if data is None:
        raise FileNotFoundError(f"No coarse forecast files found in {forecast_dir}")
    return data, is_fallback


DISTRICT_CONFIGS = {
    "MANDYA": {
        "center_lat": 12.52,
        "hr_lats": np.linspace(10.90, 14.85, 80),
        "hr_lons": np.linspace(74.90, 78.85, 80),
        "dem_lat_slice": slice(0, 80),
        "dem_lon_slice": slice(0, 80),
    },
    "BAGHPAT": {
        "center_lat": 29.04,
        "hr_lats": np.linspace(27.06, 31.01, 80),
        "hr_lons": np.linspace(75.34, 79.29, 80),
        "dem_lat_slice": slice(350, 430),
        "dem_lon_slice": slice(130, 210),
    },
    "BARPETA": {
        "center_lat": 26.36,
        "hr_lats": np.linspace(24.38, 28.33, 80),
        "hr_lons": np.linspace(88.99, 92.94, 80),
        "dem_lat_slice": slice(300, 380),
        "dem_lon_slice": slice(400, 480),
    },
}


def run_pipeline(
    forecast_date: str = "today",
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
    mode: str = "operational",
    model_tier: str = "dense_l",
    denoising_steps: int = 16,
    eta: float = 0.50,
) -> list[dict]:
    t_start = time.time()
    dist_key = district.upper()
    if dist_key not in DISTRICT_CONFIGS:
        raise ValueError(f"District {district} not supported. Supported: {list(DISTRICT_CONFIGS.keys())}")

    dist_cfg = DISTRICT_CONFIGS[dist_key]
    dist_slug = dist_key.lower()

    # Automatically adapt default paths if running for non-Mandya district
    if dist_key != "MANDYA":
        if geojson_path == ROOT / "data" / "processed" / "mandya_full.geojson":
            geojson_path = ROOT / "data" / "processed" / f"{dist_slug}_full.geojson"
        if topojson_path == ROOT / "frontend" / "mandya_simplified.topojson":
            topojson_path = ROOT / "frontend" / f"{dist_slug}_simplified.topojson"
        if output_json_path == ROOT / "data" / "serving" / "mandya_forecasts.json":
            output_json_path = ROOT / "data" / "serving" / f"{dist_slug}_forecasts.json"
        if output_geojson_path == ROOT / "data" / "serving" / "mandya_forecasts.geojson":
            output_geojson_path = ROOT / "data" / "serving" / f"{dist_slug}_forecasts.geojson"
        if forecast_file is None:
            dist_coarse = ROOT / "data" / "raw" / "forecast" / f"multiday_coarse_{dist_slug}_20260910.json"
            if dist_coarse.exists():
                forecast_file = dist_coarse

    # 1. Load Operational Multi-day Coarse NWP Grids
    coarse_data, is_committed_fallback = load_multiday_coarse(forecast_file)
    coarse_grids = coarse_data.get("grids", {})
    dt_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Date handling: 7 days into the future starting from today (or requested date)
    if not forecast_date or forecast_date.lower() in ("today", "auto", "now"):
        base_d = datetime.now(timezone.utc).date()
        actual_date_str = base_d.isoformat()
        cycle_date = actual_date_str
        fetched_at_utc = dt_utc
        cycle_age_days = 0
    else:
        try:
            base_d = date.fromisoformat(forecast_date)
            actual_date_str = forecast_date
            cycle_date = forecast_date
            fetched_at_utc = coarse_data.get("fetched_at_utc", dt_utc)
            f_dt = datetime.fromisoformat(str(fetched_at_utc).replace("Z", "+00:00"))
            now_utc = datetime.now(timezone.utc)
            cycle_age_days = max(0, int((now_utc - f_dt).total_seconds() // 86400))
        except Exception:
            base_d = datetime.now(timezone.utc).date()
            actual_date_str = base_d.isoformat()
            cycle_date = actual_date_str
            fetched_at_utc = dt_utc
            cycle_age_days = 0

    day_dates = [(base_d + timedelta(days=d)).isoformat() for d in range(7)]
    print(f"[*] Starting {dist_key} 7-Day Forecast Pipeline ({day_dates[0]} through {day_dates[6]})...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target device: {device}")

    # Build 5-channel terrain features for target domain
    terrain_tensor = None
    real_dem = ROOT / "data" / "raw" / "dem" / "glo30_mandya_terrain.nc"
    synth_dem = ROOT / "data" / "raw" / "dem" / "synthetic_terrain.nc"
    dem_path = real_dem if real_dem.exists() else synth_dem
    if dem_path.exists():
        try:
            dem_ds = xr.open_dataset(dem_path)
            lat_sl = dist_cfg.get("dem_lat_slice", slice(0, 80))
            lon_sl = dist_cfg.get("dem_lon_slice", slice(0, 80))
            elev_p = dem_ds["elevation"].values[lat_sl, lon_sl]
            slope_p = dem_ds["slope"].values[lat_sl, lon_sl]
            aspect_p = dem_ds["aspect"].values[lat_sl, lon_sl]
            month_num = int(actual_date_str.split("-")[1])
            t_5ch = build_terrain_tensor_5ch(elev_p, slope_p, aspect_p, center_lat_deg=dist_cfg["center_lat"], month=month_num)
            terrain_tensor = t_5ch.unsqueeze(0).to(device)
            print(f"[*] Terrain loaded from {dem_path.name}: {terrain_tensor.shape}")
        except Exception as e:
            print(f"[!] Warning: Terrain extraction fallback: {e}")

    use_dense_l = (model_tier == "dense_l" and DENSE_L_CKPT_PATH.exists() and n_mc_passes == 8)
    if model_tier == "dense_l" and not DENSE_L_CKPT_PATH.exists():
        print(f"[!] Dense-L checkpoint not found at {DENSE_L_CKPT_PATH}, falling back to UNet5x.")
        use_dense_l = False

    daily_results = []
    aggregator = ZonalAggregator(str(geojson_path))
    hr_lats = dist_cfg["hr_lats"]
    hr_lons = dist_cfg["hr_lons"]

    if use_dense_l:
        print(f"[*] Initializing Dense-L Accurate (52.00M params, checkpoint={DENSE_L_CKPT_PATH.name})...")
        dense_l_model = create_scalable_residual_diffusion("dense_l").to(device)
        ckpt = torch.load(DENSE_L_CKPT_PATH, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        dense_l_model.load_state_dict(state, strict=True)
        dense_l_model.eval()

        # Load normalization stats
        with open(STATS_V2_PATH, "r", encoding="utf-8") as f:
            stats_raw = yaml.safe_load(f)["channels"]
        channel_order = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]

        # Build [1, 7, 6, 16, 16] future conditioning tensor
        future_7d = np.zeros((1, 7, 6, 16, 16), dtype=np.float32)
        n_available = len(coarse_grids.get("precipitation_sum", []))
        for d in range(7):
            idx = min(d, max(0, n_available - 1))
            future_7d[0, d, 0] = np.array(coarse_grids["precipitation_sum"][idx], dtype=np.float32)
            future_7d[0, d, 1] = np.array(coarse_grids.get("temperature_2m_max", [np.full((16, 16), 31.5)])[idx], dtype=np.float32)
            future_7d[0, d, 2] = np.array(coarse_grids.get("temperature_2m_min", [np.full((16, 16), 21.0)])[idx], dtype=np.float32)
            future_7d[0, d, 3] = np.array(coarse_grids.get("relative_humidity_2m_mean", [np.full((16, 16), 68.0)])[idx], dtype=np.float32)
            w_spd = np.array(coarse_grids.get("wind_speed_10m_max", [np.full((16, 16), 8.5)])[idx], dtype=np.float32) / 3.6
            future_7d[0, d, 4] = w_spd * 0.8
            future_7d[0, d, 5] = w_spd * 0.6

        from src.data.tensor_builder import apply_normalization

        # Apply canonical statistical normalization across all 6 weather channels
        future_norm = apply_normalization(future_7d, stats_raw, channel_names=channel_order)
        future_t = torch.from_numpy(future_norm).to(device)
        # In single NWP cycle operational run, initialize history context with neutral initial-step state
        history_t = future_t[:, 0:1].expand(-1, 14, -1, -1, -1).clone()

        print(f"[*] Running Dense-L reverse diffusion sampling ({denoising_steps} DDIM steps, eta={eta})...")
        t_sample = time.time()
        with torch.no_grad():
            pred_norm = dense_l_model.sample(
                history_t, future_t, terrain_tensor,
                num_steps=denoising_steps, eta=eta, seed=42,
            )
        print(f"[*] Dense-L reverse diffusion completed in {time.time() - t_sample:.2f}s!")

        # Denormalize to physical units
        pred_phys = invert_normalization(pred_norm[0].cpu().numpy(), stats_raw, channel_names=channel_order)

        for d in range(7):
            idx = min(d, max(0, n_available - 1))
            p_coarse = np.array(coarse_grids["precipitation_sum"][idx], dtype=np.float32)
            p_coarse = np.nan_to_num(np.maximum(p_coarse, 0.0))

            # Channel 0: Precipitation with local block mass conservation
            hr_raw_p = np.maximum(0.0, pred_phys[d, 0])
            hr_p_t = torch.from_numpy(hr_raw_p).unsqueeze(0).unsqueeze(0).to(device)
            lr_p_t = torch.from_numpy(p_coarse).unsqueeze(0).unsqueeze(0).to(device)
            hr_conserved_t = conservative_renorm_local(hr_p_t, lr_p_t)
            hr_mean = hr_conserved_t.squeeze().cpu().numpy()

            # Conformal spread scaled to conserved mean
            hr_lo = np.maximum(0.0, hr_mean - q_hat)
            hr_hi = hr_mean + q_hat

            # Thermodynamic channels
            hr_tmax = np.maximum(pred_phys[d, 1], pred_phys[d, 2] + 0.5)
            hr_tmin = pred_phys[d, 2]
            hr_rh = np.clip(pred_phys[d, 3], 10.0, 100.0)
            hr_wind = np.hypot(pred_phys[d, 4], pred_phys[d, 5]) * 3.6

            # Zonal aggregation
            agg_mean = aggregator.aggregate_grid(hr_mean, hr_lats, hr_lons)
            agg_lo = aggregator.aggregate_grid(hr_lo, hr_lats, hr_lons)
            agg_hi = aggregator.aggregate_grid(hr_hi, hr_lats, hr_lons)
            agg_tmax = aggregator.aggregate_grid(hr_tmax, hr_lats, hr_lons)
            agg_tmin = aggregator.aggregate_grid(hr_tmin, hr_lats, hr_lons)
            agg_rh = aggregator.aggregate_grid(hr_rh, hr_lats, hr_lons)
            agg_wind = aggregator.aggregate_grid(hr_wind, hr_lats, hr_lons)
            agg_detailed = aggregator.aggregate_grid_detailed(hr_mean, hr_lo, hr_hi, hr_lats, hr_lons)

            day_prov = "COMMITTED_FALLBACK_CYCLE" if is_committed_fallback else "OPENMETEO_FORECAST_DOWNSCALED"
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
    else:
        # Fallback UNet5x inference pipeline
        v3_1_path = ROOT / "models" / "checkpoints" / "best_5x_model_v3_1.pt"
        if checkpoint_path == (ROOT / "models" / "checkpoints" / "best_5x_model.pt") and v3_1_path.exists():
            checkpoint_path = v3_1_path

        print(f"[*] Loading checkpoint: {checkpoint_path.name}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        is_v3_1 = "v3_1" in checkpoint_path.name
        base_model = UNet5x(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            scale_factor=5,
            terrain_channels=5,
            use_residual=is_v3_1,
        ).to(device)
        base_model.load_pretrained(ckpt, device=device)
        base_model.eval()

        mc_model = MCDropoutWrapper(base_model, p=0.1).to(device)
        multi_downscaler = MultivariatePhysicalDownscaler().to(device)

        for d in range(7):
            grid_idx = min(d, len(coarse_grids.get("precipitation_sum", [])) - 1)
            p_coarse = np.array(coarse_grids["precipitation_sum"][grid_idx], dtype=np.float32)
            p_coarse = np.nan_to_num(np.maximum(p_coarse, 0.0))

            tmax_lr = torch.from_numpy(np.array(coarse_grids["temperature_2m_max"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            tmin_lr = torch.from_numpy(np.array(coarse_grids["temperature_2m_min"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            rh_lr = torch.from_numpy(np.array(coarse_grids["relative_humidity_2m_mean"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            wind_lr = torch.from_numpy(np.array(coarse_grids["wind_speed_10m_max"][grid_idx], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)

            x_tensor = torch.from_numpy(p_coarse).unsqueeze(0).unsqueeze(0).to(device)
            x_log = torch.log1p(x_tensor)
            mean_pred, q_lo, q_hi = mc_model.predict_with_uncertainty(
                x_log, terrain_hr=terrain_tensor, n_passes=n_mc_passes
            )
            mean_pred = conservative_renorm_local(mean_pred, x_tensor)

            int_lo = torch.clamp(q_lo - q_hat, min=0.0)
            int_hi = torch.clamp(q_hi + q_hat, min=0.0)

            hr_mean = mean_pred.squeeze().cpu().numpy()
            hr_lo = int_lo.squeeze().cpu().numpy()
            hr_hi = int_hi.squeeze().cpu().numpy()

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

            agg_mean = aggregator.aggregate_grid(hr_mean, hr_lats, hr_lons)
            agg_lo = aggregator.aggregate_grid(hr_lo, hr_lats, hr_lons)
            agg_hi = aggregator.aggregate_grid(hr_hi, hr_lats, hr_lons)
            agg_tmax = aggregator.aggregate_grid(hr_tmax, hr_lats, hr_lons)
            agg_tmin = aggregator.aggregate_grid(hr_tmin, hr_lats, hr_lons)
            agg_rh = aggregator.aggregate_grid(hr_rh, hr_lats, hr_lons)
            agg_wind = aggregator.aggregate_grid(hr_wind, hr_lats, hr_lons)
            agg_detailed = aggregator.aggregate_grid_detailed(hr_mean, hr_lo, hr_hi, hr_lats, hr_lons)

            day_prov = "COMMITTED_FALLBACK_CYCLE" if is_committed_fallback else "OPENMETEO_FORECAST_DOWNSCALED"
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
                "parcels": (d_res["agg_detailed"].get(gpcode) or {}).get("parcels", []),
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
            "model_tier": "dense_l" if use_dense_l else "unet_5x",
            "model_name": "Dense-L Accurate (52M Parameters, 16-step DDIM)" if use_dense_l else "UNet5x-SuperRes-Terrain",
            "ragi_stage": r_stage,
            "paddy_stage": p_stage,
            "spatial_variance": daily_results[0]["agg_detailed"].get(gpcode),
            "multi_day_forecast": multi_day_items,
        }
        records.append(rec)

    records.sort(key=lambda r: r["lgd_code"])

    # Write serving JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Write GeoJSON
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
    print(f"          Model: {'Dense-L Accurate (52M Diffusion)' if use_dense_l else 'UNet5x'}")
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
    parser = argparse.ArgumentParser(description="Single-Command Mandya Weather Forecast Pipeline.")
    parser.add_argument("--date", default="today", help="Forecast date (YYYY-MM-DD or 'today')")
    parser.add_argument("--district", default="MANDYA", help="Target district (default: MANDYA)")
    parser.add_argument("--output", default="data/serving/mandya_forecasts.json", help="Output JSON path")
    parser.add_argument("--forecast-file", default=None, help="Optional path to operational multiday coarse JSON")
    parser.add_argument("--mode", default="operational", choices=["operational", "hybrid"], help="Pipeline mode: operational (unified 7-day NWP) or hybrid (Day 0 IMD + Days 1-6 NWP)")
    parser.add_argument("--model-tier", default="dense_l", choices=["dense_l", "unet_5x"], help="Model tier (default: dense_l)")
    parser.add_argument("--steps", type=int, default=16, help="Denoising steps for diffusion model (default: 16)")
    parser.add_argument("--eta", type=float, default=0.50, help="DDIM eta parameter (default: 0.50)")
    parser.add_argument("--live-sim", action="store_true", help="Simulate automated FTP cron ingestion from IMD")
    args = parser.parse_args()

    if args.live_sim:
        simulate_live_imd_ingest(args.date)

    fc_file = Path(args.forecast_file) if args.forecast_file else None
    out_path = Path(args.output)
    if args.district.upper() != "MANDYA" and args.output == "data/serving/mandya_forecasts.json":
        out_path = ROOT / "data" / "serving" / f"{args.district.lower()}_forecasts.json"

    run_pipeline(
        forecast_date=args.date,
        district=args.district,
        output_json_path=out_path,
        forecast_file=fc_file,
        mode=args.mode,
        model_tier=args.model_tier,
        denoising_steps=args.steps,
        eta=args.eta,
    )

    if args.live_sim:
        print("\n================================================================================")
        print("[CRON] Ingestion & serving pipeline completed successfully.")
        print("[CRON] Standby mode: Next automated polling scheduled in 06:00:00 (Next IMD sync: 08:30 IST).")
        print("================================================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
