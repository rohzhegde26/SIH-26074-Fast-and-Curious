"""
scripts/run_pipeline.py

Single-Command End-to-End Inference Orchestrator for Sprint 4.
Executes in < 5 seconds on CPU/GPU.

Pipeline Workflow:
1. Loads LR input grid (16x16) from IMD observations for the requested date.
2. Evaluates UNet5x checkpoint (best_5x_model.pt) with MC-dropout uncertainty (CQR).
3. Applies quantile calibration and calculates 90% conformal uncertainty intervals [likely_min, likely_max].
4. Computes area-weighted zonal aggregation onto Mandya's 234 panchayats using mandya_full.geojson.
5. Emits serving payload data/serving/mandya_forecasts.json matching SPRINT-3-HANDOFF-CONTRACT.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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

from src.data.zonal_aggregation import ZonalAggregator
from src.eval.cqr import MCDropoutWrapper
from src.losses.conservation import expm1_transform
from src.models.unet_5x import UNet5x


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
) -> list[dict]:
    t_start = time.time()
    if district.upper() != "MANDYA":
        raise ValueError(f"District {district} not supported. Only MANDYA is configured for Sprint 4.")

    print(f"[*] Starting Mandya End-to-End Forecast Pipeline for date: {forecast_date}...")

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
    # In IMD grid (lat: 6.5..38.5, lon: 66.5..100.0), lat_slice=(18, 34), lon_slice=(34, 50)
    lr_slice = ds.rainfall.isel(time=time_idx, lat=slice(18, 34), lon=slice(34, 50)).values.astype(np.float32)
    lr_slice = np.nan_to_num(np.maximum(lr_slice, 0.0))

    # 2. Load trained UNet5x checkpoint and setup MC-Dropout for CQR uncertainty
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_model = UNet5x(in_channels=1, out_channels=1, base_channels=32).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    base_model.eval()

    mc_model = MCDropoutWrapper(base_model, p=0.1).to(device)

    # 3. Model forward pass (log1p transform as used in training)
    x_tensor = torch.from_numpy(lr_slice).unsqueeze(0).unsqueeze(0).to(device)
    x_log = torch.log1p(x_tensor)

    mean_pred, q_lo, q_hi = mc_model.predict_with_uncertainty(x_log, n_passes=n_mc_passes)

    # Apply conformal quantile bounds (clip-at-zero)
    int_lo = torch.clamp(q_lo - q_hat, min=0.0)
    int_hi = torch.clamp(q_hi + q_hat, min=0.0)

    hr_mean = mean_pred.squeeze().cpu().numpy()
    hr_lo = int_lo.squeeze().cpu().numpy()
    hr_hi = int_hi.squeeze().cpu().numpy()

    # 4. Zonal polygon aggregation
    hr_lats = np.linspace(10.90, 14.85, 80)
    hr_lons = np.linspace(74.90, 78.85, 80)

    aggregator = ZonalAggregator(str(geojson_path))
    agg_mean = aggregator.aggregate_grid(hr_mean, hr_lats, hr_lons)
    agg_lo = aggregator.aggregate_grid(hr_lo, hr_lats, hr_lons)
    agg_hi = aggregator.aggregate_grid(hr_hi, hr_lats, hr_lons)

    # Read simplified map gpcodes to ensure exact 1-to-1 correspondence
    topology = json.loads(topojson_path.read_text(encoding="utf-8"))
    collections = [v for v in topology.get("objects", {}).values() if v.get("type") == "GeometryCollection"]
    valid_gpcodes = {
        str(g.get("properties", {}).get("gpcode", "")).strip()
        for col in collections
        for g in col.get("geometries", [])
    } - {""}

    # Load polygon attribute table
    gdf = gpd.read_file(geojson_path)
    records = []
    dt_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

        exp = round(float(agg_mean.get(gpcode, 0.0)), 1)
        l_min = round(float(agg_lo.get(gpcode, 0.0)), 1)
        l_max = round(float(agg_hi.get(gpcode, 0.0)), 1)

        # Enforce consistency bounds
        l_min = min(l_min, exp)
        l_max = max(l_max, exp)

        rec = {
            "lgd_code": gpcode,
            "panchayat_name": str(row.get("gpname", "")).strip() or f"GP-{gpcode}",
            "district": district.upper(),
            "forecast_date": actual_date_str,
            "timestamp_utc": dt_utc,
            "expected_mm": exp,
            "likely_min_mm": l_min,
            "likely_max_mm": l_max,
            "ragi_stage": r_stage,
            "paddy_stage": p_stage,
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
        for col in ["expected_mm", "likely_min_mm", "likely_max_mm", "ragi_stage", "paddy_stage", "forecast_date", "timestamp_utc"]:
            export_gdf[col] = export_gdf["gpcode"].astype(str).str.strip().map(lambda c: rec_map[c][col])
        export_gdf.to_file(output_geojson_path, driver="GeoJSON")

    elapsed = time.time() - t_start
    print(f"[SUCCESS] Pipeline generated {len(records)} panchayat forecasts in {elapsed:.2f}s!")
    print(f"          Output saved to: {output_json_path}")
    if output_geojson_path:
        print(f"          GeoJSON saved to: {output_geojson_path}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-Command Mandya Weather Forecast Pipeline.")
    parser.add_argument("--date", default="2023-07-01", help="Forecast date (YYYY-MM-DD)")
    parser.add_argument("--district", default="MANDYA", help="Target district (default: MANDYA)")
    parser.add_argument("--output", default="data/serving/mandya_forecasts.json", help="Output JSON path")
    args = parser.parse_args()

    run_pipeline(
        forecast_date=args.date,
        district=args.district,
        output_json_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
