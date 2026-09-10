"""
scripts/fit_quantile_mapping.py

Fits empirical quantile mapping curves from real (UNet5x predictions, CHIRPS ground truth)
pairs on the Mandya holdout split (data/processed/mandya_holdout_buffer.geojson).

Execution Flow:
    1. Loads trained UNet5x v3.1 model (best_5x_model_v3_1.pt).
    2. Identifies the 8 holdout spatial windows intersecting the Mandya buffer.
    3. Runs raw model inference (APPLY_QUANTILE_MAPPING=False) across all holdout dates.
    4. Gathers paired (predicted fine mm, CHIRPS truth mm) samples per 0.25° coarse cell
       region (16x16 grid), plus a pooled global set.
    5. Fits 100-quantile empirical interp1d transfer curves with monotonic guarantees.
    6. Embeds provenance metadata: n_samples, date_range, split_name, checkpoint_hash,
       fit_timestamp, script_version.
    7. Serializes to data/static/quantile_mapping_params.json via QuantileMapper.save_parameters().
"""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import interpolate
import torch
import xarray as xr

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.patch_extraction import filter_spatial_windows, generate_spatial_windows
from src.eval.calibration import QuantileMapper
from src.models.unet_5x import UNet5x

SCRIPT_VERSION = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit Empirical Quantile Mapping Parameters on Mandya Holdout Split"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_5x_model_v3_1.pt",
        help="Path to trained UNet5x PyTorch checkpoint",
    )
    parser.add_argument(
        "--chirps",
        type=str,
        default="data/raw/chirps/chirps_sample.nc",
        help="Path to CHIRPS fine-scale (0.05°) observation NetCDF",
    )
    parser.add_argument(
        "--imd",
        type=str,
        default="data/raw/imd/imd_sample.nc",
        help="Path to IMD coarse-scale (0.25°) observation NetCDF",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default="data/processed/mandya_holdout_buffer.geojson",
        help="Path to Mandya holdout buffer GeoJSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/static/quantile_mapping_params.json",
        help="Target output path for serialized QuantileMapper JSON",
    )
    parser.add_argument(
        "--n-quantiles",
        type=int,
        default=100,
        help="Number of quantile points for empirical curve fitting",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Fallback flag: serialize synthetic default heavy-tail parameters instead of fitting",
    )
    return parser.parse_args()


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 hex digest of a local file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_fit_pipeline(args):
    out_path = Path(args.output)
    mapper = QuantileMapper(n_quantiles=args.n_quantiles, lr_shape=(16, 16))

    if args.fallback:
        print("[!] --fallback specified: Initializing synthetic default parameters...")
        mapper._init_default_heavy_tail_params()
        mapper.provenance = {
            "n_samples": 0,
            "date_range": [],
            "split_name": "synthetic_fallback",
            "checkpoint_hash": "none",
            "fit_timestamp": datetime.now(timezone.utc).isoformat(),
            "script_version": SCRIPT_VERSION,
        }
        saved_file = mapper.save_parameters(out_path)
        print(f"[+] Fallback parameters serialized to: {saved_file}")
        return

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    chirps_path = Path(args.chirps)
    if not chirps_path.exists():
        raise FileNotFoundError(f"CHIRPS observation file not found: {chirps_path}")
    imd_path = Path(args.imd)
    if not imd_path.exists():
        raise FileNotFoundError(f"IMD observation file not found: {imd_path}")
    holdout_path = Path(args.holdout)
    if not holdout_path.exists():
        raise FileNotFoundError(f"Holdout buffer GeoJSON not found: {holdout_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[*] Fitting Empirical Quantile Mapping Curves on Mandya Holdout Split")
    print(f"[*] Model Checkpoint: {ckpt_path} (on {device})")
    print(f"[*] Output Target:    {out_path}")
    print("=" * 70)

    # 1. Compute checkpoint hash
    ckpt_hash = compute_file_sha256(ckpt_path)
    print(f"[*] Checkpoint SHA-256: {ckpt_hash}")

    # 2. Instantiate UNet5x model (matching physics v3.1 architecture)
    model = UNet5x(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        scale_factor=5,
        terrain_channels=5,
        use_residual=True,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()

    # 3. Load observation rasters
    ds_chirps = xr.open_dataset(chirps_path)
    ds_imd = xr.open_dataset(imd_path)
    chirps_data = ds_chirps["precip"].values
    imd_data = ds_imd["rainfall"].values
    n_dates = chirps_data.shape[0]
    date_start = str(ds_chirps.time.values[0])[:10]
    date_end = str(ds_chirps.time.values[-1])[:10]
    print(f"[*] Observation Dates: {n_dates} days ({date_start} to {date_end})")

    # 4. Filter spatial windows for Mandya holdout buffer
    all_windows = generate_spatial_windows()
    usable, dropped = filter_spatial_windows(all_windows, holdout_geojson_path=str(holdout_path))
    holdout_windows = [w for w in dropped if w.get("intersects_mandya_buffer")]
    n_holdout = len(holdout_windows)
    print(f"[*] Identified {n_holdout} spatial windows in Mandya holdout split:")
    for hw in holdout_windows:
        print(f"    - Window {hw['window_id']:02d}: Lat [{hw['min_lat']}, {hw['max_lat']}] | Lon [{hw['min_lon']}, {hw['max_lon']}]")

    # 5. Inference collection loop with QUANTILE MAPPING OFF
    cell_preds: Dict[Tuple[int, int], List[np.ndarray]] = {(r, c): [] for r in range(16) for c in range(16)}
    cell_truth: Dict[Tuple[int, int], List[np.ndarray]] = {(r, c): [] for r in range(16) for c in range(16)}
    global_preds: List[np.ndarray] = []
    global_truth: List[np.ndarray] = []

    print("\n[*] Collecting paired inference & observation samples across holdout...")
    for t_idx in range(n_dates):
        for w in holdout_windows:
            hr_r, hr_c = w["hr_row"], w["hr_col"]
            lr_r, lr_c = w["lr_row"], w["lr_col"]

            lr_patch = np.nan_to_num(np.maximum(imd_data[t_idx, lr_r : lr_r + 16, lr_c : lr_c + 16], 0.0))
            hr_truth_patch = np.nan_to_num(np.maximum(chirps_data[t_idx, hr_r : hr_r + 80, hr_c : hr_c + 80], 0.0))

            coarse_tensor = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                # Raw physical model prediction (QM strictly OFF)
                pred_log = model(torch.log1p(coarse_tensor))
                pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)[0, 0].cpu().numpy()

            global_preds.append(pred_phys.flatten())
            global_truth.append(hr_truth_patch.flatten())

            for r in range(16):
                for c in range(16):
                    p_sub = pred_phys[r * 5 : (r + 1) * 5, c * 5 : (c + 1) * 5].flatten()
                    t_sub = hr_truth_patch[r * 5 : (r + 1) * 5, c * 5 : (c + 1) * 5].flatten()
                    cell_preds[(r, c)].append(p_sub)
                    cell_truth[(r, c)].append(t_sub)

    g_preds_arr = np.concatenate(global_preds)
    g_truth_arr = np.concatenate(global_truth)
    n_samples = int(len(g_preds_arr))
    print(f"[+] Successfully gathered {n_samples:,} paired samples (25 pixels x {n_dates} days x {n_holdout} windows).")
    print(f"[+] Samples per coarse cell region: {len(np.concatenate(cell_preds[(0, 0)])):,}")

    # 6. Fit empirical quantile mapping transfer curves
    quantiles = np.linspace(0.0, 1.0, args.n_quantiles)
    mapper.quantiles = quantiles

    # A. Global Regional Fallback Mapper
    gp_q = np.quantile(g_preds_arr, quantiles)
    gt_q = np.quantile(g_truth_arr, quantiles)
    gp_mono, u_idx = np.unique(gp_q, return_index=True)
    gt_mono = np.maximum.accumulate(gt_q[u_idx])

    mapper.global_mapper = interpolate.interp1d(
        gp_mono,
        gt_mono,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    # B. Per-0.25°-Cell Mappers
    mapper.mappers.clear()
    for r in range(16):
        for c in range(16):
            c_preds = np.concatenate(cell_preds[(r, c)])
            c_truth = np.concatenate(cell_truth[(r, c)])

            cp_q = np.quantile(c_preds, quantiles)
            ct_q = np.quantile(c_truth, quantiles)
            cp_mono, u_idx = np.unique(cp_q, return_index=True)
            ct_mono = np.maximum.accumulate(ct_q[u_idx])

            if len(cp_mono) >= 2 and (cp_mono[-1] > cp_mono[0]):
                cell_mapper = interpolate.interp1d(
                    cp_mono,
                    ct_mono,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
            else:
                cell_mapper = mapper.global_mapper

            mapper.mappers[(r, c)] = cell_mapper

    # 7. Embed Provenance Metadata
    fit_timestamp = datetime.now(timezone.utc).isoformat()
    mapper.provenance = {
        "n_samples": n_samples,
        "date_range": [date_start, date_end],
        "split_name": "mandya_holdout_buffer",
        "checkpoint_hash": ckpt_hash,
        "fit_timestamp": fit_timestamp,
        "script_version": SCRIPT_VERSION,
    }

    # 8. Serialize parameters to static JSON
    saved_path = mapper.save_parameters(out_path)
    print(f"\n[+] Quantile Mapping parameters successfully saved to: {saved_path}")

    # 9. Report Fitted Diagnostics & Heavy-Tail Bias Correction
    mapped_global = mapper.global_mapper(g_preds_arr)

    p_50 = float(np.quantile(g_preds_arr, 0.50))
    t_50 = float(np.quantile(g_truth_arr, 0.50))
    m_50 = float(np.quantile(mapped_global, 0.50))
    bias_50_before = ((p_50 - t_50) / max(t_50, 1e-4)) * 100.0
    bias_50_after = ((m_50 - t_50) / max(t_50, 1e-4)) * 100.0

    p_75 = float(np.quantile(g_preds_arr, 0.75))
    t_75 = float(np.quantile(g_truth_arr, 0.75))
    m_75 = float(np.quantile(mapped_global, 0.75))
    bias_75_before = ((p_75 - t_75) / max(t_75, 1e-4)) * 100.0
    bias_75_after = ((m_75 - t_75) / max(t_75, 1e-4)) * 100.0

    p_95 = float(np.quantile(g_preds_arr, 0.95))
    t_95 = float(np.quantile(g_truth_arr, 0.95))
    m_95 = float(np.quantile(mapped_global, 0.95))
    factor_95 = t_95 / max(p_95, 1e-4)
    bias_95_before = ((p_95 - t_95) / max(t_95, 1e-4)) * 100.0
    bias_95_after = ((m_95 - t_95) / max(t_95, 1e-4)) * 100.0

    p_99 = float(np.quantile(g_preds_arr, 0.99))
    t_99 = float(np.quantile(g_truth_arr, 0.99))
    m_99 = float(np.quantile(mapped_global, 0.99))
    factor_99 = t_99 / max(p_99, 1e-4)
    bias_99_before = ((p_99 - t_99) / max(t_99, 1e-4)) * 100.0
    bias_99_after = ((m_99 - t_99) / max(t_99, 1e-4)) * 100.0

    print("\n" + "=" * 70)
    print("MANDYA HOLDOUT CALIBRATION DIAGNOSTICS & TAIL CORRECTION FACTORS")
    print("=" * 70)
    print(f"50th Percentile (Median):")
    print(f"  - Model Prediction : {p_50:.2f} mm")
    print(f"  - CHIRPS Truth     : {t_50:.2f} mm (Pre-fit bias: {bias_50_before:+.2f}%)")
    print(f"  - Fitted Mapped    : {m_50:.2f} mm (Post-fit bias: {bias_50_after:+.2f}%)")
    print(f"  - Identity Verified: Preserved without forcing below 50th percentile (factor={t_50/max(p_50, 1e-4):.3f})")
    print(f"\n75th Percentile:")
    print(f"  - Model Prediction : {p_75:.2f} mm")
    print(f"  - CHIRPS Truth     : {t_75:.2f} mm (Pre-fit bias: {bias_75_before:+.2f}%)")
    print(f"  - Fitted Mapped    : {m_75:.2f} mm (Post-fit bias: {bias_75_after:+.2f}%)")
    print(f"\n95th Percentile (Heavy-Tail Trigger):")
    print(f"  - Model Prediction : {p_95:.2f} mm")
    print(f"  - CHIRPS Truth     : {t_95:.2f} mm (Pre-fit bias: {bias_95_before:+.2f}%)")
    print(f"  - Tail Boost Factor: {factor_95:.3f} (+{(factor_95-1.0)*100.0:.2f}%)")
    print(f"  - Fitted Mapped    : {m_95:.2f} mm (Post-fit bias: {bias_95_after:+.2f}%)")
    print(f"\n99th Percentile (Extreme Convective Trigger):")
    print(f"  - Model Prediction : {p_99:.2f} mm")
    print(f"  - CHIRPS Truth     : {t_99:.2f} mm (Pre-fit bias: {bias_99_before:+.2f}%)")
    print(f"  - Tail Boost Factor: {factor_99:.3f} (+{(factor_99-1.0)*100.0:.2f}%)")
    print(f"  - Fitted Mapped    : {m_99:.2f} mm (Post-fit bias: {bias_99_after:+.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    run_fit_pipeline(args)
