"""
scripts/benchmark_multitask_baselines.py

Scientific Gridded Benchmark: MultiTaskUNet5x vs Classical Baselines.
Compares:
    1. Raw 0.25° Coarse Meteorological Input (repetition baseline)
    2. Bilinear 0.05° Spatial Interpolation Baseline
    3. MultiTaskUNet5x (Terrain-Conditioned Residual Neural Downscaler)

Evaluated across all five variables:
    - Precipitation (CHIRPS v2.0 0.25° to 0.05° Spatial Disaggregation)
    - Tmax (ERA5 0.25° -> ERA5-Land 0.05°)
    - Tmin (ERA5 0.25° -> ERA5-Land 0.05°)
    - Relative Humidity (ERA5 0.25° -> ERA5-Land 0.05°)
    - Wind Speed (ERA5 0.25° -> ERA5-Land 0.05°)

Evaluated on the 122 held-out 2023 monsoon test partition.
Zero synthetic proxy generation; strict authentic data evaluation.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.multitask_unet import MultiTaskUNet5x
from src.data.multitask_dataset import MultiTaskPanchayatDataset

CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "multitask_5x_champion.pt"

DOMAIN_LAT = (11.0, 15.0)
DOMAIN_LON = (74.0, 78.0)


def bilinear_sample_point(
    grid: np.ndarray,
    lat: float,
    lon: float,
    lat_bounds: Tuple[float, float] = DOMAIN_LAT,
    lon_bounds: Tuple[float, float] = DOMAIN_LON,
) -> float:
    """
    4-point bilinear interpolation sampling of a 2D scalar field at exact (lat, lon) coordinates.
    Exact cell-center alignment under Pixel-Is-Area geometry:
    Row 0 center is at lat_max - 0.5 * res, Row H-1 center is at lat_min + 0.5 * res.
    Col 0 center is at lon_min + 0.5 * res, Col W-1 center is at lon_max - 0.5 * res.
    """
    h, w = grid.shape
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    res_lat = (lat_max - lat_min) / float(h)
    res_lon = (lon_max - lon_min) / float(w)

    r_continuous = (lat_max - 0.5 * res_lat - lat) / res_lat
    c_continuous = (lon - (lon_min + 0.5 * res_lon)) / res_lon

    r_clamped = max(0.0, min(float(r_continuous), float(h - 1)))
    c_clamped = max(0.0, min(float(c_continuous), float(w - 1)))

    r0 = int(np.floor(r_clamped))
    c0 = int(np.floor(c_clamped))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)

    dr = r_clamped - float(r0)
    dc = c_clamped - float(c0)

    val = (
        (1.0 - dr) * (1.0 - dc) * grid[r0, c0]
        + (1.0 - dr) * dc * grid[r0, c1]
        + dr * (1.0 - dc) * grid[r1, c0]
        + dr * dc * grid[r1, c1]
    )
    return float(val)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Computes standard regression metrics (MAE, RMSE, R2) with pairwise deletion of NaNs."""
    mask = np.isfinite(pred) & np.isfinite(target)
    if not np.any(mask):
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "count": 0}

    p = pred[mask]
    t = target[mask]
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    ss_res = float(np.sum((t - p) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-6 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "count": int(len(p))}


def compute_csi(pred: np.ndarray, target: np.ndarray, threshold: float = 15.0) -> float:
    """Computes Critical Success Index (CSI) at specified threshold with NaN filtering."""
    mask = np.isfinite(pred) & np.isfinite(target)
    p = pred[mask]
    t = target[mask]
    hits = np.sum((p >= threshold) & (t >= threshold))
    misses = np.sum((p < threshold) & (t >= threshold))
    false_alarms = np.sum((p >= threshold) & (t < threshold))
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else 1.0 if (misses == 0 and false_alarms == 0) else 0.0


def run_benchmark(max_test_samples: Optional[int] = None) -> Dict[str, Any]:
    print("=" * 86)
    print("SCIENTIFIC GRIDDED BENCHMARK: MultiTaskUNet5x vs Classical Baselines")
    print(f"Dataset Partition: 2023 Held-Out Test Split (total samples: {max_test_samples or 122})")
    print(f"Spatial Domain: 11.0°N–15.0°N, 74.0°E–78.0°E (Exact 5x Scaling: 16x16 -> 80x80)")
    print("=" * 86)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiTaskPanchayatDataset(split="test", max_samples=max_test_samples)

    model = MultiTaskUNet5x().to(device)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[+] Loaded model checkpoint: {CHECKPOINT_PATH.name}")
    else:
        print("[*] Benchmark running with initialized model weights.")
    model.eval()

    var_keys = ["rain", "tmax", "tmin", "rh", "wind"]
    var_labels = {
        "rain": "Rain (mm)",
        "tmax": "Tmax (°C)",
        "tmin": "Tmin (°C)",
        "rh": "RH (%)",
        "wind": "Wind (km/h)",
    }

    raw_preds: Dict[str, List[np.ndarray]] = {k: [] for k in var_keys}
    bilinear_preds: Dict[str, List[np.ndarray]] = {k: [] for k in var_keys}
    model_preds: Dict[str, List[np.ndarray]] = {k: [] for k in var_keys}
    ground_truths: Dict[str, List[np.ndarray]] = {k: [] for k in var_keys}

    n_samples = len(dataset)
    with torch.no_grad():
        for i in range(n_samples):
            c_nwp, f_terrain, f_tgt = dataset[i]
            c_nwp_t = c_nwp.unsqueeze(0).to(device)
            f_terrain_t = f_terrain.unsqueeze(0).to(device)

            # 1. Raw 0.25° NWP (nearest-neighbor repetition 16x16 -> 80x80)
            raw_t = F.interpolate(c_nwp_t, size=(80, 80), mode="nearest")

            # 2. Bilinear baseline (16x16 -> 80x80)
            bilinear_t = F.interpolate(c_nwp_t, size=(80, 80), mode="bilinear", align_corners=False)

            # 3. MultiTaskUNet5x
            model_res = model(c_nwp_t, terrain_hr=f_terrain_t)

            for idx, k in enumerate(var_keys):
                r_val = raw_t[0, idx].cpu().numpy()
                b_val = bilinear_t[0, idx].cpu().numpy()
                m_val = model_res[k][0, 0].cpu().numpy()
                tgt_val = f_tgt[idx].numpy()

                raw_preds[k].append(r_val)
                bilinear_preds[k].append(b_val)
                model_preds[k].append(m_val)
                ground_truths[k].append(tgt_val)

    print(f"\n[1] HIGH-RESOLUTION GRIDDED VERIFICATION (80x80 reference fields across {n_samples} test days)")
    print(f"{'Variable':<12} | {'Metric':<6} | {'Raw 0.25°':<12} | {'Bilinear 0.05°':<14} | {'MultiTaskUNet5x':<15} | {'Delta vs Bilin':<14}")
    print("-" * 86)

    benchmark_summary = {}

    for k in var_keys:
        v_name = var_labels[k]
        r_arr = np.concatenate([p.ravel() for p in raw_preds[k]])
        b_arr = np.concatenate([p.ravel() for p in bilinear_preds[k]])
        m_arr = np.concatenate([p.ravel() for p in model_preds[k]])
        t_arr = np.concatenate([p.ravel() for p in ground_truths[k]])

        r_metrics = compute_metrics(r_arr, t_arr)
        b_metrics = compute_metrics(b_arr, t_arr)
        m_metrics = compute_metrics(m_arr, t_arr)

        delta_mae = ((m_metrics["mae"] - b_metrics["mae"]) / b_metrics["mae"]) * 100.0 if b_metrics["mae"] > 0 else 0.0
        delta_str = f"{delta_mae:+.1f}%"
        print(f"{v_name:<12} | {'MAE':<6} | {r_metrics['mae']:<12.3f} | {b_metrics['mae']:<14.3f} | {m_metrics['mae']:<15.3f} | {delta_str:<14}")

        delta_rmse = ((m_metrics["rmse"] - b_metrics["rmse"]) / b_metrics["rmse"]) * 100.0 if b_metrics["rmse"] > 0 else 0.0
        delta_str_rmse = f"{delta_rmse:+.1f}%"
        print(f"{'':<12} | {'RMSE':<6} | {r_metrics['rmse']:<12.3f} | {b_metrics['rmse']:<14.3f} | {m_metrics['rmse']:<15.3f} | {delta_str_rmse:<14}")

        benchmark_summary[k] = {
            "raw_mae": r_metrics["mae"],
            "bilin_mae": b_metrics["mae"],
            "model_mae": m_metrics["mae"],
            "delta_mae_pct": delta_mae,
            "raw_rmse": r_metrics["rmse"],
            "bilin_rmse": b_metrics["rmse"],
            "model_rmse": m_metrics["rmse"],
            "delta_rmse_pct": delta_rmse,
        }

        if k == "rain":
            csi_r = compute_csi(r_arr, t_arr, threshold=15.0)
            csi_b = compute_csi(b_arr, t_arr, threshold=15.0)
            csi_m = compute_csi(m_arr, t_arr, threshold=15.0)
            print(f"{'':<12} | {'CSI@15':<6} | {csi_r:<12.3f} | {csi_b:<14.3f} | {csi_m:<15.3f} | {csi_m - csi_b:+.3f}")
            benchmark_summary[k]["csi15_model"] = csi_m
            benchmark_summary[k]["csi15_bilin"] = csi_b
        else:
            print(f"{'':<12} | {'R²':<6} | {r_metrics['r2']:<12.3f} | {b_metrics['r2']:<14.3f} | {m_metrics['r2']:<15.3f} | {m_metrics['r2'] - b_metrics['r2']:+.3f}")
            benchmark_summary[k]["r2_model"] = m_metrics["r2"]
            benchmark_summary[k]["r2_bilin"] = b_metrics["r2"]
        print("-" * 86)

    print("=" * 86)
    print("[+] High-Resolution Gridded Benchmark Completed.")
    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-way gridded baseline benchmark")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples to evaluate (default: all 122)")
    args = parser.parse_args()

    run_benchmark(max_test_samples=args.max_test_samples)
