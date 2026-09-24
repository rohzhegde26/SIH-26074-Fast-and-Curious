"""
scripts/diagnose_residual_statistics.py

Diagnostic analysis of physical vs normalized residuals across the 6 weather variables
and 7 forecast leads on the SIH 26074 dataset.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
import yaml
import zarr

ROOT = Path(__file__).resolve().parent.parent

def run_diagnostics():
    store = zarr.open(str(ROOT / "datasets" / "multitask_temporal_v2_h14.zarr"), mode="r")
    df = pd.read_parquet(ROOT / "data" / "sample_index_v2_h14.parquet")
    with open(ROOT / "data" / "normalization_stats_v2.yaml", "r", encoding="utf-8") as f:
        norm_v2 = yaml.safe_load(f)["channels"]
    with open(ROOT / "data" / "normalization_stats.yaml", "r", encoding="utf-8") as f:
        norm_v1 = yaml.safe_load(f)["channels"]

    channels = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    val_indices = df[df["split"] == "val"].index.to_numpy()
    train_indices = df[df["split"] == "train"].index.to_numpy()

    print(f"Loaded {len(train_indices)} train and {len(val_indices)} val samples.")

    # Load validation subset into memory for thorough analysis
    val_targ = np.asarray(store["target"][val_indices], dtype=np.float32)  # [122, 7, 6, 80, 80]
    val_fcst = np.asarray(store["future_forecast"][val_indices], dtype=np.float32)  # [122, 7, 6, 16, 16]

    b, leads, c, h, w = val_targ.shape
    fcst_tensor = torch.from_numpy(val_fcst).reshape(b * leads, c, 16, 16)
    fcst_up_tensor = F.interpolate(fcst_tensor, size=(80, 80), mode="bilinear", align_corners=False)
    val_fcst_up = fcst_up_tensor.view(b, leads, c, 80, 80).numpy()

    # 1. Physical Residual: r_phys = val_targ - val_fcst_up
    r_phys = val_targ - val_fcst_up

    # 2. Normalized Residual under v2 stats (current Sprint 5 model-space residual)
    val_targ_norm_v2 = np.zeros_like(val_targ)
    val_fcst_norm_v2 = np.zeros_like(val_fcst_up)
    for c_idx, ch in enumerate(channels):
        mu = norm_v2[ch]["mean"]
        sd = norm_v2[ch]["std"]
        val_targ_norm_v2[:, :, c_idx] = (val_targ[:, :, c_idx] - mu) / sd
        val_fcst_norm_v2[:, :, c_idx] = (val_fcst_up[:, :, c_idx] - mu) / sd
    r_norm_v2 = val_targ_norm_v2 - val_fcst_norm_v2

    # 3. Normalized Residual under v1 stats (log1p for precip)
    val_targ_norm_v1 = np.zeros_like(val_targ)
    val_fcst_norm_v1 = np.zeros_like(val_fcst_up)
    for c_idx, ch in enumerate(channels):
        mu = norm_v1[ch]["mean"]
        sd = norm_v1[ch]["std"]
        if ch == "precipitation":
            val_targ_norm_v1[:, :, c_idx] = (np.log1p(np.maximum(0.0, val_targ[:, :, c_idx])) - mu) / sd
            val_fcst_norm_v1[:, :, c_idx] = (np.log1p(np.maximum(0.0, val_fcst_up[:, :, c_idx])) - mu) / sd
        else:
            val_targ_norm_v1[:, :, c_idx] = (val_targ[:, :, c_idx] - mu) / sd
            val_fcst_norm_v1[:, :, c_idx] = (val_fcst_up[:, :, c_idx] - mu) / sd
    r_norm_v1 = val_targ_norm_v1 - val_fcst_norm_v1

    # Analysis across channels
    stats_dict = {}
    print("\n" + "=" * 80)
    print("RESIDUAL STATISTICS BY CHANNEL (Validation 2022)")
    print("=" * 80)
    for c_idx, ch in enumerate(channels):
        r_c_phys = r_phys[:, :, c_idx].ravel()
        r_c_v2 = r_norm_v2[:, :, c_idx].ravel()
        r_c_v1 = r_norm_v1[:, :, c_idx].ravel()

        stats_dict[ch] = {
            "phys_mean": float(np.mean(r_c_phys)),
            "phys_std": float(np.std(r_c_phys)),
            "phys_q01": float(np.percentile(r_c_phys, 1)),
            "phys_q05": float(np.percentile(r_c_phys, 5)),
            "phys_q50": float(np.percentile(r_c_phys, 50)),
            "phys_q95": float(np.percentile(r_c_phys, 95)),
            "phys_q99": float(np.percentile(r_c_phys, 99)),
            "phys_skew": float(stats.skew(r_c_phys)),
            "phys_kurtosis": float(stats.kurtosis(r_c_phys)),
            "norm_v2_mean": float(np.mean(r_c_v2)),
            "norm_v2_std": float(np.std(r_c_v2)),
            "norm_v2_skew": float(stats.skew(r_c_v2)),
            "norm_v2_kurtosis": float(stats.kurtosis(r_c_v2)),
            "norm_v1_mean": float(np.mean(r_c_v1)),
            "norm_v1_std": float(np.std(r_c_v1)),
            "norm_v1_skew": float(stats.skew(r_c_v1)),
            "norm_v1_kurtosis": float(stats.kurtosis(r_c_v1)),
        }
        print(f"[{ch.upper():13s}] "
              f"Phys Mean: {stats_dict[ch]['phys_mean']:+.3f} | "
              f"Phys Std: {stats_dict[ch]['phys_std']:6.3f} | "
              f"NormV2 Std: {stats_dict[ch]['norm_v2_std']:6.3f} | "
              f"Skew: {stats_dict[ch]['phys_skew']:+6.2f} | "
              f"Kurtosis: {stats_dict[ch]['phys_kurtosis']:6.2f}")

    # Lead-time variance analysis
    lead_stats = {}
    print("\n" + "=" * 80)
    print("RESIDUAL STANDARD DEVIATION ACROSS LEADS (D+0 to D+6)")
    print("=" * 80)
    for l in range(leads):
        lead_stats[f"D+{l}"] = {}
        row = [f"D+{l}"]
        for c_idx, ch in enumerate(channels):
            r_l = r_norm_v2[:, l, c_idx].ravel()
            sd_l = float(np.std(r_l))
            lead_stats[f"D+{l}"][ch] = sd_l
            row.append(f"{ch[:4]}: {sd_l:.3f}")
        print(" | ".join(row))

    # Precipitation intensity breakdown
    p_targ = val_targ[:, :, 0].ravel()
    p_fcst = val_fcst_up[:, :, 0].ravel()
    r_p = r_phys[:, :, 0].ravel()

    regimes = {
        "dry (< 0.1 mm)": p_targ < 0.1,
        "light (0.1 - 2.5 mm)": (p_targ >= 0.1) & (p_targ < 2.5),
        "moderate (2.5 - 15 mm)": (p_targ >= 2.5) & (p_targ < 15.0),
        "heavy (15 - 30 mm)": (p_targ >= 15.0) & (p_targ < 30.0),
        "extreme (> 30 mm)": p_targ >= 30.0,
    }

    print("\n" + "=" * 80)
    print("PRECIPITATION RESIDUAL BY INTENSITY REGIME")
    print("=" * 80)
    precip_regime_stats = {}
    for r_name, mask in regimes.items():
        count = int(np.sum(mask))
        if count == 0:
            continue
        r_sub = r_p[mask]
        p_fcst_sub = p_fcst[mask]
        p_targ_sub = p_targ[mask]
        precip_regime_stats[r_name] = {
            "count": count,
            "pct_of_total": float(count / len(p_targ) * 100.0),
            "targ_mean": float(np.mean(p_targ_sub)),
            "fcst_mean": float(np.mean(p_fcst_sub)),
            "bias_mean": float(np.mean(r_sub)),
            "residual_std": float(np.std(r_sub)),
            "residual_q10": float(np.percentile(r_sub, 10)),
            "residual_q50": float(np.percentile(r_sub, 50)),
            "residual_q90": float(np.percentile(r_sub, 90)),
        }
        print(f"[{r_name:22s}] N={count:6d} ({precip_regime_stats[r_name]['pct_of_total']:4.1f}%) | "
              f"Targ: {precip_regime_stats[r_name]['targ_mean']:5.1f} | "
              f"Fcst: {precip_regime_stats[r_name]['fcst_mean']:5.1f} | "
              f"Bias: {precip_regime_stats[r_name]['bias_mean']:+6.2f} mm | "
              f"Std: {precip_regime_stats[r_name]['residual_std']:5.2f} mm")

    # Save to JSON
    out_path = ROOT / "reports" / "sprint_6_residual_audit_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "channels": stats_dict,
            "leads": lead_stats,
            "precip_regimes": precip_regime_stats,
        }, f, indent=2)
    print(f"\n[+] Wrote audit data to {out_path}")

if __name__ == "__main__":
    run_diagnostics()
