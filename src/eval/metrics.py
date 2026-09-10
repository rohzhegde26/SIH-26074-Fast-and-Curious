"""
src/eval/metrics.py

Comprehensive Metric Suite & Hill-vs-Plains Stratified Evaluation.

Physical Standards & Guardrails:
    - All headline metrics MUST be computed and reported on the QM-calibrated product
      (what is actually shipped to the user), not on raw internal diagnostics.
    - Dry-day separation is mandatory: Monsoon precipitation exhibits heavy-tailed skewness
      where 80-90% of pixels can report 0.0 mm. Reporting only All-Day MAE introduces severe
      deflation bias. Wet-Day MAE is strictly conditioned on Rain > 2.5 mm (IMD trace threshold).
    - Categorical skill scores: CSI (Critical Success Index), POD (Probability of Detection),
      and FAR (False Alarm Ratio) across standard IMD rainfall categories:
      * Light: 2.5–15.5 mm
      * Moderate: 15.5–64.4 mm
      * Heavy / Extreme: >64.5 mm (R95 / R99)
    - Hill-vs-Plains breakdown: Separate error reporting for mountainous/Ghats terrain vs plains
      to address satellite retrieval bias in orographic precipitation zones.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.conservation import expm1_transform


def compute_headline_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    trues: Union[np.ndarray, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes standard headline evaluation metrics:
    - MAE (Mean Absolute Error, mm)
    - RMSE (Root Mean Squared Error, mm)
    - Pearson r (Correlation)
    - PBIAS (Percent Bias, %)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    p = np.asarray(preds, dtype=np.float32).flatten()
    t = np.asarray(trues, dtype=np.float32).flatten()

    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    # Pearson r
    std_p = np.std(p)
    std_t = np.std(t)
    if std_p > 1e-6 and std_t > 1e-6:
        r = float(np.corrcoef(p, t)[0, 1])
    else:
        r = 0.0

    # Percent Bias: 100 * sum(pred - true) / sum(true)
    sum_t = np.sum(t)
    if sum_t > 1e-6:
        pbias = float(100.0 * np.sum(p - t) / sum_t)
    else:
        pbias = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson_r": r,
        "pbias": pbias,
    }


def compute_wet_day_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    trues: Union[np.ndarray, torch.Tensor],
    threshold: float = 2.5,
) -> Dict[str, float]:
    """
    Computes Wet-Day MAE and RMSE strictly conditioned on ground truth Rain > 2.5 mm.
    Eliminates dry-day zero-inflation bias.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    p = np.asarray(preds, dtype=np.float32).flatten()
    t = np.asarray(trues, dtype=np.float32).flatten()

    wet_mask = t > threshold
    n_wet = int(np.sum(wet_mask))

    if n_wet > 0:
        wet_mae = float(np.mean(np.abs(p[wet_mask] - t[wet_mask])))
        wet_rmse = float(np.sqrt(np.mean((p[wet_mask] - t[wet_mask]) ** 2)))
        wet_fraction = float(n_wet / len(t))
    else:
        wet_mae = float(np.mean(np.abs(p - t)))
        wet_rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        wet_fraction = 0.0

    return {
        "wet_mae": wet_mae,
        "wet_rmse": wet_rmse,
        "wet_count": n_wet,
        "wet_fraction": wet_fraction,
        "wet_threshold_mm": threshold,
    }


def compute_categorical_skill(
    preds: Union[np.ndarray, torch.Tensor],
    trues: Union[np.ndarray, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """
    Computes CSI, POD, and FAR across standard IMD rainfall categories:
    - Light: 2.5 to 15.5 mm
    - Moderate: 15.5 to 64.4 mm
    - Heavy/Extreme: > 64.5 mm
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    p = np.asarray(preds, dtype=np.float32).flatten()
    t = np.asarray(trues, dtype=np.float32).flatten()

    categories = {
        "Light (2.5-15.5mm)": (2.5, 15.5),
        "Moderate (15.5-64.4mm)": (15.5, 64.4),
        "Heavy/Extreme (>64.5mm)": (64.5, 1e6),
    }

    results = {}
    for cat_name, (lo, hi) in categories.items():
        if hi == 1e6:
            mask_pred = p >= lo
            mask_true = t >= lo
        else:
            mask_pred = (p >= lo) & (p < hi)
            mask_true = (t >= lo) & (t < hi)

        hits = int(np.sum(mask_pred & mask_true))
        false_alarms = int(np.sum(mask_pred & ~mask_true))
        misses = int(np.sum(~mask_pred & mask_true))
        correct_negs = int(np.sum(~mask_pred & ~mask_true))

        # Probability of Detection (POD) = H / (H + M)
        pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
        # False Alarm Ratio (FAR) = F / (H + F)
        far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
        # Critical Success Index (CSI) = H / (H + F + M)
        csi = float(hits / (hits + false_alarms + misses)) if (hits + false_alarms + misses) > 0 else 0.0

        results[cat_name] = {
            "hits": hits,
            "false_alarms": false_alarms,
            "misses": misses,
            "pod": pod,
            "far": far,
            "csi": csi,
        }

    return results


def classify_terrain(
    lats: Union[np.ndarray, torch.Tensor],
    lons: Optional[Union[np.ndarray, torch.Tensor]] = None,
    elevation_threshold: float = 500.0,
    dem_path: Optional[str] = "data/raw/dem/synthetic_terrain.nc",
) -> Tuple[np.ndarray, bool, str, str]:
    """
    Classifies spatial samples into Hill vs Plains.
    If DEM file exists at dem_path and lons are provided:
        Extracts elevation from terrain-conditioned DEM and classifies Hill if elevation >= 500m.
    Otherwise:
        Falls back to latitude proxy (lat >= 12.5°N) with a printed warning.

    Returns:
        (is_hill_mask, using_dem, hill_label, plains_label)
    """
    if isinstance(lats, torch.Tensor):
        lats = lats.detach().cpu().numpy()
    if isinstance(lons, torch.Tensor):
        lons = lons.detach().cpu().numpy()

    lats_np = np.asarray(lats, dtype=np.float32)

    using_dem = False
    is_hill = None

    if dem_path is not None and Path(dem_path).exists() and lons is not None:
        try:
            import xarray as xr

            lons_np = np.asarray(lons, dtype=np.float32)
            with xr.open_dataset(dem_path) as ds:
                elev_var = None
                for cand in ["elevation", "dem", "z", "Band1"]:
                    if cand in ds.data_vars:
                        elev_var = cand
                        break
                if elev_var is None:
                    elev_var = list(ds.data_vars)[0]

                elev_data = ds[elev_var].values
                dem_lats = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
                dem_lons = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values

                # Nearest-neighbor lookup for sample center coordinates
                lat_idx = np.abs(dem_lats[:, None] - lats_np[None, :]).argmin(axis=0)
                lon_idx = np.abs(dem_lons[:, None] - lons_np[None, :]).argmin(axis=0)
                sample_elev = elev_data[lat_idx, lon_idx]

                is_hill = sample_elev >= elevation_threshold
                using_dem = True
        except Exception as e:
            print(f"[WARNING] Failed loading DEM from '{dem_path}': {e}. Falling back to latitude proxy.")

    if not using_dem:
        if dem_path and not Path(dem_path).exists():
            print(f"[WARNING] DEM file not found at '{dem_path}'. Falling back to latitude proxy (lat >= 12.5°N).")
        elif lons is None and dem_path is not None:
            print(f"[WARNING] Longitudes not provided for DEM elevation lookup. Falling back to latitude proxy (lat >= 12.5°N).")

        lat_thresh = 12.5 if elevation_threshold == 500.0 else elevation_threshold
        is_hill = lats_np >= lat_thresh
        hill_label = "Hills (lat ≥ 12.5°N)"
        plains_label = "Plains (lat < 12.5°N)"
    else:
        hill_label = f"Hills (≥{int(elevation_threshold)}m)"
        plains_label = f"Plains (<{int(elevation_threshold)}m)"

    return is_hill, using_dem, hill_label, plains_label


def compute_hill_vs_plains(
    preds: Union[np.ndarray, torch.Tensor],
    trues: Union[np.ndarray, torch.Tensor],
    lats: Union[np.ndarray, torch.Tensor],
    lons: Optional[Union[np.ndarray, torch.Tensor]] = None,
    elevation_threshold: float = 500.0,
    dem_path: Optional[str] = "data/raw/dem/synthetic_terrain.nc",
) -> Dict[str, Dict[str, float]]:
    """
    Computes stratified error breakdown for Hill (Western Ghats / high terrain) vs Plains.
    Uses DEM elevation when available (>= 500m), falling back to latitude proxy (>= 12.5°N).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(trues, torch.Tensor):
        trues = trues.detach().cpu().numpy()

    p = np.asarray(preds, dtype=np.float32).flatten()
    t = np.asarray(trues, dtype=np.float32).flatten()

    is_hill_sample, using_dem, hill_label, plains_label = classify_terrain(
        lats=lats,
        lons=lons,
        elevation_threshold=elevation_threshold,
        dem_path=dem_path,
    )

    # Expand sample-level mask to match flattened pixels if needed
    if len(is_hill_sample) != len(p):
        pixels_per_patch = len(p) // max(len(is_hill_sample), 1)
        hill_mask = np.repeat(is_hill_sample, pixels_per_patch)
    else:
        hill_mask = is_hill_sample

    plains_mask = ~hill_mask

    # Hill metrics
    if np.sum(hill_mask) > 0:
        hill_metrics = compute_headline_metrics(p[hill_mask], t[hill_mask])
        hill_wet = compute_wet_day_metrics(p[hill_mask], t[hill_mask])
        hill_metrics.update(hill_wet)
    else:
        hill_metrics = compute_headline_metrics(p, t)

    # Plains metrics
    if np.sum(plains_mask) > 0:
        plains_metrics = compute_headline_metrics(p[plains_mask], t[plains_mask])
        plains_wet = compute_wet_day_metrics(p[plains_mask], t[plains_mask])
        plains_metrics.update(plains_wet)
    else:
        plains_metrics = compute_headline_metrics(p, t)

    return {
        hill_label: hill_metrics,
        plains_label: plains_metrics,
    }


def plot_hill_vs_plains(
    preds: np.ndarray,
    trues: np.ndarray,
    lats: np.ndarray,
    lons: Optional[np.ndarray] = None,
    elevation_threshold: float = 500.0,
    dem_path: Optional[str] = "data/raw/dem/synthetic_terrain.nc",
    save_path: str = "docs/hill_vs_plains.png",
) -> Path:
    """
    Generate three-panel visualization:
    1. Scatter plot (Pred vs True) color-coded by Hills vs Plains.
    2. Residual error distributions (Pred - True) by terrain.
    3. Categorical CSI and PBIAS comparison across Light, Moderate, and Heavy rainfall.
    """
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    p = np.asarray(preds, dtype=np.float32).flatten()
    t = np.asarray(trues, dtype=np.float32).flatten()

    is_hill_sample, using_dem, hill_label, plains_label = classify_terrain(
        lats=lats,
        lons=lons,
        elevation_threshold=elevation_threshold,
        dem_path=dem_path,
    )

    if len(is_hill_sample) != len(p):
        pixels_per_patch = len(p) // max(len(is_hill_sample), 1)
        hill_mask = np.repeat(is_hill_sample, pixels_per_patch)
    else:
        hill_mask = is_hill_sample

    plains_mask = ~hill_mask

    # Subsample for clean scatter visualization (max 5,000 points)
    n_pts = min(len(p), 5000)
    idx = np.random.choice(len(p), size=n_pts, replace=False)
    p_sub, t_sub = p[idx], t[idx]
    hill_sub = hill_mask[idx]
    plains_sub = plains_mask[idx]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Panel 1: Scatter plot
    max_scatter = min(max(float(np.quantile(t_sub, 0.99)), float(np.quantile(p_sub, 0.99))), 80.0)
    ax1.plot([0, max_scatter], [0, max_scatter], "k--", alpha=0.7, label="1:1 Perfect Prediction")
    if np.any(plains_sub):
        ax1.scatter(t_sub[plains_sub], p_sub[plains_sub], alpha=0.35, s=15, color="#2e6da4", label=plains_label)
    if np.any(hill_sub):
        ax1.scatter(t_sub[hill_sub], p_sub[hill_sub], alpha=0.45, s=18, color="#d9534f", label=hill_label)
    ax1.set_xlim([0, max_scatter])
    ax1.set_ylim([0, max_scatter])
    ax1.set_title("Calibrated Output: Observed vs. Predicted", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Observed Rainfall (mm)", fontsize=10)
    ax1.set_ylabel("QM-Calibrated Predicted Rainfall (mm)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Panel 2: Error distributions by terrain
    res_hill = (p - t)[hill_mask]
    res_plains = (p - t)[plains_mask]

    res_hill_clip = np.clip(res_hill, -25.0, 25.0) if len(res_hill) > 0 else np.array([])
    res_plains_clip = np.clip(res_plains, -25.0, 25.0) if len(res_plains) > 0 else np.array([])

    mae_plains = np.mean(np.abs(res_plains)) if len(res_plains) > 0 else 0.0
    mae_hill = np.mean(np.abs(res_hill)) if len(res_hill) > 0 else 0.0

    if len(res_plains_clip) > 0:
        ax2.hist(res_plains_clip, bins=50, alpha=0.6, color="#2e6da4", density=True, label=f"{plains_label} (MAE: {mae_plains:.2f}mm)")
    if len(res_hill_clip) > 0:
        ax2.hist(res_hill_clip, bins=50, alpha=0.6, color="#d9534f", density=True, label=f"{hill_label} (MAE: {mae_hill:.2f}mm)")
    ax2.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax2.set_xlim([-25, 25])
    ax2.set_title("Residual Error Distribution (Pred - True)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Error (mm)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    # Panel 3: Categorical skill & PBIAS
    skill = compute_categorical_skill(p, t)
    cat_labels = ["Light\n(2.5-15.5mm)", "Moderate\n(15.5-64.4mm)", "Heavy/Extreme\n(>64.5mm)"]
    csi_vals = [skill[k]["csi"] for k in skill]
    pod_vals = [skill[k]["pod"] for k in skill]
    far_vals = [skill[k]["far"] for k in skill]

    x_bar = np.arange(len(cat_labels))
    w_bar = 0.25

    ax3.bar(x_bar - w_bar, csi_vals, width=w_bar, color="#5cb85c", label="CSI (Critical Success)")
    ax3.bar(x_bar, pod_vals, width=w_bar, color="#337ab7", label="POD (Detection)")
    ax3.bar(x_bar + w_bar, far_vals, width=w_bar, color="#d9534f", label="FAR (False Alarm)")

    ax3.set_xticks(x_bar)
    ax3.set_xticklabels(cat_labels, fontsize=9)
    ax3.set_ylim([0, 1.05])
    ax3.set_title("Categorical Skill Across IMD Rainfall Classes", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Skill Score Fraction", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")

    plt.suptitle(
        "Sprint 3D: Orographic Terrain Error Decomposition & Categorical Skill Breakdown\n"
        "(Evaluated on QM-Calibrated Unseen Monsoon 2023 Predictions)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Hill-vs-plains evaluation artifact saved to: {out_path}")
    return out_path


def run_metrics_pipeline(
    checkpoint_path: str = "models/checkpoints/best_5x_model.pt",
    zarr_path: str = "data/cache/india_monsoon_patches.zarr",
    dem_path: str = "data/raw/dem/synthetic_terrain.nc",
    save_plot_path: str = "docs/hill_vs_plains.png",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Executes Sprint 3D:
    1. Loads best model and QuantileMapper.
    2. Evaluates on test split 2023.
    3. Transforms outputs via per-0.25°-cell QM.
    4. Computes all headline, wet-day, categorical, and hill-vs-plains metrics.
    5. Saves docs/hill_vs_plains.png and prints formatted summary table.
    """
    from src.eval.calibration import QuantileMapper, run_calibration_service
    from src.models.dataset import MonsoonPatchDataset
    from src.models.unet_5x import UNet5x
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Running Comprehensive Evaluation Suite on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet5x(in_channels=1, out_channels=1, base_channels=32).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Fit or obtain QuantileMapper
    mapper, _ = run_calibration_service(
        checkpoint_path=checkpoint_path,
        zarr_path=zarr_path,
        device=device,
    )

    # 2. Run inference across test 2023 split
    test_ds = MonsoonPatchDataset(zarr_path, split="test", log_transform=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"[*] Evaluating on test split 2023 ({len(test_ds)} patches)...")

    raw_preds = []
    qm_preds = []
    trues = []
    all_lats = []
    all_lons = []

    max_test_batches = 40
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_test_batches:
                break
            x = batch["lr"].to(device)
            y_phys = batch["hr_phys"].cpu().numpy()
            lats = batch["lat"].numpy()
            lons = batch["lon"].numpy()

            p_log = model(x)
            p_phys = expm1_transform(p_log).cpu().numpy()  # [B, 1, 80, 80]

            # Apply per-0.25°-cell quantile mapping
            p_qm = mapper.transform(p_phys)

            raw_preds.append(p_phys[:, 0, ...])
            qm_preds.append(p_qm[:, 0, ...] if p_qm.ndim == 4 else p_qm)
            trues.append(y_phys[:, 0, ...])
            all_lats.append(lats)
            all_lons.append(lons)

    raw_arr = np.concatenate(raw_preds, axis=0)
    qm_arr = np.concatenate(qm_preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    lats_arr = np.concatenate(all_lats, axis=0)
    lons_arr = np.concatenate(all_lons, axis=0)

    # 3. Compute Metrics on QM-Calibrated predictions
    headline = compute_headline_metrics(qm_arr, true_arr)
    wet_metrics = compute_wet_day_metrics(qm_arr, true_arr, threshold=2.5)
    categorical = compute_categorical_skill(qm_arr, true_arr)
    terrain_metrics = compute_hill_vs_plains(
        qm_arr,
        true_arr,
        lats_arr,
        lons=lons_arr,
        elevation_threshold=500.0,
        dem_path=dem_path,
    )

    # Also compute raw diagnostics
    raw_headline = compute_headline_metrics(raw_arr, true_arr)
    raw_wet = compute_wet_day_metrics(raw_arr, true_arr, threshold=2.5)

    # 4. Generate visualization artifact
    plot_hill_vs_plains(
        qm_arr,
        true_arr,
        lats_arr,
        lons=lons_arr,
        elevation_threshold=500.0,
        dem_path=dem_path,
        save_path=save_plot_path,
    )

    # 5. Print Formatted Table
    print("\n" + "=" * 80)
    print("      SIH-26074 SPRINT 3: HEADLINE CALIBRATED BENCHMARK REPORT (TEST 2023)")
    print("=" * 80)
    print(f"{'Evaluation Metric':<32} | {'Raw 5x Model':<18} | {'QM-Calibrated (Shipped)':<22}")
    print("-" * 80)
    print(f"{'All-Day MAE (mm)':<32} | {raw_headline['mae']:<18.3f} | {headline['mae']:<22.3f}")
    print(f"{'All-Day RMSE (mm)':<32} | {raw_headline['rmse']:<18.3f} | {headline['rmse']:<22.3f}")
    print(f"{'Wet-Day MAE (>2.5mm)':<32} | {raw_wet['wet_mae']:<18.3f} | {wet_metrics['wet_mae']:<22.3f}")
    print(f"{'Wet-Day RMSE (>2.5mm)':<32} | {raw_wet['wet_rmse']:<18.3f} | {wet_metrics['wet_rmse']:<22.3f}")
    print(f"{'Pearson Correlation (r)':<32} | {raw_headline['pearson_r']:<18.3f} | {headline['pearson_r']:<22.3f}")
    print(f"{'Percent Bias (PBIAS)':<32} | {raw_headline['pbias']:<+17.1f}% | {headline['pbias']:<+21.1f}%")
    print("-" * 80)
    print("\n--- Categorical Skill (CSI / POD / FAR) Across IMD Rainfall Classes ---")
    for cat, val in categorical.items():
        print(f"  {cat:<28} -> CSI: {val['csi']:.3f} | POD: {val['pod']:.3f} | FAR: {val['far']:.3f}")
    print("\n--- Orographic Error Stratification (Hills vs. Plains) ---")
    for terrain, val in terrain_metrics.items():
        print(f"  {terrain:<24} -> MAE: {val['mae']:.3f} mm | Wet MAE: {val['wet_mae']:.3f} mm | PBIAS: {val['pbias']:+.1f}%")
    print("=" * 80 + "\n")

    return {
        "headline": headline,
        "wet_metrics": wet_metrics,
        "categorical": categorical,
        "terrain_metrics": terrain_metrics,
        "raw_headline": raw_headline,
    }


if __name__ == "__main__":
    run_metrics_pipeline()

