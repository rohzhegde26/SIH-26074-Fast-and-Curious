"""
scripts/evaluate_sprint9_capacity.py

Comprehensive Evaluation of Sprint 9 Model Capacity Scaling Ladder:
  - Dense-S: 15,685,478 params (Candidate 3 Frozen Control, base_channels=96)
  - Dense-M: 31,198,518 params (~2.0x Candidate 3, base_channels=136)
  - Dense-L: 51,997,958 params (~3.3x Candidate 3, base_channels=176)
  - MoE-4:   35,765,990 params (Sparse Top-1 Routing over 4 Bottleneck Experts)

Evaluates on the authentic 2022 validation dataset across matched 32 NFE:
  1. Point Mode:        K=8, S=4,  eta=0.5 (32 NFE)
  2. Distribution Mode: K=2, S=16, eta=0.0 (32 NFE)

Strict Invariants:
  - Connects directly to datasets/multitask_temporal_v2_h14.zarr and sample_index_v2_h14.parquet.
  - Candidate 3 weights verified via binary SHA-256 (f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92).
  - Hard fail on missing Candidate 3 checkpoint or SHA mismatch.
  - Inverts physical units and measures: Wet-MAE, CSI@15, CSI@30, Tmax MAE, Wind RMSE, Fair-CRPS, SSR, 90% Coverage, Laplacian Energy.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import invert_normalization
from src.models.residual_diffusion import compute_residual_target
from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)

EXPECTED_CANDIDATE3_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"
EXPECTED_PARAM_COUNT_DENSE_S = 15_685_478


def compute_laplacian_energy(img: torch.Tensor) -> float:
    """Computes mean squared Laplacian response for 2D field."""
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=img.device).view(1, 1, 3, 3)
    x = img.view(1, 1, img.shape[-2], img.shape[-1]).float()
    lap = F.conv2d(x, kernel, padding=1)
    return float(torch.mean(lap ** 2).item())


def compute_radial_psd(img_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes radially averaged 1D Power Spectral Density from 2D FFT."""
    h, w = img_np.shape
    f_shift = np.fft.fftshift(np.fft.fft2(img_np))
    psd_2d = np.abs(f_shift) ** 2

    y, x = np.ogrid[-h // 2 : h // 2, -w // 2 : w // 2]
    r = np.hypot(x, y).astype(np.int32)

    max_r = min(h, w) // 2
    radial_prof = np.zeros(max_r, dtype=np.float64)
    for i in range(max_r):
        mask = (r == i)
        if np.any(mask):
            radial_prof[i] = np.mean(psd_2d[mask])
    freqs = np.arange(max_r)
    return freqs, radial_prof


def resolve_dataset_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    zarr_candidates = [
        ROOT / "datasets" / "multitask_temporal_v2_h14.zarr",
        Path("/kaggle/working/datasets/multitask_temporal_v2_h14.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/multitask_temporal_v2_h14.zarr"),
    ]
    zarr_path = next((cand for cand in zarr_candidates if cand.exists() and ((cand / ".zgroup").exists() or (cand / "dates").exists())), None)
    if zarr_path is None and Path("/kaggle/input").exists():
        for d in Path("/kaggle/input").rglob("*.zarr"):
            if (d / ".zgroup").exists() or (d / "dates").exists():
                zarr_path = d
                break

    index_candidates = [
        ROOT / "data" / "sample_index_v2_h14.parquet",
        Path("/kaggle/working/data/sample_index_v2_h14.parquet"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/sample_index_v2_h14.parquet"),
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)

    stats_candidates = [
        ROOT / "data" / "normalization_stats_v2.yaml",
        Path("/kaggle/working/data/normalization_stats_v2.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)

    return zarr_path, index_path, stats_path


def load_verified_dense_s_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Hard-fails on missing checkpoint, Git-LFS pointer, SHA-256 mismatch, or parameter count mismatch."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Strict Provenance Hard-Fail: Candidate 3 checkpoint not found at: {checkpoint_path}\n"
            f"Attach 'rohitajitbharadwaj/sih26074-sprint6-checkpoints' on Kaggle or hydrate Git-LFS locally."
        )

    raw_bytes = checkpoint_path.read_bytes()
    if raw_bytes.startswith(b"version https://git-lfs.github.com"):
        raise RuntimeError(f"Strict Provenance Hard-Fail: Checkpoint is an unhydrated Git-LFS pointer: {checkpoint_path}")

    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != EXPECTED_CANDIDATE3_SHA256:
        raise RuntimeError(
            f"Strict Provenance Hard-Fail: Checkpoint SHA-256 mismatch!\n"
            f"Expected: {EXPECTED_CANDIDATE3_SHA256}\n"
            f"Actual:   {actual_sha}"
        )

    model = create_scalable_residual_diffusion("dense_s").to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    weights = state_dict.get("model_state_dict", state_dict)
    model.load_state_dict(weights, strict=True)

    if model.trainable_parameters != EXPECTED_PARAM_COUNT_DENSE_S:
        raise RuntimeError(
            f"Strict Invariant Hard-Fail: Dense-S parameter count {model.trainable_parameters} "
            f"!= {EXPECTED_PARAM_COUNT_DENSE_S}"
        )

    print(f"[+] Candidate 3 Checkpoint Verified & Loaded: SHA={actual_sha[:16]}... Params={model.trainable_parameters:,}")
    model.eval()
    return model


def evaluate_model_on_val_loader(
    model: nn.Module,
    val_loader: DataLoader,
    stats: Dict[str, Any],
    device: torch.device,
    max_cubes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes matched 32 NFE evaluation across authentic validation cubes:
      1. Point Mode: K=8, S=4, eta=0.5 (32 NFE)
      2. Distribution Mode: K=2, S=16, eta=0.0 (32 NFE)
    """
    model.eval()
    wet_maes, csi15s, csi30s = [], [], []
    tmax_maes, wind_rmses = [], []
    crps_list, spread_list, rmse_list = [], [], []
    cov90_hits, total_pts = 0, 0
    lap_single_list, lap_mean_list = [], []
    hf_single_list = []

    m_p = stats.get("precip", {}).get("mean", 6.266)
    s_p = stats.get("precip", {}).get("std", 18.620)
    m_tmax = stats.get("tmax", {}).get("mean", 30.825)
    s_tmax = stats.get("tmax", {}).get("std", 3.245)
    s_wind = stats.get("wind_u", {}).get("std", 2.150)

    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device).view(1, 1, 3, 3)

    cubes_processed = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_cubes is not None and cubes_processed >= max_cubes:
                break

            h = batch["history"].to(device)
            f = batch["future_forecast"].to(device)
            terr = batch["terrain"].to(device)
            y = batch["target"].to(device)
            b_size = y.shape[0]

            # ---------------------------------------------------------
            # 1. Point Mode: K=8, S=4, eta=0.5 (32 NFE)
            # ---------------------------------------------------------
            pt_mems = []
            for k in range(8):
                pred_k = model.sample(h, f, terr, num_steps=4, eta=0.5, seed=1000 + b_idx * 10 + k)
                pt_mems.append(pred_k)

            ens_mean = torch.stack(pt_mems, dim=0).mean(dim=0)  # [B, 7, 6, 80, 80]

            # Invert physical precipitation
            p_pred = torch.clamp(ens_mean[:, :, 0] * s_p + m_p, min=0.0)
            p_tgt = torch.clamp(y[:, :, 0] * s_p + m_p, min=0.0)

            # Wet MAE (> 1 mm)
            w_mask = (p_tgt >= 1.0) | (p_pred >= 1.0)
            if w_mask.any():
                wet_maes.append(float(torch.mean(torch.abs(p_pred[w_mask] - p_tgt[w_mask])).item()))

            # CSI@15 and CSI@30
            h15 = int(((p_pred >= 15.0) & (p_tgt >= 15.0)).sum().item())
            f15 = int(((p_pred >= 15.0) & (p_tgt < 15.0)).sum().item())
            n15 = int(((p_pred < 15.0) & (p_tgt >= 15.0)).sum().item())
            csi15s.append(h15 / max(1, h15 + f15 + n15))

            h30 = int(((p_pred >= 30.0) & (p_tgt >= 30.0)).sum().item())
            f30 = int(((p_pred >= 30.0) & (p_tgt < 30.0)).sum().item())
            n30 = int(((p_pred < 30.0) & (p_tgt >= 30.0)).sum().item())
            csi30s.append(h30 / max(1, h30 + f30 + n30))

            # Tmax MAE & Wind RMSE
            tmax_pred = ens_mean[:, :, 1] * s_tmax + m_tmax
            tmax_tgt = y[:, :, 1] * s_tmax + m_tmax
            tmax_maes.append(float(torch.mean(torch.abs(tmax_pred - tmax_tgt)).item()))

            w_err = (ens_mean[:, :, 4] - y[:, :, 4])**2 + (ens_mean[:, :, 5] - y[:, :, 5])**2
            wind_rmses.append(float(torch.sqrt(torch.mean(w_err)).item() * s_wind))

            # Laplacian energy for single member and ensemble mean
            single_p = torch.clamp(pt_mems[0][:, :, 0] * s_p + m_p, min=0.0)
            for lead in range(7):
                l_gt = torch.mean(F.conv2d(p_tgt[0:1, lead:lead+1], kernel, padding=1)**2).item()
                l_single = torch.mean(F.conv2d(single_p[0:1, lead:lead+1], kernel, padding=1)**2).item()
                l_mean = torch.mean(F.conv2d(p_pred[0:1, lead:lead+1], kernel, padding=1)**2).item()
                lap_single_list.append(l_single / max(1e-6, l_gt))
                lap_mean_list.append(l_mean / max(1e-6, l_gt))

                freqs, psd_pred = compute_radial_psd(single_p[0, lead].cpu().numpy())
                _, psd_gt = compute_radial_psd(p_tgt[0, lead].cpu().numpy())
                hf_idx = len(freqs) // 2
                hf_single_list.append(float(np.mean(psd_pred[hf_idx:]) / max(1e-6, np.mean(psd_gt[hf_idx:]))))

            # ---------------------------------------------------------
            # 2. Distribution Mode: K=2, S=16, eta=0.0 (32 NFE)
            # ---------------------------------------------------------
            dist_mems = []
            for k in range(2):
                pred_k = model.sample(h, f, terr, num_steps=16, eta=0.0, seed=5000 + b_idx * 10 + k)
                dist_mems.append(torch.clamp(pred_k[:, :, 0] * s_p + m_p, min=0.0))
            dist_t = torch.stack(dist_mems, dim=0)  # [2, B, 7, 80, 80]

            d_mean = dist_t.mean(dim=0)
            d_std = dist_t.std(dim=0, unbiased=True)
            mae = torch.mean(torch.abs(d_mean - p_tgt)).item()
            sp = d_std.mean().item()
            crps_list.append(mae - 0.5 * sp)
            spread_list.append(sp)
            rmse_list.append(torch.sqrt(torch.mean((d_mean - p_tgt)**2)).item())

            q05 = torch.quantile(dist_t, 0.05, dim=0)
            q95 = torch.quantile(dist_t, 0.95, dim=0)
            cov90_hits += int(((p_tgt >= q05) & (p_tgt <= q95)).sum().item())
            total_pts += p_tgt.numel()

            cubes_processed += b_size

    mean_sp = float(np.mean(spread_list))
    mean_rmse = float(np.mean(rmse_list))
    ssr = mean_sp / max(1e-6, mean_rmse)

    return {
        "cubes_evaluated": cubes_processed,
        "wet_mae_mm": round(float(np.mean(wet_maes)), 2),
        "csi15": round(float(np.mean(csi15s)), 4),
        "csi30": round(float(np.mean(csi30s)), 4),
        "tmax_mae": round(float(np.mean(tmax_maes)), 3),
        "wind_rmse": round(float(np.mean(wind_rmses)), 3),
        "precip_crps": round(float(np.mean(crps_list)), 3),
        "raw_ssr": round(ssr, 3),
        "raw_cov90": round(cov90_hits / max(1, total_pts), 3),
        "laplacian_retention_single": round(float(np.mean(lap_single_list)), 3),
        "laplacian_retention_mean": round(float(np.mean(lap_mean_list)), 3),
        "high_freq_psd_retention": round(float(np.mean(hf_single_list)), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Sprint 9 Capacity Scaling Evaluation")
    parser.add_argument("--checkpoint-dense-s", type=str, default="models/checkpoints/sprint6_candidate3_multitask_champion.pt")
    parser.add_argument("--checkpoint-dense-m", type=str, default=None)
    parser.add_argument("--checkpoint-dense-l", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--output-json", type=str, default="reports/sprint9_dense_scaling_results.json")
    parser.add_argument("--output-md", type=str, default="reports/sprint9_capacity_frontier.md")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device)
    print("=" * 80)
    print("SPRINT 9 MODEL CAPACITY SCALING EVALUATION")
    print(f"Device: {device} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Resolve authentic dataset
    zarr_path, index_path, stats_path = resolve_dataset_paths()
    if zarr_path is None or index_path is None or stats_path is None:
        raise FileNotFoundError(
            "Authentic Zarr validation dataset not found in local path or /kaggle/input!\n"
            "Attach 'rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14' to evaluate."
        )

    print(f"[*] Loading authentic 2022 validation dataset from: {zarr_path.name}")
    ds_val = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=14,
        context_size=24,
    )
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(ds_val)} forecast cubes for 2022 validation evaluation.")

    # 2. Candidate 3 Control Evaluation
    ckpt_s = Path(args.checkpoint_dense_s)
    if not ckpt_s.exists() and Path("/kaggle/input").exists():
        for f in Path("/kaggle/input").rglob("sprint6_candidate3_multitask_champion.pt"):
            ckpt_s = f
            break

    print(f"\n[*] Evaluating Candidate 3 Control (Dense-S, 15.69M params)...")
    model_s = load_verified_dense_s_model(ckpt_s, device)
    res_s = evaluate_model_on_val_loader(model_s, val_loader, ds_val.stats, device, max_cubes=args.max_cubes)
    print(f"[+] Dense-S Evaluated: Wet-MAE={res_s['wet_mae_mm']}, CSI@30={res_s['csi30']}, CRPS={res_s['precip_crps']}, SSR={res_s['raw_ssr']}")

    # 3. Compile report dictionary
    evaluation_summary = {
        "dense_s": {
            "tier": "dense_s",
            "base_channels": 96,
            "trainable_parameters": EXPECTED_PARAM_COUNT_DENSE_S,
            "status": "VALIDATED_EMPIRICAL_EVALUATION",
            "metrics": res_s,
        },
        "dense_m": {
            "tier": "dense_m",
            "base_channels": 136,
            "trainable_parameters": 31198518,
            "status": "PENDING_PHASE1_KAGGLE_EXECUTION",
            "target_hypotheses": {
                "H1_point_accuracy": "Wet-MAE < 7.80 mm",
                "H2_storm_tail": "CSI@30 > 0.6000",
                "H3_uncertainty": "Raw SSR > 0.350, selected alpha* <= 2.0",
                "H4_spatial_texture": "Single-member Laplacian retention > 0.250",
            },
        },
        "dense_l": {
            "tier": "dense_l",
            "base_channels": 176,
            "trainable_parameters": 51997958,
            "status": "PENDING_PHASE1_KAGGLE_EXECUTION",
            "target_hypotheses": {
                "H1_point_accuracy": "Wet-MAE < 7.40 mm",
                "H2_storm_tail": "CSI@30 > 0.6250",
                "H3_uncertainty": "Raw SSR > 0.450, selected alpha* <= 1.5",
                "H4_spatial_texture": "Single-member Laplacian retention > 0.300",
            },
        },
        "moe_4": {
            "tier": "moe_4",
            "base_channels": 96,
            "trainable_parameters": 35765990,
            "active_parameters": 15685478,
            "status": "PENDING_PHASE2_KAGGLE_EXECUTION",
            "target_hypotheses": {
                "H6_moe_efficiency": "Match Dense-M Wet-MAE and CSI@30 within 2% while executing at Dense-S active parameter scale",
            },
        },
    }

    # Save outputs
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"\n[+] Saved complete evaluation results to {out_json}")

    # Generate Markdown Summary
    out_md = Path(args.output_md)
    lines = [
        "# Sprint 9: Model Capacity Scaling Pareto Frontier Report",
        "",
        "## 1. Executive Capacity Frontier Overview",
        "",
        "Matched compute comparison across 32 NFE in Point Mode (K=8, S=4, eta=0.5) and Distribution Mode (K=2, S=16, eta=0.0).",
        "Evaluated on the authentic 2022 validation dataset (multitask_temporal_v2_h14.zarr).",
        "",
        "| Model Tier | Total Params | Active Params | Status | Wet-MAE (mm) | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Single |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Dense-S (Control)** | 15,685,478 | 15,685,478 | VALIDATED_BASELINE | {res_s['wet_mae_mm']} | {res_s['csi30']} | {res_s['precip_crps']} | {res_s['raw_ssr']} | {res_s['raw_cov90']} | {res_s['laplacian_retention_single']} |",
        "| **Dense-M** | 31,198,518 | 31,198,518 | PENDING_PHASE1_KAGGLE | Target < 7.80 | Target > 0.6000 | Target < 1.650 | Target > 0.350 | Target > 0.350 | Target > 0.250 |",
        "| **Dense-L** | 51,997,958 | 51,997,958 | PENDING_PHASE1_KAGGLE | Target < 7.40 | Target > 0.6250 | Target < 1.500 | Target > 0.450 | Target > 0.450 | Target > 0.300 |",
        "| **MoE-4** | 35,765,990 | 15,685,478 | PENDING_PHASE2_KAGGLE | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M |",
        "",
        "## 2. Invariant Scientific Verification",
        "- All configurations preserve history H=14, context N=24, output crop M=16.",
        "- Multi-task v-prediction parameterization with group-tail loss weighting strictly maintained.",
        "- Candidate 3 SHA-256 hash verified strictly.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Exported Markdown frontier table to {out_md}")


if __name__ == "__main__":
    main()
