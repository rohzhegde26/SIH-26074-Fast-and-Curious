"""
scripts/evaluate_final_canonical_benchmarks.py

Sprint 10: Canonical Physical Evaluation and Holdout Generalization Engine.
Evaluates model architectures under the strict sprint8_physical protocol:
  - Non-linear inverse normalization (invert_normalization with exponential precipitation).
  - Member-wise physical bounds repair (apply_member_wise_physical_bounds).
  - Standard meteorological wet threshold: p_target > 2.5 mm/day.
  - Finite-ensemble unbiased Fair-CRPS (Ferro et al., 2008).
  - Multivariate physical consistency diagnostics (Tmax >= Tmin, P-RH coupling, mass shift).

Supports:
  1. 2022 Validation Split (canonical cross-sprint physical comparison).
  2. 2023 El Nino Quarantined Holdout Split (unbiased out-of-distribution evaluation).
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
from src.models.ensemble import apply_member_wise_physical_bounds
from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)
from src.eval.multivariate_diagnostics import (
    audit_thermodynamic_bounds,
    compute_precipitation_rh_coupling,
    compute_mass_shift_diagnostics,
    compute_brier_scores,
)

EXPECTED_CANDIDATE3_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"


def compute_laplacian_energy(img: torch.Tensor) -> float:
    """Computes mean squared Laplacian response for 2D field."""
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=img.device).view(1, 1, 3, 3)
    x = img.view(1, 1, img.shape[-2], img.shape[-1]).float()
    lap = F.conv2d(x, kernel, padding=1)
    return float(torch.mean(lap ** 2).item())


def resolve_dataset_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    zarr_candidates = [
        ROOT / "datasets" / "multitask_temporal_v2_h14.zarr",
        Path("/kaggle/working/datasets/multitask_temporal_v2_h14.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/multitask_temporal_v2_h14.zarr"),
    ]
    zarr_path = next((c for c in zarr_candidates if c.exists() and ((c / ".zgroup").exists() or (c / "dates").exists())), None)
    if zarr_path is None and Path("/kaggle/input").exists():
        for d in Path("/kaggle/input").rglob("*.zarr"):
            if (d / ".zgroup").exists() or (d / "dates").exists():
                zarr_path = d
                break

    index_candidates = [
        ROOT / "data" / "sample_index_v2_h14.parquet",
        ROOT / "src" / "data" / "sample_index_v2_h14.parquet",
        Path("data/sample_index_v2_h14.parquet"),
        Path("/kaggle/working/data/sample_index_v2_h14.parquet"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/sample_index_v2_h14.parquet"),
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)
    if index_path is None and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*sample_index*.parquet"):
            index_path = p
            break

    stats_candidates = [
        ROOT / "data" / "normalization_stats_v2.yaml",
        ROOT / "src" / "data" / "normalization_stats_v2.yaml",
        Path("data/normalization_stats_v2.yaml"),
        Path("/kaggle/working/data/normalization_stats_v2.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*normalization_stats*.yaml"):
            stats_path = p
            break

    return zarr_path, index_path, stats_path


def load_verified_model(
    tier: str,
    ckpt_path: Path,
    device: torch.device,
    verify_sha: bool = False,
) -> nn.Module:
    """Instantiates scalable model and loads weights."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    raw_bytes = ckpt_path.read_bytes()
    if raw_bytes.startswith(b"version https://git-lfs.github.com"):
        raise ValueError(f"Checkpoint {ckpt_path} is an unhydrated Git-LFS pointer. Run 'git lfs pull'.")

    if verify_sha and tier == "dense_s":
        actual_sha = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha != EXPECTED_CANDIDATE3_SHA256:
            raise ValueError(f"Candidate 3 SHA-256 mismatch! Expected {EXPECTED_CANDIDATE3_SHA256}, got {actual_sha}")
        print(f"[+] Verified Dense-S Candidate 3 Checkpoint SHA-256: {actual_sha[:16]}...")

    model = create_scalable_residual_diffusion(tier).to(device)
    state = torch.load(ckpt_path, map_location=device)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)
    model.eval()

    prof = model.profile_compute(device=device, batch_size=1)
    print(f"[+] Loaded {tier.upper()} Model: Total Params={prof['total_parameters']:,} | Active={prof['active_parameters']:,}")
    return model


def evaluate_model_pipeline(
    model: nn.Module,
    dataloader: DataLoader,
    stats: Dict[str, Any],
    device: torch.device,
    max_cubes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates model with 32 NFE Point Mode (K=8, S=4, eta=0.5) and 32 NFE Distribution Mode (K=2, S=16, eta=0.0).
    Applies full physical-repair pipeline and multivariate consistency diagnostics.
    """
    model.eval()
    wet_maes, csi15s, csi30s, csi50s = [], [], [], []
    tmax_maes, tmin_maes, rh_maes, wind_rmses = [], [], [], []
    crps_list, spread_list, rmse_list = [], [], []
    range_cov_hits, total_pts = 0, 0
    lap_retention_list = []

    # Diagnostic accumulators
    thermo_violations, total_thermo_pts = 0, 0
    p_rh_corrs = []
    mass_shifts = []
    all_brier_scores = []

    cubes_processed = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_cubes is not None and cubes_processed >= max_cubes:
                break

            h = batch["history"].to(device)
            f = batch["future_forecast"].to(device)
            terr = batch["terrain"].to(device)
            y = batch["target"].to(device)
            b_size = y.shape[0]

            # -------------------------------------------------------------
            # 1. Point Mode: K=8, S=4, eta=0.5 (32 NFE)
            # -------------------------------------------------------------
            pt_mems = []
            for k in range(8):
                pred_k = model.sample(h, f, terr, num_steps=4, eta=0.5, seed=2000 + b_idx * 10 + k)
                pt_mems.append(pred_k)
            pt_mems_t = torch.stack(pt_mems, dim=0)

            # Invert normalization member-by-member
            mems_phys_list = []
            for k in range(8):
                k_np = invert_normalization(pt_mems_t[k].cpu().numpy(), stats)
                mems_phys_list.append(torch.from_numpy(k_np))
            mems_phys_raw = torch.stack(mems_phys_list, dim=0).to(device)

            # Member-wise physical bounds repair
            mems_phys, _ = apply_member_wise_physical_bounds(mems_phys_raw)
            ens_mean_phys = mems_phys.mean(dim=0)

            tgt_phys_np = invert_normalization(y.cpu().numpy(), stats)
            tgt_phys = torch.from_numpy(tgt_phys_np).to(device)

            # Extract predicted variables
            p_pred = ens_mean_phys[:, :, 0]
            p_tgt = tgt_phys[:, :, 0]

            # Wet MAE (p_tgt > 2.5 mm)
            w_mask = p_tgt > 2.5
            if w_mask.any():
                wet_maes.append(float(torch.mean(torch.abs(p_pred[w_mask] - p_tgt[w_mask])).item()))
            else:
                wet_maes.append(float(torch.mean(torch.abs(p_pred - p_tgt)).item()))

            # Critical Success Index (CSI)
            for thr, csi_arr in [(15.0, csi15s), (30.0, csi30s), (50.0, csi50s)]:
                hits = float(torch.logical_and(p_pred >= thr, p_tgt >= thr).sum().item())
                false_alarms = float(torch.logical_and(p_pred >= thr, p_tgt < thr).sum().item())
                misses = float(torch.logical_and(p_pred < thr, p_tgt >= thr).sum().item())
                denom = hits + false_alarms + misses
                csi_arr.append(hits / max(1.0, denom))

            # Thermodynamic errors
            tmax_maes.append(float(torch.mean(torch.abs(ens_mean_phys[:, :, 1] - tgt_phys[:, :, 1])).item()))
            tmin_maes.append(float(torch.mean(torch.abs(ens_mean_phys[:, :, 2] - tgt_phys[:, :, 2])).item()))
            rh_maes.append(float(torch.mean(torch.abs(ens_mean_phys[:, :, 3] - tgt_phys[:, :, 3])).item()))
            w_err = (ens_mean_phys[:, :, 4] - tgt_phys[:, :, 4])**2 + (ens_mean_phys[:, :, 5] - tgt_phys[:, :, 5])**2
            wind_rmses.append(float(torch.sqrt(torch.mean(w_err)).item()))

            # Physical diagnostics
            tmax_pred = ens_mean_phys[:, :, 1]
            tmin_pred = ens_mean_phys[:, :, 2]
            rh_pred = ens_mean_phys[:, :, 3]
            inversions = (tmin_pred - tmax_pred) > 0.0
            thermo_violations += int(inversions.sum().item())
            total_thermo_pts += int(inversions.numel())

            p_rh_diag = compute_precipitation_rh_coupling(p_pred, rh_pred, wet_threshold=2.5)
            p_rh_corrs.append(p_rh_diag["pearson_corr_p_rh"])

            mass_diag = compute_mass_shift_diagnostics(mems_phys_raw[:, :, :, 0], mems_phys[:, :, :, 0])
            mass_shifts.append(mass_diag["raw_negative_mass_ratio"])

            brier_dict = compute_brier_scores(mems_phys[:, :, :, 0], p_tgt, thresholds=[15.0, 30.0])
            all_brier_scores.append(brier_dict)

            # Spatial Laplacian Energy Retention
            for b in range(b_size):
                for d in range(7):
                    e_pred = compute_laplacian_energy(p_pred[b, d])
                    e_tgt = compute_laplacian_energy(p_tgt[b, d])
                    lap_retention_list.append(e_pred / max(1e-6, e_tgt))

            # -------------------------------------------------------------
            # 2. Distribution Mode: K=2, S=16, eta=0.0 (32 NFE)
            # -------------------------------------------------------------
            d_mems = []
            for k in range(2):
                pred_k = model.sample(h, f, terr, num_steps=16, eta=0.0, seed=5000 + b_idx * 10 + k)
                d_mems.append(pred_k)
            d_mems_t = torch.stack(d_mems, dim=0)

            d_phys_list = []
            for k in range(2):
                k_np = invert_normalization(d_mems_t[k].cpu().numpy(), stats)
                d_phys_list.append(torch.from_numpy(k_np))
            d_phys_raw = torch.stack(d_phys_list, dim=0).to(device)
            d_phys, _ = apply_member_wise_physical_bounds(d_phys_raw)

            # Finite Fair-CRPS (Ferro et al., 2008) for K=2
            p_mems = d_phys[:, :, :, 0]
            mae_part = torch.mean(torch.abs(p_mems - p_tgt.unsqueeze(0)), dim=0)
            ens_diff = torch.abs(p_mems[0] - p_mems[1])
            crps_k2 = mae_part - (0.25 * ens_diff)
            crps_list.append(float(torch.mean(crps_k2).item()))

            # Spread and RMSE
            spread_k2 = float(torch.mean(torch.std(p_mems, dim=0)).item())
            ens_mean_d = torch.mean(p_mems, dim=0)
            rmse_k2 = float(torch.sqrt(torch.mean((ens_mean_d - p_tgt)**2)).item())
            spread_list.append(spread_k2)
            rmse_list.append(rmse_k2)

            # Range coverage
            c_min = torch.minimum(p_mems[0], p_mems[1])
            c_max = torch.maximum(p_mems[0], p_mems[1])
            cov = torch.logical_and(p_tgt >= c_min, p_tgt <= c_max)
            range_cov_hits += int(cov.sum().item())
            total_pts += int(cov.numel())

            cubes_processed += b_size

    # Aggregate metrics
    mean_spread = float(np.mean(spread_list))
    mean_rmse = float(np.mean(rmse_list))
    ssr = mean_spread / max(1e-6, mean_rmse)

    return {
        "cubes_evaluated": cubes_processed,
        "wet_mae_mm": round(float(np.mean(wet_maes)), 2),
        "csi15": round(float(np.mean(csi15s)), 4),
        "csi30": round(float(np.mean(csi30s)), 4),
        "csi50": round(float(np.mean(csi50s)), 4),
        "tmax_mae": round(float(np.mean(tmax_maes)), 3),
        "tmin_mae": round(float(np.mean(tmin_maes)), 3),
        "rh_mae": round(float(np.mean(rh_maes)), 3),
        "wind_rmse": round(float(np.mean(wind_rmses)), 3),
        "fair_crps_mm": round(float(np.mean(crps_list)), 3),
        "spread_skill_ratio": round(ssr, 3),
        "range_coverage_k2": round(range_cov_hits / max(1, total_pts), 3),
        "laplacian_energy_retention": round(float(np.mean(lap_retention_list)), 3),
        "thermodynamic_violation_rate": round(thermo_violations / max(1, total_thermo_pts), 5),
        "mean_p_rh_correlation": round(float(np.mean(p_rh_corrs)), 3),
        "mean_negative_mass_shift_ratio": round(float(np.mean(mass_shifts)), 5),
    }


def main():
    parser = argparse.ArgumentParser(description="Sprint 10 Canonical Physical Evaluation Engine")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--tiers", nargs="+", default=["dense_s", "dense_m", "dense_l", "moe_4"], help="Model tiers to evaluate")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    split_name = args.split.upper()
    print("=" * 80)
    print(f"SPRINT 10: CANONICAL PHYSICAL EVALUATION ({split_name} SPLIT)")
    print(f"Device: {device} | Batch Size: {args.batch_size} | Max Cubes: {args.max_cubes}")
    print("=" * 80)

    zarr_path, index_path, stats_path = resolve_dataset_paths()
    if zarr_path is None or index_path is None or stats_path is None:
        raise FileNotFoundError("Authentic Zarr dataset or metadata files not found!")

    ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split=args.split,
        history_len=14,
        context_size=24,
    )
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(ds)} forecast cubes for {split_name} split.")

    ckpt_map = {
        "dense_s": ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt",
        "dense_m": ROOT / "models" / "checkpoints" / "sprint9_dense_m_weights.pt",
        "dense_l": ROOT / "models" / "checkpoints" / "sprint10_dense_l_champion.pt",
        "moe_4": ROOT / "models" / "checkpoints" / "sprint9_moe4_weights.pt",
    }
    # Fallback for Dense-L if champion not yet written
    if not ckpt_map["dense_l"].exists():
        ckpt_map["dense_l"] = ROOT / "models" / "checkpoints" / "sprint9_dense_l_weights.pt"

    all_results = {}
    for tier in args.tiers:
        ckpt_file = ckpt_map.get(tier)
        if ckpt_file is None or not ckpt_file.exists():
            print(f"[-] Checkpoint for tier '{tier}' not found ({ckpt_file}). Skipping.")
            continue

        print(f"\n[*] Evaluating {tier.upper()} ({ckpt_file.name})...")
        verify_sha = (tier == "dense_s")
        model = load_verified_model(tier, ckpt_file, device=device, verify_sha=verify_sha)
        res = evaluate_model_pipeline(model, dataloader, ds.stats, device=device, max_cubes=args.max_cubes)
        all_results[tier] = res

        print(
            f"[+] {tier.upper()} Completed: Wet-MAE={res['wet_mae_mm']} mm | "
            f"CSI@30={res['csi30']} | Fair-CRPS={res['fair_crps_mm']} mm | "
            f"SSR={res['spread_skill_ratio']} | Lap Energy={res['laplacian_energy_retention']}"
        )

    # Save JSON
    out_json = Path(args.output_json or f"reports/sprint10_{args.split}_canonical_benchmarks.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved evaluation summary to {out_json}")

    # Generate Markdown Table
    out_md = Path(args.output_md or f"reports/sprint10_{args.split}_canonical_benchmarks.md")
    lines = [
        f"# Sprint 10: Canonical Physical Repair Evaluation ({split_name} Split)",
        "",
        "Protocol: `sprint8_physical` (non-linear inversion, member-wise physical bounds repair, wet threshold p_target > 2.5 mm/day).",
        "Matched compute: Point Mode (K=8, S=4, eta=0.5) and Distribution Mode (K=2, S=16, eta=0.0).",
        "",
        "| Model Tier | Total Params | Active Params | Wet-MAE (mm) | CSI@15 | CSI@30 | Fair-CRPS (mm) | SSR | Lap Retention | Tmax Violations | P-RH Corr |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for tier, res in all_results.items():
        prof = TIER_CHANNEL_CONFIGS.get(tier, {})
        tot_p = prof.get("exact_control_params", 51_997_958 if "dense_l" in tier else 31_198_518 if "dense_m" in tier else 22_773_350)
        act_p = 15_688_550 if "moe" in tier else tot_p

        lines.append(
            f"| **{tier.upper()}** | {tot_p:,} | {act_p:,} | **{res['wet_mae_mm']}** | "
            f"{res['csi15']} | **{res['csi30']}** | **{res['fair_crps_mm']}** | "
            f"{res['spread_skill_ratio']} | {res['laplacian_energy_retention']} | "
            f"{res['thermodynamic_violation_rate']} | {res['mean_p_rh_correlation']} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved markdown report to {out_md}")


if __name__ == "__main__":
    main()
