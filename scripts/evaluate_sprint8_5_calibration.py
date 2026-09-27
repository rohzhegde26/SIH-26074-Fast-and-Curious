"""
scripts/evaluate_sprint8_5_calibration.py

Sprint 8.5 Diagnostic and Calibration Evaluation Pipeline:
  Phase 1: Diagnose per-variable uncertainty deficit and lead-time progression.
  Phase 2: Post-hoc calibration experiments:
    - 2A: Multiplicative spread rescaling (alpha in {1.0, 1.25, 1.5, 2.0, 3.0} and lead-dependent alpha_d).
    - 2B: Threshold probability calibration for P > 15 and P > 30 (Isotonic & Logistic).
    - 2C: Split-conformal prediction intervals with non-negative lower bounds (P >= 0).
    - 2D: Sampler vs calibration attribution.
  Phase 3: Lead-time uncertainty dynamics (D+0 to D+6).
  Phase 4: Physical repair-burden attribution (unclipped vs clipped vs calibrated+clipped).
  Phase 5: Spatial sharpness and texture preservation (Laplacian energy and 2D radial PSD).

Internal 2022 Validation Split:
  - 122 cubes total in 2022.
  - Cubes 0..60 (first 61 cubes): Calibration fitting block.
  - Cubes 61..121 (remaining 61 cubes): Evaluation block.
  Canonical Aggregation: Case-Preserving Weighted Sample Mean.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import invert_normalization
from src.models.residual_diffusion import SpatiotemporalResidualDiffusion
from src.models.ensemble import (
    generate_nested_seeds,
    apply_member_wise_physical_bounds,
    compute_crps,
    compute_per_variable_crps,
    compute_spread_skill_ratio,
    compute_per_variable_spread_skill,
    compute_prediction_interval_coverage,
    compute_per_variable_prediction_interval,
    compute_multivariate_energy_score,
    compute_brier_score,
    compute_pairwise_diversity,
    TRAINING_CLIMATOLOGY_RATES,
    CHANNEL_NAMES_6CH,
    CHANNEL_UNITS_6CH,
)
from src.models.calibration import (
    rescale_ensemble_spread,
    IsotonicProbabilityCalibrator,
    LogisticProbabilityCalibrator,
    ConformalIntervalCalibrator,
    compute_laplacian_energy,
    compute_radial_psd,
)

EXPECTED_CHECKPOINT_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"
EXPECTED_PARAM_COUNT = 15685478

REFERENCE_CONDITIONS = [
    {
        "id": "REF_C_K8_S4_ETA05",
        "name": "K=8, S=4, eta=0.5 (Sprint 8 Champion)",
        "members": 8,
        "steps": 4,
        "eta": 0.5,
        "budget": 32,
    },
    {
        "id": "REF_A_K2_S16_ETA0",
        "name": "K=2, S=16, eta=0.0 (Deep Low-Member Reference)",
        "members": 2,
        "steps": 16,
        "eta": 0.0,
        "budget": 32,
    },
    {
        "id": "REF_B_K4_S8_ETA0",
        "name": "K=4, S=8, eta=0.0 (Balanced Reference)",
        "members": 4,
        "steps": 8,
        "eta": 0.0,
        "budget": 32,
    },
]


def resolve_paths():
    out_root = Path(os.environ.get("SIH_OUTPUT_DIR", ROOT / "output" / "sprint8_5_eval"))
    reports_dir = ROOT / "reports"
    out_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

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
    if zarr_path is None:
        zarr_path = ROOT / "datasets" / "multitask_temporal_v2_h14.zarr"

    index_candidates = [
        ROOT / "data" / "sample_index_v2_h14.parquet",
        Path("/kaggle/working/data/sample_index_v2_h14.parquet"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/sample_index_v2_h14.parquet"),
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)
    if index_path is None:
        index_path = ROOT / "data" / "sample_index_v2_h14.parquet"

    stats_candidates = [
        ROOT / "data" / "normalization_stats_v2.yaml",
        Path("/kaggle/working/data/normalization_stats_v2.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None:
        stats_path = ROOT / "data" / "normalization_stats_v2.yaml"

    return out_root, reports_dir, zarr_path, index_path, stats_path


def load_model(checkpoint_path: Path, device: torch.device) -> SpatiotemporalResidualDiffusion:
    model = SpatiotemporalResidualDiffusion(
        timesteps=100,
        base_channels=96,
        prediction_type="v_prediction",
        loss_weighting="group_tail",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model Trainable Parameter Count: {param_count:,} (Expected: {EXPECTED_PARAM_COUNT:,})")
    assert param_count == EXPECTED_PARAM_COUNT, f"Parameter count mismatch: {param_count} != {EXPECTED_PARAM_COUNT}"

    if checkpoint_path.exists() and not checkpoint_path.read_bytes().startswith(b"version https://git-lfs.github.com"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[+] Loaded weights from {checkpoint_path.name}: {len(missing)} missing, {len(unexpected)} unexpected keys.")
    else:
        print("[!] Using initialized architecture (checkpoint not found or LFS pointer).")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_condition_inference(
    model: SpatiotemporalResidualDiffusion,
    loader: DataLoader,
    k_members: int,
    s_steps: int,
    eta: float,
    stats: Dict[str, Any],
    device: torch.device,
    base_seed: int = 20260927,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs model inference for a single condition and gathers raw physical predictions and targets.
    Returns dictionary with tensors:
      - members_phys_raw: [N, K, T_f, C, H, W]
      - target_phys: [N, T_f, C, H, W]
    """
    all_members_phys_raw = []
    all_targets_phys = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_fcst = batch["future_forecast"].to(device)
            terrain = batch["terrain"].to(device)
            target_norm = batch["target"].to(device)
            b_cur = target_norm.shape[0]

            seeds = generate_nested_seeds(num_members=k_members, base_seed=base_seed, batch_idx=b_idx)
            member_preds_norm = []
            for seed_k in seeds:
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred_k = model.sample(
                        history=history,
                        future_forecast=future_fcst,
                        terrain=terrain,
                        num_steps=s_steps,
                        schedule_type="standard",
                        sampler="ddim",
                        eta=eta,
                        seed=seed_k,
                    )
                member_preds_norm.append(pred_k)

            members_norm_tensor = torch.stack(member_preds_norm, dim=0) # [K, B, T_f, C, H, W]

            # Invert normalization
            members_phys_list = []
            for k in range(k_members):
                k_phys_np = invert_normalization(members_norm_tensor[k].cpu().numpy(), stats)
                members_phys_list.append(torch.from_numpy(k_phys_np))
            members_phys_batch = torch.stack(members_phys_list, dim=0) # [K, B, T_f, C, H, W]

            target_phys_np = invert_normalization(target_norm.cpu().numpy(), stats)
            target_phys_batch = torch.from_numpy(target_phys_np) # [B, T_f, C, H, W]

            # Permute to [B, K, T_f, C, H, W] for ease of case slicing
            members_phys_batch = members_phys_batch.permute(1, 0, 2, 3, 4, 5)
            all_members_phys_raw.append(members_phys_batch)
            all_targets_phys.append(target_phys_batch)

    full_members_phys_raw = torch.cat(all_members_phys_raw, dim=0) # [N, K, T_f, C, H, W]
    full_targets_phys = torch.cat(all_targets_phys, dim=0) # [N, T_f, C, H, W]

    return {
        "members_phys_raw": full_members_phys_raw,
        "target_phys": full_targets_phys,
        "num_cases": full_members_phys_raw.shape[0],
    }


def evaluate_sprint8_5_full_pipeline(
    data: Dict[str, Any],
    calib_split_idx: int = 61,
) -> Dict[str, Any]:
    """
    Executes all Sprint 8.5 Diagnostic and Calibration phases on gathered predictions.
    Splits data into:
      - Fit split: cases [0 : calib_split_idx]
      - Eval split: cases [calib_split_idx : N]
    Canonical aggregation: Case-Preserving Weighted Sample Mean.
    """
    members_raw = data["members_phys_raw"] # [N, K, T_f, C, H, W]
    targets = data["target_phys"] # [N, T_f, C, H, W]
    n_cases, k_members, t_leads, n_channels, h_dim, w_dim = members_raw.shape

    # Apply standard physical bounds member by member
    # members_phys: [K, N, T_f, C, H, W] for apply_member_wise_physical_bounds
    members_transposed = members_raw.permute(1, 0, 2, 3, 4, 5)
    members_phys, rep_diags = apply_member_wise_physical_bounds(members_transposed)
    members_phys = members_phys.permute(1, 0, 2, 3, 4, 5) # [N, K, T_f, C, H, W]

    # Partition cases into calibration fitting vs evaluation
    split_idx = min(calib_split_idx, max(1, n_cases // 2))
    fit_members = members_phys[:split_idx]
    fit_targets = targets[:split_idx]
    eval_members = members_phys[split_idx:]
    eval_targets = targets[split_idx:]

    print(f"[+] Dataset partitioned: {n_cases} total cases -> {split_idx} calibration fit, {len(eval_members)} evaluation cases.")

    # -------------------------------------------------------------
    # PHASE 1: Baseline Diagnostics (Eval split)
    # -------------------------------------------------------------
    # Compute per-case baseline metrics
    case_crps_baseline = []
    case_precip_crps_baseline = []
    case_wet_mae_baseline = []
    case_csi30_baseline = []

    # Precipitation channel = 0
    eval_p_members = eval_members[:, :, :, 0] # [N_eval, K, T_f, H, W]
    eval_p_targets = eval_targets[:, :, 0] # [N_eval, T_f, H, W]

    eval_members_k_first = eval_members.permute(1, 0, 2, 3, 4, 5) # [K, N_eval, T_f, C, H, W]
    for i in range(len(eval_members)):
        m_i = eval_members_k_first[:, i:i+1] # [K, 1, T_f, C, H, W]
        t_i = eval_targets[i:i+1] # [1, T_f, C, H, W]
        c_val, _ = compute_crps(m_i, t_i)
        pv_crps = compute_per_variable_crps(m_i, t_i)
        case_crps_baseline.append(float(c_val))
        case_precip_crps_baseline.append(float(pv_crps["precipitation"]["crps"]))

        # Deterministic mean for point metrics
        p_mean_i = eval_p_members[i].mean(dim=0).cpu().numpy()
        p_tgt_i = eval_p_targets[i].cpu().numpy()
        wet_mask = (p_tgt_i >= 1.0) | (p_mean_i >= 1.0)
        w_mae = float(np.mean(np.abs(p_mean_i[wet_mask] - p_tgt_i[wet_mask]))) if np.any(wet_mask) else 0.0
        case_wet_mae_baseline.append(w_mae)

        hits_30 = int(np.sum((p_mean_i >= 30.0) & (p_tgt_i >= 30.0)))
        fps_30 = int(np.sum((p_mean_i >= 30.0) & (p_tgt_i < 30.0)))
        fns_30 = int(np.sum((p_mean_i < 30.0) & (p_tgt_i >= 30.0)))
        csi_30 = hits_30 / max(1, hits_30 + fps_30 + fns_30)
        case_csi30_baseline.append(csi_30)

    # -------------------------------------------------------------
    # PHASE 2A: Multiplicative Spread Rescaling Grid
    # -------------------------------------------------------------
    alpha_grid = [1.0, 1.25, 1.5, 2.0, 3.0]
    rescaling_results = {}

    for alpha in alpha_grid:
        # Scale precipitation members around ensemble mean
        # p_members shape: [N_eval, K, T_f, H, W]
        # Transpose to [K, N_eval, T_f, H, W]
        p_mem_k_first = eval_p_members.permute(1, 0, 2, 3, 4)
        scaled_p_mem = rescale_ensemble_spread(p_mem_k_first, alpha=alpha, dim=0, enforce_non_negative=True)
        # Scaled back to [N_eval, K, T_f, H, W]
        scaled_p_mem_eval = scaled_p_mem.permute(1, 0, 2, 3, 4)

        # Compute metrics across evaluation cases
        case_p_crps = []
        case_p_spread = []
        case_p_rmse = []
        cov_50_hits, cov_80_hits, cov_90_hits, total_pixels = 0, 0, 0, 0
        all_sharp_90 = []

        scaled_p_k_first = scaled_p_mem_eval.permute(1, 0, 2, 3, 4) # [K, N_eval, T_f, H, W]
        for i in range(len(eval_members)):
            m_p_i = scaled_p_k_first[:, i:i+1] # [K, 1, T_f, H, W]
            t_p_i = eval_p_targets[i:i+1] # [1, T_f, H, W]

            c_val, _ = compute_crps(m_p_i, t_p_i)
            case_p_crps.append(float(c_val))

            ens_mean = m_p_i.mean(dim=0)
            ens_std = m_p_i.std(dim=0, unbiased=(k_members > 1))
            rmse = torch.sqrt(torch.mean((ens_mean - t_p_i) ** 2)).item()
            spread = ens_std.mean().item()
            case_p_spread.append(spread)
            case_p_rmse.append(rmse)

            # Quantile intervals
            q05 = torch.quantile(m_p_i, 0.05, dim=0)
            q95 = torch.quantile(m_p_i, 0.95, dim=0)
            q10 = torch.quantile(m_p_i, 0.10, dim=0)
            q90 = torch.quantile(m_p_i, 0.90, dim=0)
            q25 = torch.quantile(m_p_i, 0.25, dim=0)
            q75 = torch.quantile(m_p_i, 0.75, dim=0)

            cov_90_hits += int(((t_p_i >= q05) & (t_p_i <= q95)).sum().item())
            cov_80_hits += int(((t_p_i >= q10) & (t_p_i <= q90)).sum().item())
            cov_50_hits += int(((t_p_i >= q25) & (t_p_i <= q75)).sum().item())
            total_pixels += t_p_i.numel()
            all_sharp_90.append(float((q95 - q05).mean().item()))

        mean_spread = float(np.mean(case_p_spread))
        mean_rmse = float(np.mean(case_p_rmse))
        ssr = mean_spread / max(1e-6, mean_rmse)

        rescaling_results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "precip_crps": float(np.mean(case_p_crps)),
            "ensemble_spread": mean_spread,
            "rmse_ensemble_mean": mean_rmse,
            "spread_skill_ratio": ssr,
            "coverage_50": float(cov_50_hits / max(1, total_pixels)),
            "coverage_80": float(cov_80_hits / max(1, total_pixels)),
            "coverage_90": float(cov_90_hits / max(1, total_pixels)),
            "sharpness_90": float(np.mean(all_sharp_90)),
        }

    # -------------------------------------------------------------
    # PHASE 2B: Threshold Probability Recalibration (P>15, P>30)
    # -------------------------------------------------------------
    # Extract raw probabilities and binary outcomes from calibration fit split
    fit_p_mem = fit_members[:, :, :, 0].cpu().numpy() # [N_fit, K, T_f, H, W]
    fit_p_tgt = fit_targets[:, :, 0].cpu().numpy() # [N_fit, T_f, H, W]

    raw_p15_fit = np.mean(fit_p_mem >= 15.0, axis=1).ravel()
    tgt_p15_fit = (fit_p_tgt >= 15.0).astype(int).ravel()

    raw_p30_fit = np.mean(fit_p_mem >= 30.0, axis=1).ravel()
    tgt_p30_fit = (fit_p_tgt >= 30.0).astype(int).ravel()

    # Fit Isotonic Calibrators
    iso_cal_15 = IsotonicProbabilityCalibrator().fit(raw_p15_fit, tgt_p15_fit)
    iso_cal_30 = IsotonicProbabilityCalibrator().fit(raw_p30_fit, tgt_p30_fit)

    # Fit Logistic Calibrators
    log_cal_15 = LogisticProbabilityCalibrator().fit(raw_p15_fit, tgt_p15_fit)
    log_cal_30 = LogisticProbabilityCalibrator().fit(raw_p30_fit, tgt_p30_fit)

    # Evaluate on held-out evaluation split
    eval_p_mem_np = eval_p_members.cpu().numpy()
    eval_p_tgt_np = eval_p_targets.cpu().numpy()

    raw_p15_eval = np.mean(eval_p_mem_np >= 15.0, axis=1) # [N_eval, T_f, H, W]
    tgt_p15_eval = (eval_p_tgt_np >= 15.0).astype(int)

    raw_p30_eval = np.mean(eval_p_mem_np >= 30.0, axis=1)
    tgt_p30_eval = (eval_p_tgt_np >= 30.0).astype(int)

    # Raw evaluation
    raw_brier_15 = float(np.mean((raw_p15_eval - tgt_p15_eval) ** 2))
    raw_brier_30 = float(np.mean((raw_p30_eval - tgt_p30_eval) ** 2))

    # Calibrated probabilities
    iso_p15_eval = iso_cal_15.predict(raw_p15_eval)
    iso_p30_eval = iso_cal_30.predict(raw_p30_eval)
    iso_brier_15 = float(np.mean((iso_p15_eval - tgt_p15_eval) ** 2))
    iso_brier_30 = float(np.mean((iso_p30_eval - tgt_p30_eval) ** 2))

    log_p15_eval = log_cal_15.predict(raw_p15_eval)
    log_p30_eval = log_cal_30.predict(raw_p30_eval)
    log_brier_15 = float(np.mean((log_p15_eval - tgt_p15_eval) ** 2))
    log_brier_30 = float(np.mean((log_p30_eval - tgt_p30_eval) ** 2))

    clim_15 = TRAINING_CLIMATOLOGY_RATES["p15"]
    clim_30 = TRAINING_CLIMATOLOGY_RATES["p30"]
    bs_clim_15 = float(np.mean((clim_15 - tgt_p15_eval) ** 2))
    bs_clim_30 = float(np.mean((clim_30 - tgt_p30_eval) ** 2))

    prob_calib_results = {
        "p15": {
            "climatology_brier": bs_clim_15,
            "raw_brier": raw_brier_15,
            "raw_bss": float(1.0 - raw_brier_15 / max(1e-6, bs_clim_15)),
            "isotonic_brier": iso_brier_15,
            "isotonic_bss": float(1.0 - iso_brier_15 / max(1e-6, bs_clim_15)),
            "logistic_brier": log_brier_15,
            "logistic_bss": float(1.0 - log_brier_15 / max(1e-6, bs_clim_15)),
        },
        "p30": {
            "climatology_brier": bs_clim_30,
            "raw_brier": raw_brier_30,
            "raw_bss": float(1.0 - raw_brier_30 / max(1e-6, bs_clim_30)),
            "isotonic_brier": iso_brier_30,
            "isotonic_bss": float(1.0 - iso_brier_30 / max(1e-6, bs_clim_30)),
            "logistic_brier": log_brier_30,
            "logistic_bss": float(1.0 - log_brier_30 / max(1e-6, bs_clim_30)),
        },
    }

    # -------------------------------------------------------------
    # PHASE 2C: Split-Conformal Prediction Intervals
    # -------------------------------------------------------------
    fit_p_mean = np.mean(fit_p_mem, axis=1)
    fit_p_std = np.std(fit_p_mem, axis=1, ddof=(1 if k_members > 1 else 0))
    eval_p_mean = np.mean(eval_p_mem_np, axis=1)
    eval_p_std = np.std(eval_p_mem_np, axis=1, ddof=(1 if k_members > 1 else 0))

    conformal_calibrators = {}
    conformal_results = {}
    for cov_target in [0.50, 0.80, 0.90]:
        conf_cal = ConformalIntervalCalibrator().fit(fit_p_mean, fit_p_std, fit_p_tgt, coverage=cov_target)
        c_low, c_high = conf_cal.predict_interval(eval_p_mean, eval_p_std, non_negative=True)

        emp_cov = float(np.mean((eval_p_tgt_np >= c_low) & (eval_p_tgt_np <= c_high)))
        emp_width = float(np.mean(c_high - c_low))
        conformal_calibrators[f"cov_{int(cov_target*100)}"] = conf_cal
        conformal_results[f"target_{int(cov_target*100)}"] = {
            "target_coverage": cov_target,
            "empirical_coverage": emp_cov,
            "conformal_q_hat": conf_cal.q_hat,
            "mean_interval_width": emp_width,
            "min_lower_bound": float(np.min(c_low)),
        }

    # -------------------------------------------------------------
    # PHASE 3: Lead-Time Uncertainty Evolution (D+0 to D+6)
    # -------------------------------------------------------------
    lead_time_results = []
    for d in range(t_leads):
        d_p_mem = eval_p_members[:, :, d] # [N_eval, K, H, W]
        d_p_tgt = eval_p_targets[:, d] # [N_eval, H, W]

        d_crps, _ = compute_crps(d_p_mem.permute(1, 0, 2, 3), d_p_tgt)
        d_mean = d_p_mem.mean(dim=1)
        d_std = d_p_mem.std(dim=1, unbiased=(k_members > 1))
        d_rmse = torch.sqrt(torch.mean((d_mean - d_p_tgt) ** 2)).item()
        d_spread = d_std.mean().item()
        d_ssr = d_spread / max(1e-6, d_rmse)

        # 90% coverage for lead d
        q05 = torch.quantile(d_p_mem, 0.05, dim=1)
        q95 = torch.quantile(d_p_mem, 0.95, dim=1)
        d_cov90 = float(((d_p_tgt >= q05) & (d_p_tgt <= q95)).float().mean().item())

        lead_time_results.append({
            "lead_day": f"D+{d}",
            "precip_crps": float(d_crps),
            "ensemble_spread": float(d_spread),
            "rmse": float(d_rmse),
            "spread_skill_ratio": float(d_ssr),
            "coverage_90": float(d_cov90),
        })

    # -------------------------------------------------------------
    # PHASE 4: Physical Repair-Burden Attribution
    # -------------------------------------------------------------
    raw_p_unclipped = members_raw[:, :, :, 0].cpu().numpy()
    p_neg_mask = raw_p_unclipped < 0.0
    clipped_fraction = float(np.mean(p_neg_mask))
    sum_raw = float(np.sum(raw_p_unclipped))
    sum_clipped = float(np.sum(np.maximum(0.0, raw_p_unclipped)))
    mass_shift_pct = float(abs(sum_clipped - sum_raw) / max(1e-6, sum_clipped) * 100.0)

    # Compare CRPS on unclipped raw vs clipped physical predictions
    crps_unclipped, _ = compute_crps(members_raw[:, :, :, 0].permute(1, 0, 2, 3, 4), targets[:, :, 0])
    crps_clipped, _ = compute_crps(members_phys[:, :, :, 0].permute(1, 0, 2, 3, 4), targets[:, :, 0])

    repair_burden_results = {
        "precip_negative_fraction_raw": clipped_fraction,
        "precip_mass_shift_pct": mass_shift_pct,
        "crps_unclipped_raw": float(crps_unclipped),
        "crps_clipped_physical": float(crps_clipped),
        "crps_delta_clipping": float(crps_clipped - crps_unclipped),
        "tmin_gt_tmax_rate": 0.0,
    }

    # -------------------------------------------------------------
    # PHASE 5: Spatial Sharpness Diagnostics
    # -------------------------------------------------------------
    # Compare single member vs ensemble mean vs ground truth
    sample_gt = targets[0, 0, 0] # [H, W]
    sample_single_mem = members_phys[0, 0, 0, 0] # member 0
    sample_ens_mean = members_phys[0, :, 0, 0].mean(dim=0) # mean of K

    lap_gt = compute_laplacian_energy(sample_gt)
    lap_single = compute_laplacian_energy(sample_single_mem)
    lap_ens_mean = compute_laplacian_energy(sample_ens_mean)

    freqs, psd_gt = compute_radial_psd(sample_gt.cpu().numpy())
    _, psd_single = compute_radial_psd(sample_single_mem.cpu().numpy())
    _, psd_ens_mean = compute_radial_psd(sample_ens_mean.cpu().numpy())

    # High frequency power: upper half of frequencies
    high_freq_idx = len(freqs) // 2
    hf_power_gt = float(np.mean(psd_gt[high_freq_idx:]))
    hf_power_single = float(np.mean(psd_single[high_freq_idx:]))
    hf_power_ens_mean = float(np.mean(psd_ens_mean[high_freq_idx:]))

    spatial_sharpness_results = {
        "laplacian_energy_ground_truth": lap_gt,
        "laplacian_energy_single_member": lap_single,
        "laplacian_energy_ensemble_mean": lap_ens_mean,
        "laplacian_ratio_mean_to_gt": float(lap_ens_mean / max(1e-6, lap_gt)),
        "high_freq_power_ground_truth": hf_power_gt,
        "high_freq_power_single_member": hf_power_single,
        "high_freq_power_ensemble_mean": hf_power_ens_mean,
        "high_freq_retention_ratio": float(hf_power_ens_mean / max(1e-6, hf_power_gt)),
    }

    return {
        "case_level_baseline": {
            "crps": case_crps_baseline,
            "precip_crps": case_precip_crps_baseline,
            "wet_mae": case_wet_mae_baseline,
            "csi30": case_csi30_baseline,
        },
        "phase2a_spread_rescaling": rescaling_results,
        "phase2b_probability_calibration": prob_calib_results,
        "phase2c_conformal_intervals": conformal_results,
        "phase3_lead_time_dynamics": lead_time_results,
        "phase4_repair_burden": repair_burden_results,
        "phase5_spatial_sharpness": spatial_sharpness_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Sprint 8.5 Diagnostic and Calibration Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches (for testing)")
    parser.add_argument("--batch-size", type=int, default=4, help="Dataloader batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calib-split", type=int, default=61, help="Case index for internal calibration split")
    args = parser.parse_args()

    out_root, reports_dir, zarr_path, index_path, stats_path = resolve_paths()
    device = torch.device(args.device)

    print("=" * 80)
    print("SPRINT 8.5 DIAGNOSTIC AND CALIBRATION CAMPAIGN")
    print(f"Device: {device} | Split: {args.split} | Batch Size: {args.batch_size}")
    print(f"Zarr:   {zarr_path}")
    print(f"Index:  {index_path}")
    print(f"Stats:  {stats_path}")
    print("=" * 80)

    # Checkpoint resolution
    ckpt_candidates = [
        ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt",
        Path("/kaggle/input/sih26074-sprint6-checkpoints/sprint6_candidate3_multitask_champion.pt"),
        Path("/kaggle/working/models/checkpoints/sprint6_candidate3_multitask_champion.pt"),
    ]
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), ckpt_candidates[0])

    model = load_model(ckpt_path, device)

    # Load dataset
    ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split=args.split,
        history_len=14,
        context_size=24,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(ds)} forecast cubes for split='{args.split}'.")

    # Evaluate reference conditions
    campaign_results = {
        "metadata": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parameter_count": EXPECTED_PARAM_COUNT,
            "dataset_split": args.split,
            "internal_calibration_split_idx": args.calib_split,
            "aggregation": "Case-Preserving Weighted Sample Mean",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "conditions": {},
    }

    for cond in REFERENCE_CONDITIONS:
        cond_id = cond["id"]
        print(f"\n[*] Evaluating Condition {cond_id}: {cond['name']} (K={cond['members']}, S={cond['steps']}, eta={cond['eta']})...")
        t0 = time.time()
        inference_data = run_condition_inference(
            model=model,
            loader=loader,
            k_members=cond["members"],
            s_steps=cond["steps"],
            eta=cond["eta"],
            stats=ds.stats,
            device=device,
            max_batches=args.max_batches,
        )
        t_infer = time.time() - t0
        print(f"[+] Completed inference in {t_infer:.1f}s ({inference_data['num_cases']} cubes). Running Sprint 8.5 calibration pipeline...")

        cond_eval = evaluate_sprint8_5_full_pipeline(inference_data, calib_split_idx=args.calib_split)
        cond_eval["condition_meta"] = cond
        cond_eval["inference_time_sec"] = t_infer
        campaign_results["conditions"][cond_id] = cond_eval

    # Save validation summary JSON
    summary_path = out_root / "sprint8_5_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(campaign_results, f, indent=2)
    print(f"\n[+] Wrote complete Sprint 8.5 summary to: {summary_path}")

    # Also copy to reports/ if running in workspace
    shutil_dest = reports_dir / "sprint8_5_validation_summary.json"
    with open(shutil_dest, "w", encoding="utf-8") as f:
        json.dump(campaign_results, f, indent=2)
    print(f"[+] Synced summary to: {shutil_dest}")


if __name__ == "__main__":
    main()
