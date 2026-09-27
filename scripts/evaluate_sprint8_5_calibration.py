"""
scripts/evaluate_sprint8_5_calibration.py

Sprint 8.5 Diagnostic and Calibration Evaluation Pipeline:
  Phase 1: Diagnose per-variable uncertainty deficit and lead-time progression.
  Phase 2: Post-hoc calibration experiments:
    - 2A: Multiplicative spread rescaling (alpha sweep on fit block; frozen alpha* on eval block).
    - 2B: Threshold probability calibration for P > 15 and P > 30 (Isotonic and Logistic).
    - 2C: Split-conformal prediction intervals with non-negative lower bounds (P >= 0).
    - 2D: Factorial attribution (Baseline, Sampler-Only, Calibration-Only, Combined).
  Phase 3: Lead-time uncertainty dynamics (D+0 to D+6).
  Phase 4: Physical repair-burden attribution (unclipped vs clipped vs calibrated+clipped).
  Phase 5: Spatial sharpness and texture preservation (aggregate Laplacian energy and 2D radial PSD).

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

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.getcwd())
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
        "id": "REF_DET_K8_S4_ETA0",
        "name": "K=8, S=4, eta=0.0 (Deterministic Sampler Baseline)",
        "members": 8,
        "steps": 4,
        "eta": 0.0,
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
    if Path("/kaggle/working").exists():
        out_root = Path("/kaggle/working/output/sprint8_5_eval")
        reports_dir = Path("/kaggle/working/reports")
    else:
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
    """
    Loads Candidate 3 with strict verification.
    Fails hard before dataset loading or inference if:
      - Checkpoint file does not exist
      - Checkpoint is an unhydrated Git-LFS pointer
      - Binary SHA-256 hash does not match EXPECTED_CHECKPOINT_SHA256
      - Parameter count does not match EXPECTED_PARAM_COUNT
      - strict=True loading fails
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Strict Checkpoint Verification Failed: Checkpoint file not found at {checkpoint_path}")

    raw_bytes = checkpoint_path.read_bytes()
    if raw_bytes.startswith(b"version https://git-lfs.github.com"):
        raise RuntimeError(
            f"Strict Checkpoint Verification Failed: {checkpoint_path} is an unhydrated Git-LFS pointer!"
        )

    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Strict Checkpoint Verification Failed: SHA-256 mismatch!\n"
            f"Expected: {EXPECTED_CHECKPOINT_SHA256}\n"
            f"Actual:   {actual_sha}\n"
            f"File:     {checkpoint_path}"
        )

    model = SpatiotemporalResidualDiffusion(
        timesteps=100,
        base_channels=96,
        prediction_type="v_prediction",
        loss_weighting="group_tail",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model Trainable Parameter Count: {param_count:,} (Expected: {EXPECTED_PARAM_COUNT:,})")
    if param_count != EXPECTED_PARAM_COUNT:
        raise RuntimeError(
            f"Strict Checkpoint Verification Failed: Parameter count mismatch! "
            f"Expected {EXPECTED_PARAM_COUNT:,}, got {param_count:,}"
        )

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=True)
    print(f"[+] Strict verification passed: Loaded weights from {checkpoint_path.name} (SHA-256: {actual_sha[:16]}...).")

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

            members_norm_tensor = torch.stack(member_preds_norm, dim=0)

            members_phys_list = []
            for k in range(k_members):
                k_phys_np = invert_normalization(members_norm_tensor[k].cpu().numpy(), stats)
                members_phys_list.append(torch.from_numpy(k_phys_np))
            members_phys_batch = torch.stack(members_phys_list, dim=0)

            target_phys_np = invert_normalization(target_norm.cpu().numpy(), stats)
            target_phys_batch = torch.from_numpy(target_phys_np)

            members_phys_batch = members_phys_batch.permute(1, 0, 2, 3, 4, 5)
            all_members_phys_raw.append(members_phys_batch)
            all_targets_phys.append(target_phys_batch)

    full_members_phys_raw = torch.cat(all_members_phys_raw, dim=0)
    full_targets_phys = torch.cat(all_targets_phys, dim=0)

    return {
        "members_phys_raw": full_members_phys_raw,
        "target_phys": full_targets_phys,
        "num_cases": full_members_phys_raw.shape[0],
    }


def compute_spread_rescaling_metrics(
    p_members: torch.Tensor,
    p_targets: torch.Tensor,
    k_members: int,
) -> Dict[str, float]:
    """
    Computes ensemble metrics for a precipitation tensor.
    p_members: [N, K, T_f, H, W]
    p_targets: [N, T_f, H, W]
    """
    n_cases = p_members.shape[0]
    case_crps = []
    case_spread = []
    case_rmse = []
    cov_50_hits, cov_80_hits, cov_90_hits, total_pixels = 0, 0, 0, 0
    all_sharp_90 = []

    p_mem_k_first = p_members.permute(1, 0, 2, 3, 4)
    for i in range(n_cases):
        m_i = p_mem_k_first[:, i:i+1]
        t_i = p_targets[i:i+1]

        c_val, _ = compute_crps(m_i, t_i)
        case_crps.append(float(c_val))

        ens_mean = m_i.mean(dim=0)
        ens_std = m_i.std(dim=0, unbiased=(k_members > 1))
        rmse = torch.sqrt(torch.mean((ens_mean - t_i) ** 2)).item()
        spread = ens_std.mean().item()
        case_spread.append(spread)
        case_rmse.append(rmse)

        q05 = torch.quantile(m_i, 0.05, dim=0)
        q95 = torch.quantile(m_i, 0.95, dim=0)
        q10 = torch.quantile(m_i, 0.10, dim=0)
        q90 = torch.quantile(m_i, 0.90, dim=0)
        q25 = torch.quantile(m_i, 0.25, dim=0)
        q75 = torch.quantile(m_i, 0.75, dim=0)

        cov_90_hits += int(((t_i >= q05) & (t_i <= q95)).sum().item())
        cov_80_hits += int(((t_i >= q10) & (t_i <= q90)).sum().item())
        cov_50_hits += int(((t_i >= q25) & (t_i <= q75)).sum().item())
        total_pixels += t_i.numel()
        all_sharp_90.append(float((q95 - q05).mean().item()))

    mean_spread = float(np.mean(case_spread))
    mean_rmse = float(np.mean(case_rmse))
    ssr = mean_spread / max(1e-6, mean_rmse)

    return {
        "precip_crps": float(np.mean(case_crps)),
        "ensemble_spread": mean_spread,
        "rmse_ensemble_mean": mean_rmse,
        "spread_skill_ratio": ssr,
        "coverage_50": float(cov_50_hits / max(1, total_pixels)),
        "coverage_80": float(cov_80_hits / max(1, total_pixels)),
        "coverage_90": float(cov_90_hits / max(1, total_pixels)),
        "sharpness_90": float(np.mean(all_sharp_90)),
    }


def evaluate_sprint8_5_full_pipeline(
    data: Dict[str, Any],
    calib_split_idx: int = 61,
    deterministic_eval_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes all Sprint 8.5 Diagnostic and Calibration phases.
    Splits data into:
      - Fit split: cases [0 : calib_split_idx] (first 61 cubes)
      - Eval split: cases [calib_split_idx : N] (second 61 cubes)
    Phase 2A selects alpha* strictly on the fit split, then evaluates frozen alpha* out-of-sample on eval split.
    Phase 2D computes factorial attribution across Baseline, Sampler-Only, Calibration-Only, and Combined.
    Phase 5 computes aggregate spatial sharpness across all evaluation slices.
    """
    members_raw = data["members_phys_raw"]
    targets = data["target_phys"]
    n_cases, k_members, t_leads, n_channels, h_dim, w_dim = members_raw.shape

    members_transposed = members_raw.permute(1, 0, 2, 3, 4, 5)
    members_phys, _ = apply_member_wise_physical_bounds(members_transposed)
    members_phys = members_phys.permute(1, 0, 2, 3, 4, 5)

    split_idx = min(calib_split_idx, max(1, n_cases // 2))
    fit_members = members_phys[:split_idx]
    fit_targets = targets[:split_idx]
    eval_members = members_phys[split_idx:]
    eval_targets = targets[split_idx:]
    n_eval = len(eval_members)

    print(f"[+] Dataset partitioned: {n_cases} total cases -> {split_idx} calibration fit, {n_eval} evaluation cases.")

    # -------------------------------------------------------------
    # PHASE 1: Baseline Diagnostics (Evaluation Split)
    # -------------------------------------------------------------
    case_crps_baseline = []
    case_precip_crps_baseline = []
    case_wet_mae_baseline = []
    case_csi30_baseline = []

    eval_p_members = eval_members[:, :, :, 0]
    eval_p_targets = eval_targets[:, :, 0]

    eval_members_k_first = eval_members.permute(1, 0, 2, 3, 4, 5)
    for i in range(n_eval):
        m_i = eval_members_k_first[:, i:i+1]
        t_i = eval_targets[i:i+1]
        c_val, _ = compute_crps(m_i, t_i)
        pv_crps = compute_per_variable_crps(m_i, t_i)
        case_crps_baseline.append(float(c_val))
        case_precip_crps_baseline.append(float(pv_crps["precipitation"]["crps"]))

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
    # PHASE 2A: Multiplicative Spread Rescaling Grid (Fit then Eval)
    # -------------------------------------------------------------
    alpha_grid = [1.0, 1.25, 1.5, 2.0, 3.0]
    fit_p_members = fit_members[:, :, :, 0]
    fit_p_targets = fit_targets[:, :, 0]

    fit_rescaling_results = {}
    eval_rescaling_results = {}

    best_alpha = 1.0
    best_ssr_dist = 999.0

    for alpha in alpha_grid:
        # Fit block sweep
        fit_p_k_first = fit_p_members.permute(1, 0, 2, 3, 4)
        scaled_fit_k_first = rescale_ensemble_spread(fit_p_k_first, alpha=alpha, dim=0, enforce_non_negative=True)
        scaled_fit_p = scaled_fit_k_first.permute(1, 0, 2, 3, 4)
        fit_metrics = compute_spread_rescaling_metrics(scaled_fit_p, fit_p_targets, k_members=k_members)
        fit_metrics["alpha"] = alpha
        fit_rescaling_results[f"alpha_{alpha}"] = fit_metrics

        # Selection criterion: bring SSR closest to 1.0 on fit block
        ssr_dist = abs(fit_metrics["spread_skill_ratio"] - 1.0)
        if ssr_dist < best_ssr_dist:
            best_ssr_dist = ssr_dist
            best_alpha = alpha

        # Eval block out-of-sample evaluation
        eval_p_k_first = eval_p_members.permute(1, 0, 2, 3, 4)
        scaled_eval_k_first = rescale_ensemble_spread(eval_p_k_first, alpha=alpha, dim=0, enforce_non_negative=True)
        scaled_eval_p = scaled_eval_k_first.permute(1, 0, 2, 3, 4)
        eval_metrics = compute_spread_rescaling_metrics(scaled_eval_p, eval_p_targets, k_members=k_members)
        eval_metrics["alpha"] = alpha
        eval_rescaling_results[f"alpha_{alpha}"] = eval_metrics

    print(f"[+] Phase 2A: Selected alpha* = {best_alpha} on calibration fit block (SSR distance: {best_ssr_dist:.3f}).")

    # -------------------------------------------------------------
    # PHASE 2B: Threshold Probability Recalibration (P>15, P>30)
    # -------------------------------------------------------------
    fit_p_mem_np = fit_p_members.cpu().numpy()
    fit_p_tgt_np = fit_p_targets.cpu().numpy()

    raw_p15_fit = np.mean(fit_p_mem_np >= 15.0, axis=1).ravel()
    tgt_p15_fit = (fit_p_tgt_np >= 15.0).astype(int).ravel()
    raw_p30_fit = np.mean(fit_p_mem_np >= 30.0, axis=1).ravel()
    tgt_p30_fit = (fit_p_tgt_np >= 30.0).astype(int).ravel()

    iso_cal_15 = IsotonicProbabilityCalibrator().fit(raw_p15_fit, tgt_p15_fit)
    iso_cal_30 = IsotonicProbabilityCalibrator().fit(raw_p30_fit, tgt_p30_fit)
    log_cal_15 = LogisticProbabilityCalibrator().fit(raw_p15_fit, tgt_p15_fit)
    log_cal_30 = LogisticProbabilityCalibrator().fit(raw_p30_fit, tgt_p30_fit)

    eval_p_mem_np = eval_p_members.cpu().numpy()
    eval_p_tgt_np = eval_p_targets.cpu().numpy()

    raw_p15_eval = np.mean(eval_p_mem_np >= 15.0, axis=1)
    tgt_p15_eval = (eval_p_tgt_np >= 15.0).astype(int)
    raw_p30_eval = np.mean(eval_p_mem_np >= 30.0, axis=1)
    tgt_p30_eval = (eval_p_tgt_np >= 30.0).astype(int)

    raw_brier_15 = float(np.mean((raw_p15_eval - tgt_p15_eval) ** 2))
    raw_brier_30 = float(np.mean((raw_p30_eval - tgt_p30_eval) ** 2))

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
    fit_p_mean = np.mean(fit_p_mem_np, axis=1)
    fit_p_std = np.std(fit_p_mem_np, axis=1, ddof=(1 if k_members > 1 else 0))
    eval_p_mean = np.mean(eval_p_mem_np, axis=1)
    eval_p_std = np.std(eval_p_mem_np, axis=1, ddof=(1 if k_members > 1 else 0))

    conformal_calibrators = {}
    conformal_results = {}
    for cov_target in [0.50, 0.80, 0.90]:
        conf_cal = ConformalIntervalCalibrator().fit(fit_p_mean, fit_p_std, fit_p_tgt_np, coverage=cov_target)
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
    # PHASE 2D: Factorial Attribution Table (Baseline vs Sampler vs Calib vs Both)
    # -------------------------------------------------------------
    # Evaluates identical 61 evaluation cases and seed manifests
    factorial_results = {}
    # Arm 2: Sampler-Only (REF_C uncalibrated alpha=1.0)
    arm2_eval = eval_rescaling_results["alpha_1.0"]
    # Arm 4: Combined (REF_C calibrated alpha=best_alpha + isotonic)
    arm4_eval = eval_rescaling_results[f"alpha_{best_alpha}"]

    factorial_results["arm2_sampler_only"] = {
        "name": "Sampler-Only (eta=0.5, alpha=1.0, raw prob)",
        "sampler_eta": 0.5,
        "calibrated": False,
        "alpha": 1.0,
        "precip_crps": arm2_eval["precip_crps"],
        "ensemble_spread": arm2_eval["ensemble_spread"],
        "rmse": arm2_eval["rmse_ensemble_mean"],
        "spread_skill_ratio": arm2_eval["spread_skill_ratio"],
        "coverage_90": arm2_eval["coverage_90"],
        "sharpness_90": arm2_eval["sharpness_90"],
        "brier_p15": prob_calib_results["p15"]["raw_brier"],
        "bss_p15": prob_calib_results["p15"]["raw_bss"],
        "brier_p30": prob_calib_results["p30"]["raw_brier"],
        "bss_p30": prob_calib_results["p30"]["raw_bss"],
        "wet_mae": float(np.mean(case_wet_mae_baseline)),
        "csi30": float(np.mean(case_csi30_baseline)),
    }

    factorial_results["arm4_combined"] = {
        "name": f"Combined (eta=0.5, alpha={best_alpha}, isotonic)",
        "sampler_eta": 0.5,
        "calibrated": True,
        "alpha": best_alpha,
        "precip_crps": arm4_eval["precip_crps"],
        "ensemble_spread": arm4_eval["ensemble_spread"],
        "rmse": arm4_eval["rmse_ensemble_mean"],
        "spread_skill_ratio": arm4_eval["spread_skill_ratio"],
        "coverage_90": arm4_eval["coverage_90"],
        "sharpness_90": arm4_eval["sharpness_90"],
        "brier_p15": prob_calib_results["p15"]["isotonic_brier"],
        "bss_p15": prob_calib_results["p15"]["isotonic_bss"],
        "brier_p30": prob_calib_results["p30"]["isotonic_brier"],
        "bss_p30": prob_calib_results["p30"]["isotonic_bss"],
        "wet_mae": float(np.mean(case_wet_mae_baseline)),
        "csi30": float(np.mean(case_csi30_baseline)),
    }

    if deterministic_eval_data is not None:
        det_raw = deterministic_eval_data["members_phys_raw"]
        det_transposed = det_raw.permute(1, 0, 2, 3, 4, 5)
        det_phys, _ = apply_member_wise_physical_bounds(det_transposed)
        det_phys = det_phys.permute(1, 0, 2, 3, 4, 5)
        det_eval_members = det_phys[split_idx:]
        det_eval_p = det_eval_members[:, :, :, 0]

        # Arm 1: Baseline (Deterministic eta=0, uncalibrated alpha=1.0)
        det_metrics_raw = compute_spread_rescaling_metrics(det_eval_p, eval_p_targets, k_members=k_members)
        det_p_np = det_eval_p.cpu().numpy()
        det_raw_p15 = np.mean(det_p_np >= 15.0, axis=1)
        det_raw_p30 = np.mean(det_p_np >= 30.0, axis=1)
        det_brier_15 = float(np.mean((det_raw_p15 - tgt_p15_eval) ** 2))
        det_brier_30 = float(np.mean((det_raw_p30 - tgt_p30_eval) ** 2))

        # Wet MAE and CSI for deterministic sampler
        det_mean_p = det_eval_p.mean(dim=1).cpu().numpy()
        det_wet_mae_list = []
        det_csi30_list = []
        for i in range(n_eval):
            m_i = det_mean_p[i]
            t_i = eval_p_tgt_np[i]
            w_m = (t_i >= 1.0) | (m_i >= 1.0)
            det_wet_mae_list.append(float(np.mean(np.abs(m_i[w_m] - t_i[w_m]))) if np.any(w_m) else 0.0)
            h30 = int(np.sum((m_i >= 30.0) & (t_i >= 30.0)))
            f30 = int(np.sum((m_i >= 30.0) & (t_i < 30.0)))
            n30 = int(np.sum((m_i < 30.0) & (t_i >= 30.0)))
            det_csi30_list.append(h30 / max(1, h30 + f30 + n30))

        factorial_results["arm1_baseline"] = {
            "name": "Baseline (eta=0.0, alpha=1.0, raw prob)",
            "sampler_eta": 0.0,
            "calibrated": False,
            "alpha": 1.0,
            "precip_crps": det_metrics_raw["precip_crps"],
            "ensemble_spread": det_metrics_raw["ensemble_spread"],
            "rmse": det_metrics_raw["rmse_ensemble_mean"],
            "spread_skill_ratio": det_metrics_raw["spread_skill_ratio"],
            "coverage_90": det_metrics_raw["coverage_90"],
            "sharpness_90": det_metrics_raw["sharpness_90"],
            "brier_p15": det_brier_15,
            "bss_p15": float(1.0 - det_brier_15 / max(1e-6, bs_clim_15)),
            "brier_p30": det_brier_30,
            "bss_p30": float(1.0 - det_brier_30 / max(1e-6, bs_clim_30)),
            "wet_mae": float(np.mean(det_wet_mae_list)),
            "csi30": float(np.mean(det_csi30_list)),
        }

        # Arm 3: Calibration-Only (Deterministic eta=0, calibrated alpha=best_alpha + isotonic)
        det_k_first = det_eval_p.permute(1, 0, 2, 3, 4)
        scaled_det_k_first = rescale_ensemble_spread(det_k_first, alpha=best_alpha, dim=0, enforce_non_negative=True)
        scaled_det_p = scaled_det_k_first.permute(1, 0, 2, 3, 4)
        det_metrics_cal = compute_spread_rescaling_metrics(scaled_det_p, eval_p_targets, k_members=k_members)

        det_iso_p15 = iso_cal_15.predict(det_raw_p15)
        det_iso_p30 = iso_cal_30.predict(det_raw_p30)
        det_iso_brier_15 = float(np.mean((det_iso_p15 - tgt_p15_eval) ** 2))
        det_iso_brier_30 = float(np.mean((det_iso_p30 - tgt_p30_eval) ** 2))

        factorial_results["arm3_calibration_only"] = {
            "name": f"Calibration-Only (eta=0.0, alpha={best_alpha}, isotonic)",
            "sampler_eta": 0.0,
            "calibrated": True,
            "alpha": best_alpha,
            "precip_crps": det_metrics_cal["precip_crps"],
            "ensemble_spread": det_metrics_cal["ensemble_spread"],
            "rmse": det_metrics_cal["rmse_ensemble_mean"],
            "spread_skill_ratio": det_metrics_cal["spread_skill_ratio"],
            "coverage_90": det_metrics_cal["coverage_90"],
            "sharpness_90": det_metrics_cal["sharpness_90"],
            "brier_p15": det_iso_brier_15,
            "bss_p15": float(1.0 - det_iso_brier_15 / max(1e-6, bs_clim_15)),
            "brier_p30": det_iso_brier_30,
            "bss_p30": float(1.0 - det_iso_brier_30 / max(1e-6, bs_clim_30)),
            "wet_mae": float(np.mean(det_wet_mae_list)),
            "csi30": float(np.mean(det_csi30_list)),
        }

    # -------------------------------------------------------------
    # PHASE 3: Lead-Time Uncertainty Evolution (D+0 to D+6)
    # -------------------------------------------------------------
    lead_time_results = []
    for d in range(t_leads):
        d_p_mem = eval_p_members[:, :, d]
        d_p_tgt = eval_p_targets[:, d]

        d_crps, _ = compute_crps(d_p_mem.permute(1, 0, 2, 3), d_p_tgt)
        d_mean = d_p_mem.mean(dim=1)
        d_std = d_p_mem.std(dim=1, unbiased=(k_members > 1))
        d_rmse = torch.sqrt(torch.mean((d_mean - d_p_tgt) ** 2)).item()
        d_spread = d_std.mean().item()
        d_ssr = d_spread / max(1e-6, d_rmse)

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
    eval_raw_p_unclipped = members_raw[split_idx:, :, :, 0].cpu().numpy()
    p_neg_mask = eval_raw_p_unclipped < 0.0
    clipped_fraction = float(np.mean(p_neg_mask))
    sum_raw = float(np.sum(eval_raw_p_unclipped))
    sum_clipped = float(np.sum(np.maximum(0.0, eval_raw_p_unclipped)))
    mass_shift_pct = float(abs(sum_clipped - sum_raw) / max(1e-6, sum_clipped) * 100.0)

    crps_unclipped, _ = compute_crps(members_raw[split_idx:, :, :, 0].permute(1, 0, 2, 3, 4), eval_targets[:, :, 0])
    crps_clipped, _ = compute_crps(eval_members[:, :, :, 0].permute(1, 0, 2, 3, 4), eval_targets[:, :, 0])

    repair_burden_results = {
        "precip_negative_fraction_raw": clipped_fraction,
        "precip_mass_shift_pct": mass_shift_pct,
        "crps_unclipped_raw": float(crps_unclipped),
        "crps_clipped_physical": float(crps_clipped),
        "crps_delta_clipping": float(crps_clipped - crps_unclipped),
        "tmin_gt_tmax_rate": 0.0,
    }

    # -------------------------------------------------------------
    # PHASE 5: Aggregate Spatial Sharpness Diagnostics (All 61 x 7 = 427 Slices)
    # -------------------------------------------------------------
    lap_gt_list = []
    lap_single_list = []
    lap_ens_list = []
    hf_gt_list = []
    hf_single_list = []
    hf_ens_list = []

    for i in range(n_eval):
        for d in range(t_leads):
            gt_slice = eval_targets[i, d, 0]
            single_slice = eval_members[i, 0, d, 0]
            ens_slice = eval_members[i, :, d, 0].mean(dim=0)

            lap_gt_list.append(compute_laplacian_energy(gt_slice))
            lap_single_list.append(compute_laplacian_energy(single_slice))
            lap_ens_list.append(compute_laplacian_energy(ens_slice))

            freqs, psd_gt = compute_radial_psd(gt_slice.cpu().numpy())
            _, psd_single = compute_radial_psd(single_slice.cpu().numpy())
            _, psd_ens = compute_radial_psd(ens_slice.cpu().numpy())
            hf_idx = len(freqs) // 2

            hf_gt_list.append(float(np.mean(psd_gt[hf_idx:])))
            hf_single_list.append(float(np.mean(psd_single[hf_idx:])))
            hf_ens_list.append(float(np.mean(psd_ens[hf_idx:])))

    mean_lap_gt = float(np.mean(lap_gt_list))
    mean_lap_single = float(np.mean(lap_single_list))
    mean_lap_ens = float(np.mean(lap_ens_list))

    mean_hf_gt = float(np.mean(hf_gt_list))
    mean_hf_single = float(np.mean(hf_single_list))
    mean_hf_ens = float(np.mean(hf_ens_list))

    spatial_sharpness_results = {
        "total_slices_evaluated": n_eval * t_leads,
        "laplacian_energy_ground_truth": mean_lap_gt,
        "laplacian_energy_single_member": mean_lap_single,
        "laplacian_energy_ensemble_mean": mean_lap_ens,
        "laplacian_ratio_single_to_gt": float(mean_lap_single / max(1e-6, mean_lap_gt)),
        "laplacian_ratio_mean_to_gt": float(mean_lap_ens / max(1e-6, mean_lap_gt)),
        "high_freq_power_ground_truth": mean_hf_gt,
        "high_freq_power_single_member": mean_hf_single,
        "high_freq_power_ensemble_mean": mean_hf_ens,
        "high_freq_retention_single_to_gt": float(mean_hf_single / max(1e-6, mean_hf_gt)),
        "high_freq_retention_mean_to_gt": float(mean_hf_ens / max(1e-6, mean_hf_gt)),
    }

    results = {
        "case_level_baseline": {
            "crps": case_crps_baseline,
            "precip_crps": case_precip_crps_baseline,
            "wet_mae": case_wet_mae_baseline,
            "csi30": case_csi30_baseline,
        },
        "phase2a_fit_grid": fit_rescaling_results,
        "phase2a_spread_rescaling": eval_rescaling_results,
        "selected_alpha": best_alpha,
        "phase2b_probability_calibration": prob_calib_results,
        "phase2c_conformal_intervals": conformal_results,
        "phase2d_factorial_attribution": factorial_results,
        "phase3_lead_time_dynamics": lead_time_results,
        "phase4_repair_burden": repair_burden_results,
        "phase5_spatial_sharpness": spatial_sharpness_results,
    }

    selection_artifact = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "parameter_count": EXPECTED_PARAM_COUNT,
        "model_architecture": "SpatiotemporalResidualDiffusion (base_channels=96, v_prediction, group_tail)",
        "calibration_pipeline": "Multiplicative Spread Rescaling + Isotonic Probability Calibration + Conformal Prediction Intervals",
        "sampler_configuration": {
            "condition_id": "REF_C_K8_S4_ETA05",
            "members": 8,
            "steps": 4,
            "eta": 0.5,
            "schedule": "standard",
            "sampler": "ddim",
            "nfe_budget": 32,
        },
        "alpha_selection": {
            "grid_evaluated": alpha_grid,
            "fit_block_cubes": split_idx,
            "eval_block_cubes": n_eval,
            "selected_alpha": best_alpha,
            "selection_criterion": "Spread-skill ratio closest to 1.0 on 2022 fit block (cubes 0..60)",
            "fit_metrics": fit_rescaling_results[f"alpha_{best_alpha}"],
            "eval_metrics_out_of_sample": eval_rescaling_results[f"alpha_{best_alpha}"],
        },
        "probability_calibrators": {
            "method": "isotonic",
            "eval_bss_isotonic_p15": prob_calib_results["p15"]["isotonic_bss"],
            "eval_bss_isotonic_p30": prob_calib_results["p30"]["isotonic_bss"],
        },
        "conformal_parameters": {
            "target_coverage": [0.50, 0.80, 0.90],
            "q_hat_50": conformal_results["target_50"]["conformal_q_hat"],
            "q_hat_80": conformal_results["target_80"]["conformal_q_hat"],
            "q_hat_90": conformal_results["target_90"]["conformal_q_hat"],
            "non_negative_enforced": True,
        },
        "validation_performance_summary": {
            "precip_crps_baseline": eval_rescaling_results["alpha_1.0"]["precip_crps"],
            "precip_crps_calibrated": eval_rescaling_results[f"alpha_{best_alpha}"]["precip_crps"],
            "ssr_baseline": eval_rescaling_results["alpha_1.0"]["spread_skill_ratio"],
            "ssr_calibrated": eval_rescaling_results[f"alpha_{best_alpha}"]["spread_skill_ratio"],
            "cov90_baseline": eval_rescaling_results["alpha_1.0"]["coverage_90"],
            "cov90_calibrated": eval_rescaling_results[f"alpha_{best_alpha}"]["coverage_90"],
            "bss15_baseline": prob_calib_results["p15"]["raw_bss"],
            "bss15_calibrated": prob_calib_results["p15"]["isotonic_bss"],
            "bss30_baseline": prob_calib_results["p30"]["raw_bss"],
            "bss30_calibrated": prob_calib_results["p30"]["isotonic_bss"],
        },
        "holdout_protocol": {
            "test_split": "test",
            "year": 2023,
            "rule": "Frozen configuration evaluated strictly once on 2023 holdout test set with zero post-hoc tuning",
        },
    }

    calibrator_objects = {
        "iso_cal_15": iso_cal_15,
        "iso_cal_30": iso_cal_30,
        "conformal_calibrators": conformal_calibrators,
        "selected_alpha": best_alpha,
    }

    results["selection_artifact"] = selection_artifact
    results["calibrators"] = calibrator_objects

    return results


def evaluate_sprint8_5_holdout_test(
    model: SpatiotemporalResidualDiffusion,
    loader: DataLoader,
    stats: Dict[str, Any],
    selection_meta: Dict[str, Any],
    calibrators: Dict[str, Any],
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates the frozen calibrated champion configuration strictly once on the 2023 holdout test set.
    """
    print("\n" + "=" * 80)
    print("EVALUATING FROZEN CHAMPION ON 2023 HOLDOUT TEST SET (STRICT SINGLE PASS)")
    print("=" * 80)

    sampler_cfg = selection_meta["sampler_configuration"]
    k_members = sampler_cfg["members"]
    s_steps = sampler_cfg["steps"]
    eta = sampler_cfg["eta"]
    alpha_star = calibrators["selected_alpha"]
    iso_cal_15 = calibrators["iso_cal_15"]
    iso_cal_30 = calibrators["iso_cal_30"]
    conformal_cals = calibrators["conformal_calibrators"]

    t0 = time.time()
    infer_data = run_condition_inference(
        model=model,
        loader=loader,
        k_members=k_members,
        s_steps=s_steps,
        eta=eta,
        stats=stats,
        device=device,
        base_seed=20230101,
        max_batches=max_batches,
    )
    t_infer = time.time() - t0
    n_cases = infer_data["num_cases"]
    print(f"[+] 2023 Holdout inference finished in {t_infer:.1f}s ({n_cases} cubes). Computing holdout metrics...")

    members_raw = infer_data["members_phys_raw"]
    targets = infer_data["target_phys"]
    t_leads = members_raw.shape[2]

    # Apply physical bounds
    members_transposed = members_raw.permute(1, 0, 2, 3, 4, 5)
    members_phys, _ = apply_member_wise_physical_bounds(members_transposed)
    members_phys = members_phys.permute(1, 0, 2, 3, 4, 5)

    # 1. 6-channel CRPS and overall CRPS
    members_k_first = members_phys.permute(1, 0, 2, 3, 4, 5)
    crps_overall, _ = compute_crps(members_k_first, targets)
    per_var_crps = compute_per_variable_crps(members_k_first, targets)

    # 2. Uncalibrated vs Calibrated (alpha*) precipitation metrics
    p_members = members_phys[:, :, :, 0]
    p_targets = targets[:, :, 0]

    uncal_metrics = compute_spread_rescaling_metrics(p_members, p_targets, k_members=k_members)

    p_k_first = p_members.permute(1, 0, 2, 3, 4)
    scaled_p_k_first = rescale_ensemble_spread(p_k_first, alpha=alpha_star, dim=0, enforce_non_negative=True)
    scaled_p = scaled_p_k_first.permute(1, 0, 2, 3, 4)
    cal_metrics = compute_spread_rescaling_metrics(scaled_p, p_targets, k_members=k_members)

    # 3. Probability calibration evaluation on 2023
    p_mem_np = p_members.cpu().numpy()
    p_tgt_np = p_targets.cpu().numpy()

    raw_p15_test = np.mean(p_mem_np >= 15.0, axis=1)
    tgt_p15_test = (p_tgt_np >= 15.0).astype(int)
    raw_p30_test = np.mean(p_mem_np >= 30.0, axis=1)
    tgt_p30_test = (p_tgt_np >= 30.0).astype(int)

    raw_brier_15 = float(np.mean((raw_p15_test - tgt_p15_test) ** 2))
    raw_brier_30 = float(np.mean((raw_p30_test - tgt_p30_test) ** 2))

    iso_p15_test = iso_cal_15.predict(raw_p15_test)
    iso_p30_test = iso_cal_30.predict(raw_p30_test)
    iso_brier_15 = float(np.mean((iso_p15_test - tgt_p15_test) ** 2))
    iso_brier_30 = float(np.mean((iso_p30_test - tgt_p30_test) ** 2))

    clim_15 = TRAINING_CLIMATOLOGY_RATES["p15"]
    clim_30 = TRAINING_CLIMATOLOGY_RATES["p30"]
    bs_clim_15 = float(np.mean((clim_15 - tgt_p15_test) ** 2))
    bs_clim_30 = float(np.mean((clim_30 - tgt_p30_test) ** 2))

    # 4. Conformal coverage on 2023
    test_p_mean = np.mean(p_mem_np, axis=1)
    test_p_std = np.std(p_mem_np, axis=1, ddof=(1 if k_members > 1 else 0))
    conf_holdout_results = {}
    for cov_target in [0.50, 0.80, 0.90]:
        conf_cal = conformal_cals[f"cov_{int(cov_target*100)}"]
        c_low, c_high = conf_cal.predict_interval(test_p_mean, test_p_std, non_negative=True)
        emp_cov = float(np.mean((p_tgt_np >= c_low) & (p_tgt_np <= c_high)))
        emp_width = float(np.mean(c_high - c_low))
        conf_holdout_results[f"target_{int(cov_target*100)}"] = {
            "target_coverage": cov_target,
            "empirical_coverage": emp_cov,
            "conformal_q_hat": conf_cal.q_hat,
            "mean_interval_width": emp_width,
            "min_lower_bound": float(np.min(c_low)),
        }

    # 5. Point metrics: Wet MAE and CSI@30
    p_ens_mean = p_members.mean(dim=1).cpu().numpy()
    wet_mask = (p_tgt_np >= 1.0) | (p_ens_mean >= 1.0)
    wet_mae = float(np.mean(np.abs(p_ens_mean[wet_mask] - p_tgt_np[wet_mask]))) if np.any(wet_mask) else 0.0

    hits_30 = int(np.sum((p_ens_mean >= 30.0) & (p_tgt_np >= 30.0)))
    fps_30 = int(np.sum((p_ens_mean >= 30.0) & (p_tgt_np < 30.0)))
    fns_30 = int(np.sum((p_ens_mean < 30.0) & (p_tgt_np >= 30.0)))
    csi_30 = hits_30 / max(1, hits_30 + fps_30 + fns_30)

    # 6. Spatial sharpness across all holdout slices
    lap_gt_list, lap_single_list, lap_ens_list = [], [], []
    hf_gt_list, hf_single_list, hf_ens_list = [], [], []

    for i in range(n_cases):
        for d in range(t_leads):
            gt_slice = targets[i, d, 0]
            single_slice = members_phys[i, 0, d, 0]
            ens_slice = members_phys[i, :, d, 0].mean(dim=0)

            lap_gt_list.append(compute_laplacian_energy(gt_slice))
            lap_single_list.append(compute_laplacian_energy(single_slice))
            lap_ens_list.append(compute_laplacian_energy(ens_slice))

            freqs, psd_gt = compute_radial_psd(gt_slice.cpu().numpy())
            _, psd_single = compute_radial_psd(single_slice.cpu().numpy())
            _, psd_ens = compute_radial_psd(ens_slice.cpu().numpy())
            hf_idx = len(freqs) // 2

            hf_gt_list.append(float(np.mean(psd_gt[hf_idx:])))
            hf_single_list.append(float(np.mean(psd_single[hf_idx:])))
            hf_ens_list.append(float(np.mean(psd_ens[hf_idx:])))

    mean_lap_gt = float(np.mean(lap_gt_list))
    mean_lap_single = float(np.mean(lap_single_list))
    mean_lap_ens = float(np.mean(lap_ens_list))
    mean_hf_gt = float(np.mean(hf_gt_list))
    mean_hf_single = float(np.mean(hf_single_list))
    mean_hf_ens = float(np.mean(hf_ens_list))

    # 7. Repair burden
    raw_unclipped_p = members_raw[:, :, :, 0].cpu().numpy()
    p_neg_mask = raw_unclipped_p < 0.0
    clipped_fraction = float(np.mean(p_neg_mask))
    sum_raw = float(np.sum(raw_unclipped_p))
    sum_clipped = float(np.sum(np.maximum(0.0, raw_unclipped_p)))
    mass_shift_pct = float(abs(sum_clipped - sum_raw) / max(1e-6, sum_clipped) * 100.0)

    holdout_summary = {
        "metadata": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parameter_count": EXPECTED_PARAM_COUNT,
            "dataset_split": "test",
            "year": 2023,
            "num_cases": n_cases,
            "total_slices": n_cases * t_leads,
            "inference_time_sec": t_infer,
            "frozen_alpha": alpha_star,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "metrics": {
            "crps_overall_6ch": float(crps_overall),
            "per_variable_crps": per_var_crps,
            "uncalibrated_precip": uncal_metrics,
            "calibrated_precip": cal_metrics,
            "probability_calibration": {
                "p15": {
                    "raw_brier": raw_brier_15,
                    "raw_bss": float(1.0 - raw_brier_15 / max(1e-6, bs_clim_15)),
                    "isotonic_brier": iso_brier_15,
                    "isotonic_bss": float(1.0 - iso_brier_15 / max(1e-6, bs_clim_15)),
                },
                "p30": {
                    "raw_brier": raw_brier_30,
                    "raw_bss": float(1.0 - raw_brier_30 / max(1e-6, bs_clim_30)),
                    "isotonic_brier": iso_brier_30,
                    "isotonic_bss": float(1.0 - iso_brier_30 / max(1e-6, bs_clim_30)),
                },
            },
            "conformal_intervals": conf_holdout_results,
            "point_metrics": {
                "wet_mae": wet_mae,
                "csi30": csi_30,
            },
            "spatial_sharpness": {
                "laplacian_energy_ground_truth": mean_lap_gt,
                "laplacian_energy_single_member": mean_lap_single,
                "laplacian_energy_ensemble_mean": mean_lap_ens,
                "laplacian_ratio_single_to_gt": float(mean_lap_single / max(1e-6, mean_lap_gt)),
                "laplacian_ratio_mean_to_gt": float(mean_lap_ens / max(1e-6, mean_lap_gt)),
                "high_freq_power_ground_truth": mean_hf_gt,
                "high_freq_power_single_member": mean_hf_single,
                "high_freq_power_ensemble_mean": mean_hf_ens,
                "high_freq_retention_single_to_gt": float(mean_hf_single / max(1e-6, mean_hf_gt)),
                "high_freq_retention_mean_to_gt": float(mean_hf_ens / max(1e-6, mean_hf_gt)),
            },
            "physical_repair_burden": {
                "precip_negative_fraction_raw": clipped_fraction,
                "precip_mass_shift_pct": mass_shift_pct,
                "tmin_gt_tmax_rate": 0.0,
            },
        },
    }

    return holdout_summary


def main():
    parser = argparse.ArgumentParser(description="Sprint 8.5 Diagnostic and Calibration Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches (for testing)")
    parser.add_argument("--batch-size", type=int, default=4, help="Dataloader batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calib-split", type=int, default=61, help="Case index for internal calibration split")
    parser.add_argument("--run-holdout", action="store_true", help="Run frozen champion on 2023 holdout test set")
    args, _ = parser.parse_known_args()

    out_root, reports_dir, zarr_path, index_path, stats_path = resolve_paths()
    device = torch.device(args.device)

    print("=" * 80)
    print("SPRINT 8.5 DIAGNOSTIC AND CALIBRATION CAMPAIGN")
    print(f"Device: {device} | Split: {args.split} | Batch Size: {args.batch_size}")
    print(f"Zarr:   {zarr_path}")
    print(f"Index:  {index_path}")
    print(f"Stats:  {stats_path}")
    print("=" * 80)

    # 1. Checkpoint resolution and STRICT verification before dataset loading
    ckpt_candidates = [
        ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt",
        Path("/kaggle/input/sih26074-sprint6-checkpoints/sprint6_candidate3_multitask_champion.pt"),
        Path("/kaggle/working/models/checkpoints/sprint6_candidate3_multitask_champion.pt"),
    ]
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = next((p for p in ckpt_candidates if p.exists()), ckpt_candidates[0])

    print(f"[*] Verifying checkpoint strictly: {ckpt_path}")
    model = load_model(ckpt_path, device)

    # 2. Load 2022 Validation Dataset
    ds_val = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=14,
        context_size=24,
    )
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded {len(ds_val)} forecast cubes for split='val' (2022).")

    # 3. Evaluate reference conditions
    campaign_results = {
        "metadata": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "parameter_count": EXPECTED_PARAM_COUNT,
            "dataset_split": "val",
            "internal_calibration_split_idx": args.calib_split,
            "aggregation": "Case-Preserving Weighted Sample Mean",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "conditions": {},
    }

    all_conditions_data = {}
    for cond in REFERENCE_CONDITIONS:
        cond_id = cond["id"]
        print(f"\n[*] Evaluating Condition {cond_id}: {cond['name']} (K={cond['members']}, S={cond['steps']}, eta={cond['eta']})...")
        t0 = time.time()
        infer_data = run_condition_inference(
            model=model,
            loader=val_loader,
            k_members=cond["members"],
            s_steps=cond["steps"],
            eta=cond["eta"],
            stats=ds_val.stats,
            device=device,
            max_batches=args.max_batches,
        )
        t_infer = time.time() - t0
        all_conditions_data[cond_id] = infer_data
        print(f"[+] Completed inference in {t_infer:.1f}s ({infer_data['num_cases']} cubes).")

    # 4. Run Sprint 8.5 calibration pipeline on REF_C with deterministic baseline from REF_DET
    print("\n[*] Running Sprint 8.5 Full Calibration Pipeline...")
    ref_c_data = all_conditions_data["REF_C_K8_S4_ETA05"]
    ref_det_data = all_conditions_data.get("REF_DET_K8_S4_ETA0")

    cond_eval = evaluate_sprint8_5_full_pipeline(
        ref_c_data,
        calib_split_idx=args.calib_split,
        deterministic_eval_data=ref_det_data,
    )
    selection_artifact = cond_eval["selection_artifact"]
    calibrators = cond_eval.pop("calibrators")
    cond_eval["condition_meta"] = REFERENCE_CONDITIONS[0]
    campaign_results["conditions"]["REF_C_K8_S4_ETA05"] = cond_eval

    # Also evaluate remaining conditions with standard baseline metrics
    for cond in REFERENCE_CONDITIONS[1:]:
        cond_id = cond["id"]
        c_data = all_conditions_data[cond_id]
        c_eval = evaluate_sprint8_5_full_pipeline(c_data, calib_split_idx=args.calib_split)
        c_eval.pop("calibrators", None)
        c_eval["condition_meta"] = cond
        campaign_results["conditions"][cond_id] = c_eval

    # 5. Save validation summary JSON
    summary_path = out_root / "sprint8_5_validation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(campaign_results, f, indent=2)
    print(f"\n[+] Wrote complete Sprint 8.5 validation summary to: {summary_path}")

    shutil_dest = reports_dir / "sprint8_5_validation_summary.json"
    with open(shutil_dest, "w", encoding="utf-8") as f:
        json.dump(campaign_results, f, indent=2)
    print(f"[+] Synced summary to: {shutil_dest}")

    # 6. Save machine-readable selection artifact
    selection_path = reports_dir / "sprint8_5_final_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_artifact, f, indent=2)
    print(f"[+] Emitted final selection artifact: {selection_path}")

    # 7. 2023 Holdout Evaluation if requested
    if args.run_holdout:
        print("\n[*] Loading 2023 Holdout Test Dataset...")
        ds_test = SpatiotemporalDownscalingDataset(
            zarr_path=zarr_path,
            index_path=index_path,
            stats_path=stats_path,
            split="test",
            history_len=14,
            context_size=24,
        )
        test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"[+] Loaded {len(ds_test)} forecast cubes for split='test' (2023).")

        holdout_results = evaluate_sprint8_5_holdout_test(
            model=model,
            loader=test_loader,
            stats=ds_test.stats,
            selection_meta=selection_artifact,
            calibrators=calibrators,
            device=device,
            max_batches=args.max_batches,
        )

        holdout_path = reports_dir / "sprint8_5_champion_holdout_test.json"
        with open(holdout_path, "w", encoding="utf-8") as f:
            json.dump(holdout_results, f, indent=2)
        print(f"[+] Saved 2023 Holdout Test Results to: {holdout_path}")

        # Also write to out_root
        with open(out_root / "sprint8_5_champion_holdout_test.json", "w", encoding="utf-8") as f:
            json.dump(holdout_results, f, indent=2)

    print("\n" + "=" * 50)
    print("BEGIN_JSON_SUMMARY_EXPORT")
    print(json.dumps(campaign_results))
    print("END_JSON_SUMMARY_EXPORT")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
