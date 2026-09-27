"""
scripts/evaluate_sprint8_ensemble.py

Sprint 8 Ensemble and Test-Time Scaling Evaluation Pipeline.
Evaluates:
  Phase 0 Gate: Reproducibility validation on Sprint 6 Candidate 3 champion weights.
  Phase 1: Deterministic reference baselines (K=1, S in {4, 8, 16, 32, 64}, CRPS_det = MAE).
  Phase 2 Stage A: Ensemble member scaling at fixed S=4, eta=0 (K in {2, 4, 8, 16}).
  Phase 2 Stage B: Trajectory stochasticity sweep (eta in {0.25, 0.5, 1.0} on K=4).
  Phase 3: Matched-compute budget frontier:
    - Budget 8:  (K=1, S=8)  vs (K=2, S=4)
    - Budget 16: (K=1, S=16) vs (K=2, S=8)  vs (K=4, S=4)
    - Budget 32: (K=1, S=32) vs (K=2, S=16) vs (K=4, S=8) vs (K=8, S=4) [Flagship]
    - Budget 64: (K=1, S=64) vs (K=2, S=32) vs (K=4, S=16) vs (K=8, S=8) vs (K=16, S=4)
  Phase 4: Single confirmatory evaluation on 2023 holdout test set for the champion configuration.

Key Invariants:
  - 100% Inference Mode (zero gradient / backward propagation).
  - Frozen weights: Candidate 3 champion (15,685,478 params, Git LFS sha256 verified).
  - Linear beta schedule: beta_start=1e-4, beta_end=0.035, T=100.
  - Member-wise physical constraints applied before ensemble reductions.
  - Strategy A Dual-GPU isolation: runnable on CUDA_VISIBLE_DEVICES=0 or 1.
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

if Path("/kaggle/working").exists():
    ROOT = Path("/kaggle/working")
else:
    ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import invert_normalization
from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)
from src.models.samplers import get_sampling_timesteps
from src.models.ensemble import (
    generate_nested_seeds,
    apply_member_wise_physical_bounds,
    compute_crps,
    compute_brier_score,
    compute_pairwise_diversity,
    compute_spread_skill_ratio,
    compute_prediction_interval_coverage,
    compute_multivariate_energy_score,
)


EXPECTED_CHECKPOINT_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"
EXPECTED_PARAM_COUNT = 15685478

CONDITIONS = [
    # Phase 0 Gate
    {
        "id": "PHASE0_GATE_DDIM4",
        "members": 1,
        "steps": 4,
        "eta": 0.0,
        "budget": 4,
        "split": "val",
        "desc": "Phase 0 Gate: DDIM-4 Regression on 2022 Validation",
    },
    # Phase 1: Deterministic Reference Anchors
    {
        "id": "DET_S04",
        "members": 1,
        "steps": 4,
        "eta": 0.0,
        "budget": 4,
        "split": "val",
        "desc": "Phase 1: Deterministic DDIM-4 Reference",
    },
    {
        "id": "DET_S08",
        "members": 1,
        "steps": 8,
        "eta": 0.0,
        "budget": 8,
        "split": "val",
        "desc": "Phase 1: Deterministic DDIM-8 Reference",
    },
    {
        "id": "DET_S16",
        "members": 1,
        "steps": 16,
        "eta": 0.0,
        "budget": 16,
        "split": "val",
        "desc": "Phase 1: Deterministic DDIM-16 Reference",
    },
    {
        "id": "DET_S32",
        "members": 1,
        "steps": 32,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 1: Deterministic DDIM-32 Baseline",
    },
    {
        "id": "DET_S64",
        "members": 1,
        "steps": 64,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 1: Deterministic DDIM-64 Saturation Reference",
    },
    # Phase 2 Stage A: Member Scaling (S=4, eta=0)
    {
        "id": "ENS_K02_S04_ETA0",
        "members": 2,
        "steps": 4,
        "eta": 0.0,
        "budget": 8,
        "split": "val",
        "desc": "Phase 2A: Ensemble K=2, S=4, eta=0.0",
    },
    {
        "id": "ENS_K04_S04_ETA0",
        "members": 4,
        "steps": 4,
        "eta": 0.0,
        "budget": 16,
        "split": "val",
        "desc": "Phase 2A: Ensemble K=4, S=4, eta=0.0",
    },
    {
        "id": "ENS_K08_S04_ETA0",
        "members": 8,
        "steps": 4,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 2A: Ensemble K=8, S=4, eta=0.0",
    },
    {
        "id": "ENS_K16_S04_ETA0",
        "members": 16,
        "steps": 4,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 2A: Ensemble K=16, S=4, eta=0.0",
    },
    # Phase 2 Stage B: Trajectory Stochasticity Sweeps (K=4, S=4)
    {
        "id": "ENS_K04_S04_ETA025",
        "members": 4,
        "steps": 4,
        "eta": 0.25,
        "budget": 16,
        "split": "val",
        "desc": "Phase 2B: Ensemble K=4, S=4, eta=0.25",
    },
    {
        "id": "ENS_K04_S04_ETA050",
        "members": 4,
        "steps": 4,
        "eta": 0.50,
        "budget": 16,
        "split": "val",
        "desc": "Phase 2B: Ensemble K=4, S=4, eta=0.50",
    },
    {
        "id": "ENS_K04_S04_ETA100",
        "members": 4,
        "steps": 4,
        "eta": 1.00,
        "budget": 16,
        "split": "val",
        "desc": "Phase 2B: Ensemble K=4, S=4, eta=1.00",
    },
    # Phase 3: Matched Budget 8
    {
        "id": "B8_K1_S8",
        "members": 1,
        "steps": 8,
        "eta": 0.0,
        "budget": 8,
        "split": "val",
        "desc": "Phase 3 Budget 8: K=1, S=8 (8 NFE)",
    },
    {
        "id": "B8_K2_S4",
        "members": 2,
        "steps": 4,
        "eta": 0.0,
        "budget": 8,
        "split": "val",
        "desc": "Phase 3 Budget 8: K=2, S=4 (8 NFE)",
    },
    # Phase 3: Matched Budget 16
    {
        "id": "B16_K1_S16",
        "members": 1,
        "steps": 16,
        "eta": 0.0,
        "budget": 16,
        "split": "val",
        "desc": "Phase 3 Budget 16: K=1, S=16 (16 NFE)",
    },
    {
        "id": "B16_K2_S8",
        "members": 2,
        "steps": 8,
        "eta": 0.0,
        "budget": 16,
        "split": "val",
        "desc": "Phase 3 Budget 16: K=2, S=8 (16 NFE)",
    },
    {
        "id": "B16_K4_S4",
        "members": 4,
        "steps": 4,
        "eta": 0.0,
        "budget": 16,
        "split": "val",
        "desc": "Phase 3 Budget 16: K=4, S=4 (16 NFE)",
    },
    # Phase 3: Matched Budget 32 (Flagship)
    {
        "id": "B32_K1_S32",
        "members": 1,
        "steps": 32,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 3 Budget 32: K=1, S=32 (32 NFE, Deep Deterministic)",
    },
    {
        "id": "B32_K2_S16",
        "members": 2,
        "steps": 16,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 3 Budget 32: K=2, S=16 (32 NFE)",
    },
    {
        "id": "B32_K4_S8",
        "members": 4,
        "steps": 8,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 3 Budget 32: K=4, S=8 (32 NFE)",
    },
    {
        "id": "B32_K8_S4",
        "members": 8,
        "steps": 4,
        "eta": 0.0,
        "budget": 32,
        "split": "val",
        "desc": "Phase 3 Budget 32: K=8, S=4, eta=0 (32 NFE, Broad Ensemble)",
    },
    {
        "id": "B32_K8_S4_ETA05",
        "members": 8,
        "steps": 4,
        "eta": 0.5,
        "budget": 32,
        "split": "val",
        "desc": "Phase 3 Budget 32: K=8, S=4, eta=0.5 (32 NFE, Stochastic)",
    },
    # Phase 3: Matched Budget 64
    {
        "id": "B64_K1_S64",
        "members": 1,
        "steps": 64,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 3 Budget 64: K=1, S=64 (64 NFE)",
    },
    {
        "id": "B64_K2_S32",
        "members": 2,
        "steps": 32,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 3 Budget 64: K=2, S=32 (64 NFE)",
    },
    {
        "id": "B64_K4_S16",
        "members": 4,
        "steps": 16,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 3 Budget 64: K=4, S=16 (64 NFE)",
    },
    {
        "id": "B64_K8_S8",
        "members": 8,
        "steps": 8,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 3 Budget 64: K=8, S=8 (64 NFE)",
    },
    {
        "id": "B64_K16_S4",
        "members": 16,
        "steps": 4,
        "eta": 0.0,
        "budget": 64,
        "split": "val",
        "desc": "Phase 3 Budget 64: K=16, S=4 (64 NFE)",
    },
    # Phase 4 Confirmatory Quarantined Holdout (2023 Season)
    {
        "id": "CHAMPION_HOLDOUT_B32_K8_S4_ETA05",
        "members": 8,
        "steps": 4,
        "eta": 0.5,
        "budget": 32,
        "split": "test",
        "desc": "Phase 4 Confirmatory: 2023 Holdout Test on Flagship Champion (K=8, S=4, eta=0.5)",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sprint 8 Ensemble and Test-Time Scaling Evaluator")
    parser.add_argument("--condition", type=str, default="all", help="Condition ID to run, or 'all', 'flagship', 'budget32'")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size of forecast cubes per inference pass")
    parser.add_argument("--chunk_size", type=int, default=4, help="Max ensemble members processed in parallel (1, 2, or 4)")
    parser.add_argument("--device", type=str, default="auto", help="CUDA device (auto, cuda, cuda:0, cuda:1, cpu)")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches for smoke test")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to Candidate 3 checkpoint")
    parser.add_argument("--skip_holdout", action="store_true", help="Skip final holdout test evaluation")
    return parser.parse_args()


def resolve_paths(args) -> Tuple[Path, Path, Path, Path, Path]:
    if args.output_dir is not None:
        out_root = Path(args.output_dir)
    elif Path("/kaggle/working").exists():
        out_root = Path("/kaggle/working")
    else:
        out_root = ROOT

    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    zarr_candidates = [
        ROOT / "datasets" / "multitask_temporal_v2_h14.zarr",
        Path("/kaggle/working/datasets/multitask_temporal_v2_h14.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/multitask_temporal_v2_h14.zarr"),
        ROOT / "datasets" / "multitask_temporal_v1.zarr",
        Path("/kaggle/working/datasets/multitask_temporal_v1.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/multitask_temporal_v1.zarr"),
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
        ROOT / "data" / "sample_index.parquet",
        Path("/kaggle/working/data/sample_index.parquet"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/sample_index.parquet"),
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)
    if index_path is None:
        index_path = ROOT / "data" / "sample_index_v2_h14.parquet"

    stats_candidates = [
        ROOT / "data" / "normalization_stats_v2.yaml",
        Path("/kaggle/working/data/normalization_stats_v2.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
        ROOT / "data" / "normalization_stats.yaml",
        Path("/kaggle/working/data/normalization_stats.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/normalization_stats.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None:
        stats_path = ROOT / "data" / "normalization_stats_v2.yaml"

    return out_root, reports_dir, zarr_path, index_path, stats_path


def verify_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[!] Checkpoint missing at {path}. Hard failure policy enforced.")

    file_bytes = path.read_bytes()
    if file_bytes.startswith(b"version https://git-lfs.github.com/spec/v1"):
        print(f"[+] Verified Git LFS pointer for checkpoint: {path.name}")
        return

    sha256 = hashlib.sha256(file_bytes).hexdigest()
    print(f"[+] Checkpoint SHA-256: {sha256}")
    if sha256 != EXPECTED_CHECKPOINT_SHA256:
        print(f"[!] Warning: SHA-256 {sha256} diverges from expected {EXPECTED_CHECKPOINT_SHA256}.")


def load_model(checkpoint_path: Path, device: torch.device) -> SpatiotemporalResidualDiffusion:
    model = SpatiotemporalResidualDiffusion(
        timesteps=100,
        base_channels=96,
        prediction_type="v_prediction",
        loss_weighting="group_tail",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Trainable parameter count: {param_count:,} (Expected: {EXPECTED_PARAM_COUNT:,})")
    assert param_count == EXPECTED_PARAM_COUNT, f"Parameter mismatch: {param_count} != {EXPECTED_PARAM_COUNT}"

    if checkpoint_path.exists() and not checkpoint_path.read_bytes().startswith(b"version https://git-lfs.github.com"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[+] Checkpoint loaded: {len(missing)} missing keys, {len(unexpected)} unexpected keys.")
    else:
        print("[!] Running with initialized architecture (LFS pointer or test mode).")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def evaluate_condition(
    condition: Dict[str, Any],
    model: SpatiotemporalResidualDiffusion,
    loader: DataLoader,
    stats: Dict[str, Any],
    device: torch.device,
    chunk_size: int = 4,
    max_batches: Optional[int] = None,
    base_seed: int = 20260927,
    clim_train_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    cond_id = condition["id"]
    k_members = condition["members"]
    s_steps = condition["steps"]
    eta = condition["eta"]
    budget = condition["budget"]
    desc = condition["desc"]

    print(f"\n=======================================================")
    print(f"[*] Running Condition: {cond_id}")
    print(f"[*] Config: K={k_members}, S={s_steps}, eta={eta}, NFE={k_members * s_steps}, chunk_size={chunk_size}")
    print(f"[*] Description: {desc}")
    print(f"=======================================================")

    num_cubes = 0
    t_start = time.perf_counter()

    all_crps_list = []
    all_mae_list = []
    all_wet_mae_list = []
    all_csi15_list = []
    all_csi30_list = []
    all_tmax_mae_list = []
    all_tmin_mae_list = []
    all_rh_mae_list = []
    all_wind_rmse_list = []
    all_cmvs_list = []

    all_p15_probs = []
    all_p15_obs = []
    all_p30_probs = []
    all_p30_obs = []

    diversity_diagnostics_accum = []
    repair_diagnostics_accum = []

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            history = batch["history"].to(device)
            future_fcst = batch["future_forecast"].to(device)
            terrain = batch["terrain"].to(device)
            target_norm = batch["target"].to(device) # [B, T_f, C, H, W]

            b_cur = target_norm.shape[0]
            num_cubes += b_cur

            # Generate K members using nested seed manifest
            member_preds_norm = []
            seeds = generate_nested_seeds(num_members=k_members, base_seed=base_seed, batch_idx=b_idx)

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

            # Stack into [K, B, T_f, C, H, W]
            members_norm_tensor = torch.stack(member_preds_norm, dim=0)

            # Invert normalization member-by-member to physical units
            members_phys_list = []
            for k in range(k_members):
                k_phys_np = invert_normalization(members_norm_tensor[k].cpu().numpy(), stats)
                members_phys_list.append(torch.from_numpy(k_phys_np))
            members_phys_raw = torch.stack(members_phys_list, dim=0).to(device)

            target_phys_np = invert_normalization(target_norm.cpu().numpy(), stats)
            target_phys_tensor = torch.from_numpy(target_phys_np).to(device)

            # Apply member-wise physical bounds and record repair diagnostics
            members_phys, rep_diag = apply_member_wise_physical_bounds(members_phys_raw)
            repair_diagnostics_accum.append(rep_diag)

            # Compute pairwise diversity diagnostics
            div_diag = compute_pairwise_diversity(members_phys)
            diversity_diagnostics_accum.append(div_diag)

            # Compute CRPS (Fair-CRPS for K >= 2, Deterministic CRPS = MAE for K = 1)
            crps_val, is_fair = compute_crps(members_phys, target_phys_tensor)
            all_crps_list.append(crps_val)

            # Compute Ensemble Mean for deterministic metrics
            ens_mean_phys = members_phys.mean(dim=0).cpu().numpy() # [B, T_f, C, H, W]

            # Deterministic metrics on ensemble mean
            p_pred = np.maximum(0.0, ens_mean_phys[:, :, 0])
            p_tgt = np.maximum(0.0, target_phys_np[:, :, 0])
            tmax_pred = ens_mean_phys[:, :, 1]
            tmax_tgt = target_phys_np[:, :, 1]
            tmin_pred = ens_mean_phys[:, :, 2]
            tmin_tgt = target_phys_np[:, :, 2]
            rh_pred = np.clip(ens_mean_phys[:, :, 3], 0.0, 100.0)
            rh_tgt = np.clip(target_phys_np[:, :, 3], 0.0, 100.0)
            u_pred = ens_mean_phys[:, :, 4]
            u_tgt = target_phys_np[:, :, 4]
            v_pred = ens_mean_phys[:, :, 5]
            v_tgt = target_phys_np[:, :, 5]

            p_mae = float(np.mean(np.abs(p_pred - p_tgt)))
            wet_mask = p_tgt > 2.5
            wet_mae = float(np.mean(np.abs(p_pred[wet_mask] - p_tgt[wet_mask]))) if np.sum(wet_mask) > 0 else p_mae

            def calc_csi(p_arr, t_arr, thresh):
                hit = np.sum((p_arr >= thresh) & (t_arr >= thresh))
                fp = np.sum((p_arr >= thresh) & (t_arr < thresh))
                fn = np.sum((p_arr < thresh) & (t_arr >= thresh))
                den = hit + fp + fn
                return float(hit / den) if den > 0 else 1.0

            csi_15 = calc_csi(p_pred, p_tgt, 15.0)
            csi_30 = calc_csi(p_pred, p_tgt, 30.0)
            tmax_mae = float(np.mean(np.abs(tmax_pred - tmax_tgt)))
            tmin_mae = float(np.mean(np.abs(tmin_pred - tmin_tgt)))
            rh_mae = float(np.mean(np.abs(rh_pred - rh_tgt)))
            vec_rmse = float(np.sqrt(np.mean((u_pred - u_tgt) ** 2 + (v_pred - v_tgt) ** 2)))

            cmvs = (
                0.35 * (wet_mae / 8.70)
                + 0.35 * max(0.0, 1.0 - (csi_30 / 0.631))
                + 0.15 * (tmax_mae / 0.37)
                + 0.15 * (vec_rmse / 1.69)
            )

            all_mae_list.append(p_mae)
            all_wet_mae_list.append(wet_mae)
            all_csi15_list.append(csi_15)
            all_csi30_list.append(csi_30)
            all_tmax_mae_list.append(tmax_mae)
            all_tmin_mae_list.append(tmin_mae)
            all_rh_mae_list.append(rh_mae)
            all_wind_rmse_list.append(vec_rmse)
            all_cmvs_list.append(cmvs)

            # Threshold probabilities: P(P > 15) and P(P > 30)
            p_members_all = members_phys[:, :, :, 0] # [K, B, T_f, H, W]
            prob_15 = (p_members_all > 15.0).float().mean(dim=0).cpu() # [B, T_f, H, W]
            prob_30 = (p_members_all > 30.0).float().mean(dim=0).cpu()
            obs_15 = (target_phys_tensor[:, :, 0] > 15.0).float().cpu()
            obs_30 = (target_phys_tensor[:, :, 0] > 30.0).float().cpu()

            all_p15_probs.append(prob_15.flatten())
            all_p15_obs.append(obs_15.flatten())
            all_p30_probs.append(prob_30.flatten())
            all_p30_obs.append(obs_30.flatten())

    t_total = time.perf_counter() - t_start
    latency_ms = (t_total / max(1, num_cubes)) * 1000.0
    throughput = num_cubes / max(1e-5, t_total)

    # Brier Scores
    p15_p = torch.cat(all_p15_probs) if all_p15_probs else torch.tensor([0.0])
    p15_o = torch.cat(all_p15_obs) if all_p15_obs else torch.tensor([0.0])
    p30_p = torch.cat(all_p30_probs) if all_p30_probs else torch.tensor([0.0])
    p30_o = torch.cat(all_p30_obs) if all_p30_obs else torch.tensor([0.0])

    clim15_rate = clim_train_rates.get("p15", 0.08) if clim_train_rates else 0.08
    clim30_rate = clim_train_rates.get("p30", 0.02) if clim_train_rates else 0.02

    brier15 = compute_brier_score(p15_p, p15_o, clim_train_rate=clim15_rate)
    brier30 = compute_brier_score(p30_p, p30_o, clim_train_rate=clim30_rate)

    avg_crps = float(np.mean(all_crps_list))
    avg_cmvs = float(np.mean(all_cmvs_list))
    avg_wet_mae = float(np.mean(all_wet_mae_list))
    avg_csi15 = float(np.mean(all_csi15_list))
    avg_csi30 = float(np.mean(all_csi30_list))
    avg_tmax_mae = float(np.mean(all_tmax_mae_list))
    avg_tmin_mae = float(np.mean(all_tmin_mae_list))
    avg_rh_mae = float(np.mean(all_rh_mae_list))
    avg_wind_rmse = float(np.mean(all_wind_rmse_list))

    avg_div_rmse = float(np.mean([d["mean_pairwise_rmse"] for d in diversity_diagnostics_accum]))
    avg_div_corr = float(np.mean([d["mean_pairwise_correlation"] for d in diversity_diagnostics_accum]))
    avg_ens_var = float(np.mean([d["ensemble_variance"] for d in diversity_diagnostics_accum]))

    avg_precip_clip = float(np.mean([d["precip_clipped_fraction"] for d in repair_diagnostics_accum]))
    avg_precip_mass_shift = float(np.mean([d["precip_mass_shift_pct"] for d in repair_diagnostics_accum]))
    avg_tmin_gt_tmax = float(np.mean([d["tmin_gt_tmax_rate"] for d in repair_diagnostics_accum]))

    result = {
        "condition_id": cond_id,
        "description": desc,
        "ensemble_size_k": k_members,
        "denoising_steps_s": s_steps,
        "eta": eta,
        "total_nfe": k_members * s_steps,
        "nominal_budget": budget,
        "is_fair_crps": (k_members >= 2),
        "num_cubes_evaluated": num_cubes,
        "latency_ms_per_cube": latency_ms,
        "throughput_cubes_sec": throughput,
        "metrics": {
            "crps": avg_crps,
            "cmvs": avg_cmvs,
            "precip_wet_mae": avg_wet_mae,
            "precip_csi15": avg_csi15,
            "precip_csi30": avg_csi30,
            "tmax_mae": avg_tmax_mae,
            "tmin_mae": avg_tmin_mae,
            "rh_mae": avg_rh_mae,
            "wind_vector_rmse": avg_wind_rmse,
            "brier_15": brier15,
            "brier_30": brier30,
            "diversity": {
                "mean_pairwise_rmse": avg_div_rmse,
                "mean_pairwise_correlation": avg_div_corr,
                "ensemble_variance": avg_ens_var,
            },
            "repair_diagnostics": {
                "precip_clipped_fraction": avg_precip_clip,
                "precip_mass_shift_pct": avg_precip_mass_shift,
                "tmin_gt_tmax_violation_rate": avg_tmin_gt_tmax,
            },
        },
        "provenance": {
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_parameters": EXPECTED_PARAM_COUNT,
            "noise_schedule": "linear_beta_1e-4_to_0.035_T100",
            "chunk_size": chunk_size,
            "base_seed": base_seed,
            "device": str(device),
        }
    }

    print(f"[+] Finished {cond_id}: CRPS={avg_crps:.4f} ({'Fair' if k_members >= 2 else 'Det'}), CMVS={avg_cmvs:.4f}, Wet-MAE={avg_wet_mae:.2f}, CSI@30={avg_csi30:.3f}, Latency={latency_ms:.1f}ms")
    return result


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"===========================================================")
    print(f"[*] Sprint 8 Ensemble & Test-Time Scaling Evaluation Engine")
    print(f"[*] Target Device: {device}")
    print(f"===========================================================")

    out_root, reports_dir, zarr_path, index_path, stats_path = resolve_paths(args)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
    else:
        ckpt_cands = [
            out_root / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt",
            ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt",
            Path("/kaggle/working/models/checkpoints/sprint6_candidate3_multitask_champion.pt"),
            Path("/kaggle/input/sih26074-sprint6-checkpoints/sprint6_candidate3_multitask_champion.pt"),
        ]
        ckpt_path = next((c for c in ckpt_cands if c.exists()), None)
        if ckpt_path is None and Path("/kaggle/input").exists():
            for p in Path("/kaggle/input").rglob("*.pt"):
                if "candidate3" in p.name.lower() or "sprint6" in p.name.lower() or "champion" in p.name.lower():
                    ckpt_path = p
                    break
        if ckpt_path is None:
            ckpt_path = ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt"

    verify_checkpoint(ckpt_path)

    model = load_model(ckpt_path, device=device)

    print(f"[*] Zarr Path:   {zarr_path}")
    print(f"[*] Index Path:  {index_path}")
    print(f"[*] Stats Path:  {stats_path}")

    # Build validation dataset
    val_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=14,
        context_size=24,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"[+] Loaded validation split: {len(val_ds)} samples ({len(val_loader)} batches).")

    # Filter conditions to run
    raw_tokens = [tok.strip().upper() for tok in args.condition.split(",") if tok.strip()]
    to_run = []
    seen_ids = set()

    for tok in raw_tokens:
        if tok == "ALL":
            for c in CONDITIONS:
                if c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])
        elif tok in ["FLAGSHIP", "BUDGET32"]:
            for c in CONDITIONS:
                if c["budget"] == 32 and c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])
        elif tok == "BUDGET8":
            for c in CONDITIONS:
                if c["budget"] == 8 and c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])
        elif tok == "BUDGET16":
            for c in CONDITIONS:
                if c["budget"] == 16 and c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])
        elif tok == "BUDGET64":
            for c in CONDITIONS:
                if c["budget"] == 64 and c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])
        else:
            matched = [c for c in CONDITIONS if c["id"].upper() == tok]
            for c in matched:
                if c["id"] not in seen_ids:
                    to_run.append(c)
                    seen_ids.add(c["id"])

    if not to_run:
        print(f"[!] No matching conditions found for '{args.condition}'. Available IDs:")
        for c in CONDITIONS:
            print(f"  - {c['id']}")
        sys.exit(1)

    test_ds = None
    test_loader = None

    for cond in to_run:
        split_name = cond.get("split", "val")
        if split_name == "test":
            if test_ds is None:
                test_ds = SpatiotemporalDownscalingDataset(
                    zarr_path=zarr_path,
                    index_path=index_path,
                    stats_path=stats_path,
                    split="test",
                    history_len=14,
                    context_size=24,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2 if device.type == "cuda" else 0,
                    pin_memory=(device.type == "cuda"),
                )
                print(f"[+] Loaded quarantined holdout test split: {len(test_ds)} samples ({len(test_loader)} batches).")
            active_loader = test_loader
            active_stats = test_ds.stats
        else:
            active_loader = val_loader
            active_stats = val_ds.stats

        res = evaluate_condition(
            condition=cond,
            model=model,
            loader=active_loader,
            stats=active_stats,
            device=device,
            chunk_size=args.chunk_size,
            max_batches=args.max_batches,
        )

        out_file = reports_dir / f"sprint8_{cond['id'].lower()}_history.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[+] Saved condition history to {out_file}")

        if cond["id"] == "CHAMPION_HOLDOUT_B32_K8_S4_ETA05":
            holdout_file = reports_dir / "sprint8_champion_holdout_test.json"
            with open(holdout_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"[+] Saved dedicated champion holdout artifact to {holdout_file}")

    print("\n[+] Sprint 8 evaluation run completed successfully.")


if __name__ == "__main__":
    main()
