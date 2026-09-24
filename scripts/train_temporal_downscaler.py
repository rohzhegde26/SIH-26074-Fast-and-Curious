"""
scripts/train_temporal_downscaler.py

Unified Training Pipeline for Sprint 3 Spatiotemporal Downscalers.
Supports:
  1. Deterministic Mode: Trains TemporalMultiTaskUNet5x with SpatiotemporalMultiTaskLoss.
  2. Diffusion Mode: Trains SpatiotemporalResidualDiffusion with MSE epsilon prediction.
  3. Dynamic History Lengths: H in {1, 2, 3}.
  4. Automatic Mixed Precision (torch.amp.autocast) with GradScaler.
  5. Per-Variable & Per-Lead-Day Metric Evaluation on Validation (2022).
  6. Automated Checkpointing with Experiment Metadata.
"""

import argparse
import json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import invert_normalization
from src.losses.spatiotemporal_multitask_loss import SpatiotemporalMultiTaskLoss
from src.models.temporal_multitask_baseline import TemporalMultiTaskUNet5x
from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)
from src.losses.conservation import coarsen_hr_to_lr_torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sprint 4 Scaled Spatiotemporal Downscaler")
    parser.add_argument("--mode", type=str, default="diffusion", choices=["deterministic", "diffusion"])
    parser.add_argument("--history_len", type=int, default=3, choices=[1, 2, 3, 5, 7, 10, 14], help="Antecedent history days")
    parser.add_argument("--model_size", type=str, default="ultra", choices=["small", "base", "large", "ultra"])
    parser.add_argument("--base_channels", type=int, default=None, help="Explicit override for model base channels")
    parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr dataset store")
    parser.add_argument("--index_path", type=str, default=None, help="Path to sample index parquet")
    parser.add_argument("--stats_path", type=str, default=None, help="Path to normalization stats YAML")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory for checkpoints & reports")
    parser.add_argument("--ddim_steps", type=int, default=32, help="Sampling steps for diffusion evaluation")
    parser.add_argument("--eval_sampling_interval", type=int, default=5, help="Epoch interval to run full DDIM reverse diffusion sampling on validation")
    parser.add_argument("--early_stopping_patience", type=int, default=7, help="Epoch patience for early stop")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for quick smoke test")
    parser.add_argument("--context_size", type=int, default=16, choices=[16, 20, 24, 32], help="Spatial context dimension N (N/M experiments)")
    parser.add_argument("--spatial_mode", action="store_true", help="Sprint 5 spatial context experiment mode")
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "v_prediction"], help="Diffusion prediction parameterization")
    parser.add_argument("--loss_weighting", type=str, default="uniform", choices=["uniform", "group_tail"], help="Multi-task loss weighting scheme")
    parser.add_argument("--checkpoint_criterion", type=str, default="loss", choices=["loss", "cmvs"], help="Checkpoint selection criterion")
    parser.add_argument("--exp_name", type=str, default=None, help="Explicit experiment name prefix for checkpoints and reports")
    return parser.parse_args()


def get_model_channels(model_size: str, base_channels: Optional[int] = None) -> int:
    if base_channels is not None:
        return int(base_channels)
    sizes = {"small": 16, "base": 24, "large": 32, "ultra": 96}
    return sizes.get(model_size, 96)


def compute_lead_metrics(
    preds_phys: np.ndarray,
    targets_phys: np.ndarray,
    coarse_phys: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes per-variable and per-lead metrics:
      - Precipitation: All-Day MAE, Wet-Day MAE (> 2.5 mm), RMSE, CSI@15, CSI@30
      - Tmax / Tmin: MAE, RMSE, Diurnal spread violation rate
      - RH: MAE, out-of-range rate (< 0 or > 100%)
      - Wind: U MAE, V MAE, Vector RMSE
    """
    b, num_leads, num_channels, h, w = preds_phys.shape
    per_lead = []

    for l in range(num_leads):
        p_pred = preds_phys[:, l, 0]
        p_tgt = targets_phys[:, l, 0]
        tmax_pred = preds_phys[:, l, 1]
        tmax_tgt = targets_phys[:, l, 1]
        tmin_pred = preds_phys[:, l, 2]
        tmin_tgt = targets_phys[:, l, 2]
        rh_pred = preds_phys[:, l, 3]
        rh_tgt = targets_phys[:, l, 3]
        u_pred = preds_phys[:, l, 4]
        u_tgt = targets_phys[:, l, 4]
        v_pred = preds_phys[:, l, 5]
        v_tgt = targets_phys[:, l, 5]

        # Precipitation metrics
        p_mae = float(np.mean(np.abs(p_pred - p_tgt)))
        p_rmse = float(np.sqrt(np.mean((p_pred - p_tgt) ** 2)))
        wet_mask = p_tgt > 2.5
        wet_mae = float(np.mean(np.abs(p_pred[wet_mask] - p_tgt[wet_mask]))) if np.sum(wet_mask) > 0 else p_mae

        def calc_csi(pred_arr, tgt_arr, thresh):
            hit = np.sum((pred_arr >= thresh) & (tgt_arr >= thresh))
            fp = np.sum((pred_arr >= thresh) & (tgt_arr < thresh))
            fn = np.sum((pred_arr < thresh) & (tgt_arr >= thresh))
            den = hit + fp + fn
            return float(hit / den) if den > 0 else 1.0

        csi_15 = calc_csi(p_pred, p_tgt, 15.0)
        csi_30 = calc_csi(p_pred, p_tgt, 30.0)

        # Temperature metrics
        tmax_mae = float(np.mean(np.abs(tmax_pred - tmax_tgt)))
        tmin_mae = float(np.mean(np.abs(tmin_pred - tmin_tgt)))
        diurnal_violations = float(np.mean(tmin_pred > tmax_pred))

        # RH metrics
        rh_mae = float(np.mean(np.abs(rh_pred - rh_tgt)))
        rh_out_of_range = float(np.mean((rh_pred < 0.0) | (rh_pred > 100.0)))

        # Wind metrics
        u_mae = float(np.mean(np.abs(u_pred - u_tgt)))
        v_mae = float(np.mean(np.abs(v_pred - v_tgt)))
        vec_rmse = float(np.sqrt(np.mean((u_pred - u_tgt) ** 2 + (v_pred - v_tgt) ** 2)))

        lead_summary = {
            "lead_day": f"D+{l}",
            "precip_mae": p_mae,
            "precip_wet_mae": wet_mae,
            "precip_rmse": p_rmse,
            "precip_csi15": csi_15,
            "precip_csi30": csi_30,
            "tmax_mae": tmax_mae,
            "tmin_mae": tmin_mae,
            "diurnal_violation_rate": diurnal_violations,
            "rh_mae": rh_mae,
            "rh_out_of_range_rate": rh_out_of_range,
            "wind_u_mae": u_mae,
            "wind_v_mae": v_mae,
            "wind_vector_rmse": vec_rmse,
        }
        per_lead.append(lead_summary)

    # 7-day aggregate mean
    agg = {
        k: float(np.mean([m[k] for m in per_lead]))
        for k in per_lead[0]
        if k != "lead_day"
    }

    return {"per_lead": per_lead, "aggregate": agg}


def run_training():
    args = parse_args()
    print("=" * 70)
    print(f"SIH 26074 Sprint 3 Training: Mode={args.mode.upper()} | History=H{args.history_len}")
    print("=" * 70)

    # Select execution device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Execution device: {device}")
    if device.type == "cuda":
        print(f"    GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"    Available GPUs: {torch.cuda.device_count()}")
        print(f"    CUDA VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Output paths
    if args.output_dir is not None:
        out_root = Path(args.output_dir)
    elif Path("/kaggle/working").exists():
        out_root = Path("/kaggle/working")
    else:
        out_root = ROOT

    models_dir = out_root / "models" / "checkpoints"
    reports_dir = out_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("[*] Resolving datasets and contracts...")
    if args.zarr_path:
        zarr_path = Path(args.zarr_path)
    else:
        zarr_candidates = [
            ROOT / "datasets" / "multitask_temporal_v2_h14.zarr",
            Path("/kaggle/working/datasets/multitask_temporal_v2_h14.zarr"),
            Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/multitask_temporal_v2_h14.zarr"),
            ROOT / "datasets" / "multitask_temporal_v1.zarr",
            Path("/kaggle/working/datasets/multitask_temporal_v1.zarr"),
            Path("/kaggle/input/sih26074-multitask-temporal-v1/multitask_temporal_v1.zarr"),
        ]
        zarr_path = None
        for cand in zarr_candidates:
            if cand.exists() and ((cand / ".zgroup").exists() or (cand / "dates").exists()):
                zarr_path = cand
                break
        if zarr_path is None and Path("/kaggle/input").exists():
            for d in Path("/kaggle/input").rglob("*.zarr"):
                if (d / ".zgroup").exists() or (d / "dates").exists():
                    zarr_path = d
                    break
        if zarr_path is None:
            zarr_path = ROOT / "datasets" / "multitask_temporal_v2_h14.zarr"

    if args.index_path:
        index_path = Path(args.index_path)
    else:
        index_candidates = [
            ROOT / "data" / "sample_index_v2_h14.parquet",
            Path("/kaggle/working/data/sample_index_v2_h14.parquet"),
            Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/sample_index_v2_h14.parquet"),
            ROOT / "data" / "sample_index.parquet",
            Path("/kaggle/working/data/sample_index.parquet"),
            Path("/kaggle/input/sih26074-multitask-temporal-v1/sample_index.parquet"),
        ]
        index_path = next((p for p in index_candidates if p.exists()), None)
        if index_path is None and Path("/kaggle/input").exists():
            found = list(Path("/kaggle/input").rglob("sample_index*.parquet"))
            if found:
                index_path = found[0]
        if index_path is None:
            index_path = ROOT / "data" / "sample_index_v2_h14.parquet"

    if args.stats_path:
        stats_path = Path(args.stats_path)
    else:
        stats_candidates = [
            ROOT / "data" / "normalization_stats_v2.yaml",
            Path("/kaggle/working/data/normalization_stats_v2.yaml"),
            Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
            ROOT / "data" / "normalization_stats.yaml",
            Path("/kaggle/working/data/normalization_stats.yaml"),
            Path("/kaggle/input/sih26074-multitask-temporal-v1/normalization_stats.yaml"),
        ]
        stats_path = next((p for p in stats_candidates if p.exists()), None)
        if stats_path is None and Path("/kaggle/input").exists():
            found = list(Path("/kaggle/input").rglob("normalization_stats*.yaml"))
            if found:
                stats_path = found[0]
        if stats_path is None:
            stats_path = ROOT / "data" / "normalization_stats_v2.yaml"

    print(f"[+] Resolved Zarr store:  {zarr_path}")
    print(f"[+] Resolved Index table: {index_path}")
    print(f"[+] Resolved Norm stats:  {stats_path}")

    train_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="train",
        history_len=args.history_len,
        context_size=args.context_size,
        normalize=True,
    )
    val_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=args.history_len,
        context_size=args.context_size,
        normalize=True,
    )
    test_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="test",
        history_len=args.history_len,
        context_size=args.context_size,
        normalize=True,
    )

    print(f"[+] Loaded Train samples: {len(train_ds)} (2015-2021)")
    print(f"[+] Loaded Val samples:   {len(val_ds)} (2022)")
    print(f"[+] Loaded Test samples:  {len(test_ds)} (2023)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    # Build model and optimizer
    base_ch = get_model_channels(args.model_size, args.base_channels)
    embed_dim = base_ch if base_ch >= 48 else 32
    num_heads = 8 if base_ch >= 48 else 4
    if args.mode == "deterministic":
        model = TemporalMultiTaskUNet5x(
            base_channels=base_ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        criterion = SpatiotemporalMultiTaskLoss().to(device)
    else:
        model = SpatiotemporalResidualDiffusion(
            timesteps=100,
            base_channels=base_ch,
            prediction_type=args.prediction_type,
            loss_weighting=args.loss_weighting,
        )
        criterion = None

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Model initialized: {model.__class__.__name__} ({param_count:,} parameters)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Training Loop
    history_log = []
    best_val_loss = float("inf")
    best_cmvs = float("inf")
    patience_counter = 0

    print("=" * 70)
    print(f"[*] Starting training loop ({args.epochs} epoch(s))...")
    print("=" * 70)

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            history = batch["history"].to(device)  # [B, H, 6, 16, 16]
            fcst = batch["future_forecast"].to(device)  # [B, 7, 6, 16, 16]
            terrain = batch["terrain"].to(device)  # [B, 5, 80, 80]
            target = batch["target"].to(device)  # [B, 7, 6, 80, 80]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if args.mode == "deterministic":
                    preds = model(history=history, future_forecast=fcst, terrain=terrain)
                    loss_dict = criterion(preds=preds, targets=target, coarse_fcst=fcst)
                    loss = loss_dict["loss"]
                else:
                    r_0 = compute_residual_target(target, fcst)
                    loss, _, _ = model.compute_training_loss(
                        r_0=r_0,
                        history=history,
                        future_forecast=fcst,
                        terrain=terrain,
                        target_norm=target,
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1
            if args.max_batches is not None and train_batches >= args.max_batches:
                break

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        avg_train_loss = train_loss_sum / max(1, train_batches)

        # Validation Phase
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_preds_list = []
        val_targets_list = []
        val_coarse_list = []

        should_sample = (args.mode == "deterministic") or (epoch % args.eval_sampling_interval == 0) or (epoch == args.epochs)

        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device)
                fcst = batch["future_forecast"].to(device)
                terrain = batch["terrain"].to(device)
                target = batch["target"].to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    if args.mode == "deterministic":
                        preds = model(history=history, future_forecast=fcst, terrain=terrain)
                        loss_dict = criterion(preds=preds, targets=target, coarse_fcst=fcst)
                        loss = loss_dict["loss"]
                    else:
                        r_0 = compute_residual_target(target, fcst)
                        loss, _, _ = model.compute_training_loss(
                            r_0=r_0,
                            history=history,
                            future_forecast=fcst,
                            terrain=terrain,
                            target_norm=target,
                        )
                        if should_sample:
                            preds = model.sample(
                                history=history,
                                future_forecast=fcst,
                                terrain=terrain,
                                num_steps=args.ddim_steps,
                            )

                val_loss_sum += loss.item()
                val_batches += 1
                if should_sample:
                    val_preds_list.append(preds.cpu().numpy())
                    val_targets_list.append(target.cpu().numpy())
                    val_coarse_list.append(fcst.cpu().numpy())
                if args.max_batches is not None and val_batches >= args.max_batches:
                    break

        avg_val_loss = val_loss_sum / max(1, val_batches)

        metrics = None
        if should_sample:
            # De-normalize validation samples to physical space for meteorological metrics
            val_preds_all = np.concatenate(val_preds_list, axis=0)
            val_targets_all = np.concatenate(val_targets_list, axis=0)
            val_coarse_all = np.concatenate(val_coarse_list, axis=0)

            preds_phys = invert_normalization(val_preds_all, train_ds.stats)
            targets_phys = invert_normalization(val_targets_all, train_ds.stats)
            coarse_phys = invert_normalization(val_coarse_all, train_ds.stats)

            metrics = compute_lead_metrics(preds_phys, targets_phys, coarse_phys)
            agg_m = metrics["aggregate"]

            # Compute Composite Meteorological Validation Score (CMVS)
            cmvs_val = (
                0.35 * (agg_m["precip_wet_mae"] / 8.70)
                + 0.35 * max(0.0, 1.0 - (agg_m["precip_csi30"] / 0.631))
                + 0.15 * (agg_m["tmax_mae"] / 0.37)
                + 0.15 * (agg_m["wind_vector_rmse"] / 1.69)
            )
            agg_m["cmvs"] = float(cmvs_val)

            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"CMVS: {cmvs_val:.4f} | "
                f"Wet-MAE: {agg_m['precip_wet_mae']:.2f} mm | "
                f"CSI@30: {agg_m['precip_csi30']:.3f} | "
                f"Tmax-MAE: {agg_m['tmax_mae']:.2f} C | "
                f"Wind-RMSE: {agg_m['wind_vector_rmse']:.2f} m/s | "
                f"Time: {epoch_duration:.1f}s"
            )
        else:
            print(
                f"[Epoch {epoch:02d}/{args.epochs:02d}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Time: {epoch_duration:.1f}s (fast val loss pass)"
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_duration_sec": epoch_duration,
            "metrics": metrics,
        }
        history_log.append(epoch_record)

        # Checkpoint Selection: either CMVS or validation loss
        is_best = False
        current_cmvs = (
            agg_m["cmvs"]
            if (metrics is not None and "aggregate" in metrics and "cmvs" in metrics["aggregate"])
            else None
        )

        if args.checkpoint_criterion == "cmvs":
            if current_cmvs is not None and current_cmvs < best_cmvs:
                best_cmvs = current_cmvs
                is_best = True
            elif current_cmvs is None and avg_val_loss < best_val_loss and best_cmvs == float("inf"):
                best_val_loss = avg_val_loss
                is_best = True
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                is_best = True

        if is_best:
            patience_counter = 0
            if args.exp_name:
                ckpt_path = models_dir / f"{args.exp_name}_champion.pt"
            elif args.spatial_mode or args.context_size != 16:
                ckpt_path = models_dir / f"spatial_{args.mode}_n{args.context_size:02d}_champion.pt"
            else:
                ckpt_path = models_dir / f"temporal_{args.mode}_h{args.history_len:02d}_champion.pt"
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "cmvs": current_cmvs,
                "best_cmvs": best_cmvs,
                "metrics": metrics,
                "args": vars(args),
                "normalization_stats": train_ds.stats,
            }
            torch.save(state_dict, ckpt_path)
            torch.save(state_dict, models_dir / f"temporal_{args.mode}_champion.pt")
            print(f"    [+] Saved champion checkpoint: {ckpt_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break

    total_training_sec = time.time() - start_total_time
    print("=" * 70)
    print(f"[+] Training completed in {total_training_sec:.1f} seconds ({total_training_sec/60:.2f} minutes).")
    print(f"    Measured 1-Epoch Rate: {history_log[0]['epoch_duration_sec']:.2f}s/epoch.")
    print("=" * 70)

    # Final Test Set Evaluation on Champion Checkpoint (2023 Season Holdout)
    print("=" * 70)
    print("[*] Running final DDIM-32 evaluation on Test Set (2023 holdout) with Champion Checkpoint...")
    print("=" * 70)
    champion_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(champion_ckpt["model_state_dict"])
    model.eval()

    test_preds_list = []
    test_targets_list = []
    test_coarse_list = []
    test_eval_start = time.time()

    with torch.no_grad():
        for batch in test_loader:
            history = batch["history"].to(device)
            fcst = batch["future_forecast"].to(device)
            terrain = batch["terrain"].to(device)
            target = batch["target"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if args.mode == "deterministic":
                    preds = model(history=history, future_forecast=fcst, terrain=terrain)
                else:
                    preds = model.sample(
                        history=history,
                        future_forecast=fcst,
                        terrain=terrain,
                        num_steps=args.ddim_steps,
                    )

            test_preds_list.append(preds.cpu().numpy())
            test_targets_list.append(target.cpu().numpy())
            test_coarse_list.append(fcst.cpu().numpy())
            if args.max_batches is not None and len(test_preds_list) >= args.max_batches:
                break

    test_preds_all = np.concatenate(test_preds_list, axis=0)
    test_targets_all = np.concatenate(test_targets_list, axis=0)
    test_coarse_all = np.concatenate(test_coarse_list, axis=0)

    test_preds_phys = invert_normalization(test_preds_all, train_ds.stats)
    test_targets_phys = invert_normalization(test_targets_all, train_ds.stats)
    test_coarse_phys = invert_normalization(test_coarse_all, train_ds.stats)

    test_metrics = compute_lead_metrics(test_preds_phys, test_targets_phys, test_coarse_phys)
    test_agg = test_metrics["aggregate"]
    test_duration = time.time() - test_eval_start

    print(
        f"[+] Test Evaluation (2023 Season): "
        f"Wet-MAE: {test_agg['precip_wet_mae']:.2f} mm | "
        f"Tmax-MAE: {test_agg['tmax_mae']:.2f} C | "
        f"Wind-RMSE: {test_agg['wind_vector_rmse']:.2f} m/s | "
        f"Duration: {test_duration:.1f}s"
    )

    # Persist report JSON
    if args.exp_name:
        report_file = reports_dir / f"{args.exp_name}_history.json"
    elif args.spatial_mode or args.context_size != 16:
        report_file = reports_dir / f"training_spatial_n{args.context_size:02d}_history.json"
    else:
        report_file = reports_dir / f"training_{args.mode}_h{args.history_len:02d}_history.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "history_len": args.history_len,
                "context_size": args.context_size,
                "linear_ratio": float(args.context_size) / 16.0,
                "area_ratio": float((args.context_size / 16.0) ** 2),
                "model_size": args.model_size,
                "total_epochs": len(history_log),
                "total_seconds": total_training_sec,
                "seconds_per_epoch": history_log[0]["epoch_duration_sec"] if history_log else 0.0,
                "best_val_loss": best_val_loss,
                "history": history_log,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
        )
    print(f"[+] Persisted metrics report: {report_file}")


if __name__ == "__main__":
    run_training()
