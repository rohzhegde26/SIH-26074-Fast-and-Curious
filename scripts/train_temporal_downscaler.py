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
    parser = argparse.ArgumentParser(description="Train Sprint 3 Spatiotemporal Downscaler")
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "diffusion"])
    parser.add_argument("--history_len", type=int, default=3, choices=[1, 2, 3], help="Antecedent history days")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (1 for probe)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Root directory for checkpoints & reports")
    parser.add_argument("--ddim_steps", type=int, default=16, help="Sampling steps for diffusion evaluation")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Epoch patience for early stop")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for quick smoke test")
    return parser.parse_args()


def get_model_channels(model_size: str) -> int:
    sizes = {"small": 16, "base": 24, "large": 32}
    return sizes.get(model_size, 24)


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

    # Load frozen datasets
    print("[*] Loading frozen datasets (multitask_temporal_v1.zarr)...")
    # Resolve Zarr path (support Kaggle attached input or local workspace)
    # Dynamic Zarr path resolution
    zarr_candidates = [
        ROOT / "datasets" / "multitask_temporal_v1.zarr",
        Path("/kaggle/working/datasets/multitask_temporal_v1.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/multitask_temporal_v1.zarr"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/multitask_temporal_v1.zarr/multitask_temporal_v1.zarr"),
    ]
    zarr_path = None
    for cand in zarr_candidates:
        if cand.exists() and ((cand / ".zgroup").exists() or (cand / "dates").exists()):
            zarr_path = cand
            break
    if zarr_path is None and Path("/kaggle/input").exists():
        for d in Path("/kaggle/input").rglob("multitask_temporal_v1.zarr"):
            if (d / ".zgroup").exists() or (d / "dates").exists():
                zarr_path = d
                break
    if zarr_path is None:
        zarr_path = ROOT / "datasets" / "multitask_temporal_v1.zarr"

    index_candidates = [
        ROOT / "data" / "sample_index.parquet",
        Path("/kaggle/working/data/sample_index.parquet"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/sample_index.parquet"),
    ]
    index_path = next((p for p in index_candidates if p.exists()), None)
    if index_path is None and Path("/kaggle/input").exists():
        found = list(Path("/kaggle/input").rglob("sample_index.parquet"))
        if found:
            index_path = found[0]
    if index_path is None:
        index_path = ROOT / "data" / "sample_index.parquet"

    stats_candidates = [
        ROOT / "data" / "normalization_stats.yaml",
        Path("/kaggle/working/data/normalization_stats.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v1/normalization_stats.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None and Path("/kaggle/input").exists():
        found = list(Path("/kaggle/input").rglob("normalization_stats.yaml"))
        if found:
            stats_path = found[0]
    if stats_path is None:
        stats_path = ROOT / "data" / "normalization_stats.yaml"

    print(f"[+] Resolved Zarr store:  {zarr_path}")
    print(f"[+] Resolved Index table: {index_path}")
    print(f"[+] Resolved Norm stats:  {stats_path}")

    train_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="train",
        history_len=args.history_len,
        normalize=True,
    )
    val_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=args.history_len,
        normalize=True,
    )

    print(f"[+] Loaded Train samples: {len(train_ds)} (2015-2021)")
    print(f"[+] Loaded Val samples:   {len(val_ds)} (2022)")

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

    # Build model and optimizer
    base_ch = get_model_channels(args.model_size)
    if args.mode == "deterministic":
        model = TemporalMultiTaskUNet5x(base_channels=base_ch)
        criterion = SpatiotemporalMultiTaskLoss().to(device)
    else:
        model = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=base_ch)
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
                        )
                        # For evaluation on validation, sample with DDIM
                        preds = model.sample(
                            history=history,
                            future_forecast=fcst,
                            terrain=terrain,
                            num_steps=args.ddim_steps,
                        )

                val_loss_sum += loss.item()
                val_batches += 1
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(target.cpu().numpy())
                val_coarse_list.append(fcst.cpu().numpy())
                if args.max_batches is not None and val_batches >= args.max_batches:
                    break

        avg_val_loss = val_loss_sum / max(1, val_batches)

        # De-normalize validation samples to physical space for meteorological metrics
        val_preds_all = np.concatenate(val_preds_list, axis=0)
        val_targets_all = np.concatenate(val_targets_list, axis=0)
        val_coarse_all = np.concatenate(val_coarse_list, axis=0)

        preds_phys = invert_normalization(val_preds_all, train_ds.stats)
        targets_phys = invert_normalization(val_targets_all, train_ds.stats)
        coarse_phys = invert_normalization(val_coarse_all, train_ds.stats)

        metrics = compute_lead_metrics(preds_phys, targets_phys, coarse_phys)
        agg_m = metrics["aggregate"]

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Wet-MAE: {agg_m['precip_wet_mae']:.2f} mm | "
            f"Tmax-MAE: {agg_m['tmax_mae']:.2f} C | "
            f"Wind-RMSE: {agg_m['wind_vector_rmse']:.2f} m/s | "
            f"Time: {epoch_duration:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_duration_sec": epoch_duration,
            "metrics": metrics,
        }
        history_log.append(epoch_record)

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            ckpt_path = models_dir / f"temporal_{args.mode}_champion.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "metrics": metrics,
                    "args": vars(args),
                    "normalization_stats": train_ds.stats,
                },
                ckpt_path,
            )
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

    # Persist report JSON
    report_file = reports_dir / f"training_{args.mode}_h{args.history_len}_history.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "history_len": args.history_len,
                "model_size": args.model_size,
                "total_epochs": len(history_log),
                "total_seconds": total_training_sec,
                "seconds_per_epoch": history_log[0]["epoch_duration_sec"],
                "best_val_loss": best_val_loss,
                "history": history_log,
            },
            f,
            indent=2,
        )
    print(f"[+] Persisted metrics report: {report_file}")


if __name__ == "__main__":
    run_training()
