"""
scripts/train_sprint9_dense.py

Training Pipeline for Sprint 9 Model Capacity Scaling Ladder:
  - Dense-S: 15.69M params (base_channels=96)
  - Dense-M: 31.20M params (base_channels=136)
  - Dense-L: 52.00M params (base_channels=176)
  - MoE-4:   Sparse Top-k routing (base_channels=96)

Preserves exact scientific invariants:
  - Multi-task v-prediction formulation
  - Convective group-tail loss weighting (lambda_precip=1.0, lambda_thermo=1.2, lambda_wind=1.1)
  - H=14 history days, N=24 coarse context, M=16 target central crop, 80x80 fine target
  - Mixed-precision AMP autocast with GradScaler
  - Checkpoint integrity tracking and SHA-256 recording
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.models.residual_diffusion import compute_residual_target
from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)

EXPECTED_CANDIDATE3_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    t0 = time.time()

    for batch in dataloader:
        history = batch["history"].to(device)
        future = batch["future_forecast"].to(device)
        terrain = batch["terrain"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        r_0 = compute_residual_target(target, future)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            loss, _, _ = model.compute_training_loss(
                r_0=r_0,
                history=history,
                future_forecast=future,
                terrain=terrain,
                target_norm=target,
            )

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    elapsed = max(1e-5, time.time() - t0)
    avg_loss = total_loss / max(1, num_batches)
    throughput = (num_batches * dataloader.batch_size) / elapsed
    return avg_loss, throughput


def main():
    parser = argparse.ArgumentParser(description="Sprint 9 Model Capacity Training Pipeline")
    parser.add_argument("--tier", type=str, default="dense_m", choices=list(TIER_CHANNEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch synthetic shape validation")
    parser.add_argument("--save-dir", type=str, default="models/checkpoints")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device)
    print(f"[*] Initializing Sprint 9 Capacity Training for tier: {args.tier.upper()} on {device}")

    # Build model with exact configured channels
    model = create_scalable_residual_diffusion(tier=args.tier).to(device)
    profile = model.profile_compute(device=device, batch_size=args.batch_size)

    print(f"[+] Total Parameters:    {profile['total_parameters']:,}")
    print(f"[+] Trainable Parameters: {profile['trainable_parameters']:,}")
    print(f"[+] Active Parameters:    {profile['active_parameters']:,}")
    print(f"[+] Ratio vs Candidate 3: {profile['parameter_ratio_vs_candidate3']:.2f}x")
    print(f"[+] Est. Peak VRAM:       {profile['estimated_vram_gb']:.2f} GB")

    if args.smoke_test:
        print("[*] Running 1-sample synthetic shape check...")
        b = 1
        h = torch.randn(b, 14, 6, 16, 16, device=device)
        f = torch.randn(b, 7, 6, 16, 16, device=device)
        terr = torch.randn(b, 5, 80, 80, device=device)
        y = torch.randn(b, 7, 6, 80, 80, device=device)
        r0 = compute_residual_target(y, f)
        loss, pred, _ = model.compute_training_loss(r0, h, f, terr, target_norm=y)
        loss.backward()
        print(f"[+] Smoke test PASSED: Loss={loss.item():.4f}, Pred shape={pred.shape}")
        return

    zarr_path, index_path, stats_path = resolve_dataset_paths()
    if zarr_path is None or index_path is None or stats_path is None:
        raise FileNotFoundError(
            "Authentic Zarr training dataset not found in local path or /kaggle/input!\n"
            "Attach 'rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14' to train on authentic data."
        )

    ds_train = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="train",
        history_len=14,
        context_size=24,
    )
    dataloader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"[+] Loaded authentic training dataset with {len(ds_train)} cubes (2015-2021).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss, throughput = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] Loss: {loss:.4f} | LR: {cur_lr:.6f} | Throughput: {throughput:.2f} samples/s")

        if loss < best_loss:
            best_loss = loss
            ckpt_file = save_path / f"sprint9_{args.tier}_best.pt"
            torch.save({
                "epoch": epoch,
                "tier": args.tier,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": loss,
                "param_count": profile["trainable_parameters"],
            }, ckpt_file)
            print(f"[+] Saved checkpoint to {ckpt_file} (SHA-256: {compute_sha256(ckpt_file)[:16]}...)")

    print(f"[+] Sprint 9 Training Completed for {args.tier.upper()}.")


if __name__ == "__main__":
    main()
