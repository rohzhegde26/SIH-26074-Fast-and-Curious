"""
scripts/train_sprint10_dense_l.py

Sprint 10: Champion Dense-L Denoiser Convergence and Extended Training Pipeline.
Features:
  - Resumes directly from Sprint 9 Dense-L weights (epoch 30, loss ~0.41065).
  - Unconstrained epoch budget with validation-loss tracking on the 2022 validation split.
  - Early stopping with patience=5 to prevent overfitting and capture the global optimum.
  - Saves the resulting champion model as sprint10_dense_l_champion.pt.
  - Exports training history to training_history.json.
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

curr_dir = Path(__file__).resolve().parent
if (curr_dir / "src").is_dir():
    ROOT = curr_dir
    if str(curr_dir) not in sys.path:
        sys.path.insert(0, str(curr_dir))
else:
    ROOT = curr_dir.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.models.residual_diffusion import compute_residual_target
from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)


def compute_sha256(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def resolve_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
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
        Path("src/data/sample_index_v2_h14.parquet"),
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
        Path("src/data/normalization_stats_v2.yaml"),
        Path("/kaggle/working/data/normalization_stats_v2.yaml"),
        Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml"),
    ]
    stats_path = next((p for p in stats_candidates if p.exists()), None)
    if stats_path is None and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*normalization_stats*.yaml"):
            stats_path = p
            break

    ckpt_candidates = [
        ROOT / "models" / "checkpoints" / "sprint9_dense_l_weights.pt",
        Path("models/checkpoints/sprint9_dense_l_weights.pt"),
        Path("sprint9_dense_l_weights.pt"),
        Path("/kaggle/input/sih26074-sprint9-checkpoints/sprint9_dense_l_weights.pt"),
        Path("/kaggle/working/models/checkpoints/sprint9_dense_l_weights.pt"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt_path is None and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob("*sprint9_dense_l*.pt"):
            ckpt_path = p
            break

    return zarr_path, index_path, stats_path, ckpt_path


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

        if not torch.isfinite(loss):
            print(f"    [WARNING] Non-finite loss ({loss.item()}). Skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"    [WARNING] Non-finite grad_norm ({grad_norm}). Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"    [WARNING] Non-finite grad_norm ({grad_norm}). Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    elapsed = max(1e-5, time.time() - t0)
    avg_loss = total_loss / max(1, num_batches)
    throughput = (num_batches * dataloader.batch_size) / elapsed
    return avg_loss, throughput


def evaluate_val_loss(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if max_batches is not None and b_idx >= max_batches:
                break
            history = batch["history"].to(device)
            future = batch["future_forecast"].to(device)
            terrain = batch["terrain"].to(device)
            target = batch["target"].to(device)

            r_0 = compute_residual_target(target, future)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                loss, _, _ = model.compute_training_loss(
                    r_0=r_0,
                    history=history,
                    future_forecast=future,
                    terrain=terrain,
                    target_norm=target,
                )

            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    parser = argparse.ArgumentParser(description="Sprint 10 Dense-L Convergence Pipeline")
    parser.add_argument("--epochs", type=int, default=30, help="Number of additional epochs to train (total 30 + epochs)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5, help="Fine-tuning learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience in epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models/checkpoints")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch synthetic test")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device)
    print("=" * 80)
    print("SPRINT 10: DENSE-L CHAMPION CONVERGENCE & EXTENDED TRAINING")
    print(f"Device: {device} | Fine-tuning LR: {args.lr} | Patience: {args.patience}")
    print("=" * 80)

    # Initialize model
    model = create_scalable_residual_diffusion(tier="dense_l").to(device)
    profile = model.profile_compute(device=device, batch_size=args.batch_size)
    print(f"[+] Total Parameters:    {profile['total_parameters']:,}")
    print(f"[+] Active Parameters:   {profile['active_parameters']:,}")

    if args.smoke_test:
        print("[*] Running 1-sample synthetic test...")
        b = 1
        h = torch.randn(b, 14, 6, 16, 16, device=device)
        f = torch.randn(b, 7, 6, 16, 16, device=device)
        terr = torch.randn(b, 5, 80, 80, device=device)
        y = torch.randn(b, 7, 6, 80, 80, device=device)
        r0 = compute_residual_target(y, f)
        loss, pred, _ = model.compute_training_loss(r0, h, f, terr, target_norm=y)
        loss.backward()
        print(f"[+] Smoke test PASSED: Loss={loss.item():.4f}")
        return

    zarr_path, index_path, stats_path, ckpt_path = resolve_paths()
    if zarr_path is None or index_path is None or stats_path is None:
        raise FileNotFoundError(
            "Authentic Zarr dataset not found! Ensure 'rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14' is attached."
        )

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "Sprint 9 Dense-L checkpoint not found! Ensure 'ssachithananthan/sih26074-sprint9-checkpoints' is attached."
        )

    print(f"[*] Resuming from Sprint 9 Checkpoint: {ckpt_path.name}")
    state = torch.load(ckpt_path, map_location=device)
    weights = state.get("model_state_dict", state)
    model.load_state_dict(weights, strict=True)

    start_epoch = int(state.get("epochs", state.get("epoch", 30)))
    prev_loss = float(state.get("loss", state.get("train_loss", 0.41065)))
    print(f"[+] Successfully loaded Dense-L weights from Epoch {start_epoch} (Prior Loss: {prev_loss:.5f})")

    ds_train = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="train",
        history_len=14,
        context_size=24,
    )
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    ds_val = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=14,
        context_size=24,
    )
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[+] Loaded Authentic Data: {len(ds_train)} train cubes (2015-2021) | {len(ds_val)} val cubes (2022)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"), init_scale=2048.0)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initial validation evaluation before fine-tuning
    print("[*] Evaluating baseline validation loss prior to fine-tuning...")
    initial_val_loss = evaluate_val_loss(model, val_loader, device, use_amp=(device.type == "cuda"))
    print(f"[+] Epoch {start_epoch:02d} Baseline Validation Loss: {initial_val_loss:.5f}")

    best_val_loss = initial_val_loss
    patience_counter = 0
    history = {
        "start_epoch": start_epoch,
        "initial_val_loss": initial_val_loss,
        "epochs": [],
    }

    total_target_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, total_target_epochs + 1):
        t_epoch_start = time.time()
        train_loss, throughput = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        val_loss = evaluate_val_loss(
            model=model,
            dataloader=val_loader,
            device=device,
            use_amp=(device.type == "cuda"),
        )
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        epoch_sec = time.time() - t_epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": cur_lr,
            "throughput_samples_sec": round(throughput, 2),
            "epoch_duration_sec": round(epoch_sec, 1),
        }
        history["epochs"].append(epoch_record)

        print(
            f"[Epoch {epoch:02d}/{total_target_epochs:02d}] "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"LR: {cur_lr:.6f} | Throughput: {throughput:.2f} s/s | Time: {epoch_sec:.1f}s"
        )

        # Check for improvement
        if val_loss < best_val_loss - 1e-4:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0

            champion_path = save_path / "sprint10_dense_l_champion.pt"
            ckpt_payload = {
                "epoch": epoch,
                "tier": "dense_l",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "param_count": profile["trainable_parameters"],
            }
            torch.save(ckpt_payload, champion_path)
            # Also save to current directory for guaranteed Kaggle output capture
            torch.save(ckpt_payload, "sprint10_dense_l_champion.pt")
            print(f"    [+] NEW CHAMPION! Val loss improved by {improvement:.5f}. Saved to {champion_path}")
        else:
            patience_counter += 1
            print(f"    [-] No validation improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print(f"[!] Early stopping triggered at Epoch {epoch}! Validation loss converged without overfitting.")
                break

    # Save final training history
    history_file = save_path / "sprint10_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open("sprint10_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[+] Training history exported to {history_file}")
    print("[+] Sprint 10 Model Convergence Phase Complete.")


if __name__ == "__main__":
    main()
