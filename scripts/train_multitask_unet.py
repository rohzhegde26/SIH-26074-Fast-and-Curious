"""
scripts/train_multitask_unet.py

Deterministic One-Shot Training Pipeline for MultiTaskUNet5x.
Trains the 5-variable weather downscaling network (Rain, Tmax, Tmin, RH, Wind)
with physical constraints and homoscedastic uncertainty loss balancing.

Supports:
    - Robust single-GPU / CPU default with Automatic Mixed Precision (AMP).
    - Multi-GPU DistributedDataParallel (DDP) via --distributed flag.
    - Checkpoints saved to models/checkpoints/multitask_5x_champion.pt.
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.multitask_unet import MultiTaskUNet5x
from src.losses.multitask_loss import MultiTaskPhysicalLoss
from src.data.multitask_dataset import get_multitask_dataloaders

CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiTaskUNet5x Weather Downscaler")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel dimension")
    parser.add_argument("--distributed", action="store_true", help="Enable PyTorch DDP")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset samples for rapid testing")
    parser.add_argument("--save_path", type=str, default=str(CHECKPOINTS_DIR / "multitask_5x_champion.pt"))
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates validation loss and meteorological error metrics."""
    model.eval()
    total_loss = 0.0
    rain_mae_sum = 0.0
    tmax_mae_sum = 0.0
    tmin_mae_sum = 0.0
    rh_mae_sum = 0.0
    wind_mae_sum = 0.0
    mass_err_pct_sum = 0.0
    count = 0

    with torch.no_grad():
        for coarse_nwp, fine_terrain, fine_targets in val_loader:
            coarse_nwp = coarse_nwp.to(device)
            fine_terrain = fine_terrain.to(device)
            fine_targets = fine_targets.to(device)

            preds = model(coarse_nwp, terrain_hr=fine_terrain)
            loss_dict = criterion(preds, fine_targets, coarse_nwp, fine_terrain)
            total_loss += loss_dict["loss"].item()

            p_pred = preds["rain"]
            p_tgt = fine_targets[:, 0:1]
            # Wet-cell MAE (target > 0.5 mm)
            wet_mask = p_tgt > 0.5
            if torch.any(wet_mask):
                rain_mae_sum += torch.mean(torch.abs(p_pred[wet_mask] - p_tgt[wet_mask])).item()
            else:
                rain_mae_sum += torch.mean(torch.abs(p_pred - p_tgt)).item()

            tmax_mae_sum += torch.mean(torch.abs(preds["tmax"] - fine_targets[:, 1:2])).item()
            tmin_mae_sum += torch.mean(torch.abs(preds["tmin"] - fine_targets[:, 2:3])).item()
            rh_mae_sum += torch.mean(torch.abs(preds["rh"] - fine_targets[:, 3:4])).item()
            wind_mae_sum += torch.mean(torch.abs(preds["wind"] - fine_targets[:, 4:5])).item()

            # Mass conservation error %
            p_pooled = criterion.pooler(p_pred)
            p_lr = coarse_nwp[:, 0:1]
            mass_err_pct_sum += (
                torch.mean(torch.abs(p_pooled - p_lr) / (p_lr + 1e-4)).item() * 100.0
            )

            count += 1

    return {
        "val_loss": total_loss / max(count, 1),
        "rain_wet_mae": rain_mae_sum / max(count, 1),
        "tmax_mae": tmax_mae_sum / max(count, 1),
        "tmin_mae": tmin_mae_sum / max(count, 1),
        "rh_mae": rh_mae_sum / max(count, 1),
        "wind_mae": wind_mae_sum / max(count, 1),
        "mass_err_pct": mass_err_pct_sum / max(count, 1),
    }


def train_multitask_model(args):
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] Training on device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_multitask_dataloaders(
        batch_size=args.batch_size,
        num_workers=0 if device.type == "cpu" else 2,
        pin_memory=(device.type == "cuda"),
        distributed=args.distributed,
        max_samples=args.max_samples,
    )

    print(f"[*] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model and loss
    model = MultiTaskUNet5x(base_channels=args.base_channels).to(device)
    criterion = MultiTaskPhysicalLoss().to(device)

    # All parameters including uncertainty weights
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    print("=" * 70)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<10} {'Rain MAE':<10} {'Tmax MAE':<10} {'RH MAE':<8} {'Time':<6}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for coarse_nwp, fine_terrain, fine_targets in train_loader:
            coarse_nwp = coarse_nwp.to(device)
            fine_terrain = fine_terrain.to(device)
            fine_targets = fine_targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds = model(coarse_nwp, terrain_hr=fine_terrain)
                loss_dict = criterion(preds, fine_targets, coarse_nwp, fine_terrain)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss_sum / max(train_batches, 1)

        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["val_loss"]

        print(
            f"{epoch:<6} {avg_train_loss:<12.4f} {val_loss:<10.4f} "
            f"{val_metrics['rain_wet_mae']:<10.2f} {val_metrics['tmax_mae']:<10.2f} "
            f"{val_metrics['rh_mae']:<8.2f} {epoch_time:<6.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(ckpt_dict, save_path)
            cwd_path = Path("multitask_5x_champion.pt")
            if cwd_path.resolve() != save_path.resolve():
                try:
                    torch.save(ckpt_dict, cwd_path)
                except Exception:
                    pass
            print(f"    [+] New best checkpoint saved to: {save_path.name} (val_loss={best_val_loss:.4f})")

    print("=" * 70)
    print(f"[+] Multi-task training completed! Best val loss: {best_val_loss:.4f}")
    return save_path


if __name__ == "__main__":
    args = parse_args()
    train_multitask_model(args)
