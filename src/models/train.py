"""
src/models/train.py

Training pipeline for 5x direct super-resolution U-Net with physical mass conservation.
Features:
- Dual-domain Composite Log Conservation Loss (L1 log-error + linear mass conservation).
- Automatic Mixed Precision (torch.cuda.amp / torch.amp) for RTX 3060 / Kaggle T4.
- Honest evaluation metrics: All-Day MAE and Wet-Day MAE (Rain > 2.5 mm).
- Checkpoint management saving best models to models/checkpoints/.
"""

import os
import time
import argparse
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.unet_5x import UNet5x
from src.models.baselines import BilinearInterpolationBaseline, DeepSDBaseline
from src.models.dataset import get_dataloaders
from src.losses.conservation import CompositeLogConservationLoss, expm1_transform


def compute_metrics(
    pred_phys: torch.Tensor,
    true_phys: torch.Tensor,
    wet_threshold: float = 2.5,
) -> Dict[str, float]:
    """
    Computes All-Day MAE, Wet-Day MAE (> 2.5 mm), RMSE, and Pearson correlation.
    Strictly avoids zero-rain dry day inflation bias.
    """
    pred_np = pred_phys.detach().cpu().numpy().flatten()
    true_np = true_phys.detach().cpu().numpy().flatten()

    # All-day metrics
    all_mae = float(np.mean(np.abs(pred_np - true_np)))
    all_rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))

    # Wet-day metrics (condition on true rainfall > 2.5 mm)
    wet_mask = true_np > wet_threshold
    if np.sum(wet_mask) > 0:
        wet_mae = float(np.mean(np.abs(pred_np[wet_mask] - true_np[wet_mask])))
        wet_rmse = float(np.sqrt(np.mean((pred_np[wet_mask] - true_np[wet_mask]) ** 2)))
    else:
        wet_mae = all_mae
        wet_rmse = all_rmse

    # Pearson r
    if np.std(pred_np) > 1e-6 and np.std(true_np) > 1e-6:
        r_corr = float(np.corrcoef(pred_np, true_np)[0, 1])
    else:
        r_corr = 0.0

    return {
        "all_mae": all_mae,
        "all_rmse": all_rmse,
        "wet_mae": wet_mae,
        "wet_rmse": wet_rmse,
        "pearson_r": r_corr,
    }


def evaluate_model(
    model: nn.Module,
    val_loader,
    device: torch.device,
    is_log_model: bool = True,
    max_batches: Optional[int] = 50,
) -> Dict[str, float]:
    """Evaluates a model across the validation dataloader."""
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            lr = batch["lr"].to(device)
            hr_phys = batch["hr_phys"].to(device)

            if is_log_model:
                pred_log = model(lr)
                pred_phys = expm1_transform(pred_log)
            else:
                pred_phys = model(lr)

            all_preds.append(pred_phys.cpu())
            all_trues.append(hr_phys.cpu())

    all_preds_t = torch.cat(all_preds, dim=0)
    all_trues_t = torch.cat(all_trues, dim=0)

    return compute_metrics(all_preds_t, all_trues_t)


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: CompositeLogConservationLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    max_batches: int = 500,
) -> Tuple[float, float, float]:
    """Train single epoch with AMP fp16 and loss breakdown."""
    model.train()
    total_loss = 0.0
    total_log_err = 0.0
    total_cons_err = 0.0
    steps = 0

    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break

        lr_log = batch["lr"].to(device)
        hr_log = batch["hr"].to(device)
        lr_phys = batch["lr_phys"].to(device)
        lats = batch["lat"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred_log = model(lr_log)
            loss, loss_dict = criterion(
                pred_log=pred_log,
                target_log=hr_log,
                lr_phys=lr_phys,
                hr_lats_deg=lats,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss_dict["loss_total"])
        total_log_err += float(loss_dict.get("loss_log", loss_dict.get("loss_recon_l1", 0.0)))
        total_cons_err += float(loss_dict.get("loss_cons", loss_dict.get("loss_conservation", 0.0)))
        steps += 1

    return total_loss / steps, total_log_err / steps, total_cons_err / steps


def run_training(
    zarr_path: str = "data/cache/india_monsoon_patches.zarr",
    output_dir: str = "models/checkpoints",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    lambda_cons: float = 0.1,
    max_steps_per_epoch: int = 250,
):
    """Executes the full training and baseline comparison loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Training Loop on device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"[*] Connecting to Zarr patch cube at: {zarr_path}")
    train_loader, val_loader, _ = get_dataloaders(
        zarr_path=zarr_path,
        batch_size=batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        log_transform=True,
    )
    print(f"[*] Train patches: {len(train_loader.dataset)}, Val patches: {len(val_loader.dataset)}")

    # 2. Evaluate Baselines First to Set Benchmark Floors
    print("\n--- Evaluating Baseline Models on Val 2021 ---")
    bilinear = BilinearInterpolationBaseline().to(device)
    bilinear_metrics = evaluate_model(bilinear, val_loader, device=device, is_log_model=False)
    print(f"Bilinear Baseline  -> All MAE: {bilinear_metrics['all_mae']:.3f} mm | Wet MAE (>2.5mm): {bilinear_metrics['wet_mae']:.3f} mm | r: {bilinear_metrics['pearson_r']:.3f}")

    deepsd = DeepSDBaseline().to(device)
    deepsd_metrics = evaluate_model(deepsd, val_loader, device=device, is_log_model=False)
    print(f"DeepSD CNN Baseline -> All MAE: {deepsd_metrics['all_mae']:.3f} mm | Wet MAE (>2.5mm): {deepsd_metrics['wet_mae']:.3f} mm | r: {deepsd_metrics['pearson_r']:.3f}")

    # 3. Initialize 5x U-Net and Optimizer
    print("\n--- Training 5x Terrain-Conditioned U-Net ---")
    model = UNet5x(in_channels=1, out_channels=1, base_channels=32).to(device)
    criterion = CompositeLogConservationLoss(lambda_cons=lambda_cons)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val_mae = float("inf")
    best_checkpoint_path = os.path.join(output_dir, "best_5x_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_log_err, train_cons_err = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            max_batches=max_steps_per_epoch,
        )
        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device=device, is_log_model=True)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{dt:.1f}s] "
            f"Loss: {train_loss:.4f} (Log: {train_log_err:.4f}, Cons: {train_cons_err:.4f}) | "
            f"Val All MAE: {val_metrics['all_mae']:.3f} mm | "
            f"Val Wet MAE: {val_metrics['wet_mae']:.3f} mm | "
            f"Val r: {val_metrics['pearson_r']:.3f}",
            flush=True,
        )

        # Save checkpoint if best wet-day skill
        if val_metrics["wet_mae"] < best_val_mae:
            best_val_mae = val_metrics["wet_mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "lambda_cons": lambda_cons,
                },
                best_checkpoint_path,
            )
            print(f"  [+] Saved new best model checkpoint to {best_checkpoint_path}", flush=True)

    print(f"\n[SUCCESS] Training loop completed. Best Val Wet MAE: {best_val_mae:.3f} mm")
    return {
        "bilinear": bilinear_metrics,
        "deepsd": deepsd_metrics,
        "unet_5x_val_mae": best_val_mae,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5x U-Net with mass conservation")
    parser.add_argument("--zarr", default="data/cache/india_monsoon_patches.zarr", help="Path to Zarr patch store")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--steps", type=int, default=250, help="Max steps per epoch")
    args = parser.parse_args()

    run_training(
        zarr_path=args.zarr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps_per_epoch=args.steps,
    )
