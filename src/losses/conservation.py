"""
src/losses/conservation.py

Physical Mass Conservation Grid-to-Grid (Kernel=5, Scale 5x) & Log-Domain Transform.

Mathematical Formulation:
    w_HR = cos(lat_HR_rad)
    C[HR] = avg_pool2d(HR * w_HR, k=5, s=5, count_include_pad=False) /
            avg_pool2d(w_HR, k=5, s=5, count_include_pad=False)
    L_cons = MSE(C[HR], LR)

Log-Domain Training Safeguard:
    1. Training in log-domain:
       x_log = log1p(x_phys)
       y_log = log1p(y_phys)
       pred_log = model(x_log)
       pred_phys = torch.clamp(expm1(pred_log), min=0.0)
    2. Composite Loss:
       L_total = L1(pred_log, y_log) + lambda_cons * L_cons(pred_phys, x_phys)
    3. CRITICAL: Conservation loss MUST be computed after expm1 in FP32 physical mm space.
       Average pooling in log space produces a geometric mean, violating mass conservation.
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log1p_transform(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Log-transform rainfall: log(1 + x). Stabilizes extreme heavy-tail skewness."""
    if isinstance(x, torch.Tensor):
        return torch.log1p(torch.clamp(x, min=0.0))
    return np.log1p(np.maximum(x, 0.0))


def expm1_transform(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Inverse log-transform: exp(x) - 1. Maps back to physical precipitation (mm)."""
    if isinstance(x, torch.Tensor):
        return torch.clamp(torch.expm1(x), min=0.0)
    return np.maximum(np.expm1(x), 0.0)


def coarsen_hr_to_lr_numpy(
    hr_grid: np.ndarray,
    hr_lats_deg: np.ndarray,
    kernel_size: int = 5,
    stride: int = 5,
) -> np.ndarray:
    """
    Area-weighted coarsening of HR (0.05°) grid to LR (0.25°) using cosine weights
    at HR centers via NumPy.
    
    Args:
        hr_grid: [H, W] or [B, H, W] array of HR precipitation.
        hr_lats_deg: [H] 1D array of latitude coordinates at HR centers.
        kernel_size: pooling kernel size (5).
        stride: pooling stride (5).
        
    Returns:
        Coarsened LR grid: [H/5, W/5] or [B, H/5, W/5].
    """
    has_batch = hr_grid.ndim == 3
    if not has_batch:
        hr_grid = hr_grid[np.newaxis, ...]  # [1, H, W]

    b, h, w = hr_grid.shape
    assert h % kernel_size == 0 and w % kernel_size == 0, (
        f"Dimensions ({h}, {w}) must be divisible by kernel size {kernel_size}"
    )

    out_h = h // kernel_size
    out_w = w // kernel_size

    orig_dtype = hr_grid.dtype
    # Cosine weights at HR centers in float64 for exact precision
    cos_lats = np.cos(np.radians(hr_lats_deg, dtype=np.float64))[np.newaxis, :, np.newaxis]
    w_hr = np.broadcast_to(cos_lats, (b, h, w))

    weighted_hr = hr_grid.astype(np.float64) * w_hr

    # Reshape for exact block average pooling: [B, out_h, k, out_w, k]
    w_hr_blocks = w_hr.reshape(b, out_h, kernel_size, out_w, kernel_size)
    hr_blocks = weighted_hr.reshape(b, out_h, kernel_size, out_w, kernel_size)

    # Sum over block axes (2, 4)
    sum_weighted = hr_blocks.sum(axis=(2, 4))
    sum_weights = w_hr_blocks.sum(axis=(2, 4))

    coarsened = sum_weighted / np.maximum(sum_weights, 1e-12)

    if not has_batch:
        return coarsened[0].astype(orig_dtype)
    return coarsened.astype(orig_dtype)


def coarsen_hr_to_lr_torch(
    hr_tensor: torch.Tensor,
    hr_lats_deg: torch.Tensor,
    kernel_size: int = 5,
    stride: int = 5,
) -> torch.Tensor:
    """
    Area-weighted coarsening of HR tensor to LR using cosine weights at HR centers via PyTorch.
    
    Args:
        hr_tensor: [B, C, H, W] tensor of HR predictions in physical mm.
        hr_lats_deg: [H] 1D tensor of HR center latitudes in degrees.
        kernel_size: 5
        stride: 5
        
    Returns:
        Coarsened LR tensor: [B, C, H/5, W/5].
    """
    b, c, h, w = hr_tensor.shape
    device = hr_tensor.device

    # Ensure hr_lats is on same device
    hr_lats_deg = hr_lats_deg.to(device)

    # Compute cosine weights at HR centers: [1, 1, H, 1]
    cos_lats = torch.cos(torch.deg2rad(hr_lats_deg)).view(1, 1, h, 1)
    w_hr = cos_lats.expand(b, c, h, w)

    # Numerator: avg_pool2d(HR * w_hr)
    num = F.avg_pool2d(
        hr_tensor * w_hr,
        kernel_size=kernel_size,
        stride=stride,
        count_include_pad=False,
    )

    # Denominator: avg_pool2d(w_hr)
    den = F.avg_pool2d(
        w_hr,
        kernel_size=kernel_size,
        stride=stride,
        count_include_pad=False,
    )

    coarsened = num / torch.clamp(den, min=1e-8)
    return coarsened


def conservation_loss_grid(
    hr_pred: Union[torch.Tensor, np.ndarray],
    lr_true: Union[torch.Tensor, np.ndarray],
    hr_lats_deg: Union[torch.Tensor, np.ndarray],
    kernel_size: int = 5,
    stride: int = 5,
) -> Union[torch.Tensor, float]:
    """
    Compute physical mass conservation loss between coarsened HR predictions and LR true input.
    
    L_cons = MSE(C[HR], LR)
    """
    if isinstance(hr_pred, torch.Tensor):
        if not isinstance(hr_lats_deg, torch.Tensor):
            hr_lats_deg = torch.tensor(hr_lats_deg, dtype=torch.float32, device=hr_pred.device)
        coarsened_lr = coarsen_hr_to_lr_torch(hr_pred, hr_lats_deg, kernel_size, stride)
        loss = F.mse_loss(coarsened_lr, lr_true)
        return loss
    else:
        coarsened_lr = coarsen_hr_to_lr_numpy(hr_pred, hr_lats_deg, kernel_size, stride)
        loss = float(np.mean((coarsened_lr - lr_true) ** 2))
        return loss


class CompositeLogConservationLoss(nn.Module):
    """
    Dual-Domain Composite Loss:
        - Reconstruction Loss in Log Domain: L1(pred_log, target_log)
        - Conservation Loss in Physical Domain: MSE(Coarsen(expm1(pred_log)), lr_phys)
    """

    def __init__(self, lambda_cons: float = 0.1, kernel_size: int = 5, stride: int = 5):
        super().__init__()
        self.lambda_cons = lambda_cons
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(
        self,
        pred_log: torch.Tensor,
        target_log: torch.Tensor,
        lr_phys: torch.Tensor,
        hr_lats_deg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Reconstruction loss in log domain (stabilizes gradients for extremes)
        loss_recon = F.l1_loss(pred_log, target_log)

        # 2. Invert to physical domain in FP32 and clamp non-negative
        pred_phys = torch.clamp(torch.expm1(pred_log.float()), min=0.0)

        # 3. Mass conservation loss in physical space (mm)
        loss_cons = conservation_loss_grid(
            pred_phys,
            lr_phys.float(),
            hr_lats_deg,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        total_loss = loss_recon + self.lambda_cons * loss_cons

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_recon_l1": loss_recon.detach(),
            "loss_conservation": loss_cons.detach(),
        }
        return total_loss, metrics
