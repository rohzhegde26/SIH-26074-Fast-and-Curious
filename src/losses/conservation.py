"""
src/losses/conservation.py

Physical Mass Conservation Grid-to-Grid (Kernel=5, Scale 5x).

Mathematical Formulation:
    w_HR = cos(lat_HR_rad)
    C[HR] = avg_pool2d(HR * w_HR, k=5, s=5, count_include_pad=False) /
            avg_pool2d(w_HR, k=5, s=5, count_include_pad=False)
    L_cons = MSE(C[HR], LR)

Critical Guardrails:
    1. Direct 5x scaling (0.25° / 0.05° = 5, k=5, s=5).
    2. Cosine weights computed at HR pixel centers, NOT LR.
    3. Never use naive summation (avoids the 25x Conservation-as-Sum Bug).
    4. count_include_pad=False prevents boundary zero-leakage.
"""

from typing import Union
import numpy as np
import torch
import torch.nn.functional as F


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

    # Reshape for exact block average pooling
    # [B, out_h, k, out_w, k]
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
        hr_tensor: [B, C, H, W] tensor of HR predictions.
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
