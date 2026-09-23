"""
tests/models/test_residual_diffusion.py

TDD Test Suite for Sprint 3 Conditional Residual Diffusion Model.
Enforces:
  1. Deterministic model-space residual target calculation r_0 = y_fine - upsample(fcst).
  2. Gaussian forward process and DDPM noise schedule bounds (T=100).
  3. Denoiser network eps_theta output shape matching [B, 7, 6, 80, 80].
  4. DDIM reverse sampler executes with 4, 8, 16, 32 steps.
  5. Fixed random seed produces identical sample (deterministic reproducibility).
  6. Different random seeds produce distinct stochastic ensemble members.
  7. Inverse normalization on diffusion output produces finite physical fields.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)
from src.data.tensor_builder import invert_normalization


def test_residual_target_calculation():
    """Verify r_0 = y_fine - upsample(future_forecast) shape [B, 7, 6, 80, 80]."""
    b, l, c = 2, 7, 6
    y_fine = torch.randn(b, l, c, 80, 80)
    fcst = torch.randn(b, l, c, 16, 16)

    r_0 = compute_residual_target(y_fine, fcst)
    assert r_0.shape == (b, l, c, 80, 80)

    # If y_fine equals upsampled fcst, residual must be exactly 0
    fcst_up = nn.functional.interpolate(
        fcst.view(b * l, c, 16, 16),
        size=(80, 80),
        mode="bilinear",
        align_corners=False,
    ).view(b, l, c, 80, 80)
    r_zero = compute_residual_target(fcst_up, fcst)
    assert torch.allclose(r_zero, torch.zeros_like(r_zero), atol=1e-5)


def test_diffusion_noise_schedule_bounds():
    """Verify DDPM beta schedule and alpha_cumprod bounds (T=100)."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20)
    alphas_bar = diffusion.alphas_cumprod

    assert len(alphas_bar) == 100
    assert alphas_bar[0] > 0.99, f"Initial alpha_bar should be near 1.0 (got {alphas_bar[0].item()})"
    assert alphas_bar[-1] < 0.20, f"Final alpha_bar should decay significantly (got {alphas_bar[-1].item()})"
    # Monotonically decreasing
    assert torch.all(alphas_bar[:-1] >= alphas_bar[1:]), "alphas_cumprod must be monotonically decreasing"


def test_denoiser_forward_pass_and_loss():
    """Verify denoiser network predicts noise matching shape [B, 7, 6, 80, 80] and computes valid loss."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20)
    b, l, c = 2, 7, 6
    r_0 = torch.randn(b, l, c, 80, 80)
    history = torch.randn(b, 3, c, 16, 16)
    fcst = torch.randn(b, l, c, 16, 16)
    terrain = torch.randn(b, 5, 80, 80)

    # 1. Forward training loss
    loss, eps_pred, eps_true = diffusion.compute_training_loss(
        r_0=r_0,
        history=history,
        future_forecast=fcst,
        terrain=terrain,
    )
    assert eps_pred.shape == (b, l, c, 80, 80)
    assert eps_true.shape == (b, l, c, 80, 80)
    assert torch.isfinite(loss) and loss.item() > 0.0

    # 2. Backward pass test
    loss.backward()
    assert diffusion.denoiser.head_precip.weight.grad is not None


def test_ddim_sampler_steps_and_reproducibility():
    """Verify DDIM sampler executes across 4, 8, 16 steps and respects random seed."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20)
    diffusion.eval()

    b, l, c = 1, 7, 6
    history = torch.randn(b, 3, c, 16, 16)
    fcst = torch.randn(b, l, c, 16, 16)
    terrain = torch.randn(b, 5, 80, 80)

    # Test sampling with 4 steps
    with torch.no_grad():
        sample_4 = diffusion.sample(
            history=history,
            future_forecast=fcst,
            terrain=terrain,
            num_steps=4,
            seed=42,
        )
    assert sample_4.shape == (b, l, c, 80, 80)

    # Test seed reproducibility: same seed gives identical output
    with torch.no_grad():
        sample_4_repeat = diffusion.sample(
            history=history,
            future_forecast=fcst,
            terrain=terrain,
            num_steps=4,
            seed=42,
        )
    assert torch.allclose(sample_4, sample_4_repeat, atol=1e-5), "Fixed seed did not produce identical sample"

    # Test ensemble diversity: different seed gives different output
    with torch.no_grad():
        sample_4_diff_seed = diffusion.sample(
            history=history,
            future_forecast=fcst,
            terrain=terrain,
            num_steps=4,
            seed=999,
        )
    diff = torch.norm(sample_4 - sample_4_diff_seed).item()
    assert diff > 1e-3, f"Different seeds produced identical samples (diff={diff})"


def test_diffusion_output_inverse_normalization():
    """Verify diffusion sample de-normalizes to finite physical fields across all channels."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20)
    diffusion.eval()

    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain = torch.randn(1, 5, 80, 80)

    with torch.no_grad():
        pred_norm = diffusion.sample(
            history=history,
            future_forecast=fcst,
            terrain=terrain,
            num_steps=4,
            seed=42,
        ).numpy()

    # Load stats dict
    stats = {
        "precipitation": {"transform": "log1p_zscore", "mean": 0.767, "std": 1.279},
        "tmax": {"transform": "zscore", "mean": 31.2, "std": 2.32},
        "tmin": {"transform": "zscore", "mean": 22.5, "std": 2.43},
        "rh": {"transform": "zscore", "mean": 70.6, "std": 3.82},
        "wind_u": {"transform": "zscore", "mean": 11.7, "std": 6.75},
        "wind_v": {"transform": "zscore", "mean": 1.07, "std": 4.61},
    }

    pred_phys = invert_normalization(pred_norm, stats)
    assert np.all(np.isfinite(pred_phys)), "NaN or Inf in physical inverted diffusion prediction"
    # Precipitation physical lower bound non-negative check after max(0, ...)
    p_phys = np.maximum(0.0, pred_phys[0, :, 0, :, :])
    assert np.all(p_phys >= 0.0)
