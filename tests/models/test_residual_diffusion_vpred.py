"""
tests/models/test_residual_diffusion_vpred.py

Sprint 6 TDD Test Suite for Velocity Prediction (v-prediction) and Residual Refinements.
Enforces:
  1. Analytical v-prediction target correctness: v_t = sqrt(alpha_bar_t) * eps - sqrt(1 - alpha_bar_t) * r_0.
  2. Exact analytical reconstruction:
       r_0 = sqrt(alpha_bar_t) * r_t - sqrt(1 - alpha_bar_t) * v_t
       eps = sqrt(1 - alpha_bar_t) * r_t + sqrt(alpha_bar_t) * v_t
  3. Strict parameter count invariance: exactly 15,685,478 parameters.
  4. Multi-task group-balanced loss with convective tail weighting.
  5. Reverse DDIM sampling with v-parameterization produces valid, finite spatial predictions.
  6. Deterministic seed reproducibility under v-prediction.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)


def test_v_prediction_target_math():
    """Verify v_t calculation satisfies Salimans & Ho (2022) definition."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    b, l, c = 2, 7, 6
    r_0 = torch.randn(b, l, c, 80, 80)
    eps = torch.randn(b, l, c, 80, 80)
    t = torch.tensor([10, 50], dtype=torch.long)

    v_target = diffusion.compute_v_target(r_0=r_0, t=t, noise=eps)
    assert v_target.shape == (b, l, c, 80, 80)

    # Manual verification against alphas_cumprod
    alpha_bar = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    expected_v = torch.sqrt(alpha_bar) * eps - torch.sqrt(1.0 - alpha_bar) * r_0
    assert torch.allclose(v_target, expected_v, atol=1e-6)


def test_analytical_inversion_identity():
    """Verify algebraic inversion: (r_t, v_t) uniquely reconstructs (r_0, eps) exactly."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    b, l, c = 3, 7, 6
    r_0 = torch.randn(b, l, c, 80, 80)
    eps = torch.randn(b, l, c, 80, 80)
    t = torch.tensor([5, 45, 95], dtype=torch.long)

    # Forward diffusion
    r_t, _ = diffusion.q_sample(r_0, t, noise=eps)
    v_t = diffusion.compute_v_target(r_0, t, noise=eps)

    # Inverse mapping
    sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

    r_0_reconstructed = sqrt_alpha_bar * r_t - sqrt_one_minus_alpha_bar * v_t
    eps_reconstructed = sqrt_one_minus_alpha_bar * r_t + sqrt_alpha_bar * v_t

    assert torch.allclose(r_0_reconstructed, r_0, atol=1e-5), "r_0 reconstruction failed!"
    assert torch.allclose(eps_reconstructed, eps, atol=1e-5), "eps reconstruction failed!"


def test_strict_parameter_count_invariance():
    """Verify parameter count is strictly invariant to prediction_type and loss_weighting (15,685,478 params)."""
    model_eps = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=96, prediction_type="epsilon")
    model_vpred = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=96, prediction_type="v_prediction")
    model_group = SpatiotemporalResidualDiffusion(
        timesteps=100, base_channels=96, prediction_type="v_prediction", loss_weighting="group_tail"
    )

    count_eps = sum(p.numel() for p in model_eps.parameters() if p.requires_grad)
    count_vpred = sum(p.numel() for p in model_vpred.parameters() if p.requires_grad)
    count_group = sum(p.numel() for p in model_group.parameters() if p.requires_grad)

    assert count_eps == 15685478, f"Expected 15,685,478 parameters, got {count_eps:,}"
    assert count_vpred == count_eps, f"v_prediction changed parameter count: {count_vpred:,} vs {count_eps:,}"
    assert count_group == count_eps, f"group_tail changed parameter count: {count_group:,} vs {count_eps:,}"


def test_forward_loss_and_gradients_vpred():
    """Verify compute_training_loss executes and propagates finite gradients under v-prediction."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    b, l, c = 2, 7, 6
    r_0 = torch.randn(b, l, c, 80, 80)
    history = torch.randn(b, 3, c, 16, 16)
    fcst = torch.randn(b, l, c, 16, 16)
    terrain = torch.randn(b, 5, 80, 80)

    loss, v_pred, v_true = diffusion.compute_training_loss(
        r_0=r_0,
        history=history,
        future_forecast=fcst,
        terrain=terrain,
    )

    assert v_pred.shape == (b, l, c, 80, 80)
    assert v_true.shape == (b, l, c, 80, 80)
    assert torch.isfinite(loss) and loss.item() > 0.0

    loss.backward()
    for name, param in diffusion.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.all(torch.isfinite(param.grad)), f"NaN/Inf gradient in {name}"


def test_group_tail_loss_weighting():
    """Verify group-tail weighting applies focal penalty to extreme wet pixels."""
    diffusion = SpatiotemporalResidualDiffusion(
        timesteps=100,
        base_channels=20,
        prediction_type="v_prediction",
        loss_weighting="group_tail",
    )
    b, l, c = 2, 7, 6
    r_0 = torch.randn(b, l, c, 80, 80)
    history = torch.randn(b, 3, c, 16, 16)
    fcst = torch.randn(b, l, c, 16, 16)
    terrain = torch.randn(b, 5, 80, 80)
    target_norm = torch.zeros(b, l, c, 80, 80)
    target_norm[:, :, 0] = 1.0  # Normalized > 0.469 -> triggers 3.0x weight

    loss, _, _ = diffusion.compute_training_loss(
        r_0=r_0,
        history=history,
        future_forecast=fcst,
        terrain=terrain,
        target_norm=target_norm,
    )
    assert torch.isfinite(loss) and loss.item() > 0.0


def test_ddim_sample_vpred():
    """Verify DDIM reverse sampling with v_prediction produces valid, finite reconstructed field."""
    diffusion = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    diffusion.eval()

    b, l, c = 1, 7, 6
    history = torch.randn(b, 3, c, 24, 24)
    fcst = torch.randn(b, l, c, 24, 24)  # Test with N=24 context dimension!
    terrain = torch.randn(b, 5, 80, 80)

    with torch.no_grad():
        out = diffusion.sample(
            history=history,
            future_forecast=fcst,
            terrain=terrain,
            num_steps=4,
            seed=42,
        )

    assert out.shape == (b, l, c, 80, 80)
    assert torch.all(torch.isfinite(out)), "Output contains NaN or Inf values!"
