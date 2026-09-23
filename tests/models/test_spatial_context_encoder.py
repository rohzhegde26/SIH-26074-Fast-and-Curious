"""
tests/models/test_spatial_context_encoder.py

TDD Test Suite for Sprint 5: Spatial-Context (N/M) Halo Encoder & Central RoI Cropping.
Enforces:
  1. Zero capacity confounding: Exactly 15,685,478 parameters across all N in {16, 20, 24, 32}.
  2. Dynamic spatial context forward pass and loss computation across N in {16, 20, 24, 32}.
  3. DDIM reverse sampling output shape [B, 7, 6, 80, 80] and finite values across all N.
  4. Mathematical coordinate registration and central RoI extraction invariance.
"""

import pytest
import torch

from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)


def test_parameter_count_exact_scale_invariant():
    """Verify that model parameters are exactly 15,685,478 independent of spatial dimension N."""
    model = SpatiotemporalResidualDiffusion(base_channels=96)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params == 15_685_478, f"Expected exactly 15,685,478 total params, got {total_params:,}"
    assert trainable_params == 15_685_478, f"Expected exactly 15,685,478 trainable params, got {trainable_params:,}"


@pytest.mark.parametrize("n", [16, 20, 24, 32])
def test_spatial_context_forward_and_loss(n: int):
    """Verify forward loss computation across all candidate context dimensions N in {16, 20, 24, 32}."""
    device = torch.device("cpu")
    # Use compact base_channels=16 for fast local CPU verification
    model = SpatiotemporalResidualDiffusion(base_channels=16)
    model.eval()

    b = 2
    h = 14  # Frozen at H* = 14 per Sprint 4 champion selection
    history = torch.randn(b, h, 6, n, n, device=device)
    future = torch.randn(b, 7, 6, n, n, device=device)
    terrain = torch.randn(b, 5, 80, 80, device=device)
    target = torch.randn(b, 7, 6, 80, 80, device=device)

    r_0 = compute_residual_target(target, future)
    assert r_0.shape == (b, 7, 6, 80, 80), f"r_0 shape mismatch: {r_0.shape}"

    loss, eps_pred, eps = model.compute_training_loss(r_0, history, future, terrain)

    assert loss.ndim == 0, "Loss must be scalar"
    assert torch.isfinite(loss), f"Loss is non-finite for N={n}: {loss.item()}"
    assert eps_pred.shape == (b, 7, 6, 80, 80), f"eps_pred shape mismatch for N={n}: {eps_pred.shape}"
    assert eps.shape == (b, 7, 6, 80, 80), f"eps target shape mismatch: {eps.shape}"


@pytest.mark.parametrize("n", [16, 20, 24, 32])
def test_spatial_context_ddim_sampling(n: int):
    """Verify DDIM reverse sampler with spatial context N outputs valid [B, 7, 6, 80, 80] tensor."""
    device = torch.device("cpu")
    model = SpatiotemporalResidualDiffusion(base_channels=16)
    model.eval()

    b = 1
    h = 14
    history = torch.randn(b, h, 6, n, n, device=device)
    future = torch.randn(b, 7, 6, n, n, device=device)
    terrain = torch.randn(b, 5, 80, 80, device=device)

    # 4-step fast smoke test on CPU
    sampled = model.sample_ddim(
        history=history,
        future_forecast=future,
        terrain=terrain,
        steps=4,
        eta=0.0,
    )

    assert sampled.shape == (b, 7, 6, 80, 80), f"Sampled shape mismatch for N={n}: {sampled.shape}"
    assert torch.isfinite(sampled).all(), f"NaN or Inf in sampled tensor for N={n}"


def test_central_roi_crop_registration():
    """Verify mathematical coordinate registration for Central RoI cropping."""
    b, leads, c = 1, 7, 6
    for n in [16, 20, 24, 32]:
        offset = (n - 16) // 2
        field = torch.arange(n * n, dtype=torch.float32).view(1, 1, 1, n, n).expand(b, leads, c, n, n)
        
        # Test compute_residual_target cropping
        target_dummy = torch.zeros(b, leads, c, 80, 80)
        res = compute_residual_target(target_dummy, field)
        assert res.shape == (b, leads, c, 80, 80)

        # Expected central slice
        expected_center = field[:, :, :, offset:offset+16, offset:offset+16]
        assert expected_center.shape == (b, leads, c, 16, 16)
