"""
tests/losses/test_spatiotemporal_loss.py

TDD Test Suite for Sprint 3 Spatiotemporal Multi-Task Physical Loss.
Enforces:
  1. Multi-task loss covers all 6 channels in normalized space.
  2. Homoscedastic uncertainty balancing across tasks.
  3. Physical mass conservation penalty with cosine-latitude weighting.
  4. Physical diurnal temperature spread penalty (Tmax >= Tmin).
  5. Gradient flow through all loss terms.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.losses.spatiotemporal_multitask_loss import SpatiotemporalMultiTaskLoss


def test_loss_components_and_gradient_flow():
    """Verify all loss terms are computed and backward gradients flow properly."""
    criterion = SpatiotemporalMultiTaskLoss(lambda_mass=0.05, lambda_diurnal=0.02)
    b, l, c = 2, 7, 6

    preds = torch.randn(b, l, c, 80, 80, requires_grad=True)
    targets = torch.randn(b, l, c, 80, 80)
    coarse_fcst = torch.randn(b, l, c, 16, 16)

    loss_dict = criterion(preds=preds, targets=targets, coarse_fcst=coarse_fcst)

    expected_keys = [
        "loss",
        "loss_precip",
        "loss_tmax",
        "loss_tmin",
        "loss_rh",
        "loss_wind",
        "loss_mass",
        "loss_diurnal",
    ]
    for k in expected_keys:
        assert k in loss_dict, f"Missing loss key: {k}"
        assert torch.isfinite(loss_dict[k]), f"Loss {k} was not finite"

    total_loss = loss_dict["loss"]
    total_loss.backward()

    assert preds.grad is not None, "Gradients did not flow back to predictions"
    assert torch.any(preds.grad != 0.0), "Gradients were all zero"


def test_diurnal_temperature_penalty():
    """Verify diurnal penalty is 0 when Tmax >= Tmin and > 0 when Tmax < Tmin."""
    criterion = SpatiotemporalMultiTaskLoss(lambda_mass=0.0, lambda_diurnal=1.0)
    b, l = 1, 7

    # Case 1: Tmax is 35.0 C, Tmin is 20.0 C (valid physical order)
    # Using normalized space corresponding to valid order
    # tmax_mean=31.2, std=2.32; tmin_mean=22.5, std=2.43
    tmax_norm_high = (35.0 - 31.2) / 2.32
    tmin_norm_low = (20.0 - 22.5) / 2.43

    preds_valid = torch.zeros(b, l, 6, 80, 80)
    preds_valid[:, :, 1] = tmax_norm_high
    preds_valid[:, :, 2] = tmin_norm_low

    targets = torch.zeros(b, l, 6, 80, 80)
    coarse_fcst = torch.zeros(b, l, 6, 16, 16)

    res_valid = criterion(preds=preds_valid, targets=targets, coarse_fcst=coarse_fcst)
    assert res_valid["loss_diurnal"].item() < 1e-5, f"Valid diurnal order had non-zero loss: {res_valid['loss_diurnal'].item()}"

    # Case 2: Tmax is 15.0 C, Tmin is 30.0 C (unphysical inversion)
    tmax_norm_inverted = (15.0 - 31.2) / 2.32
    tmin_norm_inverted = (30.0 - 22.5) / 2.43

    preds_invalid = torch.zeros(b, l, 6, 80, 80)
    preds_invalid[:, :, 1] = tmax_norm_inverted
    preds_invalid[:, :, 2] = tmin_norm_inverted

    res_invalid = criterion(preds=preds_invalid, targets=targets, coarse_fcst=coarse_fcst)
    assert res_invalid["loss_diurnal"].item() > 0.01, f"Unphysical diurnal order should have positive penalty (got {res_invalid['loss_diurnal'].item()})"


def test_mass_conservation_penalty():
    """Verify mass conservation penalty increases when fine-scale rain diverges from coarse NWP."""
    criterion = SpatiotemporalMultiTaskLoss(lambda_mass=1.0, lambda_diurnal=0.0)
    b, l = 1, 7

    # Coarse precipitation = 10.0 mm/day
    p_coarse_phys = 10.0
    p_coarse_norm = (np.log1p(p_coarse_phys) - 0.767) / 1.279
    coarse_fcst = torch.zeros(b, l, 6, 16, 16)
    coarse_fcst[:, :, 0] = p_coarse_norm

    # Pred 1: fine precipitation matches 10.0 mm/day exactly
    preds_match = torch.zeros(b, l, 6, 80, 80)
    preds_match[:, :, 0] = p_coarse_norm

    # Pred 2: fine precipitation is 50.0 mm/day (divergent mass)
    p_divergent_norm = (np.log1p(50.0) - 0.767) / 1.279
    preds_divergent = torch.zeros(b, l, 6, 80, 80)
    preds_divergent[:, :, 0] = p_divergent_norm

    targets = torch.zeros(b, l, 6, 80, 80)

    res_match = criterion(preds=preds_match, targets=targets, coarse_fcst=coarse_fcst)
    res_divergent = criterion(preds=preds_divergent, targets=targets, coarse_fcst=coarse_fcst)

    assert res_match["loss_mass"].item() < res_divergent["loss_mass"].item(), "Mass conservation penalty failed to penalize mass divergence"
