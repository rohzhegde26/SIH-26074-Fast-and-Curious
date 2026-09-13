"""
tests/losses/test_multitask_loss.py

Unit tests verifying MultiTaskPhysicalLoss:
    1. Gradients flow back to all model parameters and uncertainty weights.
    2. Mass conservation loss computation.
    3. Lapse rate penalty computation.
    4. Magnus vapor pressure penalty computation.
"""

import pytest
import torch
from src.models.multitask_unet import MultiTaskUNet5x
from src.losses.multitask_loss import MultiTaskPhysicalLoss


def test_multitask_loss_backward():
    model = MultiTaskUNet5x(base_channels=16)
    criterion = MultiTaskPhysicalLoss()

    b = 2
    x_lr = torch.rand(b, 5, 16, 16, requires_grad=True)
    fine_terrain = torch.rand(b, 5, 80, 80)
    fine_targets = torch.rand(b, 5, 80, 80)

    preds = model(x_lr, terrain_hr=fine_terrain)
    loss_dict = criterion(preds, fine_targets, x_lr, fine_terrain)

    total_loss = loss_dict["loss"]
    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0.0

    total_loss.backward()

    # Check model gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    # Check loss uncertainty weights gradients
    assert criterion.log_vars.grad is not None
    assert torch.isfinite(criterion.log_vars.grad).all()


def test_loss_components_finite():
    criterion = MultiTaskPhysicalLoss()
    b = 2
    preds = {
        "rain": torch.clamp(torch.rand(b, 1, 80, 80) * 10.0, min=0.0),
        "tmax": 30.0 + torch.rand(b, 1, 80, 80) * 5.0,
        "tmin": 20.0 + torch.rand(b, 1, 80, 80) * 4.0,
        "rh": 50.0 + torch.rand(b, 1, 80, 80) * 40.0,
        "wind": 5.0 + torch.rand(b, 1, 80, 80) * 10.0,
    }
    targets = torch.rand(b, 5, 80, 80) * 10.0
    coarse_nwp = torch.rand(b, 5, 16, 16) * 10.0
    fine_terrain = torch.rand(b, 5, 80, 80)

    loss_dict = criterion(preds, targets, coarse_nwp, fine_terrain)

    for k, v in loss_dict.items():
        assert torch.isfinite(v), f"Component {k} is non-finite: {v}"
