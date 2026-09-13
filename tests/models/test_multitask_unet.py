"""
tests/models/test_multitask_unet.py

Unit tests verifying MultiTaskUNet5x invariants:
    1. Shape consistency [B, 1, 80, 80] for all five variables.
    2. Physical invariant: Tmin <= Tmax strictly for 100% of pixels.
    3. Physical invariant: RH in [2.0, 100.0]% for 100% of pixels.
    4. Physical invariant: Wind >= 0.5 km/h for 100% of pixels.
    5. Physical invariant: Non-negative Rain (>= 0.0) with non-zero gradient for dry pixels.
"""

import pytest
import torch
from src.models.multitask_unet import (
    MultiTaskUNet5x,
    StraightThroughNonNegative,
    straight_through_non_negative,
)


def test_straight_through_non_negative():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5, 5.0], requires_grad=True)
    out = straight_through_non_negative(x)

    # Forward invariant: all outputs >= 0.0
    assert torch.all(out >= 0.0)
    assert out[0].item() == 0.0
    assert out[1].item() == 0.0
    assert out[2].item() == 0.0
    assert out[3].item() == 1.5
    assert out[4].item() == 5.0

    # Backward gradient: leaky 0.01 for x < 0, 1.0 for x >= 0
    loss = torch.sum(out)
    loss.backward()

    assert pytest.approx(x.grad[0].item(), rel=1e-4) == 0.01
    assert pytest.approx(x.grad[1].item(), rel=1e-4) == 0.01
    assert pytest.approx(x.grad[3].item(), rel=1e-4) == 1.0
    assert pytest.approx(x.grad[4].item(), rel=1e-4) == 1.0


def test_multitask_unet_forward_shapes():
    model = MultiTaskUNet5x(base_channels=16)
    model.eval()

    b = 2
    x_lr = torch.rand(b, 5, 16, 16)
    terrain_hr = torch.rand(b, 5, 80, 80)

    with torch.no_grad():
        preds = model(x_lr, terrain_hr=terrain_hr)

    assert "rain" in preds
    assert "tmax" in preds
    assert "tmin" in preds
    assert "rh" in preds
    assert "wind" in preds
    assert "tensor" in preds

    for k in ["rain", "tmax", "tmin", "rh", "wind"]:
        assert preds[k].shape == (b, 1, 80, 80), f"Mismatch for {k}: {preds[k].shape}"

    assert preds["tensor"].shape == (b, 5, 80, 80)


def test_multitask_physical_invariants():
    model = MultiTaskUNet5x(base_channels=16)
    model.eval()

    b = 4
    x_lr = torch.rand(b, 5, 16, 16)
    # Set realistic coarse values: Tmax ~ 30, Tmin ~ 20, RH ~ 60, Wind ~ 10
    x_lr[:, 0] = torch.rand(b, 16, 16) * 15.0  # Rain 0-15 mm
    x_lr[:, 1] = 28.0 + torch.rand(b, 16, 16) * 6.0  # Tmax 28-34
    x_lr[:, 2] = 18.0 + torch.rand(b, 16, 16) * 4.0  # Tmin 18-22
    x_lr[:, 3] = 40.0 + torch.rand(b, 16, 16) * 50.0  # RH 40-90
    x_lr[:, 4] = 2.0 + torch.rand(b, 16, 16) * 15.0  # Wind 2-17

    terrain_hr = torch.rand(b, 5, 80, 80)

    with torch.no_grad():
        preds = model(x_lr, terrain_hr=terrain_hr)

    rain = preds["rain"]
    tmax = preds["tmax"]
    tmin = preds["tmin"]
    rh = preds["rh"]
    wind = preds["wind"]

    # 1. Non-negative rain
    assert torch.all(rain >= 0.0), f"Found negative rain: min={rain.min().item()}"

    # 2. Strict diurnal temperature order: Tmax >= Tmin + 0.5
    spread = tmax - tmin
    assert torch.all(spread >= 0.499), f"Found Tmin > Tmax violation: min spread={spread.min().item()}"

    # 3. Bounded relative humidity
    assert torch.all(rh >= 2.0), f"RH underflow: min={rh.min().item()}"
    assert torch.all(rh <= 100.0), f"RH overflow: max={rh.max().item()}"

    # 4. Strictly positive surface wind
    assert torch.all(wind >= 0.5), f"Wind underflow: min={wind.min().item()}"
