"""
tests/models/test_capacity_scaling.py

TDD Unit Tests for Sprint 9 Model Capacity Scaling Ladder:
  1. Dense-S: 15,685,478 parameters (exact Candidate 3 match).
  2. Dense-M: 25M to 35M parameters (31,198,518 measured).
  3. Dense-L: 45M to 65M parameters (51,997,958 measured).
  4. Active vs. Total parameter accounting for MoE.
  5. Tensor forward shape [B, 7, 6, 80, 80] and gradient backpropagation.
  6. Strict state_dict compatibility with Candidate 3 checkpoint.
"""

from pathlib import Path
import pytest
import torch

from src.models.residual_diffusion import compute_residual_target
from src.models.scalable_residual_diffusion import (
    ScalableSpatiotemporalResidualDiffusion,
    create_scalable_residual_diffusion,
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "models" / "checkpoints" / "sprint6_candidate3_multitask_champion.pt"


def test_dense_s_exact_parameter_count():
    """Verify Dense-S strictly matches Candidate 3 parameter count of 15,685,478."""
    model = create_scalable_residual_diffusion("dense_s")
    assert model.trainable_parameters == 15_685_478, (
        f"Dense-S must have exactly 15,685,478 parameters, got {model.trainable_parameters:,}"
    )


def test_dense_m_parameter_range():
    """Verify Dense-M satisfies target capacity range of [25M, 35M]."""
    model = create_scalable_residual_diffusion("dense_m")
    assert 25_000_000 <= model.trainable_parameters <= 35_000_000, (
        f"Dense-M parameters {model.trainable_parameters:,} outside [25M, 35M]"
    )
    assert model.trainable_parameters == 31_198_518


def test_dense_l_parameter_range():
    """Verify Dense-L satisfies target capacity range of [45M, 65M]."""
    model = create_scalable_residual_diffusion("dense_l")
    assert 45_000_000 <= model.trainable_parameters <= 65_000_000, (
        f"Dense-L parameters {model.trainable_parameters:,} outside [45M, 65M]"
    )
    assert model.trainable_parameters == 51_997_958


def test_moe_active_vs_total_parameters():
    """Verify MoE-4 has total > active parameters and top-k routing reduces compute."""
    model = create_scalable_residual_diffusion("moe_4")
    assert model.total_parameters > model.active_parameters
    profile = model.profile_compute()
    assert profile["use_moe"] is True
    assert profile["active_parameters"] < profile["total_parameters"]


@pytest.mark.parametrize("tier", ["dense_s", "dense_m"])
def test_forward_backward_pass(tier: str):
    """Verify forward and backward passes produce correct shapes and finite gradients."""
    device = torch.device("cpu")
    # For quick CPU testing, instantiate with compact channels
    if tier == "dense_s":
        model = ScalableSpatiotemporalResidualDiffusion(tier=tier, base_channels=16)
    else:
        model = ScalableSpatiotemporalResidualDiffusion(tier=tier, base_channels=24)

    model.train()
    b = 1
    h = 14
    history = torch.randn(b, h, 6, 16, 16, device=device)
    future = torch.randn(b, 7, 6, 16, 16, device=device)
    terrain = torch.randn(b, 5, 80, 80, device=device)
    target = torch.randn(b, 7, 6, 80, 80, device=device)

    r_0 = compute_residual_target(target, future)
    loss, eps_pred, target_val = model.compute_training_loss(r_0, history, future, terrain)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    assert eps_pred.shape == (b, 7, 6, 80, 80)
    assert target_val.shape == (b, 7, 6, 80, 80)

    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN or Inf gradient in {name}"


def test_checkpoint_strict_state_dict_loading():
    """Verify Candidate 3 checkpoint loads into Dense-S with strict=True if present."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not present locally: {CHECKPOINT_PATH}")

    raw_bytes = CHECKPOINT_PATH.read_bytes()
    if raw_bytes.startswith(b"version https://git-lfs.github.com"):
        pytest.skip("Checkpoint is unhydrated Git-LFS pointer on local machine")

    model = create_scalable_residual_diffusion("dense_s")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "model_state_dict" in state_dict:
        weights = state_dict["model_state_dict"]
    else:
        weights = state_dict

    # Must load strictly without missing or unexpected keys
    model.load_state_dict(weights, strict=True)
    assert model.trainable_parameters == 15_685_478
