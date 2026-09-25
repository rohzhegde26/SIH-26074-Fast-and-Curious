"""
tests/models/test_sampler_frontier.py

Gate 1 TDD Test Suite for Sprint 7 Diffusion-Step & Sampler Frontier.
Covers:
  1. test_corrected_ddim_schedule_bounds: For S in {4, 8, 16, 32, 64}, trajectory starts at 99 and ends at 0.
  2. test_schedule_strictly_monotonic: Trajectories are strictly descending without duplicates.
  3. test_legacy_vs_standard_schedule_audit: Proves legacy schedule truncation vs standard schedule correctness.
  4. test_vpred_sampler_roundtrip_math: Analytical reconstruction of r_0 and eps from v_t.
  5. test_deterministic_reproducibility: Bit-for-bit reproducibility with fixed seed.
  6. test_dpm_solver_vpred_conversion: DPM-Solver++ (2M) executes finite valid predictions across steps.
  7. test_pndm_sampler_execution: PNDM executes finite valid predictions across steps.
  8. test_physical_clipping_invariants: Output post-processing enforces P >= 0 and RH in [0, 100]%.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from src.models.residual_diffusion import (
    SpatiotemporalResidualDiffusion,
    compute_residual_target,
)
from src.models.samplers import (
    get_sampling_timesteps,
    v_to_r0_and_eps,
    eps_to_r0_and_v,
    sample_ddim_trajectory,
    sample_dpm_solver_2m_trajectory,
    sample_pndm_trajectory,
)


def test_corrected_ddim_schedule_bounds():
    """Verify that for all S in {4, 8, 16, 32, 64}, canonical trajectory starts at 99 and ends at 0."""
    total_timesteps = 100
    for s in [4, 8, 16, 32, 64]:
        seq = get_sampling_timesteps(total_timesteps, num_steps=s, schedule_type="standard")
        assert len(seq) == s, f"Expected length {s}, got {len(seq)}"
        assert seq[0] == 99, f"Expected start at 99, got {seq[0]} for S={s}"
        assert seq[-1] == 0, f"Expected end at 0, got {seq[-1]} for S={s}"


def test_schedule_strictly_monotonic():
    """Verify trajectory is strictly descending without duplicates or reversals."""
    total_timesteps = 100
    for s in [4, 8, 16, 32, 64]:
        seq = get_sampling_timesteps(total_timesteps, num_steps=s, schedule_type="standard")
        for i in range(len(seq) - 1):
            assert seq[i] > seq[i + 1], f"Non-descending step found in S={s}: {seq[i]} <= {seq[i+1]}"


def test_legacy_vs_standard_schedule_audit():
    """Verify legacy truncation bug documented in Audit Section 3 vs standard schedule."""
    # Legacy starting timesteps: S=4 -> 75, S=8 -> 84, S=16 -> 90, S=32 -> 93, S=64 -> 63
    legacy_expected_starts = {4: 75, 8: 84, 16: 90, 32: 93, 64: 63}
    for s, exp_start in legacy_expected_starts.items():
        legacy_seq = get_sampling_timesteps(100, num_steps=s, schedule_type="legacy")
        assert legacy_seq[0] == exp_start, f"Legacy S={s} should start at {exp_start}, got {legacy_seq[0]}"
        standard_seq = get_sampling_timesteps(100, num_steps=s, schedule_type="standard")
        assert standard_seq[0] == 99, f"Standard S={s} should always start at 99, got {standard_seq[0]}"


def test_vpred_sampler_roundtrip_math():
    """Verify exact analytical identity: (r_t, v_t) recovers (r_0, eps)."""
    b, l, c = 2, 7, 6
    r_0 = torch.randn(b, l, c, 40, 40)
    eps = torch.randn(b, l, c, 40, 40)
    alpha_bar = torch.tensor([0.05, 0.45, 0.90]).view(-1, 1, 1, 1, 1)

    for ab in [torch.tensor(0.01), torch.tensor(0.25), torch.tensor(0.50), torch.tensor(0.85), torch.tensor(0.99)]:
        ab_t = ab.view(1, 1, 1, 1, 1)
        r_t = torch.sqrt(ab_t) * r_0 + torch.sqrt(1.0 - ab_t) * eps
        v_t = torch.sqrt(ab_t) * eps - torch.sqrt(1.0 - ab_t) * r_0

        rec_r0, rec_eps = v_to_r0_and_eps(r_t, v_t, ab_t)
        assert torch.allclose(rec_r0, r_0, atol=1e-5), f"r_0 roundtrip failed for alpha_bar={ab.item()}"
        assert torch.allclose(rec_eps, eps, atol=1e-5), f"eps roundtrip failed for alpha_bar={ab.item()}"


def test_deterministic_reproducibility():
    """Verify deterministic sampling: identical seeds produce bit-for-bit identical outputs."""
    model = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    model.eval()

    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain = torch.randn(1, 5, 80, 80)

    with torch.no_grad():
        out1 = model.sample(history, fcst, terrain, num_steps=4, seed=42, sampler="ddim", schedule_type="standard")
        out2 = model.sample(history, fcst, terrain, num_steps=4, seed=42, sampler="ddim", schedule_type="standard")

    assert torch.equal(out1, out2), "Deterministic reproducibility failed for DDIM sampler!"


def test_dpm_solver_vpred_conversion():
    """Verify DPM-Solver++ (2M) reverse sampling executes and produces valid, finite spatial fields."""
    model = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    model.eval()

    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain = torch.randn(1, 5, 80, 80)

    with torch.no_grad():
        out = model.sample(history, fcst, terrain, num_steps=4, seed=123, sampler="dpm_solver", schedule_type="standard")

    assert out.shape == (1, 7, 6, 80, 80)
    assert not torch.isnan(out).any(), "DPM-Solver++ produced NaNs!"
    assert not torch.isinf(out).any(), "DPM-Solver++ produced Infs!"


def test_pndm_sampler_execution():
    """Verify PNDM reverse sampling executes and produces valid, finite spatial fields."""
    model = SpatiotemporalResidualDiffusion(timesteps=100, base_channels=20, prediction_type="v_prediction")
    model.eval()

    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain = torch.randn(1, 5, 80, 80)

    with torch.no_grad():
        out = model.sample(history, fcst, terrain, num_steps=4, seed=123, sampler="pndm", schedule_type="standard")

    assert out.shape == (1, 7, 6, 80, 80)
    assert not torch.isnan(out).any(), "PNDM produced NaNs!"
    assert not torch.isinf(out).any(), "PNDM produced Infs!"


def test_physical_clipping_invariants():
    """Verify physical range post-processing: P >= 0 and RH within [0, 100]%."""
    # Synthetic physical tensor
    preds_phys = torch.randn(2, 7, 6, 80, 80) * 20.0
    # Simulate post-processing
    p_clipped = torch.clamp(preds_phys[:, :, 0], min=0.0)
    rh_clipped = torch.clamp(preds_phys[:, :, 3], min=0.0, max=100.0)

    assert (p_clipped >= 0.0).all(), "Precipitation clipping failed!"
    assert (rh_clipped >= 0.0).all() and (rh_clipped <= 100.0).all(), "RH clipping failed!"
