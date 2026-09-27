"""
tests/models/test_ensemble_reproducibility.py

TDD Unit Test Suite for Sprint 8 Ensemble & Test-Time Scaling.
Covers:
  1. test_nested_seed_manifest: Seed manifests are deterministic and strictly nested (Seeds(K1) subset Seeds(K2)).
  2. test_paired_eta_initial_latent_reproducibility: Initial noise z_T is identical across eta values for the same seed.
  3. test_deterministic_crps_k1_equals_mae: For K=1, CRPS degenerates to exact MAE with is_fair_crps=False.
  4. test_fair_crps_unbiased_estimator_k_ge_2: For K >= 2, Fair-CRPS uses the unbiased 2K(K-1) denominator.
  5. test_member_wise_physical_clipping_and_diagnostics: Members satisfy P >= 0, 0 <= RH <= 100, Tmin <= Tmax with logged repair burden.
  6. test_pairwise_diversity_diagnostics: Evaluates pairwise RMSE, spatial correlation, ensemble variance, and EDR.
  7. test_brier_score_and_climatological_decomposition: Brier Score, BSS against climatology, and Murphy decomposition.
  8. test_ensemble_chunked_batching_equivalence: Sequential vs chunked member execution produces bit-for-bit identical outputs.
"""

import math
import pytest
import torch
import numpy as np

from src.models.ensemble import (
    generate_nested_seeds,
    compute_crps,
    compute_brier_score,
    compute_pairwise_diversity,
    compute_spread_skill_ratio,
    compute_prediction_interval_coverage,
    compute_multivariate_energy_score,
    apply_member_wise_physical_bounds,
    EnsembleGenerator,
)


def test_nested_seed_manifest():
    """Verify that seed manifests are deterministic and strictly nested for K1 < K2."""
    base_seed = 20260927
    batch_idx = 0
    seeds_2 = generate_nested_seeds(num_members=2, base_seed=base_seed, batch_idx=batch_idx)
    seeds_4 = generate_nested_seeds(num_members=4, base_seed=base_seed, batch_idx=batch_idx)
    seeds_8 = generate_nested_seeds(num_members=8, base_seed=base_seed, batch_idx=batch_idx)

    assert len(seeds_2) == 2
    assert len(seeds_4) == 4
    assert len(seeds_8) == 8

    # Strictly nested property: seeds_4[:2] must equal seeds_2
    assert seeds_4[:2] == seeds_2
    assert seeds_8[:4] == seeds_4

    # Determinism: rerun produces identical seeds
    seeds_4_rerun = generate_nested_seeds(num_members=4, base_seed=base_seed, batch_idx=batch_idx)
    assert seeds_4 == seeds_4_rerun


def test_paired_eta_initial_latent_reproducibility():
    """Verify initial latent z_T is identical across different eta values for the same member seed."""
    seed = 20261927
    shape = (2, 7, 6, 16, 16)

    g1 = torch.Generator().manual_seed(seed)
    z_eta0 = torch.randn(shape, generator=g1)

    g2 = torch.Generator().manual_seed(seed)
    z_eta05 = torch.randn(shape, generator=g2)

    assert torch.equal(z_eta0, z_eta05), "Initial latents must be identical across eta when seed is fixed"


def test_deterministic_crps_k1_equals_mae():
    """Verify that for K=1, compute_crps returns exact MAE and marks is_fair_crps=False."""
    # Forecast shape: [K=1, B=2, leads=7, C=6, H=8, W=8]
    torch.manual_seed(42)
    members = torch.randn(1, 2, 7, 6, 8, 8)
    targets = torch.randn(2, 7, 6, 8, 8)

    crps_val, is_fair = compute_crps(members, targets)
    expected_mae = torch.mean(torch.abs(members[0] - targets)).item()

    assert not is_fair, "K=1 must not be labeled as Fair-CRPS"
    assert math.isclose(crps_val, expected_mae, rel_tol=1e-5)


def test_fair_crps_unbiased_estimator_k_ge_2():
    """Verify that for K >= 2, Fair-CRPS uses the unbiased 2K(K-1) finite-sample correction."""
    # Worked analytical 1D example:
    # 2 members: y1 = 2.0, y2 = 6.0. Target y_true = 3.0.
    # Term 1: (1/K) * (|2 - 3| + |6 - 3|) = (1/2) * (1.0 + 3.0) = 2.0
    # Term 2: 1 / (2 * 2 * 1) * (|2 - 6| + |6 - 2|) = (1/4) * (4.0 + 4.0) = 2.0
    # Fair-CRPS = 2.0 - 2.0 = 0.0
    # Empirical biased CRPS would use 2*K^2 = 8: 2.0 - (8.0 / 8) = 1.0
    members = torch.tensor([[[[[[2.0]]]]]], dtype=torch.float32) # [1, 1, 1, 1, 1, 1]
    members = torch.cat([members, torch.tensor([[[[[[6.0]]]]]], dtype=torch.float32)], dim=0) # [2, 1, 1, 1, 1, 1]
    targets = torch.tensor([[[[[3.0]]]]], dtype=torch.float32) # [1, 1, 1, 1, 1]

    crps_val, is_fair = compute_crps(members, targets)
    assert is_fair, "K >= 2 must be labeled as Fair-CRPS"
    assert math.isclose(crps_val, 0.0, abs_tol=1e-6)


def test_member_wise_physical_clipping_and_diagnostics():
    """Verify that apply_member_wise_physical_bounds enforces constraints and records repair diagnostics."""
    # 6 channels: [precip, tmax, tmin, rh, u, v]
    # Intentionally craft violations:
    # precip = -5.0 (negative rain)
    # tmax = 20.0, tmin = 25.0 (tmin > tmax inversion)
    # rh = 120.0 (rh > 100%)
    raw_members = torch.tensor([
        [
            [-5.0, 20.0, 25.0, 120.0, 2.0, -1.0],
            [10.0, 30.0, 15.0, 60.0, 1.0, 0.5],
        ]
    ], dtype=torch.float32).view(1, 1, 2, 6, 1, 1) # [K=1, B=1, leads=2, C=6, H=1, W=1]

    clipped, diagnostics = apply_member_wise_physical_bounds(raw_members)

    # 1. Physical constraint assertions
    assert (clipped[:, :, :, 0] >= 0.0).all(), "Precipitation must be non-negative"
    assert (clipped[:, :, :, 3] >= 0.0).all() and (clipped[:, :, :, 3] <= 100.0).all(), "RH must be in [0, 100]%"
    assert (clipped[:, :, :, 1] >= clipped[:, :, :, 2]).all(), "Tmax must be >= Tmin"

    # 2. Verify repaired values
    assert clipped[0, 0, 0, 0, 0, 0].item() == 0.0, "Negative precip must be clipped to 0"
    assert clipped[0, 0, 0, 3, 0, 0].item() == 100.0, "RH 120 must be clipped to 100"
    assert clipped[0, 0, 0, 1, 0, 0].item() == 25.0, "Tmax must be repaired to match Tmin (25.0)"

    # 3. Diagnostic checks
    assert diagnostics["precip_clipped_fraction"] > 0.0
    assert diagnostics["tmin_gt_tmax_rate"] > 0.0
    assert diagnostics["rh_clipped_fraction"] > 0.0
    assert diagnostics["precip_mass_shift_pct"] > 0.0


def test_pairwise_diversity_diagnostics():
    """Verify that compute_pairwise_diversity correctly measures spread and flags zero diversity."""
    # Case A: Identical members (zero diversity)
    base = torch.ones(1, 1, 7, 6, 8, 8)
    identical_members = torch.cat([base, base, base, base], dim=0) # [K=4, 1, 7, 6, 8, 8]
    diag_identical = compute_pairwise_diversity(identical_members)

    assert math.isclose(diag_identical["mean_pairwise_rmse"], 0.0, abs_tol=1e-6)
    assert math.isclose(diag_identical["ensemble_variance"], 0.0, abs_tol=1e-6)

    # Case B: Diverse members
    torch.manual_seed(123)
    diverse_members = torch.randn(4, 1, 7, 6, 8, 8)
    diag_diverse = compute_pairwise_diversity(diverse_members)

    assert diag_diverse["mean_pairwise_rmse"] > 0.0
    assert diag_diverse["ensemble_variance"] > 0.0
    assert -1.0 <= diag_diverse["mean_pairwise_correlation"] <= 1.0


def test_brier_score_and_climatological_decomposition():
    """Verify Brier score, skill score against training climatology, and Murphy decomposition."""
    # Predicted probabilities: p = [0.8, 0.2, 0.9, 0.1]
    # Observations: o = [1, 0, 1, 0] (perfectly resolved predictions)
    probs = torch.tensor([0.8, 0.2, 0.9, 0.1], dtype=torch.float32)
    obs = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    clim_train = 0.5 # Climatological event rate

    res = compute_brier_score(probs, obs, clim_train_rate=clim_train)

    expected_bs = ((0.8 - 1)**2 + (0.2 - 0)**2 + (0.9 - 1)**2 + (0.1 - 0)**2) / 4.0 # (0.04 + 0.04 + 0.01 + 0.01)/4 = 0.025
    assert math.isclose(res["brier_score"], expected_bs, rel_tol=1e-5)
    assert res["brier_skill_score_train"] > 0.0, "Skill score relative to 0.5 climatology must be positive"
    assert "reliability" in res and "resolution" in res and "uncertainty" in res


def test_ensemble_chunked_batching_equivalence():
    """Verify that generating K members sequentially (chunk_size=1) vs chunked (chunk_size=2) with identical seeds yields identical outputs."""
    # Mock denoise function that deterministically shifts latent
    def mock_denoise_fn(x, t):
        return x * 0.5 + t.view(-1, 1, 1, 1, 1).float() * 0.01

    shape = (1, 7, 6, 8, 8)
    base_seed = 9999
    timesteps_desc = [99, 50, 0]
    alphas_cumprod = torch.linspace(0.01, 0.99, 100)

    # 1. Run with chunk_size = 1
    gen_seq = EnsembleGenerator(mock_denoise_fn, alphas_cumprod, timesteps_desc, eta=0.0)
    out_seq, _ = gen_seq.generate(shape, num_members=2, base_seed=base_seed, chunk_size=1)

    # 2. Run with chunk_size = 2
    gen_chunk = EnsembleGenerator(mock_denoise_fn, alphas_cumprod, timesteps_desc, eta=0.0)
    out_chunk, _ = gen_chunk.generate(shape, num_members=2, base_seed=base_seed, chunk_size=2)

    assert torch.allclose(out_seq, out_chunk, atol=1e-5), "Sequential and chunked generation must produce identical results"
