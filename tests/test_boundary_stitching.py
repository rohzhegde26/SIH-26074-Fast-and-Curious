"""
tests/test_boundary_stitching.py

Unit tests for C^1 complementary Hann window boundary stitching across adjacent NWP blocks.
Verifies continuity, zero checkerboard step jump (< 0.05mm), and partition of unity.
"""

import numpy as np
import pytest
import torch

from src.api.inference_service import stitch_overlapping_blocks, compute_hann_blend_weights


def test_hann_weights_partition_of_unity():
    """Verify that complementary Hann weights sum to 1.0 everywhere."""
    K = 10
    w_left, w_right = compute_hann_blend_weights(overlap=K)
    assert len(w_left) == K
    assert len(w_right) == K
    np.testing.assert_allclose(w_left + w_right, 1.0, atol=1e-6)
    # C1 smoothness checks: endpoints are close to 0 and 1
    assert w_left[0] > 0.95
    assert w_left[-1] < 0.05
    assert w_right[0] < 0.05
    assert w_right[-1] > 0.95


def test_boundary_stitching_constant_field():
    """Verify that stitching two identical constant blocks yields a perfectly uniform field."""
    # Two 80x80 blocks of value 5.0
    block_a = np.full((80, 80), 5.0, dtype=np.float32)
    block_b = np.full((80, 80), 5.0, dtype=np.float32)

    stitched = stitch_overlapping_blocks(block_a, block_b, overlap=10, axis=1)
    # Output width should be 80 + 80 - 10 = 150
    assert stitched.shape == (80, 150)
    np.testing.assert_allclose(stitched, 5.0, atol=1e-6)


def test_boundary_stitching_step_jump_under_threshold():
    """
    Verify that blending two slightly mismatched adjacent blocks produces
    a maximum adjacent cell step jump < 0.05 mm across the entire overlap zone.
    """
    # Block A: 80x80 with rainfall ~2.0 mm
    block_a = np.full((80, 80), 2.0, dtype=np.float32)
    # Block B: 80x80 with rainfall ~2.3 mm (a 0.3 mm synoptic mismatch at boundary)
    block_b = np.full((80, 80), 2.3, dtype=np.float32)

    stitched = stitch_overlapping_blocks(block_a, block_b, overlap=10, axis=1)
    # Seam region is columns 70 to 80
    diffs = np.abs(np.diff(stitched[:, 69:81], axis=1))
    max_step_jump = float(np.max(diffs))

    assert max_step_jump < 0.05, f"Step jump {max_step_jump:.4f} exceeds 0.05mm threshold!"


def test_boundary_stitching_vertical_axis():
    """Verify vertical boundary stitching along axis=0."""
    block_a = np.full((80, 80), 4.0, dtype=np.float32)
    block_b = np.full((80, 80), 4.2, dtype=np.float32)

    stitched = stitch_overlapping_blocks(block_a, block_b, overlap=10, axis=0)
    assert stitched.shape == (150, 80)
    diffs = np.abs(np.diff(stitched[69:81, :], axis=0))
    assert float(np.max(diffs)) < 0.05


def test_boundary_stitching_torch_tensor():
    """Verify that PyTorch tensors [B, C, H, W] can be stitched directly."""
    t_a = torch.full((1, 1, 80, 80), 10.0, dtype=torch.float32)
    t_b = torch.full((1, 1, 80, 80), 10.3, dtype=torch.float32)

    stitched = stitch_overlapping_blocks(t_a, t_b, overlap=10, axis=3)
    assert stitched.shape == (1, 1, 80, 150)
    assert isinstance(stitched, torch.Tensor)
    diffs = torch.abs(stitched[:, :, :, 1:] - stitched[:, :, :, :-1])
    assert float(torch.max(diffs)) < 0.05
