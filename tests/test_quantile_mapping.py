"""
tests/test_quantile_mapping.py

Unit tests for Sprint 3A Quantile Mapping service:
1. Verify QM is applied per 0.25° cell (not per 0.05° cell).
2. Verify intra-cell spatial variance is preserved after mapping (no texture flattening).
3. Verify mapped values are strictly non-negative (physical rainfall >= 0).
"""

import numpy as np
import pytest
import torch

from src.eval.calibration import QuantileMapper


def test_qm_applied_per_lr_cell():
    """
    Verify that QuantileMapper fits and transforms per 0.25° LR cell (16x16 grid)
    rather than per 0.05° HR cell.
    """
    np.random.seed(42)
    n_days = 100
    h_lr, w_lr = 16, 16

    # Create synthetic IMD LR and Model Coarsened LR with different cell biases
    imd_history = np.random.gamma(2.0, 5.0, size=(n_days, h_lr, w_lr)).astype(np.float32)
    # Cell (0, 0) has 2x model overestimation, Cell (15, 15) has 0.5x underestimation
    model_history = imd_history.copy()
    model_history[:, 0, 0] *= 2.0
    model_history[:, 15, 15] *= 0.5

    mapper = QuantileMapper(n_quantiles=50, lr_shape=(h_lr, w_lr))
    mapper.fit(imd_history, model_history)

    assert len(mapper.mappers) == h_lr * w_lr, f"Expected {h_lr*w_lr} cell mappers, got {len(mapper.mappers)}"
    assert (0, 0) in mapper.mappers
    assert (15, 15) in mapper.mappers

    # Test transforming coarse inputs: cell (0, 0) should be scaled down, (15, 15) scaled up
    test_input = np.ones((1, h_lr, w_lr), dtype=np.float32) * 20.0
    transformed = mapper.transform_lr(test_input)

    # Cell (0, 0) overestimation should be corrected downward
    assert transformed[0, 0, 0] < test_input[0, 0, 0], "Cell (0, 0) overestimation was not corrected"
    # Cell (15, 15) underestimation should be corrected upward
    assert transformed[0, 15, 15] > test_input[0, 15, 15], "Cell (15, 15) underestimation was not corrected"


def test_intra_cell_spatial_variance_preserved():
    """
    Verify that high-resolution 5x5 intra-cell texture is preserved
    and not flattened to a single constant block value.
    """
    np.random.seed(42)
    n_days = 50
    h_lr, w_lr = 16, 16
    h_hr, w_hr = 80, 80

    imd_lr = np.random.gamma(2.0, 5.0, size=(n_days, h_lr, w_lr)).astype(np.float32)
    model_lr = imd_lr * 1.5  # 50% systematic overestimation

    mapper = QuantileMapper(n_quantiles=50, lr_shape=(h_lr, w_lr))
    mapper.fit(imd_lr, model_lr)

    # Create synthetic HR patch with known non-zero spatial variance in 5x5 block
    hr_patch = np.zeros((h_hr, w_hr), dtype=np.float32)
    # Block 0: rows 0..5, cols 0..5 has strong orographic gradient
    hr_patch[:5, :5] = np.array([
        [2.0, 4.0, 6.0, 8.0, 10.0],
        [3.0, 5.0, 7.0, 9.0, 11.0],
        [4.0, 6.0, 8.0, 10.0, 12.0],
        [5.0, 7.0, 9.0, 11.0, 13.0],
        [6.0, 8.0, 10.0, 12.0, 14.0],
    ], dtype=np.float32)

    var_before = float(np.var(hr_patch[:5, :5]))
    assert var_before > 0.0, "Initial block must have non-zero spatial variance"

    mapped_hr = mapper.transform(hr_patch)
    var_after = float(np.var(mapped_hr[:5, :5]))

    # Variance must remain strictly positive (intra-cell texture not destroyed)
    assert var_after > 0.0, "Intra-cell spatial variance was collapsed to zero"
    # The relative variance pattern should be preserved
    assert abs(var_after) > 0.1 * var_before, "Intra-cell texture was excessively smoothed"


def test_mapped_values_non_negative():
    """
    Verify that all mapped outputs are strictly >= 0.0 (physical rainfall cannot be negative).
    """
    np.random.seed(42)
    n_days = 50
    h_lr, w_lr = 16, 16

    imd_lr = np.random.exponential(5.0, size=(n_days, h_lr, w_lr)).astype(np.float32)
    model_lr = np.random.exponential(8.0, size=(n_days, h_lr, w_lr)).astype(np.float32)

    mapper = QuantileMapper(n_quantiles=50, lr_shape=(h_lr, w_lr))
    mapper.fit(imd_lr, model_lr)

    # Test with negative / zero / positive inputs
    test_hr = np.random.uniform(-10.0, 50.0, size=(80, 80)).astype(np.float32)
    mapped_hr = mapper.transform(test_hr)

    assert np.all(mapped_hr >= 0.0), f"Found negative values in mapped HR: min = {np.min(mapped_hr)}"
    assert np.all(np.isfinite(mapped_hr)), "Found non-finite values in mapped HR"
