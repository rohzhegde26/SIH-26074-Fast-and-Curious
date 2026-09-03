"""
test_conservation.py
Unit tests for mass conservation laws, detection of the 25x sum bug,
and clean polygon zonal aggregation without double-counting.
"""

import numpy as np
import pytest
import torch
from src.losses.conservation import (
    coarsen_hr_to_lr_numpy,
    coarsen_hr_to_lr_torch,
    conservation_loss_grid,
)
from src.data.zonal_aggregation import ZonalAggregator, compute_spherical_cell_area_m2


def test_conservation_exact():
    """
    On a uniform constant field (e.g., 25.0 mm), downscaling then coarsening
    must match the input exactly within 1e-6.
    """
    h_hr, w_hr = 80, 80
    constant_val = 25.0
    hr_constant = np.full((h_hr, w_hr), constant_val, dtype=np.float32)
    hr_lats = np.linspace(12.0, 16.0, h_hr, dtype=np.float32)

    lr_coarsened = coarsen_hr_to_lr_numpy(hr_constant, hr_lats, kernel_size=5, stride=5)

    assert lr_coarsened.shape == (16, 16)
    max_err = float(np.max(np.abs(lr_coarsened - constant_val)))
    assert max_err < 1e-6, f"Conservation on constant field failed: max error {max_err} >= 1e-6"


def test_conservation_detects_sum_bug():
    """
    Assert that naive summation of 5x5 HR cells results in a ~25x error,
    proving that naive summation violates physical conservation.
    """
    h_hr, w_hr = 25, 25
    constant_val = 10.0
    hr_field = np.full((h_hr, w_hr), constant_val, dtype=np.float32)

    # Naive sum pooling over 5x5 blocks
    naive_sum = hr_field.reshape(5, 5, 5, 5).sum(axis=(1, 3))
    expected_sum_val = constant_val * 25.0  # 250.0 mm!

    assert np.allclose(naive_sum, expected_sum_val)
    # Discrepancy ratio is exactly 25x
    discrepancy_ratio = float(np.mean(naive_sum) / constant_val)
    assert np.isclose(discrepancy_ratio, 25.0), f"Expected 25x discrepancy, got {discrepancy_ratio}"


def test_cosine_variation_across_india():
    """
    Assert that cos(lat) varies by ~19% between 8°N (Kerala/Tamil Nadu) and 37°N (Kashmir),
    demonstrating that cosine weighting is physically mandatory across All-India.
    """
    lat_south = 8.0
    lat_north = 37.0

    cos_south = np.cos(np.radians(lat_south))
    cos_north = np.cos(np.radians(lat_north))

    variation_pct = float((cos_south - cos_north) / cos_south) * 100.0
    # cos(8°) ≈ 0.99026, cos(37°) ≈ 0.79863 -> variation ≈ 19.35%
    assert 18.0 <= variation_pct <= 21.0, f"Expected ~19% variation, got {variation_pct:.2f}%"


def test_torch_numpy_conservation_consistency():
    """
    Verify exact equivalence between PyTorch and NumPy implementations of conservation pooling.
    """
    np.random.seed(42)
    hr_np = np.random.uniform(0.0, 100.0, (1, 1, 40, 40)).astype(np.float32)
    lats_np = np.linspace(10.0, 12.0, 40, dtype=np.float32)

    coarsened_np = coarsen_hr_to_lr_numpy(hr_np[0, 0], lats_np, kernel_size=5, stride=5)

    hr_torch = torch.from_numpy(hr_np)
    lats_torch = torch.from_numpy(lats_np)
    coarsened_torch = coarsen_hr_to_lr_torch(hr_torch, lats_torch, kernel_size=5, stride=5).numpy()[0, 0]

    assert np.allclose(coarsened_np, coarsened_torch, atol=1e-5), "Torch and NumPy coarsening diverged"


def test_zonal_polygon_aggregation_constant():
    """
    Verify that zonal polygon aggregation on a uniform field yields the exact
    constant field value for all Mandya panchayats.
    """
    aggregator = ZonalAggregator(geojson_path="data/processed/mandya_full.geojson")
    test_val = 42.0
    results = aggregator.aggregate_constant(value=test_val)

    assert len(results) >= 80, f"Aggregated {len(results)} panchayats, expected >= 80"
    for gpcode, val in results.items():
        assert np.isclose(val, test_val, atol=1e-5), f"GP {gpcode} got {val} != {test_val}"


def test_spherical_cell_area_consistency():
    """Verify analytic spherical cell area at 13°N matches ~30.1 km^2."""
    area_m2 = compute_spherical_cell_area_m2(lat_deg=13.0, d_lat_deg=0.05, d_lon_deg=0.05)
    area_km2 = area_m2 / 1e6
    assert 29.0 <= area_km2 <= 31.0, f"Expected ~30.1 km^2, got {area_km2:.2f} km^2"
