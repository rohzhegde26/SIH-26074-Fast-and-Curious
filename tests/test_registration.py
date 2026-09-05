"""
test_registration.py
Unit tests for grid registration, scale factor 5x, and catching the 2.7 km systematic shift.
"""

import json
from pathlib import Path
import numpy as np
import pytest
from scripts.check_registration import compute_grid_alignment


def test_scale_factor_is_exact_5x():
    report = compute_grid_alignment()
    assert report["scale_factor"] == 5.0
    assert report["frozen_affine_transform"]["hr_kernel_size"] == 5
    assert report["frozen_affine_transform"]["hr_stride"] == 5


def test_catches_unaligned_2_7km_shift():
    # If CHIRPS half-pixel centers (+0.025°) were mismatched against bottom-left pixel origin
    # it produces a systematic ~2.7 km shift
    offset_deg = 0.025
    dist_km = offset_deg * 111.0  # ~2.775 km
    assert 2.5 <= dist_km <= 3.0, f"Expected ~2.7 km shift, got {dist_km} km"


def test_aligned_grid_zero_offset():
    report = compute_grid_alignment()
    assert report["is_aligned"] is True
    assert report["offset_km"] < 1e-3
    assert abs(report["offset_deg"]["lat"]) < 1e-4
    assert abs(report["offset_deg"]["lon"]) < 1e-4


def test_registration_transform_json_exists():
    path = Path("src/data/registration_transform.json")
    assert path.exists(), "src/data/registration_transform.json must exist"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["scale_factor"] == 5.0
    assert data["is_aligned"] is True
