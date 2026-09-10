"""
tests/test_multivariate_lapse_rate.py

Verification of thermodynamic lapse rate physics, diurnal range constraints,
psychrometric relative humidity (Magnus-Tetens formula), and topographic wind steering.
"""

import numpy as np
import pytest
import torch

from src.models.multivariate import (
    MultivariatePhysicalDownscaler,
    CLIMATOLOGY_DEFAULTS,
    PROVENANCE_TAG,
    magnus_tetens_es,
)


def test_magnus_tetens_es():
    """Verify saturation vapor pressure increases monotonically with temperature."""
    temps = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    es = magnus_tetens_es(temps)
    # At 0°C, es should be ~6.112 hPa
    assert torch.isclose(es[0], torch.tensor(6.112), atol=0.01)
    # Monotonic increase
    assert (es[1:] > es[:-1]).all()


def test_environmental_lapse_rate():
    """
    Verify environmental lapse rate dT/dh = -0.0065°C/m.
    A 1000m rise must produce a 6.5°C drop.
    """
    downscaler = MultivariatePhysicalDownscaler()

    # Create uniform coarse grids [1, 1, 16, 16]
    tmax_lr = torch.full((1, 1, 16, 16), 30.0, dtype=torch.float32)
    tmin_lr = torch.full((1, 1, 16, 16), 20.0, dtype=torch.float32)

    # Synthetic HR elevation: flat 500m on left half, 1500m on right half (delta = 1000m)
    elevation_hr = torch.full((1, 1, 80, 80), 500.0, dtype=torch.float32)
    elevation_hr[:, :, :, 40:] = 1500.0

    res = downscaler(
        tmax_lr=tmax_lr,
        tmin_lr=tmin_lr,
        elevation_hr=elevation_hr,
    )

    tmax_hr = res["tmax_hr"]
    tmin_hr = res["tmin_hr"]
    tmean_hr = res["tmean_hr"]

    # Mean drop between 500m and 1500m must equal 6.5°C
    delta_t_mean = (tmean_hr[:, :, :, :40].mean() - tmean_hr[:, :, :, 40:].mean()).item()
    assert abs(delta_t_mean - 6.5) < 0.2, f"Expected 6.5°C drop, got {delta_t_mean:.3f}°C"

    delta_t_max = (tmax_hr[:, :, :, :40].mean() - tmax_hr[:, :, :, 40:].mean()).item()
    assert abs(delta_t_max - 6.5) < 0.2, f"Expected 6.5°C drop in Tmax, got {delta_t_max:.3f}°C"


def test_melukote_ridge_cooler_than_valley():
    """
    Melukote ridge (~800m) must be strictly cooler than Kaveri valley (~650m)
    under uniform synoptic conditions by ~0.975°C.
    """
    downscaler = MultivariatePhysicalDownscaler()
    # High resolution elevation grid
    elev = torch.full((1, 1, 80, 80), 650.0, dtype=torch.float32)
    # Melukote ridge in center
    elev[:, :, 35:45, 35:45] = 800.0

    res = downscaler(elevation_hr=elev)
    tmax = res["tmax_hr"].squeeze()

    valley_temp = tmax[10, 10].item()
    ridge_temp = tmax[40, 40].item()

    assert ridge_temp < valley_temp, "Ridge must be cooler than valley"
    expected_diff = (800.0 - 650.0) * 0.0065  # 0.975°C
    actual_diff = valley_temp - ridge_temp
    assert abs(actual_diff - expected_diff) < 0.15


def test_diurnal_range_ordering_strict():
    """
    Verify Tmin <= Tmax on 100% of cells for uniform and noisy/extreme inputs.
    """
    downscaler = MultivariatePhysicalDownscaler()
    torch.manual_seed(42)

    # Test with random coarse grids and elevation anomalies
    for _ in range(5):
        tmax_lr = torch.randn(2, 1, 16, 16) * 5.0 + 30.0
        tmin_lr = tmax_lr - torch.rand(2, 1, 16, 16) * 8.0 - 1.0  # Tmin strictly less than Tmax initially
        elev = torch.rand(2, 1, 80, 80) * 1000.0 + 300.0

        res = downscaler(tmax_lr=tmax_lr, tmin_lr=tmin_lr, elevation_hr=elev)
        tmax = res["tmax_hr"]
        tmin = res["tmin_hr"]

        assert (tmin <= tmax).all(), "Thermodynamic order violation: Tmin > Tmax detected!"
        # Spread must be strictly positive
        spread = tmax - tmin
        assert (spread > 0.0).all(), "Zero or negative diurnal spread detected!"


def test_psychrometric_rh_magnus_saturation():
    """
    Verify that RH increases over elevated ridges due to adiabatic cooling,
    and remains strictly bounded in [10.0%, 100.0%].
    """
    downscaler = MultivariatePhysicalDownscaler()
    elev = torch.full((1, 1, 80, 80), 600.0, dtype=torch.float32)
    elev[:, :, 30:50, 30:50] = 1400.0  # 800m hill

    rh_lr = torch.full((1, 1, 16, 16), 70.0, dtype=torch.float32)

    res = downscaler(rh_lr=rh_lr, elevation_hr=elev)
    rh_hr = res["rh_hr"].squeeze()

    valley_rh = rh_hr[10, 10].item()
    hill_rh = rh_hr[40, 40].item()

    assert hill_rh > valley_rh, f"Hill RH ({hill_rh:.1f}%) should exceed valley RH ({valley_rh:.1f}%)"
    assert (rh_hr >= 10.0).all() and (rh_hr <= 100.0).all(), "RH outside valid bounds [10, 100]"


def test_topographic_wind_steering():
    """
    Verify wind speed accelerates over steep, exposed terrain slopes and
    is clamped within physical limits [0.5, 120.0] km/h.
    """
    downscaler = MultivariatePhysicalDownscaler()
    slope = torch.zeros((1, 1, 80, 80), dtype=torch.float32)
    slope[:, :, 30:50, 30:50] = 25.0  # Steep ridge

    w_orog = torch.zeros((1, 1, 80, 80), dtype=torch.float32)
    w_orog[:, :, 30:50, 30:50] = 0.8  # Strong orographic ascent

    wind_lr = torch.full((1, 1, 16, 16), 10.0, dtype=torch.float32)

    res = downscaler(wind_lr=wind_lr, slope_hr=slope, w_orog_hr=w_orog)
    wind_hr = res["wind_hr"].squeeze()

    flat_wind = wind_hr[5, 5].item()
    ridge_wind = wind_hr[40, 40].item()

    assert ridge_wind > flat_wind, f"Ridge wind ({ridge_wind:.1f}) should exceed flat wind ({flat_wind:.1f})"
    assert (wind_hr >= 0.5).all() and (wind_hr <= 120.0).all()


def test_climatology_fallback_and_provenance():
    """
    Verify that when coarse fields are missing, downscaler synthesizes climatological
    defaults and provides the required provenance tag.
    """
    downscaler = MultivariatePhysicalDownscaler()
    res = downscaler()

    assert res["provenance"] == PROVENANCE_TAG
    assert "tmax_hr" in res and "tmin_hr" in res and "rh_hr" in res and "wind_hr" in res
    assert res["tmax_hr"].shape == (1, 1, 80, 80)
    assert abs(res["tmax_hr"].mean().item() - CLIMATOLOGY_DEFAULTS["tmax"]) < 2.0
