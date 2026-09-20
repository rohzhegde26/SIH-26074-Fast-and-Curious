"""
tests/test_data_pipelines.py

Unit and physics validation tests for the real data ingestion, terrain processing,
loss functions, and gridded benchmark pipelines.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import xarray as xr

from src.data.glo30_terrain import compute_terrain_derivatives
from src.data.terrain_features import (
    build_terrain_tensor_5ch,
    compute_orographic_velocity,
    DEM_MEAN,
    DEM_STD,
)
from src.losses.multitask_loss import MultiTaskPhysicalLoss
from scripts.benchmark_multitask_baselines import bilinear_sample_point
from src.data.multitask_dataset import MultiTaskPanchayatDataset
from src.data.real_data_ingestion import REAL_CACHE_FILE

ROOT = Path(__file__).resolve().parents[1]
DEM_FILE = ROOT / "data" / "raw" / "dem" / "glo30_mandya_terrain.nc"


def test_glo30_terrain_bounds_and_shapes():
    """
    Assert that Copernicus GLO-30 DSM terrain dataset exists and matches
    the canonical 4°x4° Peninsular domain (11-15°N, 74-78°E, 80x80).
    """
    assert DEM_FILE.exists(), f"Missing {DEM_FILE}"
    with xr.open_dataset(DEM_FILE) as ds:
        assert ds["elevation"].shape == (80, 80)
        assert np.isclose(float(ds["lat"].values[0]), 14.975, atol=0.01)
        assert np.isclose(float(ds["lat"].values[-1]), 11.025, atol=0.01)
        assert np.isclose(float(ds["lon"].values[0]), 74.025, atol=0.01)
        assert np.isclose(float(ds["lon"].values[-1]), 77.975, atol=0.01)


def test_orographic_lift_physical_ascent():
    """
    Assert that a southerly wind (v > 0) blowing against terrain that rises northward
    produces positive orographic lift (forced ascent) in both PyTorch and NumPy branches.
    """
    # Create synthetic elevation ramp rising from South (row 79 = 100m) to North (row 0 = 2000m)
    ramp_y = np.linspace(2000.0, 100.0, 80, dtype=np.float32)
    elev_grid = np.tile(ramp_y[:, None], (1, 80))

    # 1. NumPy derivatives from glo30_terrain
    slope, aspect, curv, w_orog_np = compute_terrain_derivatives(elev_grid)
    # Southerly wind: u = 0, v = +5 m/s hitting north-rising terrain -> positive lift
    assert np.all(w_orog_np > 0.0), "Expected positive orographic lift for windward ascent"

    # 2. PyTorch compute_orographic_velocity from terrain_features
    elev_t = torch.from_numpy(elev_grid)
    w_orog_torch = compute_orographic_velocity(
        elev_t,
        center_lat_deg=13.0,
        u_wind=0.0,
        v_wind=5.0,
    )
    # PyTorch normalized velocity should be strictly positive (ascent) in the interior
    interior = w_orog_torch[2:-2, 2:-2]
    assert torch.all(interior > 0.0), "PyTorch orographic velocity must be positive for ascent"


def test_multitask_loss_elevation_unnormalization():
    """
    Verify that dem_norm is correctly unnormalized to meters in MultiTaskPhysicalLoss
    and that a physically realistic environmental lapse rate (6.5 K/km) produces 0 penalty.
    """
    criterion = MultiTaskPhysicalLoss()

    # Domain elevations within unclipped normalization range [-2.5, 3.5]
    elevs = np.array([0.0, 382.5, 1000.0, 1980.0], dtype=np.float32)
    # Standard normalization: clamp((h - 382.5)/458.2, -2.5, 3.5)/3.5
    dem_norm = np.clip((elevs - DEM_MEAN) / DEM_STD, -2.5, 3.5) / 3.5

    # MultiTaskPhysicalLoss unnormalization: elev_norm * 1603.7 + 382.5
    elev_unnorm = dem_norm * (3.5 * DEM_STD) + DEM_MEAN
    assert np.allclose(elevs, elev_unnorm, atol=1e-3), "Unnormalization did not invert normalization"

    # Verify a physically realistic lapse rate of 6.5 K/km over a mountain ridge
    # Terrain rises by 500m across 10 cells
    b = 1
    y, x = np.mgrid[0:80, 0:80].astype(np.float32)
    terrain_m = 500.0 + 15.0 * y  # dz/dy = 15 m/cell (~2.7 K/km)
    t_mean_grid = 30.0 - 0.0975 * y  # dT/dy = -0.0975 C/cell -> dT/dz = -0.0975 / 15 = -0.0065 C/m = 6.5 K/km

    t_norm = np.clip((terrain_m - DEM_MEAN) / DEM_STD, -2.5, 3.5) / 3.5
    fine_terrain = torch.zeros(b, 5, 80, 80)
    fine_terrain[:, 0] = torch.from_numpy(t_norm)

    preds = {
        "rain": torch.zeros(b, 1, 80, 80),
        "tmax": torch.from_numpy(t_mean_grid + 5.0).unsqueeze(0).unsqueeze(0),
        "tmin": torch.from_numpy(t_mean_grid - 5.0).unsqueeze(0).unsqueeze(0),
        "rh": torch.full((b, 1, 80, 80), 75.0),
        "wind": torch.full((b, 1, 80, 80), 10.0),
    }
    targets = torch.zeros(b, 5, 80, 80)
    coarse_nwp = torch.zeros(b, 5, 16, 16)

    loss_dict = criterion(preds, targets, coarse_nwp, fine_terrain)
    # Lapse rate loss should be 0.0 because 6.5 K/km is within standard [4.0, 9.8] K/km bounds
    assert torch.isclose(loss_dict["loss_lapse"], torch.tensor(0.0), atol=1e-5)


def test_bilinear_sample_point_exactness():
    """
    Verify 4-point bilinear sampling against analytical coordinates and boundaries.
    """
    grid = np.zeros((80, 80), dtype=np.float32)
    # Cell center coordinates:
    # row 0 center is 14.975N, row 79 center is 11.025N
    # col 0 center is 74.025E, col 79 center is 77.975E
    grid[0, 0] = 42.0
    grid[0, 1] = 50.0
    grid[79, 79] = 100.0

    # 1. Exact cell center 0, 0
    val_00 = bilinear_sample_point(grid, lat=14.975, lon=74.025)
    assert np.isclose(val_00, 42.0, atol=1e-4)

    # 2. Midpoint between cell (0, 0) and (0, 1)
    val_mid = bilinear_sample_point(grid, lat=14.975, lon=74.050)
    assert np.isclose(val_mid, 46.0, atol=1e-4)

    # 3. Exact cell center 79, 79
    val_7979 = bilinear_sample_point(grid, lat=11.025, lon=77.975)
    assert np.isclose(val_7979, 100.0, atol=1e-4)

    # 4. Safe clamping for boundary points outside
    val_out = bilinear_sample_point(grid, lat=15.1, lon=73.9)
    assert np.isclose(val_out, 42.0, atol=1e-4)


def test_dataset_partition_sizes():
    """
    Assert that MultiTaskPanchayatDataset has exactly 1,220 samples partitioned into:
    Train: 976 (80.0%), Val: 122 (10.0%), Test: 122 (10.0%).
    """
    if not REAL_CACHE_FILE.exists():
        pytest.skip("multitask_real.npz cache file not materialized yet")

    ds_train = MultiTaskPanchayatDataset(split="train")
    ds_val = MultiTaskPanchayatDataset(split="val")
    ds_test = MultiTaskPanchayatDataset(split="test")

    assert len(ds_train) == 976, f"Train count {len(ds_train)} != 976"
    assert len(ds_val) == 122, f"Val count {len(ds_val)} != 122"
    assert len(ds_test) == 122, f"Test count {len(ds_test)} != 122"
    assert len(ds_train) + len(ds_val) + len(ds_test) == 1220
