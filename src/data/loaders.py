"""
src/data/loaders.py

Standardized gridded data loaders for IMD 0.25° and CHIRPS 0.05°.

Temporal Cutoff Documentation (03:00 UTC vs. Calendar Day):
    IMD daily observations represent 08:30 IST to 08:30 IST (03:00 UTC) accumulation,
    whereas CHIRPS is computed on a calendar-day window.
    Downstream per-0.25°-cell quantile mapping aligns the climatological cumulative
    distribution function (CDF), absorbing bulk offsets, but daily convective timing
    (15:00–19:00 IST) is an inherent boundary condition of 24-hour gridded data.

Grid Registration & Affine Transform:
    Direct 5x scaling: 0.25° / 0.05° = 5.0
    Kernel size: 5, Stride: 5
    HR Centers are strictly aligned with LR boundaries without systematic offset.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import xarray as xr


# Domain constants
INDIA_BBOX = {
    "min_lon": 68.0,
    "max_lon": 97.0,
    "min_lat": 8.0,
    "max_lat": 37.0,
}

LR_RES = 0.25
HR_RES = 0.05
SCALE_FACTOR = 5


def get_frozen_transform(
    config_path: str = "src/data/registration_transform.json",
) -> Dict:
    """Load the frozen registration affine transform parameters."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "scale_factor": 5.0,
        "hr_kernel_size": 5,
        "hr_stride": 5,
        "is_aligned": True,
        "frozen_affine_transform": {
            "scale_x": 5,
            "scale_y": 5,
            "shift_x_deg": 0.0,
            "shift_y_deg": 0.0,
            "hr_kernel_size": 5,
            "hr_stride": 5,
        },
    }


def create_india_grid_coords() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate coordinate arrays for LR (0.25°) and HR (0.05°) over the India monsoon domain.
    
    Returns:
        lr_lats, lr_lons, hr_lats, hr_lons
    """
    # LR: 0.25° cells
    lr_lats = np.arange(INDIA_BBOX["min_lat"] + LR_RES / 2.0, INDIA_BBOX["max_lat"], LR_RES)
    lr_lons = np.arange(INDIA_BBOX["min_lon"] + LR_RES / 2.0, INDIA_BBOX["max_lon"], LR_RES)

    # HR: 0.05° cells (nested with exact 5x factor)
    hr_lats = np.arange(INDIA_BBOX["min_lat"] + HR_RES / 2.0, INDIA_BBOX["max_lat"], HR_RES)
    hr_lons = np.arange(INDIA_BBOX["min_lon"] + HR_RES / 2.0, INDIA_BBOX["max_lon"], HR_RES)

    return (
        lr_lats.astype(np.float32),
        lr_lons.astype(np.float32),
        hr_lats.astype(np.float32),
        hr_lons.astype(np.float32),
    )


def compute_hr_cosine_weights(hr_lats_deg: np.ndarray, hr_shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute area-weighting matrix based on cosine of latitude at HR centers.
    
    w_HR = cos(lat_HR_rad)
    Broadcast across longitude dimension (shape: [H_hr, W_hr]).
    """
    lats_rad = np.radians(hr_lats_deg)
    cos_lats = np.cos(lats_rad).astype(np.float32)  # [H_hr]
    weights = np.repeat(cos_lats[:, np.newaxis], hr_shape[1], axis=1)
    return weights


class GriddedDataLoader:
    """
    Lazy loader for IMD (LR) and CHIRPS (HR) grids.
    Supports windowed patch slicing and casting to float32.
    """

    def __init__(
        self,
        imd_ds: Optional[xr.Dataset] = None,
        chirps_ds: Optional[xr.Dataset] = None,
        transform_config: Optional[Dict] = None,
    ):
        self.imd_ds = imd_ds
        self.chirps_ds = chirps_ds
        self.transform = transform_config or get_frozen_transform()
        self.lr_lats, self.lr_lons, self.hr_lats, self.hr_lons = create_india_grid_coords()

    def get_patch_pair(
        self,
        hr_row_start: int,
        hr_col_start: int,
        hr_size: int = 80,
        lr_size: int = 16,
    ) -> Dict[str, np.ndarray]:
        """
        Extract an 80x80 HR patch and its corresponding 16x16 LR context.
        Enforces 5x direct resolution scaling.
        """
        assert hr_size == lr_size * SCALE_FACTOR, (
            f"HR size ({hr_size}) must be exactly {SCALE_FACTOR}x LR size ({lr_size})"
        )

        lr_row_start = hr_row_start // SCALE_FACTOR
        lr_col_start = hr_col_start // SCALE_FACTOR

        patch_hr_lats = self.hr_lats[hr_row_start : hr_row_start + hr_size]
        patch_hr_lons = self.hr_lons[hr_col_start : hr_col_start + hr_size]
        patch_lr_lats = self.lr_lats[lr_row_start : lr_row_start + lr_size]
        patch_lr_lons = self.lr_lons[lr_col_start : lr_col_start + lr_size]

        weights_hr = compute_hr_cosine_weights(patch_hr_lats, (hr_size, hr_size))

        return {
            "hr_row_start": hr_row_start,
            "hr_col_start": hr_col_start,
            "lr_row_start": lr_row_start,
            "lr_col_start": lr_col_start,
            "hr_lats": patch_hr_lats,
            "hr_lons": patch_hr_lons,
            "lr_lats": patch_lr_lats,
            "lr_lons": patch_lr_lons,
            "weights_hr": weights_hr,
        }
