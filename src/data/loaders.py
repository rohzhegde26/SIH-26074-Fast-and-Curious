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

Header & NoData Sanitization Safeguards:
    - Auto-alias coordinate dimensions: lat in ['lat','latitude','lats','y'],
      lon in ['lon','longitude','lons','x'], time in ['time','valid_time','date'].
    - Ascending coordinate order enforcement (flips latitude if north-to-south).
    - Longitude normalization (converts [0, 360) -> [-180, 180) if required).
    - Sentinel NoData conversion (-9999.0, -9999, -999.0, < 0.0 -> 0.0).
    - Automatic int16 unpack with scale_factor (e.g. CHIRPS 0.1x).
    - Strict float32 casting.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr


# Domain constants (India monsoon domain)
INDIA_BBOX = {
    "min_lon": 68.0,
    "max_lon": 97.0,
    "min_lat": 8.0,
    "max_lat": 37.0,
}

LR_RES = 0.25
HR_RES = 0.05
SCALE_FACTOR = 5

DIM_ALIASES = {
    "lat": ["lat", "latitude", "lats", "y"],
    "lon": ["lon", "longitude", "lons", "x"],
    "time": ["time", "valid_time", "date"],
}

PRECIP_VAR_ALIASES = [
    "precip", "precipitation", "rainfall", "rain", "prcp", "rf", "tp"
]


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
    lr_lats = np.arange(INDIA_BBOX["min_lat"] + LR_RES / 2.0, INDIA_BBOX["max_lat"], LR_RES)
    lr_lons = np.arange(INDIA_BBOX["min_lon"] + LR_RES / 2.0, INDIA_BBOX["max_lon"], LR_RES)

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


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Sanitize and standardize an xarray Dataset:
    1. Aliases coordinates (lat, lon, time).
    2. Enforces ascending latitude order.
    3. Normalizes longitude from [0, 360) to [-180, 180) if required.
    4. Identifies precipitation variable, unpacks int16 scale factors,
       and maps negative / sentinel NoData to 0.0 in float32.
    """
    rename_map = {}
    for canonical, aliases in DIM_ALIASES.items():
        found = False
        for a in aliases:
            if a in ds.coords or a in ds.dims:
                if a != canonical:
                    rename_map[a] = canonical
                found = True
                break

    if rename_map:
        ds = ds.rename(rename_map)

    # 2. Check and enforce ascending latitude
    if "lat" in ds.coords:
        lat_vals = ds["lat"].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            ds = ds.reindex(lat=ds["lat"][::-1])

    # 3. Check and normalize longitude 0..360 -> -180..180
    if "lon" in ds.coords:
        lon_vals = ds["lon"].values
        if np.any(lon_vals > 180.0):
            ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))
            ds = ds.sortby("lon")

    # 4. Identify precipitation variable
    precip_var = None
    for cand in PRECIP_VAR_ALIASES:
        if cand in ds.data_vars:
            precip_var = cand
            break

    if precip_var is None:
        # Fallback to first non-coordinate variable
        non_coords = list(ds.data_vars)
        if non_coords:
            precip_var = non_coords[0]

    if precip_var is not None:
        da = ds[precip_var]
        
        # Scale factor handling (e.g. CHIRPS int16 stored with 0.1 scale factor)
        scale_factor = da.attrs.get("scale_factor", None)
        data = da.values.astype(np.float32)
        if scale_factor is not None and da.dtype in (np.int16, np.int32):
            data = data * float(scale_factor)
        elif da.dtype in (np.int16, np.int32) and "chirps" in str(ds.attrs.get("source", "")).lower():
            data = data * 0.1

        # NoData masking: convert sentinels (-9999, -999, < 0) to 0.0
        data[data < 0.0] = 0.0
        data[np.isnan(data)] = 0.0
        data[np.isinf(data)] = 0.0

        ds[precip_var] = (da.dims, data.astype(np.float32), da.attrs)

    return ds


def load_and_sanitize_netcdf(filepath: Union[str, Path]) -> xr.Dataset:
    """Load a NetCDF file with lazy xarray engine and apply full sanitization."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    ds = xr.open_dataset(path)
    return standardize_dataset(ds)


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
        self.imd_ds = standardize_dataset(imd_ds) if imd_ds is not None else None
        self.chirps_ds = standardize_dataset(chirps_ds) if chirps_ds is not None else None
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

        result = {
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

        # Extract data slices if datasets are attached
        if self.chirps_ds is not None:
            for v in PRECIP_VAR_ALIASES:
                if v in self.chirps_ds.data_vars:
                    da = self.chirps_ds[v]
                    # Select slice by index
                    hr_slice = da.isel(
                        lat=slice(hr_row_start, hr_row_start + hr_size),
                        lon=slice(hr_col_start, hr_col_start + hr_size),
                    ).values
                    result["hr_precip"] = hr_slice.astype(np.float32)
                    break

        if self.imd_ds is not None:
            for v in PRECIP_VAR_ALIASES:
                if v in self.imd_ds.data_vars:
                    da = self.imd_ds[v]
                    lr_slice = da.isel(
                        lat=slice(lr_row_start, lr_row_start + lr_size),
                        lon=slice(lr_col_start, lr_col_start + lr_size),
                    ).values
                    result["lr_precip"] = lr_slice.astype(np.float32)
                    break

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and sanitize gridded raster files")
    parser.add_argument("--test-file", type=str, required=True, help="Path to NetCDF raster to sanitize and test")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    print(f"Testing and sanitizing: {test_path}")
    ds = load_and_sanitize_netcdf(test_path)
    print("\n--- Standardized Dataset Structure ---")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Data Variables: {list(ds.data_vars)}")
    
    for v in ds.data_vars:
        arr = ds[v].values
        print(f"\nVariable '{v}':")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Min: {float(np.min(arr)):.4f}, Max: {float(np.max(arr)):.4f}, Mean: {float(np.mean(arr)):.4f}")
        print(f"  NaN count: {int(np.isnan(arr).sum())}")
        print(f"  Negative count: {int((arr < 0).sum())}")
        assert np.min(arr) >= 0.0, "Sanitization failed: negative values found!"
        assert np.isnan(arr).sum() == 0, "Sanitization failed: NaNs found!"

    print("\n[SUCCESS] Dataset passed all sanitization assertions.")
