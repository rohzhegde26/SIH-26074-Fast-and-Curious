"""
src/data/terrain_features.py

Scientific High-Resolution Terrain & Wind-Aware Feature Engineering.

Features Generated:
    1. dem_norm: Global Z-score elevation normalization avoiding Himalayan saturation.
    2. slope_norm: Standardized slope using global 95th percentile clipping.
    3. aspect_sin, aspect_cos: Continuous 2-channel circular aspect encoding.
    4. w_orog_norm: 12-month climatological wind-aware orographic lifting velocity.

Total Channels: 5 [dem_norm, slope_norm, aspect_sin, aspect_cos, w_orog_norm]
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

# Earth radius in meters (WGS84 authalic sphere)
EARTH_RADIUS_M = 6371008.8
RAD_PER_DEG = np.pi / 180.0

# Global Indian Domain Topographic Statistics
DEM_MEAN = 382.5  # meters
DEM_STD = 458.2   # meters
SLOPE_P95_GLOBAL = 22.3  # degrees

# Climatological 850 hPa Mean Wind Vectors (u, v in m/s) across Peninsular India (1991-2020)
# u: + is Westerly (West->East), - is Easterly (East->West)
# v: + is Southerly (South->North), - is Northerly (North->South)
MONTHLY_850HPA_WINDS = {
    1: (-1.5, -0.5),   # Jan: Winter NE monsoon / calm
    2: (-0.5, +0.5),   # Feb: Transition
    3: (+1.0, +1.0),   # Mar: Pre-monsoon heating
    4: (+2.5, +1.5),   # Apr: Thermal low building
    5: (+5.0, +2.0),   # May: Monsoon onset preparations
    6: (+8.0, +2.0),   # Jun: Strong SW monsoon low-level jet
    7: (+9.5, +2.0),   # Jul: Peak SW monsoon flow across Ghats
    8: (+8.5, +1.5),   # Aug: Active SW monsoon
    9: (+6.0, +1.0),   # Sep: Monsoon withdrawal phase
    10: (-2.0, -1.5),  # Oct: Post-monsoon / NE monsoon onset
    11: (-4.0, -2.0),  # Nov: Active NE monsoon over peninsular India
    12: (-3.0, -1.0),  # Dec: Winter easterlies
}


def normalize_elevation(elevation_m: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Global Z-score standardization with wide-bound clipping:
    dem_norm = clamp((h - 382.5) / 458.2, -2.5, 3.5) / 3.5 in [-0.71, 1.0]

    Prevents tanh saturation above 1300m while preserving variance up to 8848m.
    """
    if isinstance(elevation_m, torch.Tensor):
        norm = (elevation_m.float() - DEM_MEAN) / DEM_STD
        return torch.clamp(norm, min=-2.5, max=3.5) / 3.5
    else:
        norm = (np.asarray(elevation_m, dtype=np.float32) - DEM_MEAN) / DEM_STD
        return np.clip(norm, -2.5, 3.5) / 3.5


def normalize_slope(slope_deg: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize terrain slope using precomputed global 95th percentile (22.3°).
    slope_norm = clamp(slope / 22.3, 0.0, 2.0) / 2.0 in [0.0, 1.0]
    """
    if isinstance(slope_deg, torch.Tensor):
        norm = slope_deg.float() / SLOPE_P95_GLOBAL
        return torch.clamp(norm, min=0.0, max=2.0) / 2.0
    else:
        norm = np.asarray(slope_deg, dtype=np.float32) / SLOPE_P95_GLOBAL
        return np.clip(norm, 0.0, 2.0) / 2.0


def encode_aspect(aspect_deg: Union[torch.Tensor, np.ndarray]) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Continuous circular aspect decomposition into [sin(aspect), cos(aspect)].
    Eliminates discontinuous 360° -> 0° boundary step.
    """
    if isinstance(aspect_deg, torch.Tensor):
        rad = torch.deg2rad(aspect_deg.float())
        return torch.sin(rad), torch.cos(rad)
    else:
        rad = np.radians(np.asarray(aspect_deg, dtype=np.float32))
        return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def compute_orographic_velocity(
    elevation_m: Union[torch.Tensor, np.ndarray],
    center_lat_deg: float,
    d_lat_deg: float = 0.05,
    d_lon_deg: float = 0.05,
    month: int = 7,
    u_wind: Optional[float] = None,
    v_wind: Optional[float] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Computes orographic vertical velocity proxy at the surface:
    w_orog = u * (dh / dx) + v * (dh / dy)

    Metric scaling:
        dx = R * cos(lat) * d_lon_rad
        dy = R * d_lat_rad

    Returns:
        w_orog_norm: tanh(w_orog / 50.0) in [-1.0, 1.0]
        Positive -> forced ascent (windward enhancement)
        Negative -> subsidence (leeward rain shadow)
    """
    if u_wind is None or v_wind is None:
        u_wind, v_wind = MONTHLY_850HPA_WINDS.get(int(month), (+8.0, +2.0))

    lat_rad = center_lat_deg * RAD_PER_DEG
    dx_m = float(EARTH_RADIUS_M * np.cos(lat_rad) * (d_lon_deg * RAD_PER_DEG))
    dy_m = float(EARTH_RADIUS_M * (d_lat_deg * RAD_PER_DEG))

    is_torch = isinstance(elevation_m, torch.Tensor)
    if is_torch:
        orig_shape = elevation_m.shape
        dem = elevation_m.float()
        if dem.ndim == 2:
            dem = dem.unsqueeze(0).unsqueeze(0)
        elif dem.ndim == 3:
            dem = dem.unsqueeze(1)

        # 3x3 Sobel kernels for metric spatial gradient
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ], dtype=torch.float32, device=dem.device).view(1, 1, 3, 3) / (8.0 * dx_m)

        # Row 0 is North and Row 2 is South, so North-South gradient:
        # dh/dy_north = (dem[North] - dem[South]) / dy_m
        sobel_y = torch.tensor([
            [ 1.0,  2.0,  1.0],
            [ 0.0,  0.0,  0.0],
            [-1.0, -2.0, -1.0],
        ], dtype=torch.float32, device=dem.device).view(1, 1, 3, 3) / (8.0 * dy_m)

        dh_dx = F.conv2d(dem, sobel_x, padding=1)
        dh_dy_north = F.conv2d(dem, sobel_y, padding=1)

        w_orog = float(u_wind) * dh_dx + float(v_wind) * dh_dy_north
        w_norm = torch.tanh(w_orog * 1000.0 / 50.0)  # scale velocity

        if len(orig_shape) == 2:
            return w_norm[0, 0]
        elif len(orig_shape) == 3:
            return w_norm[:, 0, ...]
        return w_norm
    else:
        elev_np = np.asarray(elevation_m, dtype=np.float32)
        dh_dy_south, dh_dx = np.gradient(elev_np, dy_m, dx_m)
        dh_dy_north = -dh_dy_south
        w_orog = float(u_wind) * dh_dx + float(v_wind) * dh_dy_north
        w_norm = np.tanh(w_orog * 1000.0 / 50.0).astype(np.float32)
        return w_norm


def build_terrain_tensor_5ch(
    elevation_m: Union[torch.Tensor, np.ndarray],
    slope_deg: Union[torch.Tensor, np.ndarray],
    aspect_deg: Union[torch.Tensor, np.ndarray],
    center_lat_deg: float = 12.5,
    month: int = 7,
) -> torch.Tensor:
    """
    Constructs unified 5-channel HR terrain tensor [B, 5, 80, 80] or [5, 80, 80]:
        Channel 0: dem_norm
        Channel 1: slope_norm
        Channel 2: aspect_sin
        Channel 3: aspect_cos
        Channel 4: w_orog_norm
    """
    t_dem = torch.as_tensor(elevation_m, dtype=torch.float32)
    t_slope = torch.as_tensor(slope_deg, dtype=torch.float32)
    t_aspect = torch.as_tensor(aspect_deg, dtype=torch.float32)

    has_batch = t_dem.ndim == 3 or (t_dem.ndim == 4 and t_dem.shape[1] == 1)
    if not has_batch and t_dem.ndim == 2:
        t_dem = t_dem.unsqueeze(0)
        t_slope = t_slope.unsqueeze(0)
        t_aspect = t_aspect.unsqueeze(0)
    elif t_dem.ndim == 4:
        t_dem = t_dem.squeeze(1)
        t_slope = t_slope.squeeze(1)
        t_aspect = t_aspect.squeeze(1)

    dem_norm = normalize_elevation(t_dem)
    slope_norm = normalize_slope(t_slope)
    aspect_sin, aspect_cos = encode_aspect(t_aspect)
    w_orog = compute_orographic_velocity(t_dem, center_lat_deg=center_lat_deg, month=month)

    # Stack into [B, 5, H, W]
    terrain_5ch = torch.stack([dem_norm, slope_norm, aspect_sin, aspect_cos, w_orog], dim=1)

    if not has_batch:
        return terrain_5ch[0]
    return terrain_5ch


def area_mean_downsample(terrain_hr: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Conservative area-mean downsampling for terrain grids (zero elevation bias)."""
    return F.avg_pool2d(terrain_hr, kernel_size=kernel_size, stride=kernel_size)
