"""
src/data/agera5_loader.py

AgERA5 & ERA5 NWP Reanalysis Data Loader and Preprocessor.
Pairs 0.25° coarse numerical weather prediction inputs with 0.05° high-resolution targets
(CHIRPS for precipitation, AgERA5 reanalysis for thermodynamics, GLO-30 DEM for terrain).

Variables Handled:
    - Temperature-Air-2m-Max-24h (Kelvin -> Celsius)
    - Temperature-Air-2m-Min-24h (Kelvin -> Celsius)
    - Relative-Humidity-2m-12h (percentage 2-100%)
    - Wind-Speed-10m-Mean (m/s -> km/h, multiply by 3.6)
    - Precipitation-Flux (kg/m^2/s -> mm/day, multiply by 86400)

Strict Contiguous Non-Overlapping Geographic Tiles across Peninsular India:
    - Tile 1 (Western Ghats & Coast): 11.0°N-15.0°N, 74.0°E-78.0°E (80x80 cells)
    - Tile 2 (Deccan & Mandya Plateau): 11.0°N-15.0°N, 78.0°E-82.0°E (80x80 cells)
    - Tile 3 (Eastern Ghats & Bay of Bengal): 11.0°N-15.0°N, 82.0°E-86.0°E (80x80 cells)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
AGERA5_RAW_DIR = ROOT / "data" / "raw" / "agera5"
CACHE_DIR = ROOT / "data" / "cache"

# Spatial bounding boxes for the 3 non-overlapping 4°x4° tiles
GEOGRAPHIC_TILES = {
    1: {"name": "Western_Ghats_Coast", "lat_min": 11.0, "lat_max": 15.0, "lon_min": 74.0, "lon_max": 78.0},
    2: {"name": "Deccan_Mandya_Plateau", "lat_min": 11.0, "lat_max": 15.0, "lon_min": 78.0, "lon_max": 82.0},
    3: {"name": "Eastern_Plains_Coast", "lat_min": 11.0, "lat_max": 15.0, "lon_min": 82.0, "lon_max": 86.0},
}


def kelvin_to_celsius(k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts Kelvin to Celsius."""
    return k - 273.15


def mps_to_kph(v: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts wind speed from meters per second to kilometers per hour."""
    return v * 3.6


def precip_flux_to_mm_day(flux: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts precipitation flux (kg m-2 s-1) to mm/day."""
    return flux * 86400.0


class LatitudeWeightedCoarsePool2d(nn.Module):
    """
    Computes cell-area weighted 5x pooling (80x80 -> 16x16) accounting for spherical
    Earth geometry via latitude cosine weighting w(phi) = cos(phi).
    Matches 0.25° NWP coarse grid cell centers and boundaries under Pixel-Is-Area geometry.
    """

    def __init__(self, lat_min: float = 11.0, lat_max: float = 15.0, grid_h: int = 80, pool_factor: int = 5):
        super().__init__()
        self.pool_factor = pool_factor
        # Generate 1D latitude array for each row of the fine grid (cell centers)
        lats = np.linspace(lat_max - 0.025, lat_min + 0.025, grid_h, dtype=np.float32)
        # Cosine weights for each row
        cos_weights = np.cos(np.radians(lats)).astype(np.float32)
        self.register_buffer("row_weights", torch.from_numpy(cos_weights).view(1, 1, grid_h, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, H, W] where H = W = 80.
        Returns:
            Coarse tensor of shape [B, C, H // 5, W // 5] (16x16).
        """
        w = self.row_weights.expand_as(x)
        weighted_x = x * w
        sum_x = F.avg_pool2d(weighted_x, kernel_size=self.pool_factor, stride=self.pool_factor) * (self.pool_factor ** 2)
        sum_w = F.avg_pool2d(w, kernel_size=self.pool_factor, stride=self.pool_factor) * (self.pool_factor ** 2)
        return sum_x / torch.clamp(sum_w, min=1e-6)


def area_weighted_coarse_pool(
    fine_grid: np.ndarray,
    lat_min: float = 11.0,
    lat_max: float = 15.0,
    pool_factor: int = 5,
) -> np.ndarray:
    """Numpy convenience wrapper for LatitudeWeightedCoarsePool2d."""
    is_2d = fine_grid.ndim == 2
    if is_2d:
        grid = fine_grid[np.newaxis, np.newaxis, :, :]
    elif fine_grid.ndim == 3:
        grid = fine_grid[np.newaxis, :, :, :]
    else:
        grid = fine_grid

    t = torch.from_numpy(grid.astype(np.float32))
    pooler = LatitudeWeightedCoarsePool2d(lat_min, lat_max, grid_h=fine_grid.shape[-2], pool_factor=pool_factor)
    coarse = pooler(t)
    res = coarse.squeeze(0).numpy()
    return res[0] if is_2d else res


def apply_nwp_forecast_bias_augmentation(
    coarse_nwp: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Applies realistic operational forecast bias and noise to simulated coarse NWP inputs:
        - Tmax / Tmin: Gaussian jitter (sigma=1.2 °C) + regional bias (+-1.0 °C).
        - RH: Multiplicative scale (0.95 to 1.05) + additive jitter (+-4.0 %).
        - Wind: Jitter (+-2.5 km/h).
        - Rain: Multiplicative calibration error (0.85 to 1.15).
    Prevents training-deployment domain shift.
    """
    if rng is None:
        rng = np.random.default_rng()

    aug = coarse_nwp.copy()
    # Channel 0: Rain
    aug[0] = np.clip(aug[0] * rng.uniform(0.85, 1.15), 0.0, 500.0)
    # Channel 1: Tmax
    t_bias = rng.uniform(-1.0, 1.0)
    aug[1] = aug[1] + t_bias + rng.normal(0.0, 1.2, aug[1].shape)
    # Channel 2: Tmin
    aug[2] = aug[2] + t_bias + rng.normal(0.0, 1.2, aug[2].shape)
    # Ensure coarse Tmax > Tmin
    aug[1] = np.maximum(aug[1], aug[2] + 0.5)
    # Channel 3: RH
    aug[3] = np.clip(aug[3] * rng.uniform(0.95, 1.05) + rng.normal(0.0, 3.5, aug[3].shape), 5.0, 99.0)
    # Channel 4: Wind
    aug[4] = np.clip(aug[4] + rng.normal(0.0, 2.0, aug[4].shape), 0.5, 100.0)
    return aug.astype(np.float32)


def generate_synthetic_multitask_tile(
    seed: Optional[int] = None,
    tile_id: int = 2,
    elev_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates a physically consistent paired 80x80 weather tile for peninsular India.
    Employs realistic lapse rates, diurnal spread, orographic moisture, and ridge wind acceleration
    tailored to the selected geographic tile.
    """
    rng = np.random.default_rng(seed)
    tile_info = GEOGRAPHIC_TILES.get(tile_id, GEOGRAPHIC_TILES[2])

    if elev_grid is None or elev_grid.shape != (80, 80):
        y, x = np.mgrid[0:80, 0:80]
        if tile_id == 1:
            # Western Ghats: steep ridge on east edge (up to 1800m), sea on west
            base_elev = 50.0 + 1200.0 * np.clip((x - 20) / 45.0, 0.0, 1.0) ** 1.8
            elev_grid = base_elev + 250.0 * np.sin(y / 10.0) + rng.normal(0, 25.0, (80, 80))
        elif tile_id == 2:
            # Mandya / Deccan plateau: rolling plateau 600m-1200m
            elev_grid = 680.0 + 180.0 * np.sin(x / 14.0) + 160.0 * np.cos(y / 16.0) + rng.normal(0, 20.0, (80, 80))
        else:
            # Eastern Plains: descending towards Bay of Bengal (300m down to 20m)
            elev_grid = 350.0 - 280.0 * (x / 80.0) + 60.0 * np.cos(y / 15.0) + rng.normal(0, 15.0, (80, 80))
        elev_grid = np.clip(elev_grid, 10.0, 2000.0).astype(np.float32)

    elev_km = (elev_grid - np.mean(elev_grid)) / 1000.0

    # Synoptic regional drivers
    syn_tmax = rng.uniform(28.0, 35.0)
    syn_tmin = syn_tmax - rng.uniform(8.0, 14.0)
    syn_rh = rng.uniform(55.0, 85.0)
    syn_wind = rng.uniform(8.0, 22.0)
    syn_rain_prob = rng.uniform(0.15, 0.75) if tile_id == 1 else rng.uniform(0.10, 0.55)

    # 1. Temperature with realistic lapse rates (Gamma ~ 6.5 °C/km)
    lapse_rate = rng.uniform(6.0, 7.0)
    tmin = syn_tmin - lapse_rate * elev_km + rng.normal(0, 0.35, (80, 80))
    spread = rng.uniform(6.0, 12.0, (80, 80)) + rng.normal(0, 0.3, (80, 80))
    spread = np.clip(spread, 3.0, 18.0)
    tmax = tmin + spread

    # 2. Relative humidity
    rh = syn_rh - 3.0 * elev_km + rng.normal(0, 2.5, (80, 80))
    rh = np.clip(rh, 15.0, 99.0).astype(np.float32)

    # 3. Wind speed with slope exposure
    slope = np.hypot(np.gradient(elev_grid, axis=0), np.gradient(elev_grid, axis=1))
    wind = syn_wind + 0.08 * slope + rng.normal(0, 1.2, (80, 80))
    wind = np.clip(wind, 1.0, 80.0).astype(np.float32)

    # 4. Precipitation
    rain = np.zeros((80, 80), dtype=np.float32)
    if rng.random() < syn_rain_prob:
        num_cells = rng.integers(1, 4)
        y, x = np.mgrid[0:80, 0:80]
        for _ in range(num_cells):
            cx, cy = rng.integers(10, 70), rng.integers(10, 70)
            sigma = rng.uniform(6.0, 16.0)
            peak = rng.uniform(8.0, 75.0)
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            cell = peak * np.exp(-dist_sq / (2.0 * sigma ** 2))
            rain += cell.astype(np.float32)
        rain += np.clip(rng.normal(0.5, 0.8, (80, 80)), 0.0, 5.0).astype(np.float32)
    rain = np.clip(rain, 0.0, 350.0)

    return {
        "rain": rain.astype(np.float32),
        "tmax": tmax.astype(np.float32),
        "tmin": tmin.astype(np.float32),
        "rh": rh.astype(np.float32),
        "wind": wind.astype(np.float32),
        "elevation": elev_grid.astype(np.float32),
        "tile_id": tile_id,
        "lat_min": tile_info["lat_min"],
        "lat_max": tile_info["lat_max"],
        "lon_min": tile_info["lon_min"],
        "lon_max": tile_info["lon_max"],
    }


def prepare_multitask_training_sample(
    tile_dict: Dict[str, np.ndarray],
    augment_nwp_bias: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a paired training sample:
        - coarse_nwp: [5, 16, 16] (Rain, Tmax, Tmin, RH, Wind) with latitude-weighted pooling & NWP bias.
        - fine_terrain: [5, 80, 80] (Elev, Slope, Aspect, Curvature, Windward Lift).
        - fine_targets: [5, 80, 80] (Rain, Tmax, Tmin, RH, Wind).
    """
    fine_targets = np.stack([
        tile_dict["rain"],
        tile_dict["tmax"],
        tile_dict["tmin"],
        tile_dict["rh"],
        tile_dict["wind"],
    ], axis=0).astype(np.float32)  # [5, 80, 80]

    lat_min = tile_dict.get("lat_min", 11.0)
    lat_max = tile_dict.get("lat_max", 15.0)

    # Coarse pooling with latitude cosine weighting
    coarse_nwp = area_weighted_coarse_pool(fine_targets, lat_min=lat_min, lat_max=lat_max, pool_factor=5)  # [5, 16, 16]

    if augment_nwp_bias:
        coarse_nwp = apply_nwp_forecast_bias_augmentation(coarse_nwp, rng=rng)

    # Terrain features
    elev = tile_dict["elevation"]
    dy, dx = np.gradient(elev)
    slope = np.hypot(dx, dy)
    aspect = np.arctan2(-dx, dy)
    d2y, _ = np.gradient(dy)
    _, d2x = np.gradient(dx)
    curvature = d2x + d2y

    # Monsoon windward lift index (westerly 250°)
    u_wind = np.cos(np.radians(250.0))
    v_wind = np.sin(np.radians(250.0))
    lift = np.clip(u_wind * dx + v_wind * dy, -20.0, 20.0)

    # Normalize terrain channels
    elev_norm = (elev - 700.0) / 400.0
    slope_norm = slope / 20.0
    aspect_norm = aspect / np.pi
    curv_norm = np.clip(curvature / 5.0, -1.0, 1.0)
    lift_norm = lift / 10.0

    fine_terrain = np.stack([
        elev_norm,
        slope_norm,
        aspect_norm,
        curv_norm,
        lift_norm,
    ], axis=0).astype(np.float32)  # [5, 80, 80]

    return coarse_nwp, fine_terrain, fine_targets
