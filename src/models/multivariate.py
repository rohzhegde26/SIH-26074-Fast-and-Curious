"""
src/models/multivariate.py

Thermodynamic & Multi-Variable Physics-Informed Downscaling Module.

Key Physical Constraints Enforced:
1. Environmental Lapse Rate:
   dT/dh = -0.0065 °C/m (-6.5 °C / 1000m standard troposphere).
2. Diurnal Thermodynamic Ordering:
   T_min <= T_max strictly enforced via softplus diurnal spread (Delta T > 0).
3. Psychrometric Saturation via Magnus-Tetens Equation:
   e_s(T) = 6.112 * exp((17.67 * T) / (T + 243.5))
   RH_hr = clamp(RH_interp * (e_s(T_interp) / e_s(T_hr)), 10.0, 100.0)
4. Topographic Wind Acceleration:
   S = 1.0 + 0.4 * clamp(slope / 20°, 0, 1) + 0.3 * clamp(w_orog, 0, 1)
   Wind_hr = clamp(Wind_interp * S, 0.5, 120.0)
5. Climatological Data Contract Fallback:
   Synthesizes Mandya monsoon climatology when coarse data is precipitation-only,
   tagged with explicit provenance: "SYNTHETIC_ERA5_CLIMATOLOGY_COUPLING".
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Mandya summer monsoon baseline climatology (1991-2020 IMD/ERA5)
CLIMATOLOGY_DEFAULTS = {
    "tmax": 31.5,   # °C
    "tmin": 21.0,   # °C
    "rh": 68.0,     # %
    "wind": 8.5,    # km/h
}

PROVENANCE_TAG = "SYNTHETIC_ERA5_CLIMATOLOGY_COUPLING"

# Global topographic scaling constants matching terrain_features.py
DEM_MEAN = 382.5
DEM_STD = 458.2
SLOPE_P95 = 22.3


def magnus_tetens_es(t_celsius: torch.Tensor) -> torch.Tensor:
    """
    Computes saturation vapor pressure e_s (hPa) using the Magnus-Tetens formula.
    Valid for temperatures between -40°C and +50°C.
    e_s(T) = 6.112 * exp((17.67 * T) / (T + 243.5))
    """
    return 6.112 * torch.exp((17.67 * t_celsius) / (t_celsius + 243.5))


class MultivariatePhysicalDownscaler(nn.Module):
    """
    Downscales multi-variable meteorological fields (Tmax, Tmin, RH, Wind)
    from coarse 0.25° NWP to 0.05° high-resolution terrain.
    """

    def __init__(self, lapse_rate_deg_per_m: float = 0.0065):
        super().__init__()
        self.gamma = lapse_rate_deg_per_m  # 6.5°C per 1000m

    def forward(
        self,
        tmax_lr: Optional[torch.Tensor] = None,
        tmin_lr: Optional[torch.Tensor] = None,
        rh_lr: Optional[torch.Tensor] = None,
        wind_lr: Optional[torch.Tensor] = None,
        elevation_hr: Optional[torch.Tensor] = None,
        slope_hr: Optional[torch.Tensor] = None,
        w_orog_hr: Optional[torch.Tensor] = None,
        terrain_5ch: Optional[torch.Tensor] = None,
        reference_elevation_m: Optional[Union[float, torch.Tensor]] = None,
        target_size: Tuple[int, int] = (80, 80),
    ) -> Dict[str, Any]:
        """
        Executes physical downscaling for Tmax, Tmin, RH, and Wind.

        Args:
            tmax_lr: [B, 1, H_lr, W_lr] Coarse maximum temperature (°C)
            tmin_lr: [B, 1, H_lr, W_lr] Coarse minimum temperature (°C)
            rh_lr: [B, 1, H_lr, W_lr] Coarse relative humidity (%)
            wind_lr: [B, 1, H_lr, W_lr] Coarse surface wind speed (km/h)
            elevation_hr: [B, 1, H_hr, W_hr] or [H_hr, W_hr] Elevation (meters)
            slope_hr: [B, 1, H_hr, W_hr] or [H_hr, W_hr] Slope (degrees)
            w_orog_hr: [B, 1, H_hr, W_hr] or [H_hr, W_hr] Orographic ascent proxy
            terrain_5ch: [B, 5, H_hr, W_hr] Unified 5-channel terrain features
            reference_elevation_m: Domain or station reference elevation (meters)
            target_size: (H_hr, W_hr) Output grid resolution (default 80x80)

        Returns:
            Dict containing downscaled tensors:
                'tmax_hr': [B, 1, H_hr, W_hr]
                'tmin_hr': [B, 1, H_hr, W_hr]
                'tmean_hr': [B, 1, H_hr, W_hr]
                'rh_hr': [B, 1, H_hr, W_hr]
                'wind_hr': [B, 1, H_hr, W_hr]
                'provenance': str provenance metadata
        """
        # Determine batch size and device
        batch_size = 1
        device = torch.device("cpu")
        for tensor in (tmax_lr, tmin_lr, rh_lr, wind_lr, elevation_hr, terrain_5ch):
            if tensor is not None and isinstance(tensor, torch.Tensor):
                batch_size = tensor.shape[0] if tensor.ndim >= 3 else 1
                device = tensor.device
                break

        # Check if fallback climatology is needed
        is_synthetic = any(x is None for x in (tmax_lr, tmin_lr, rh_lr, wind_lr))
        provenance = PROVENANCE_TAG if is_synthetic else "COARSE_NWP_OBSERVATION"

        # 1. Standardize Coarse Inputs
        lr_h, lr_w = target_size[0] // 5, target_size[1] // 5  # default (16, 16)
        if tmax_lr is None:
            tmax_lr = torch.full((batch_size, 1, lr_h, lr_w), CLIMATOLOGY_DEFAULTS["tmax"], dtype=torch.float32, device=device)
        if tmin_lr is None:
            tmin_lr = torch.full((batch_size, 1, lr_h, lr_w), CLIMATOLOGY_DEFAULTS["tmin"], dtype=torch.float32, device=device)
        if rh_lr is None:
            rh_lr = torch.full((batch_size, 1, lr_h, lr_w), CLIMATOLOGY_DEFAULTS["rh"], dtype=torch.float32, device=device)
        if wind_lr is None:
            wind_lr = torch.full((batch_size, 1, lr_h, lr_w), CLIMATOLOGY_DEFAULTS["wind"], dtype=torch.float32, device=device)

        # 2. Extract Terrain Features
        if terrain_5ch is not None:
            if terrain_5ch.ndim == 3:
                terrain_5ch = terrain_5ch.unsqueeze(0)
            if elevation_hr is None:
                # Invert normalization: dem_norm = clamp((h - DEM_MEAN)/DEM_STD, -2.5, 3.5)/3.5
                dem_norm = terrain_5ch[:, 0:1, :, :]
                elevation_hr = dem_norm * 3.5 * DEM_STD + DEM_MEAN
            if slope_hr is None:
                # slope_norm = clamp(slope/SLOPE_P95, 0, 2)/2
                slope_norm = terrain_5ch[:, 1:2, :, :]
                slope_hr = slope_norm * 2.0 * SLOPE_P95
            if w_orog_hr is None:
                w_orog_hr = terrain_5ch[:, 4:5, :, :]

        if elevation_hr is None:
            # Default to flat Mandya plain (650m)
            elevation_hr = torch.full((batch_size, 1, *target_size), 650.0, dtype=torch.float32, device=device)
        elif elevation_hr.ndim == 2:
            elevation_hr = elevation_hr.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        elif elevation_hr.ndim == 3:
            elevation_hr = elevation_hr.unsqueeze(1)

        if slope_hr is None:
            slope_hr = torch.zeros((batch_size, 1, *target_size), dtype=torch.float32, device=device)
        elif slope_hr.ndim == 2:
            slope_hr = slope_hr.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        elif slope_hr.ndim == 3:
            slope_hr = slope_hr.unsqueeze(1)

        if w_orog_hr is None:
            w_orog_hr = torch.zeros((batch_size, 1, *target_size), dtype=torch.float32, device=device)
        elif w_orog_hr.ndim == 2:
            w_orog_hr = w_orog_hr.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        elif w_orog_hr.ndim == 3:
            w_orog_hr = w_orog_hr.unsqueeze(1)

        # 3. Environmental Lapse Rate Downscaling
        # Mean coarse temperature and diurnal range
        tmean_lr = 0.5 * (tmax_lr + tmin_lr)
        delta_t_lr = torch.clamp(tmax_lr - tmin_lr, min=0.5)

        # Bilinear interpolation of coarse synoptic temperature
        tmean_interp = F.interpolate(tmean_lr, size=target_size, mode="bilinear", align_corners=False)
        delta_t_interp = F.interpolate(delta_t_lr, size=target_size, mode="bilinear", align_corners=False)
        delta_t_eff = F.softplus(delta_t_interp)  # Guarantees strictly positive diurnal range

        # Reference elevation for lapse rate correction
        if reference_elevation_m is not None:
            if isinstance(reference_elevation_m, (int, float)):
                h_ref = torch.tensor(reference_elevation_m, dtype=torch.float32, device=device).view(1, 1, 1, 1)
            else:
                h_ref = reference_elevation_m
        else:
            h_ref = elevation_hr.mean(dim=(-2, -1), keepdim=True)

        delta_h = elevation_hr - h_ref

        # Lapse rate temperature adjustment: dT = -Gamma * delta_h
        tmean_hr = tmean_interp - (self.gamma * delta_h)
        tmax_hr = tmean_hr + 0.5 * delta_t_eff
        tmin_hr = tmean_hr - 0.5 * delta_t_eff

        # 4. Psychrometric Relative Humidity via Magnus-Tetens Equation
        rh_interp = F.interpolate(rh_lr, size=target_size, mode="bilinear", align_corners=False)
        es_interp = magnus_tetens_es(tmean_interp)
        es_hr = magnus_tetens_es(tmean_hr)
        # As air ascends and cools, es_hr < es_interp -> RH increases
        rh_ratio = es_interp / torch.clamp(es_hr, min=1e-6)
        rh_hr = torch.clamp(rh_interp * rh_ratio, min=10.0, max=100.0)

        # 5. Topographic Wind Steering & Ridge Boost
        wind_interp = F.interpolate(wind_lr, size=target_size, mode="bilinear", align_corners=False)
        slope_factor = 0.4 * torch.clamp(slope_hr / 20.0, min=0.0, max=1.0)
        orog_factor = 0.3 * torch.clamp(w_orog_hr, min=0.0, max=1.0)
        steering_multiplier = 1.0 + slope_factor + orog_factor
        wind_hr = torch.clamp(wind_interp * steering_multiplier, min=0.5, max=120.0)

        return {
            "tmean_hr": tmean_hr,
            "tmax_hr": tmax_hr,
            "tmin_hr": tmin_hr,
            "rh_hr": rh_hr,
            "wind_hr": wind_hr,
            "provenance": provenance,
        }
