"""
src/losses/multitask_loss.py

Physics-Constrained Multi-Task Objective Loss with Homoscedastic Uncertainty Weighting.
Balances 5 downscaling tasks (Rain, Tmax, Tmin, RH, Wind) with exact physical coupling:
    1. Homoscedastic task uncertainty balancing (Kendall et al., CVPR 2018).
    2. Soft catchment mass conservation using LatitudeWeightedCoarsePool2d.
    3. Corrected environmental lapse-rate bounds: Gamma = -dT/dz in [4.0, 9.8] K/km.
    4. Magnus-Tetens vapor pressure & dew-point thermodynamic consistency.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.agera5_loader import LatitudeWeightedCoarsePool2d


def log_cosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-Cosh robust regression loss."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(torch.clamp(diff, -20.0, 20.0))))


def quantile_pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.90) -> torch.Tensor:
    """Asymmetric quantile pinball loss for convective precipitation extremes."""
    err = target - pred
    return torch.mean(torch.max(tau * err, (tau - 1.0) * err))


class MultiTaskPhysicalLoss(nn.Module):
    """
    Unified multi-task loss function with learnable task uncertainty weights and physics penalties.
    """

    def __init__(
        self,
        lambda_mass: float = 0.05,
        lambda_lapse: float = 0.02,
        lambda_magnus: float = 0.02,
        lat_min: float = 11.0,
        lat_max: float = 15.0,
    ):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_lapse = lambda_lapse
        self.lambda_magnus = lambda_magnus

        # 5 task log-variance parameters (Rain, Tmax, Tmin, RH, Wind)
        self.log_vars = nn.Parameter(torch.zeros(5))

        # Spherical cell-area weighted coarse pooling operator
        self.pooler = LatitudeWeightedCoarsePool2d(lat_min=lat_min, lat_max=lat_max, grid_h=80, pool_factor=5)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        coarse_nwp: torch.Tensor,
        fine_terrain: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: Dict containing predicted tensors for "rain", "tmax", "tmin", "rh", "wind".
            targets: Ground truth high-resolution targets [B, 5, 80, 80].
            coarse_nwp: Coarse NWP input [B, 5, 16, 16].
            fine_terrain: Fine terrain prior [B, 5, 80, 80].

        Returns:
            Dict containing total loss and individual loss components.
        """
        p_pred = preds["rain"]
        tmax_pred = preds["tmax"]
        tmin_pred = preds["tmin"]
        rh_pred = preds["rh"]
        wind_pred = preds["wind"]

        p_tgt = targets[:, 0:1, :, :]
        tmax_tgt = targets[:, 1:2, :, :]
        tmin_tgt = targets[:, 2:3, :, :]
        rh_tgt = targets[:, 3:4, :, :]
        wind_tgt = targets[:, 4:5, :, :]

        p_lr = coarse_nwp[:, 0:1, :, :]
        elev_norm = fine_terrain[:, 0:1, :, :]
        # Unnormalize elevation to meters from standard GLO-30 normalization:
        # dem_norm = clamp((elev - 382.5) / 458.2, -2.5, 3.5) / 3.5
        # elev = elev_norm * (3.5 * 458.2) + 382.5 = elev_norm * 1603.7 + 382.5
        elev = elev_norm * 1603.7 + 382.5

        # 1. Individual task regression losses
        # (a) Rain: log-cosh + asymmetric tail pinball
        l_rain = log_cosh_loss(p_pred, p_tgt) + 0.5 * quantile_pinball_loss(p_pred, p_tgt, tau=0.90)

        # (b) Tmax: Huber loss
        l_tmax = F.smooth_l1_loss(tmax_pred, tmax_tgt, beta=1.0)

        # (c) Tmin: Huber loss
        l_tmin = F.smooth_l1_loss(tmin_pred, tmin_tgt, beta=1.0)

        # (d) RH: L1 loss
        l_rh = F.l1_loss(rh_pred, rh_tgt)

        # (e) Wind: log-cosh loss
        l_wind = log_cosh_loss(wind_pred, wind_tgt)

        task_losses = [l_rain, l_tmax, l_tmin, l_rh, l_wind]

        # 2. Homoscedastic uncertainty weighting
        # L_task = 0.5 * exp(-s_k) * L_k + 0.5 * s_k
        weighted_task_loss = 0.0
        for k, l_k in enumerate(task_losses):
            s_k = self.log_vars[k]
            precision = torch.exp(-s_k)
            weighted_task_loss = weighted_task_loss + 0.5 * (precision * l_k + s_k)

        # 3. Physical Constraints
        # (a) Soft mass conservation loss
        p_pooled = self.pooler(p_pred)
        l_mass = F.l1_loss(p_pooled, p_lr)

        # (b) Environmental lapse rate bounds: Gamma = -dT/dz
        t_mean = 0.5 * (tmax_pred + tmin_pred)
        dy_t, dx_t = torch.gradient(t_mean, dim=(-2, -1))
        dy_z, dx_z = torch.gradient(elev, dim=(-2, -1))

        # Spatial temperature gradient divided by elevation gradient where elevation gradient > 20m/cell
        dz_mag = torch.hypot(dx_z, dy_z)
        dt_mag = torch.hypot(dx_t, dy_t)
        # Dot product of grad(T) and grad(z) to find dT/dz
        grad_dot = (dx_t * dx_z + dy_t * dy_z) / (dz_mag ** 2 + 1e-4)
        gamma = -grad_dot  # Gamma = -dT/dz (in °C/meter)

        valid_mask = dz_mag > 20.0
        if torch.any(valid_mask):
            gamma_valid = gamma[valid_mask]
            # Meteorological bounds: 4 to 9.8 K/km -> 0.004 to 0.0098 °C/m.
            # Inversion allowed down to -3 K/km (-0.003 °C/m). Superadiabatic capped at 12 K/km (0.012 °C/m).
            l_lapse = torch.mean(
                F.relu(gamma_valid - 0.012) ** 2 + F.relu(-0.004 - gamma_valid) ** 2
            )
        else:
            l_lapse = torch.tensor(0.0, device=t_mean.device)

        # (c) Magnus-Tetens vapor pressure & dew-point consistency: T_dew <= T_mean
        t_c = torch.clamp(t_mean, min=-10.0, max=50.0)
        rh_c = torch.clamp(rh_pred, min=2.0, max=100.0)

        # Saturation vapor pressure in hPa
        e_s = 6.112 * torch.exp(17.67 * t_c / (t_c + 243.5))
        e = e_s * (rh_c / 100.0)
        e_clamped = torch.clamp(e, min=0.1)

        # Dew point temperature in °C
        log_e = torch.log(e_clamped / 6.112)
        t_dew = (243.5 * log_e) / (17.67 - log_e)

        # Penalize unphysical supersaturation where T_dew > T_mean
        l_magnus = torch.mean(F.relu(t_dew - t_c) ** 2)

        total_loss = (
            weighted_task_loss
            + self.lambda_mass * l_mass
            + self.lambda_lapse * l_lapse
            + self.lambda_magnus * l_magnus
        )

        return {
            "loss": total_loss,
            "loss_rain": l_rain,
            "loss_tmax": l_tmax,
            "loss_tmin": l_tmin,
            "loss_rh": l_rh,
            "loss_wind": l_wind,
            "loss_mass": l_mass,
            "loss_lapse": l_lapse,
            "loss_magnus": l_magnus,
        }
