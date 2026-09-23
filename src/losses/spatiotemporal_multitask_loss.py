"""
src/losses/spatiotemporal_multitask_loss.py

Sprint 3 Physics-Informed Multi-Task Spatiotemporal Loss.
Enforces:
  1. Multi-Task Regression in Normalized Model Space:
     - Precipitation: Log-Cosh + Quantile Pinball (tau=0.90) for convective tails
     - Tmax / Tmin: Smooth L1 Huber (beta=1.0)
     - RH: L1 loss
     - Wind: Vector magnitude Log-Cosh + component Huber
  2. Homoscedastic Uncertainty Balancing (Kendall et al., CVPR 2018).
  3. Physical Area-Weighted Mass Conservation in physical mm (with cosine latitude weights).
  4. Physical Diurnal Temperature Ordering in Celsius (Tmax >= Tmin).
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.conservation import coarsen_hr_to_lr_torch


def log_cosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Robust Log-Cosh regression loss with clipping to avoid float overflow."""
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(torch.clamp(diff, -20.0, 20.0))))


def quantile_pinball(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.90) -> torch.Tensor:
    """Asymmetric quantile pinball loss focusing on upper-tail convective extremes."""
    err = target - pred
    return torch.mean(torch.max(tau * err, (tau - 1.0) * err))


class SpatiotemporalMultiTaskLoss(nn.Module):
    """
    Unified Spatiotemporal Multi-Task Physical Loss for 7-Day Downscaling.
    """

    def __init__(
        self,
        lambda_mass: float = 0.05,
        lambda_diurnal: float = 0.02,
        lat_min: float = 11.0,
        lat_max: float = 15.0,
        grid_h: int = 80,
    ):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_diurnal = lambda_diurnal

        # 5 task log-variances for uncertainty weighting (P, Tmax, Tmin, RH, Wind)
        self.log_vars = nn.Parameter(torch.zeros(5))

        # Cosine latitude vector for 80x80 domain
        lats_deg = torch.linspace(lat_max, lat_min, grid_h, dtype=torch.float32)
        self.register_buffer("lats_deg", lats_deg)

        # Train normalization constants (from data/normalization_stats.yaml)
        # Precipitation: log1p-zscore
        self.register_buffer("p_mean", torch.tensor(0.767003, dtype=torch.float32))
        self.register_buffer("p_std", torch.tensor(1.279277, dtype=torch.float32))
        # Tmax: zscore
        self.register_buffer("tmax_mean", torch.tensor(31.20812, dtype=torch.float32))
        self.register_buffer("tmax_std", torch.tensor(2.32532, dtype=torch.float32))
        # Tmin: zscore
        self.register_buffer("tmin_mean", torch.tensor(22.56778, dtype=torch.float32))
        self.register_buffer("tmin_std", torch.tensor(2.43432, dtype=torch.float32))

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        coarse_fcst: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: [B, 7, 6, 80, 80] normalized predictions
            targets: [B, 7, 6, 80, 80] normalized targets
            coarse_fcst: [B, 7, 6, 16, 16] normalized coarse GFS forecast
        Returns:
            Dict containing total loss and individual diagnostics
        """
        b, num_leads, _, _, _ = preds.shape

        # Slices for channels: [B, 7, 1, 80, 80]
        p_pred = preds[:, :, 0:1, :, :]
        p_tgt = targets[:, :, 0:1, :, :]

        tmax_pred = preds[:, :, 1:2, :, :]
        tmax_tgt = targets[:, :, 1:2, :, :]

        tmin_pred = preds[:, :, 2:3, :, :]
        tmin_tgt = targets[:, :, 2:3, :, :]

        rh_pred = preds[:, :, 3:4, :, :]
        rh_tgt = targets[:, :, 3:4, :, :]

        u_pred = preds[:, :, 4:5, :, :]
        u_tgt = targets[:, :, 4:5, :, :]

        v_pred = preds[:, :, 5:6, :, :]
        v_tgt = targets[:, :, 5:6, :, :]

        # 1. Channel task losses (in normalized space)
        # (a) Precipitation: Log-cosh + upper-quantile pinball
        l_precip = log_cosh(p_pred, p_tgt) + 0.5 * quantile_pinball(p_pred, p_tgt, tau=0.90)

        # (b) Temperatures: Smooth L1 (Huber)
        l_tmax = F.smooth_l1_loss(tmax_pred, tmax_tgt, beta=1.0)
        l_tmin = F.smooth_l1_loss(tmin_pred, tmin_tgt, beta=1.0)

        # (c) Relative Humidity: L1
        l_rh = F.l1_loss(rh_pred, rh_tgt)

        # (d) Wind: Joint vector difference magnitude + component Huber
        diff_u = u_pred - u_tgt
        diff_v = v_pred - v_tgt
        vec_mag = torch.sqrt(diff_u ** 2 + diff_v ** 2 + 1e-6)
        l_wind = torch.mean(torch.log(torch.cosh(torch.clamp(vec_mag, -20.0, 20.0)))) + 0.25 * (
            F.smooth_l1_loss(u_pred, u_tgt, beta=1.0) + F.smooth_l1_loss(v_pred, v_tgt, beta=1.0)
        )

        task_losses = [l_precip, l_tmax, l_tmin, l_rh, l_wind]

        # 2. Homoscedastic task uncertainty weighting: 0.5 * exp(-s) * L + 0.5 * s
        weighted_task_loss = torch.tensor(0.0, device=preds.device)
        for k, l_k in enumerate(task_losses):
            s_k = self.log_vars[k]
            precision = torch.exp(-s_k)
            weighted_task_loss = weighted_task_loss + 0.5 * (precision * l_k + s_k)

        # 3. Physical Diurnal Temperature Ordering in Celsius (Tmax >= Tmin)
        # De-normalize Tmax and Tmin to degrees Celsius
        tmax_phys = tmax_pred * self.tmax_std + self.tmax_mean
        tmin_phys = tmin_pred * self.tmin_std + self.tmin_mean
        # Inversion violation penalty where Tmin > Tmax
        l_diurnal = torch.mean(F.relu(tmin_phys - tmax_phys) ** 2)

        # 4. Physical Area-Weighted Mass Conservation in physical mm
        # De-normalize fine and coarse precipitation
        p_pred_phys = torch.clamp(torch.expm1(p_pred * self.p_std + self.p_mean), min=0.0)  # [B, 7, 1, 80, 80]
        p_coarse_phys = torch.clamp(
            torch.expm1(coarse_fcst[:, :, 0:1, :, :] * self.p_std + self.p_mean), min=0.0
        )  # [B, 7, 1, 16, 16]

        # Coarsen fine predicted precipitation across 80x80 to 16x16 with cosine latitude weighting
        p_pred_flat = p_pred_phys.view(b * num_leads, 1, 80, 80)
        p_coarsened = coarsen_hr_to_lr_torch(
            hr_tensor=p_pred_flat,
            hr_lats_deg=self.lats_deg,
            kernel_size=5,
            stride=5,
        ).view(b, num_leads, 1, 16, 16)

        l_mass = F.mse_loss(p_coarsened, p_coarse_phys)

        # 5. Composite Total Loss
        total_loss = weighted_task_loss + self.lambda_diurnal * l_diurnal + self.lambda_mass * l_mass

        return {
            "loss": total_loss,
            "loss_precip": l_precip,
            "loss_tmax": l_tmax,
            "loss_tmin": l_tmin,
            "loss_rh": l_rh,
            "loss_wind": l_wind,
            "loss_diurnal": l_diurnal,
            "loss_mass": l_mass,
        }
