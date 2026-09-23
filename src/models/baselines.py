"""
src/models/baselines.py

Baseline models for 5x spatial downscaling (0.25° to 0.05°):
1. BilinearInterpolationBaseline:
   - Analytical, parameter-free reference establishing performance floor.
   - Evaluates direct bilinear upsampling with conservation evaluation.
2. DeepSDBaseline:
   - Implementation of SRCNN-style architecture conditioned on high-resolution elevation,
     following Vandal et al. (2017) "DeepSD: Generating High Resolution Climate Datasets".
   - Explicitly cited as prior art in docs and presentations (never claimed as team novelty).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearInterpolationBaseline(nn.Module):
    """
    Parameter-free bilinear interpolation baseline.
    Upsamples 16x16 LR input directly to 80x80 (exact 5x factor).
    """

    def __init__(self, scale_factor: float = 5.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x_lr: torch.Tensor, dem_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_lr: [B, 1, 16, 16] low-resolution precipitation.
            dem_hr: Optional high-resolution DEM (ignored for bilinear).
        Returns:
            [B, 1, 80, 80] upsampled precipitation.
        """
        # Interpolate with align_corners=False to preserve area-representative cell values
        out = F.interpolate(
            x_lr,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        return torch.clamp(out, min=0.0)


class DeepSDBaseline(nn.Module):
    """
    DeepSD-style CNN baseline (Vandal et al., 2017).
    
    Structure:
    1. Bilinearly upsamples 16x16 LR rainfall to 80x80.
    2. Concatenates high-resolution terrain/elevation channel (80x80).
    3. Passes through 3 convolutional layers with non-linear activations:
       - Conv1: 9x9 kernel, 64 channels, ReLU (Patch extraction & representation)
       - Conv2: 5x5 kernel, 32 channels, ReLU (Non-linear mapping)
       - Conv3: 5x5 kernel, 1 channel, ReLU (Reconstruction)
    """

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64):
        """
        Args:
            in_channels: 2 (1 channel upsampled LR rainfall + 1 channel HR elevation).
            hidden_channels: Base filter count for intermediate representations.
        """
        super().__init__()
        self.in_channels = in_channels

        # Conv1: 9x9 receptive field
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # Conv2: 5x5 receptive field
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # Conv3: 5x5 reconstruction
        self.conv3 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU(inplace=True)  # Enforce non-negative precipitation

    def forward(self, x_lr: torch.Tensor, dem_hr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_lr: [B, 1, 16, 16] low-resolution precipitation.
            dem_hr: [B, 1, 80, 80] optional high-resolution elevation.
                    If None, a zero-filled tensor is used.
        Returns:
            [B, 1, 80, 80] downscaled precipitation.
        """
        b, _, _, _ = x_lr.shape

        # 1. Direct bilinear upsampling to 80x80
        x_upsampled = F.interpolate(
            x_lr,
            size=(80, 80),
            mode="bilinear",
            align_corners=False,
        )

        # 2. Condition on elevation if provided
        if dem_hr is None:
            dem_hr = torch.zeros_like(x_upsampled)

        x_cat = torch.cat([x_upsampled, dem_hr], dim=1)

        # 3. 3-layer SRCNN pass
        feat1 = self.relu1(self.conv1(x_cat))
        feat2 = self.relu2(self.conv2(feat1))
        out = self.relu3(self.conv3(feat2))

        return out


class BilinearTemporalBaseline(nn.Module):
    """
    Baseline 0B: All-Bilinear 5x Temporal Interpolation.
    Directly upsamples all 7 leads and 6 variables from 16x16 to 80x80.
    """

    def __init__(self, scale_factor: float = 5.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, future_forecast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            future_forecast: [B, 7, 6, 16, 16]
        Returns:
            [B, 7, 6, 80, 80]
        """
        b, leads, c, h, w = future_forecast.shape
        flat = future_forecast.view(b * leads, c, h, w)
        up = F.interpolate(flat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return up.view(b, leads, c, 80, 80)


class ChannelAwareTemporalBaseline(nn.Module):
    """
    Baseline 0A: Channel-Aware Coarse-to-Fine Interpolation.
    - Precipitation (ch 0): Conservative nearest 5x block disaggregation preserving cell mean.
    - Thermodynamics and Wind (ch 1..5): Bilinear 5x interpolation.
    """

    def __init__(self, scale_factor: float = 5.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, future_forecast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            future_forecast: [B, 7, 6, 16, 16]
        Returns:
            [B, 7, 6, 80, 80]
        """
        b, leads, c, h, w = future_forecast.shape
        flat = future_forecast.view(b * leads, c, h, w)

        # Precipitation: conservative block nearest disaggregation
        p_coarse = flat[:, 0:1, :, :]
        p_fine = F.interpolate(p_coarse, scale_factor=self.scale_factor, mode="nearest")

        # Thermodynamics & Wind (Tmax, Tmin, RH, U, V): bilinear
        other_coarse = flat[:, 1:, :, :]
        other_fine = F.interpolate(other_coarse, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        combined = torch.cat([p_fine, other_fine], dim=1)
        return combined.view(b, leads, c, 80, 80)


class PersistenceTemporalBaseline(nn.Module):
    """
    Baseline 0C: Persistence Baseline.
    Repeats the latest historical observation day (D-1) across all 7 future leads.
    """

    def __init__(self, scale_factor: float = 5.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history: [B, H, 6, 16, 16]
        Returns:
            [B, 7, 6, 80, 80]
        """
        b, h_len, c, h, w = history.shape
        latest_day = history[:, -1, :, :, :]  # [B, 6, 16, 16] (D-1)
        latest_up = F.interpolate(latest_day, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)  # [B, 6, 80, 80]
        return latest_up.unsqueeze(1).expand(b, 7, c, 80, 80).contiguous()

