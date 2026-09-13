"""
src/models/multitask_unet.py

Multi-Task Neural Downscaling Network (MultiTaskUNet5x).
Downscales 5 coarse atmospheric numerical weather prediction variables:
    [Rain, Tmax, Tmin, RH, Wind] from 0.25° (16x16) to 0.05° (80x80)
conditioned on 5 high-resolution terrain prior channels:
    [Elev, Slope, Aspect, Curvature, Windward Lift] (80x80).

Key Architectural Innovations:
    1. Clean Residual Formulation around bilinearly upsampled coarse baseline.
    2. Lightweight ConvFusion Block (< 20 MB VRAM, replacing global 80x80 attention).
    3. Custom StraightThroughNonNegative Autograd for precipitation.
    4. Exact Zero-Bias Diurnal Temperature Spread guaranteeing Tmin < Tmax everywhere.
    5. Memory footprint < 3.2M parameters, fully compatible with Kaggle 2xT4 GPUs.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughNonNegative(torch.autograd.Function):
    """
    Straight-Through Estimator enforcing strict physical non-negativity (>= 0.0)
    in the forward pass while preserving a 0.01 leaky gradient in the backward pass
    for negative pre-activations, preventing dead neurons on dry pixels.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad_x = torch.where(x >= 0.0, grad_output, 0.01 * grad_output)
        return grad_x


def straight_through_non_negative(x: torch.Tensor) -> torch.Tensor:
    return StraightThroughNonNegative.apply(x)


class ConvNeXtBlock(nn.Module):
    """
    Depthwise separable ConvNeXt inverted bottleneck block.
    7x7 depthwise conv -> GroupNorm -> 1x1 conv (expansion) -> GELU -> 1x1 conv (projection).
    """

    def __init__(self, dim: int, expansion: int = 2, num_groups: int = 8):
        super().__init__()
        assert dim % num_groups == 0, f"dim ({dim}) must be divisible by num_groups ({num_groups})"
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return res + out


class ConvFusion(nn.Module):
    """
    Lightweight Multi-Scale Convolutional Fusion Block.
    Fuses 32-channel upsampled atmospheric latent features with 32-channel terrain prior.
    Uses 7x7 depthwise convolution with Squeeze-and-Excitation channel gating.
    Linear memory complexity O(HW) requiring under 20 MB VRAM on 80x80 grids.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, num_groups: int = 8):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
        )
        # Squeeze-and-Excitation channel gate
        self.se_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fusion_conv(x)
        gate = self.se_gate(feat)
        return feat * gate


class MultiTaskUNet5x(nn.Module):
    """
    Multi-Task 5x Weather Downscaling Network with Exact Physical Residual Heads.

    Inputs:
        x_lr: [B, 5, 16, 16] (Rain, Tmax, Tmin, RH, Wind)
        terrain_hr: [B, 5, 80, 80] (Elev, Slope, Aspect, Curvature, Windward Lift)

    Outputs:
        Dict with keys "rain", "tmax", "tmin", "rh", "wind", each [B, 1, 80, 80],
        and "tensor" packed [B, 5, 80, 80].
    """

    def __init__(
        self,
        in_channels: int = 5,
        terrain_channels: int = 5,
        base_channels: int = 32,
        num_groups: int = 8,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.scale_factor = 5

        # 1. Coarse atmospheric encoder (16x16)
        self.coarse_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
        )
        self.coarse_block1 = ConvNeXtBlock(base_channels, expansion=2, num_groups=num_groups)
        self.coarse_block2 = ConvNeXtBlock(base_channels, expansion=2, num_groups=num_groups)

        # 2. High-resolution terrain prior encoder (80x80)
        self.terrain_stem = nn.Sequential(
            nn.Conv2d(terrain_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
            ConvNeXtBlock(base_channels, expansion=2, num_groups=num_groups),
        )

        # 3. Lightweight ConvFusion block (64 channels -> 64 channels)
        trunk_channels = base_channels * 2  # 64
        self.fusion = ConvFusion(in_channels=trunk_channels, out_channels=trunk_channels, num_groups=num_groups)

        # 4. Shared atmospheric trunk (80x80)
        self.trunk_blocks = nn.Sequential(
            ConvNeXtBlock(trunk_channels, expansion=2, num_groups=num_groups),
            ConvNeXtBlock(trunk_channels, expansion=2, num_groups=num_groups),
            ConvNeXtBlock(trunk_channels, expansion=2, num_groups=num_groups),
        )

        # 5. Dedicated physical residual output heads
        # (a) Precipitation residual head: Delta P
        self.rain_head = nn.Sequential(
            nn.Conv2d(trunk_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

        # (b) Diurnal temperature residual head: Delta Tmin and Delta Spread
        self.temp_head = nn.Sequential(
            nn.Conv2d(trunk_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 2, kernel_size=1),  # channel 0: Delta Tmin, channel 1: Delta Spread
        )

        # (c) Relative humidity residual head: Delta RH
        self.rh_head = nn.Sequential(
            nn.Conv2d(trunk_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

        # (d) Surface wind residual head: Delta Wind
        self.wind_head = nn.Sequential(
            nn.Conv2d(trunk_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        x_lr: torch.Tensor,
        terrain_hr: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass executing clean residual downscaling.
        
        Args:
            x_lr: Coarse NWP input [B, 5, 16, 16] or [B, 1, 16, 16] (rain only)
            terrain_hr: Fine terrain prior [B, 5, 80, 80]
            return_dict: If True returns dict of outputs; if False returns Tensor.
                         If None: returns dict when input has 5 channels, or rain log-tensor when input has 1 channel.
            
        Returns:
            Dict containing 80x80 tensors for "rain", "tmax", "tmin", "rh", "wind", "tensor",
            or torch.Tensor if return_dict is False.
        """
        b = x_lr.shape[0]
        device = x_lr.device
        dtype = x_lr.dtype

        is_single_ch = (x_lr.shape[1] == 1)
        if is_single_ch:
            # Climatological baseline for missing thermodynamic channels
            tmax_default = torch.full_like(x_lr, 31.5)
            tmin_default = torch.full_like(x_lr, 21.0)
            rh_default = torch.full_like(x_lr, 68.0)
            wind_default = torch.full_like(x_lr, 8.5)
            x_lr = torch.cat([x_lr, tmax_default, tmin_default, rh_default, wind_default], dim=1)

        if terrain_hr is None:
            terrain_hr = torch.zeros(b, 5, 80, 80, device=device, dtype=dtype)

        # 1. Bilinear upsampling of coarse baseline (Pixel-Is-Area geometry)
        coarse_up = F.interpolate(x_lr, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        p_lr_up = coarse_up[:, 0:1, :, :]
        tmax_lr_up = coarse_up[:, 1:2, :, :]
        tmin_lr_up = coarse_up[:, 2:3, :, :]
        rh_lr_up = coarse_up[:, 3:4, :, :]
        wind_lr_up = coarse_up[:, 4:5, :, :]

        # 2. Encode coarse atmospheric features (16x16)
        c_feat = self.coarse_stem(x_lr)
        c_feat = self.coarse_block1(c_feat)
        c_feat = self.coarse_block2(c_feat)

        # 3. 5x Upsample atmospheric features to 80x80
        c_feat_up = F.interpolate(c_feat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        # 4. Encode high-resolution terrain prior (80x80)
        t_feat = self.terrain_stem(terrain_hr)

        # 5. Lightweight ConvFusion
        cat_feat = torch.cat([c_feat_up, t_feat], dim=1)  # [B, 64, 80, 80]
        fused = self.fusion(cat_feat)

        # 6. Shared atmospheric trunk
        trunk = self.trunk_blocks(fused)

        # 7. Physical residual heads
        # (a) Precipitation: StraightThroughNonNegative(P_lr_up + Delta P)
        delta_p = self.rain_head(trunk)
        rain_pred = straight_through_non_negative(p_lr_up + delta_p)

        # (b) Diurnal Temperature: Zero-Bias Spread Clamp
        temp_residuals = self.temp_head(trunk)
        delta_tmin = temp_residuals[:, 0:1, :, :]
        delta_spread = temp_residuals[:, 1:2, :, :]

        coarse_spread = tmax_lr_up - tmin_lr_up
        pred_spread = torch.clamp(coarse_spread + delta_spread, min=0.5)
        tmin_pred = tmin_lr_up + delta_tmin
        tmax_pred = tmin_pred + pred_spread

        # (c) Relative Humidity: clamp to [2.0, 100.0]
        delta_rh = self.rh_head(trunk)
        rh_pred = torch.clamp(rh_lr_up + delta_rh, min=2.0, max=100.0)

        # (d) Surface Wind: clamp to min=0.5 km/h
        delta_wind = self.wind_head(trunk)
        wind_pred = torch.clamp(wind_lr_up + delta_wind, min=0.5)

        # Packed 5-channel tensor [B, 5, 80, 80]
        packed = torch.cat([rain_pred, tmax_pred, tmin_pred, rh_pred, wind_pred], dim=1)

        if return_dict is True:
            return {
                "rain": rain_pred,
                "tmax": tmax_pred,
                "tmin": tmin_pred,
                "rh": rh_pred,
                "wind": wind_pred,
                "tensor": packed,
            }
        elif return_dict is False:
            return packed if not is_single_ch else torch.log1p(rain_pred)

        # Default when return_dict is None:
        if is_single_ch:
            return torch.log1p(rain_pred)

        return {
            "rain": rain_pred,
            "tmax": tmax_pred,
            "tmin": tmin_pred,
            "rh": rh_pred,
            "wind": wind_pred,
            "tensor": packed,
        }
