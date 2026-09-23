"""
src/models/temporal_multitask_baseline.py

Sprint 3 Deterministic Spatiotemporal Downscaler (TemporalMultiTaskUNet5x).
Architectural Specifications:
  1. Historical Context Encoder: Spatial projection at 16x16 with temporal self-attention over H in {1, 2, 3}.
  2. Future GFS Forecast Conditioning: 16x16 projection with explicit learned lead-day embeddings (D...D+6).
  3. History-to-Future Cross-Attention: Future lead queries attend to historical keys/values.
  4. Spatial ConvNeXt/U-Net Hierarchy: 80x80 -> 40x40 -> 20x20 -> 10x10 with skip connections.
  5. Bottleneck Temporal Self-Attention: Joint temporal attention coupling the seven future leads at 10x10.
  6. High-Resolution Terrain Conditioning: 5-channel topography fused at 80x80.
  7. Joint Multivariate Prediction: 6 channels [P, Tmax, Tmin, RH, U, V] with separate U/V heads.
  8. Model-Space Residual Formulation: pred = upsample(future_forecast) + residual_correction.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """
    Depthwise separable ConvNeXt block with 7x7 depthwise conv,
    GroupNorm, inverted bottleneck expansion (2x), and GELU.
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


class TemporalMultiTaskUNet5x(nn.Module):
    """
    7-Day Deterministic Multivariate Downscaling Network.
    Downscales coarse numerical weather prediction fields from 0.25° (16x16) to 0.05° (80x80)
    across 7 forecast lead days (D through D+6), conditioned on H antecedent days and terrain.
    """

    def __init__(
        self,
        in_weather_channels: int = 6,
        terrain_channels: int = 5,
        num_leads: int = 7,
        base_channels: int = 24,
        embed_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_weather_channels = in_weather_channels
        self.terrain_channels = terrain_channels
        self.num_leads = num_leads
        self.base_channels = base_channels
        self.embed_dim = embed_dim

        # 1. Lead Day Embedding for D ... D+6
        self.lead_embed = nn.Embedding(num_leads, embed_dim)

        # 2. History & Future Coarse Projections (16x16)
        self.hist_proj = nn.Sequential(
            nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.fcst_proj = nn.Sequential(
            nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        # 3. History Temporal Self-Attention & History-to-Future Cross-Attention
        self.hist_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.hist_future_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # 4. Spatial 5x Upsampling Adapter: 16x16 -> 80x80
        # Inverted bottleneck upsampler
        self.spatial_up = nn.Sequential(
            nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # 5. High-Resolution Terrain Encoder (80x80)
        self.terrain_enc = nn.Sequential(
            nn.Conv2d(terrain_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(base_channels, num_groups=8),
        )

        # 6. Combined 80x80 Stem: fuses atmospheric features + terrain features
        c1 = base_channels * 2
        self.stem = nn.Sequential(
            nn.Conv2d(c1, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(base_channels, num_groups=8),
        )

        # 7. U-Net Encoder Stages (applied per lead)
        # Stage 1: 80x80 -> 40x40
        c2 = base_channels * 2  # 48
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, c2, kernel_size=3, stride=2, padding=1),
            ConvNeXtBlock(c2, num_groups=8),
        )
        # Stage 2: 40x40 -> 20x20
        c3 = base_channels * 4  # 96
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            ConvNeXtBlock(c3, num_groups=8),
        )
        # Stage 3: 20x20 -> 10x10
        c4 = base_channels * 8  # 192
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
            ConvNeXtBlock(c4, num_groups=8),
        )

        # 8. Bottleneck Temporal Self-Attention across 7 Leads at 10x10
        self.bottleneck_temporal_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=num_heads, batch_first=True)
        self.bottleneck_norm = nn.LayerNorm(c4)

        # 9. U-Net Decoder Stages with Skip Connections
        # Up 3: 10x10 -> 20x20
        self.up3 = nn.Upsample(size=(20, 20), mode="bilinear", align_corners=False)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(c4 + c3, c3, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(c3, num_groups=8),
        )
        # Up 2: 20x20 -> 40x40
        self.up2 = nn.Upsample(size=(40, 40), mode="bilinear", align_corners=False)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(c2, num_groups=8),
        )
        # Up 1: 40x40 -> 80x80
        self.up1 = nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(c2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(base_channels, num_groups=8),
        )

        # 10. Multi-Task Output Heads (Predicting residual corrections)
        self.head_precip = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmax = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmin = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_rh = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_u = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_v = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
        history_forecast_aux: Optional[torch.Tensor] = None,
        return_residual: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            history: [B, H, 6, 16, 16] antecedent observations (H in {1, 2, 3})
            future_forecast: [B, 7, 6, 16, 16] coarse NWP forecast leads
            terrain: [B, 5, 80, 80] high-resolution terrain prior
            history_forecast_aux: optional future extension tensor (ignored if None)
            return_residual: if True, returns pure residual correction without baseline addition
        Returns:
            [B, 7, 6, 80, 80] downscaled normalized weather fields
        """
        b, h_len, _, _, _ = history.shape
        _, num_leads, _, _, _ = future_forecast.shape

        # 1. Bilinear upsampling of coarse forecast baseline to 80x80: [B, 7, 6, 80, 80]
        coarse_flat = future_forecast.view(b * num_leads, self.in_weather_channels, 16, 16)
        baseline_up = F.interpolate(
            coarse_flat,
            size=(80, 80),
            mode="bilinear",
            align_corners=False,
        ).view(b, num_leads, self.in_weather_channels, 80, 80)

        # 2. Encode history: [B * H, 6, 16, 16] -> [B, H, embed_dim]
        hist_flat = history.view(b * h_len, self.in_weather_channels, 16, 16)
        hist_feats = self.hist_proj(hist_flat)  # [B * H, embed_dim, 16, 16]
        # Pool spatial dimensions to obtain per-day historical context vector
        hist_tokens = F.adaptive_avg_pool2d(hist_feats, 1).view(b, h_len, self.embed_dim)  # [B, H, embed_dim]
        hist_context, _ = self.hist_self_attn(hist_tokens, hist_tokens, hist_tokens)  # [B, H, embed_dim]

        # 3. Encode future forecast leads with lead embeddings: [B, 7, embed_dim]
        fcst_feats = self.fcst_proj(coarse_flat)  # [B * 7, embed_dim, 16, 16]
        fcst_tokens = F.adaptive_avg_pool2d(fcst_feats, 1).view(b, num_leads, self.embed_dim)  # [B, 7, embed_dim]
        lead_idx = torch.arange(num_leads, device=future_forecast.device)
        lead_emb = self.lead_embed(lead_idx).unsqueeze(0).expand(b, num_leads, self.embed_dim)  # [B, 7, embed_dim]
        fcst_tokens = fcst_tokens + lead_emb

        # 4. History-to-Future Cross-Attention: Queries=Future, Keys/Values=History
        cross_context, _ = self.hist_future_cross_attn(
            query=fcst_tokens,
            key=hist_context,
            value=hist_context,
        )  # [B, 7, embed_dim]
        fused_tokens = self.cross_norm(fcst_tokens + cross_context)  # [B, 7, embed_dim]

        # 5. Broadcast temporal context tokens into spatial features: [B * 7, embed_dim, 16, 16]
        fused_spatial = fcst_feats + fused_tokens.view(b * num_leads, self.embed_dim, 1, 1)

        # 6. Spatial upsampling to 80x80: [B * 7, base_channels, 80, 80]
        atmos_80 = self.spatial_up(fused_spatial)

        # 7. Terrain encoding (80x80) expanded across 7 leads
        terrain_feats = self.terrain_enc(terrain)  # [B, base_channels, 80, 80]
        terrain_expanded = terrain_feats.unsqueeze(1).expand(b, num_leads, self.base_channels, 80, 80)
        terrain_flat = terrain_expanded.reshape(b * num_leads, self.base_channels, 80, 80)

        # 8. Fuse atmospheric features with terrain prior at 80x80
        cat_80 = torch.cat([atmos_80, terrain_flat], dim=1)  # [B * 7, 2 * base_channels, 80, 80]
        s1 = self.stem(cat_80)  # [B * 7, base_channels, 80, 80]

        # 9. Downsampling encoder
        s2 = self.down1(s1)  # [B * 7, c2, 40, 40]
        s3 = self.down2(s2)  # [B * 7, c3, 20, 20]
        s4 = self.down3(s3)  # [B * 7, c4, 10, 10]

        # 10. Bottleneck Temporal Self-Attention across 7 Leads at 10x10
        _, c4, h10, w10 = s4.shape
        # Reshape to treat each spatial pixel at 10x10 across 7 leads: [B * 100, 7, c4]
        s4_perm = s4.view(b, num_leads, c4, h10 * w10).permute(0, 3, 1, 2).reshape(b * h10 * w10, num_leads, c4)
        s4_temporal, _ = self.bottleneck_temporal_attn(s4_perm, s4_perm, s4_perm)
        s4_temporal = self.bottleneck_norm(s4_perm + s4_temporal)
        # Reshape back to [B * 7, c4, 10, 10]
        s4_fused = s4_temporal.view(b, h10 * w10, num_leads, c4).permute(0, 2, 3, 1).reshape(b * num_leads, c4, h10, w10)

        # 11. Upsampling decoder with skip connections
        u3 = self.up3(s4_fused)  # [B * 7, c4, 20, 20]
        d3 = self.dec_conv3(torch.cat([u3, s3], dim=1))  # [B * 7, c3, 20, 20]

        u2 = self.up2(d3)  # [B * 7, c3, 40, 40]
        d2 = self.dec_conv2(torch.cat([u2, s2], dim=1))  # [B * 7, c2, 40, 40]

        u1 = self.up1(d2)  # [B * 7, c2, 80, 80]
        d1 = self.dec_conv1(torch.cat([u1, s1], dim=1))  # [B * 7, base_channels, 80, 80]

        # 12. Predict 6 individual weather channel residual corrections
        r_precip = self.head_precip(d1)
        r_tmax = self.head_tmax(d1)
        r_tmin = self.head_tmin(d1)
        r_rh = self.head_rh(d1)
        r_wind_u = self.head_wind_u(d1)
        r_wind_v = self.head_wind_v(d1)

        residual = torch.cat([r_precip, r_tmax, r_tmin, r_rh, r_wind_u, r_wind_v], dim=1)  # [B * 7, 6, 80, 80]
        residual_cube = residual.view(b, num_leads, self.in_weather_channels, 80, 80)

        if return_residual:
            return residual_cube

        # Model-space residual combination: prediction = upsample(coarse) + residual
        pred_cube = baseline_up + residual_cube
        return pred_cube
