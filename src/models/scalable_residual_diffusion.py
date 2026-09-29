"""
src/models/scalable_residual_diffusion.py

Scalable Conditional Spatiotemporal Residual Diffusion Downscaler.
Implements the Sprint 9 Model Capacity Scaling Ladder:
  1. Dense-S:  base_channels = 96   (~15.69M params, exact Candidate 3 control)
  2. Dense-M:  base_channels = 136  (~31.20M params, target [25M, 35M])
  3. Dense-L:  base_channels = 176  (~52.00M params, target [45M, 65M])
  4. MoE-4:    base_channels = 96   (E=4 experts, Top-k in {1, 2})

Strict Architectural Principles:
  - Preserves history H=14, context N=24, output crop M=16.
  - Preserves v-prediction parameterization and group-tail multi-task loss.
  - Preserves linear beta schedule (T=100, beta_start=1e-4, beta_end=0.035).
  - Exact state_dict key alignment with Candidate 3 checkpoint under Dense-S.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.moe import MoETimeConditionedConvNeXtBlock
from src.models.residual_diffusion import (
    SinusoidalPositionalEmbedding,
    SpatiotemporalDenoiser,
    SpatiotemporalResidualDiffusion,
    TimeConditionedConvNeXtBlock,
    compute_residual_target,
)


TIER_CHANNEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "dense_s": {
        "base_channels": 96,
        "target_param_range": (15_500_000, 16_000_000),
        "exact_control_params": 15_685_478,
        "use_moe": False,
        "description": "Dense-S (Candidate 3 Frozen Control, 15.69M params)",
    },
    "dense_m": {
        "base_channels": 136,
        "target_param_range": (25_000_000, 35_000_000),
        "exact_control_params": None,
        "use_moe": False,
        "description": "Dense-M (Scaled Intermediate Width, 31.20M params)",
    },
    "dense_l": {
        "base_channels": 176,
        "target_param_range": (45_000_000, 65_000_000),
        "exact_control_params": None,
        "use_moe": False,
        "description": "Dense-L (Large Width Denoiser, 52.00M params)",
    },
    "moe_4": {
        "base_channels": 96,
        "target_param_range": (15_500_000, 45_000_000),
        "exact_control_params": None,
        "use_moe": True,
        "num_experts": 4,
        "top_k": 1,
        "description": "MoE-4 (Sparse Top-1 Routing over 4 Bottleneck Experts)",
    },
    "moe_4_top2": {
        "base_channels": 96,
        "target_param_range": (15_500_000, 45_000_000),
        "exact_control_params": None,
        "use_moe": True,
        "num_experts": 4,
        "top_k": 2,
        "description": "MoE-4-Top2 (Sparse Top-2 Routing over 4 Bottleneck Experts)",
    },
}


class ScalableSpatiotemporalDenoiser(nn.Module):
    """
    Parametric U-Net Denoiser supporting arbitrary base_channels and optional MoE blocks.
    When use_moe=False, architecture is structurally identical to SpatiotemporalDenoiser.
    """

    def __init__(
        self,
        in_weather_channels: int = 6,
        terrain_channels: int = 5,
        num_leads: int = 7,
        base_channels: int = 96,
        time_emb_dim: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.in_weather_channels = in_weather_channels
        self.num_leads = num_leads
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.embed_dim = embed_dim
        self.use_moe = use_moe

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Lead Day Embedding
        self.lead_embed = nn.Embedding(num_leads, embed_dim)

        # History & Forecast Coarse Projections (16x16)
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

        # Cross-Attention
        self.hist_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.hist_future_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # 5x Upsampling for coarse conditioning
        self.spatial_up = nn.Sequential(
            nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Terrain Encoder (80x80)
        self.terrain_enc = nn.Sequential(
            nn.Conv2d(terrain_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # Noisy Residual Input Stem (80x80)
        in_stem_dim = in_weather_channels + base_channels + base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_stem_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stem_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        # Downsampling Stages
        c2 = base_channels * 2
        self.down1_conv = nn.Conv2d(base_channels, c2, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        c3 = base_channels * 4
        self.down2_conv = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        c4 = base_channels * 8
        self.down3_conv = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)

        # Deep Bottleneck: Standard ConvNeXt or Sparse MoE
        if use_moe:
            self.down3_block = MoETimeConditionedConvNeXtBlock(
                dim=c4,
                time_emb_dim=time_emb_dim,
                num_groups=8,
                num_experts=num_experts,
                top_k=top_k,
                aux_loss_weight=aux_loss_weight,
            )
        else:
            self.down3_block = TimeConditionedConvNeXtBlock(c4, time_emb_dim, num_groups=8)

        # Bottleneck Temporal Attention across 7 leads at 10x10
        self.bottleneck_temporal_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=num_heads, batch_first=True)
        self.bottleneck_norm = nn.LayerNorm(c4)

        # Decoder Stages with Skip Connections
        self.up3 = nn.Upsample(size=(20, 20), mode="bilinear", align_corners=False)
        self.dec3_conv = nn.Conv2d(c4 + c3, c3, kernel_size=3, padding=1)
        self.dec3_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        self.up2 = nn.Upsample(size=(40, 40), mode="bilinear", align_corners=False)
        self.dec2_conv = nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1)
        self.dec2_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        self.up1 = nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False)
        self.dec1_conv = nn.Conv2d(c2 + base_channels, base_channels, kernel_size=3, padding=1)
        self.dec1_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        # 6-channel Noise Prediction Output Heads
        self.head_precip = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmax = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmin = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_rh = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_u = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_v = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        r_t: torch.Tensor,
        t: torch.Tensor,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
    ) -> torch.Tensor:
        b, num_leads, _, _, _ = r_t.shape
        h_len = history.shape[1]

        # 1. Timestep embedding
        t_emb = self.time_mlp(t)
        t_emb_flat = t_emb.unsqueeze(1).expand(b, num_leads, self.time_emb_dim).reshape(b * num_leads, self.time_emb_dim)

        # 2. History encoding
        _, _, _, n_lat, n_lon = history.shape
        hist_flat = history.reshape(b * h_len, self.in_weather_channels, n_lat, n_lon)
        hist_feats = self.hist_proj(hist_flat)
        hist_tokens = F.adaptive_avg_pool2d(hist_feats, 1).view(b, h_len, self.embed_dim)
        hist_context, _ = self.hist_self_attn(hist_tokens, hist_tokens, hist_tokens)

        # 3. Forecast encoding with lead embeddings
        coarse_flat = future_forecast.reshape(b * num_leads, self.in_weather_channels, n_lat, n_lon)
        fcst_feats = self.fcst_proj(coarse_flat)
        fcst_tokens = F.adaptive_avg_pool2d(fcst_feats, 1).view(b, num_leads, self.embed_dim)
        lead_idx = torch.arange(num_leads, device=r_t.device)
        lead_emb = self.lead_embed(lead_idx).unsqueeze(0).expand(b, num_leads, self.embed_dim)
        fcst_tokens = fcst_tokens + lead_emb

        # 4. Cross-attention
        cross_context, _ = self.hist_future_cross_attn(query=fcst_tokens, key=hist_context, value=hist_context)
        fused_tokens = self.cross_norm(fcst_tokens + cross_context)

        # 5. Spatial context fusion and central crop
        fused_spatial = fcst_feats + fused_tokens.view(b * num_leads, self.embed_dim, 1, 1)
        if n_lat > 16 or n_lon > 16:
            crop_lat = (n_lat - 16) // 2
            crop_lon = (n_lon - 16) // 2
            fused_spatial_16 = fused_spatial[:, :, crop_lat : crop_lat + 16, crop_lon : crop_lon + 16]
        else:
            fused_spatial_16 = fused_spatial

        atmos_80 = self.spatial_up(fused_spatial_16)

        # 6. Terrain encoding
        terr_feats = self.terrain_enc(terrain)
        terr_flat = terr_feats.unsqueeze(1).expand(b, num_leads, self.base_channels, 80, 80).reshape(b * num_leads, self.base_channels, 80, 80)

        # 7. Residual denoiser stem & encoder stages
        r_t_flat = r_t.view(b * num_leads, self.in_weather_channels, 80, 80)
        stem_in = torch.cat([r_t_flat, atmos_80, terr_flat], dim=1)

        s1 = self.stem(stem_in)
        s1 = self.stem_block(s1, t_emb_flat)

        s2 = self.down1_conv(s1)
        s2 = self.down1_block(s2, t_emb_flat)

        s3 = self.down2_conv(s2)
        s3 = self.down2_block(s3, t_emb_flat)

        s4 = self.down3_conv(s3)
        s4 = self.down3_block(s4, t_emb_flat)

        # 8. Bottleneck temporal attention
        _, c4, h10, w10 = s4.shape
        s4_perm = s4.view(b, num_leads, c4, h10 * w10).permute(0, 3, 1, 2).reshape(b * h10 * w10, num_leads, c4)
        s4_temporal, _ = self.bottleneck_temporal_attn(s4_perm, s4_perm, s4_perm)
        s4_temporal = self.bottleneck_norm(s4_perm + s4_temporal)
        s4_fused = s4_temporal.view(b, h10 * w10, num_leads, c4).permute(0, 2, 3, 1).reshape(b * num_leads, c4, h10, w10)

        # 9. Decoder stages with skip connections
        u3 = self.up3(s4_fused)
        d3 = self.dec3_block(self.dec3_conv(torch.cat([u3, s3], dim=1)), t_emb_flat)

        u2 = self.up2(d3)
        d2 = self.dec2_block(self.dec2_conv(torch.cat([u2, s2], dim=1)), t_emb_flat)

        u1 = self.up1(d2)
        d1 = self.dec1_block(self.dec1_conv(torch.cat([u1, s1], dim=1)), t_emb_flat)

        # 10. Predict noise for all 6 meteorological channels
        eps_p = self.head_precip(d1)
        eps_tmax = self.head_tmax(d1)
        eps_tmin = self.head_tmin(d1)
        eps_rh = self.head_rh(d1)
        eps_u = self.head_wind_u(d1)
        eps_v = self.head_wind_v(d1)

        eps = torch.cat([eps_p, eps_tmax, eps_tmin, eps_rh, eps_u, eps_v], dim=1)
        return eps.view(b, num_leads, self.in_weather_channels, 80, 80)


class ScalableSpatiotemporalResidualDiffusion(SpatiotemporalResidualDiffusion):
    """
    Parametric Spatiotemporal Residual Diffusion Downscaler for Sprint 9.
    Provides complete capacity profiling, active compute tracking, and MoE integration.
    """

    def __init__(
        self,
        tier: str = "dense_s",
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.035,
        base_channels: Optional[int] = None,
        prediction_type: str = "v_prediction",
        loss_weighting: str = "group_tail",
        use_moe: Optional[bool] = None,
        num_experts: int = 4,
        top_k: int = 1,
        aux_loss_weight: float = 0.01,
    ):
        # Resolve tier configuration
        tier_cfg = TIER_CHANNEL_CONFIGS.get(tier, {})
        resolved_channels = base_channels or tier_cfg.get("base_channels", 96)
        resolved_use_moe = use_moe if use_moe is not None else tier_cfg.get("use_moe", False)
        resolved_experts = tier_cfg.get("num_experts", num_experts)
        resolved_top_k = tier_cfg.get("top_k", top_k)

        # Initialize base diffusion class with resolved parameters
        super().__init__(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            base_channels=resolved_channels,
            prediction_type=prediction_type,
            loss_weighting=loss_weighting,
        )

        self.tier = tier
        self.use_moe = resolved_use_moe
        self.num_experts = resolved_experts
        self.top_k = resolved_top_k
        self.aux_loss_weight = aux_loss_weight

        # If MoE or custom base_channels requested, instantiate ScalableSpatiotemporalDenoiser
        if self.use_moe:
            self.denoiser = ScalableSpatiotemporalDenoiser(
                base_channels=resolved_channels,
                use_moe=True,
                num_experts=resolved_experts,
                top_k=resolved_top_k,
                aux_loss_weight=aux_loss_weight,
            )

        # Verify Dense-S control invariant
        if tier == "dense_s" and not self.use_moe and resolved_channels == 96:
            p_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            assert p_count == 15_685_478, (
                f"Strict Invariant Violation: Dense-S parameter count {p_count} does not match Candidate 3 (15,685,478)!"
            )

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def active_parameters(self) -> int:
        """Calculates active parameter count during inference forward pass."""
        if not self.use_moe:
            return self.trainable_parameters

        total = self.trainable_parameters
        if hasattr(self.denoiser.down3_block, "get_active_parameters"):
            moe_block = self.denoiser.down3_block
            block_total = sum(p.numel() for p in moe_block.parameters())
            block_active = moe_block.get_active_parameters()
            return total - block_total + block_active
        return total

    def compute_training_loss(
        self,
        r_0: torch.Tensor,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
        target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss, model_pred, target_val = super().compute_training_loss(
            r_0=r_0,
            history=history,
            future_forecast=future_forecast,
            terrain=terrain,
            target_norm=target_norm,
        )

        # Add auxiliary MoE load-balancing loss if enabled
        if self.use_moe and hasattr(self.denoiser.down3_block, "last_aux_loss"):
            aux_loss = self.denoiser.down3_block.last_aux_loss
            loss = loss + aux_loss

        return loss, model_pred, target_val

    def profile_compute(
        self,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Profiles parameter counts, active compute, and estimated memory footprint.
        """
        dev = torch.device(device)
        total_p = self.total_parameters
        trainable_p = self.trainable_parameters
        active_p = self.active_parameters
        candidate3_p = 15_685_478

        ratio = trainable_p / candidate3_p
        checkpoint_fp32_mb = (total_p * 4) / (1024 * 1024)
        checkpoint_fp16_mb = (total_p * 2) / (1024 * 1024)

        # Estimate activation memory for batch_size cubes of [B, 7, 6, 80, 80]
        # At FP16: activations roughly scale with base_channels
        estimated_vram_gb = (checkpoint_fp16_mb * 3 + (self.denoiser.base_channels * 80 * 80 * 7 * batch_size * 2 * 12) / (1024 * 1024)) / 1024

        return {
            "tier": self.tier,
            "base_channels": self.denoiser.base_channels,
            "use_moe": self.use_moe,
            "num_experts": getattr(self, "num_experts", 1),
            "top_k": getattr(self, "top_k", 1),
            "total_parameters": total_p,
            "trainable_parameters": trainable_p,
            "active_parameters": active_p,
            "parameter_ratio_vs_candidate3": round(ratio, 3),
            "checkpoint_fp32_mb": round(checkpoint_fp32_mb, 2),
            "checkpoint_fp16_mb": round(checkpoint_fp16_mb, 2),
            "estimated_vram_gb": round(estimated_vram_gb, 2),
        }


def create_scalable_residual_diffusion(
    tier: str = "dense_s",
    **kwargs,
) -> ScalableSpatiotemporalResidualDiffusion:
    """
    Factory function to instantiate scalable residual diffusion models.
    Tiers: 'dense_s' (15.69M), 'dense_m' (31.20M), 'dense_l' (52.00M), 'moe_4'.
    """
    assert tier in TIER_CHANNEL_CONFIGS, f"Unknown tier: {tier}. Available: {list(TIER_CHANNEL_CONFIGS.keys())}"
    return ScalableSpatiotemporalResidualDiffusion(tier=tier, **kwargs)
