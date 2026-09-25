"""
src/models/residual_diffusion.py

Sprint 3 Conditional Spatiotemporal Residual Diffusion Downscaler.
Mathematical Formulation:
  1. Model-Space Residual Target:
     r_0 = y_fine_norm - upsample(future_forecast_norm)  [B, 7, 6, 80, 80]
  2. Forward Gaussian Diffusion Process (T=100):
     q(r_t | r_0) = N(r_t; sqrt(alpha_bar_t) * r_0, (1 - alpha_bar_t) * I)
  3. Denoiser Network eps_theta(r_t, t, history, forecast, terrain):
     Predicts noise residual eps with AdaGN timestep modulation,
     history-to-future cross-attention, and lead temporal attention.
  4. DDIM Fast Reverse Sampler:
     Configurable sampling steps (4, 8, 16, 32) for fast calibrated generation.
  5. Final Reconstruction:
     y_downscaled_norm = upsample(future_forecast_norm) + r_0_sampled
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.samplers import (
    get_sampling_timesteps,
    sample_ddim_trajectory,
    sample_dpm_solver_2m_trajectory,
    sample_pndm_trajectory,
)


def compute_residual_target(y_fine_norm: torch.Tensor, future_forecast_norm: torch.Tensor) -> torch.Tensor:
    """
    Computes fine spatial residual in normalized model space:
    r_0 = y_fine_norm - upsample(future_forecast_central_16).

    Args:
        y_fine_norm: [B, 7, 6, 80, 80]
        future_forecast_norm: [B, 7, 6, N, N] where N >= 16
    Returns:
        r_0: [B, 7, 6, 80, 80]
    """
    b, leads, c, h, w = future_forecast_norm.shape
    if h > 16 or w > 16:
        offset_h = (h - 16) // 2
        offset_w = (w - 16) // 2
        fcst_crop = future_forecast_norm[:, :, :, offset_h : offset_h + 16, offset_w : offset_w + 16]
    else:
        fcst_crop = future_forecast_norm

    fcst_flat = fcst_crop.reshape(b * leads, c, 16, 16)
    fcst_up = F.interpolate(
        fcst_flat,
        size=(80, 80),
        mode="bilinear",
        align_corners=False,
    ).view(b, leads, c, 80, 80)
    return y_fine_norm - fcst_up


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal timestep embeddings for continuous diffusion steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeConditionedConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with Adaptive Group Normalization (AdaGN) timestep conditioning.
    """

    def __init__(self, dim: int, time_emb_dim: int, expansion: int = 2, num_groups: int = 8):
        super().__init__()
        assert dim % num_groups == 0
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.time_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim * 2),
        )
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)

        # Time modulation: scale and shift
        scale_shift = self.time_proj(time_emb)  # [B * leads, 2 * dim]
        scale, shift = scale_shift.chunk(2, dim=-1)
        out = out * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return res + out


class SpatiotemporalDenoiser(nn.Module):
    """
    U-Net Noise Predictor eps_theta(r_t, t, history, forecast, terrain).
    """

    def __init__(
        self,
        in_weather_channels: int = 6,
        terrain_channels: int = 5,
        num_leads: int = 7,
        base_channels: int = 20,
        time_emb_dim: int = 64,
        embed_dim: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.in_weather_channels = in_weather_channels
        self.num_leads = num_leads
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.embed_dim = embed_dim

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

        # Cross-Attention: future queries attend to historical keys/values
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

        # Noisy Residual Input Stem (80x80):
        # Concatenates: r_t (6 ch) + upscaled atmospheric conditioning (base_channels) + terrain (base_channels)
        in_stem_dim = in_weather_channels + base_channels + base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_stem_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stem_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        # Downsampling Stages
        c2 = base_channels * 2  # 40
        self.down1_conv = nn.Conv2d(base_channels, c2, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        c3 = base_channels * 4  # 80
        self.down2_conv = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        c4 = base_channels * 8  # 160
        self.down3_conv = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
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
        """
        Predicts added noise eps_theta with shape [B, 7, 6, 80, 80].
        """
        b, num_leads, _, _, _ = r_t.shape
        h_len = history.shape[1]

        # 1. Timestep embedding: [B, time_emb_dim] -> expanded across leads [B * leads, time_emb_dim]
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]
        t_emb_flat = t_emb.unsqueeze(1).expand(b, num_leads, self.time_emb_dim).reshape(b * num_leads, self.time_emb_dim)

        # 2. Encode history: [B, H, embed_dim]
        _, _, _, n_lat, n_lon = history.shape
        hist_flat = history.reshape(b * h_len, self.in_weather_channels, n_lat, n_lon)
        hist_feats = self.hist_proj(hist_flat)
        hist_tokens = F.adaptive_avg_pool2d(hist_feats, 1).view(b, h_len, self.embed_dim)
        hist_context, _ = self.hist_self_attn(hist_tokens, hist_tokens, hist_tokens)

        # 3. Encode future forecast leads with lead embeddings
        coarse_flat = future_forecast.reshape(b * num_leads, self.in_weather_channels, n_lat, n_lon)
        fcst_feats = self.fcst_proj(coarse_flat)
        fcst_tokens = F.adaptive_avg_pool2d(fcst_feats, 1).view(b, num_leads, self.embed_dim)
        lead_idx = torch.arange(num_leads, device=r_t.device)
        lead_emb = self.lead_embed(lead_idx).unsqueeze(0).expand(b, num_leads, self.embed_dim)
        fcst_tokens = fcst_tokens + lead_emb

        # 4. History-to-future cross-attention
        cross_context, _ = self.hist_future_cross_attn(query=fcst_tokens, key=hist_context, value=hist_context)
        fused_tokens = self.cross_norm(fcst_tokens + cross_context)

        # 5. Spatial context fusion and Central RoI Cropping
        fused_spatial = fcst_feats + fused_tokens.view(b * num_leads, self.embed_dim, 1, 1)

        # Central RoI Crop from N x N to fixed target footprint M x M (16 x 16)
        if n_lat > 16 or n_lon > 16:
            crop_lat = (n_lat - 16) // 2
            crop_lon = (n_lon - 16) // 2
            fused_spatial_16 = fused_spatial[:, :, crop_lat : crop_lat + 16, crop_lon : crop_lon + 16]
        else:
            fused_spatial_16 = fused_spatial

        # 5x Upsampling from 16x16 to target 80x80: [B * leads, base_channels, 80, 80]
        atmos_80 = self.spatial_up(fused_spatial_16)

        # 6. Terrain encoding expanded across 7 leads
        terr_feats = self.terrain_enc(terrain)  # [B, base_channels, 80, 80]
        terr_flat = terr_feats.unsqueeze(1).expand(b, num_leads, self.base_channels, 80, 80).reshape(b * num_leads, self.base_channels, 80, 80)

        # 7. Concatenate noisy residual r_t with atmospheric and terrain conditioning
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

        # 8. Bottleneck Temporal Attention across 7 leads at 10x10
        _, c4, h10, w10 = s4.shape
        s4_perm = s4.view(b, num_leads, c4, h10 * w10).permute(0, 3, 1, 2).reshape(b * h10 * w10, num_leads, c4)
        s4_temporal, _ = self.bottleneck_temporal_attn(s4_perm, s4_perm, s4_perm)
        s4_temporal = self.bottleneck_norm(s4_perm + s4_temporal)
        s4_fused = s4_temporal.view(b, h10 * w10, num_leads, c4).permute(0, 2, 3, 1).reshape(b * num_leads, c4, h10, w10)

        # 9. Decoder with Skip Connections
        u3 = self.up3(s4_fused)
        d3 = self.dec3_block(self.dec3_conv(torch.cat([u3, s3], dim=1)), t_emb_flat)

        u2 = self.up2(d3)
        d2 = self.dec2_block(self.dec2_conv(torch.cat([u2, s2], dim=1)), t_emb_flat)

        u1 = self.up1(d2)
        d1 = self.dec1_block(self.dec1_conv(torch.cat([u1, s1], dim=1)), t_emb_flat)

        # 10. Predict noise for each channel
        eps_p = self.head_precip(d1)
        eps_tmax = self.head_tmax(d1)
        eps_tmin = self.head_tmin(d1)
        eps_rh = self.head_rh(d1)
        eps_u = self.head_wind_u(d1)
        eps_v = self.head_wind_v(d1)

        eps = torch.cat([eps_p, eps_tmax, eps_tmin, eps_rh, eps_u, eps_v], dim=1)
        return eps.view(b, num_leads, self.in_weather_channels, 80, 80)


class SpatiotemporalResidualDiffusion(nn.Module):
    """
    Complete Continuous Residual Diffusion Downscaler with DDIM Reverse Sampler.
    """

    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.035,
        base_channels: int = 20,
        prediction_type: str = "epsilon",
        loss_weighting: str = "uniform",
    ):
        super().__init__()
        assert prediction_type in ["epsilon", "v_prediction"], f"Unknown prediction_type: {prediction_type}"
        assert loss_weighting in ["uniform", "group_tail"], f"Unknown loss_weighting: {loss_weighting}"
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.loss_weighting = loss_weighting

        # Linear beta schedule (Nichol & Dhariwal / Ho et al.)
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Denoiser network
        self.denoiser = SpatiotemporalDenoiser(base_channels=base_channels)

    def q_sample(self, r_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse data r_0 to timestep t."""
        if noise is None:
            noise = torch.randn_like(r_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)

        r_t = sqrt_alpha_bar * r_0 + sqrt_one_minus_alpha_bar * noise
        return r_t, noise

    def compute_v_target(self, r_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Salimans & Ho (2022) velocity parameterization:
        v_t = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * r_0
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * r_0

    def compute_training_loss(
        self,
        r_0: torch.Tensor,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
        target_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples random timestep t, adds noise, and computes training loss (epsilon or v_prediction).
        """
        b = r_0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=r_0.device)
        r_t, noise = self.q_sample(r_0, t)

        model_pred = self.denoiser(
            r_t=r_t,
            t=t,
            history=history,
            future_forecast=future_forecast,
            terrain=terrain,
        )

        if self.prediction_type == "v_prediction":
            target_val = self.compute_v_target(r_0, t, noise)
        else:
            target_val = noise

        if self.loss_weighting == "group_tail":
            # Multi-Task Variable-Aware Noise Weighting with Convective-Tail Calibration
            # Channel 0: Precipitation (focal tail weighting on extreme wet cells > 15 mm)
            # In normalized space: norm_thresh = (15.0 - 6.266) / 18.620 approx 0.469
            ref_precip = target_norm[:, :, 0:1] if target_norm is not None else r_0[:, :, 0:1]
            tail_mask = (ref_precip > 0.469).float()
            precip_weights = 1.0 + 2.0 * tail_mask  # 3.0 for extreme wet cells (> 15mm)
            precip_diff_sq = (model_pred[:, :, 0:1] - target_val[:, :, 0:1]) ** 2
            precip_loss = (precip_weights * precip_diff_sq).mean()

            # Channels 1-3: Thermodynamic variables (Tmax, Tmin, RH)
            thermo_loss = F.mse_loss(model_pred[:, :, 1:4], target_val[:, :, 1:4])

            # Channels 4-5: Dynamic Wind Vector (U, V)
            wind_loss = F.mse_loss(model_pred[:, :, 4:6], target_val[:, :, 4:6])

            # Group Balancing Weights (Plan Section 5.1 EXP-03):
            # lambda_precip = 1.0, lambda_thermo = 1.2, lambda_wind = 1.1
            loss = 1.0 * precip_loss + 1.2 * thermo_loss + 1.1 * wind_loss
        else:
            loss = F.mse_loss(model_pred, target_val)

        return loss, model_pred, target_val

    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
        num_steps: int = 16,
        seed: Optional[int] = None,
        init_residual: Optional[torch.Tensor] = None,
        init_noise_level: Optional[int] = None,
        schedule_type: str = "standard",
        sampler: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Reverse Sampling across configurable steps (e.g. 4, 8, 16, 32, 64) and ODE solvers:
          - "ddim": 1st-order pseudo-Euler ODE
          - "dpm_solver": 2nd-order DPM-Solver++(2M) in data space
          - "pndm": 4th-order Adams-Bashforth multi-step in eps space
        Schedule types:
          - "standard": Canonical uniform schedule tau_k = round(k * (T - 1) / (S - 1)) spanning 99 to 0.
          - "legacy": Sprint 5/6 stride slicing schedule.
        Returns final predicted weather cube [B, 7, 6, 80, 80] in normalized space.
        """
        if seed is not None:
            torch.manual_seed(seed)

        b, leads, c, h, w = future_forecast.shape
        device = future_forecast.device

        # Coarse baseline upsampled to 80x80 from central target footprint
        if h > 16 or w > 16:
            offset_h = (h - 16) // 2
            offset_w = (w - 16) // 2
            fcst_crop = future_forecast[:, :, :, offset_h : offset_h + 16, offset_w : offset_w + 16]
        else:
            fcst_crop = future_forecast

        fcst_flat = fcst_crop.reshape(b * leads, c, 16, 16)
        baseline_up = F.interpolate(
            fcst_flat,
            size=(80, 80),
            mode="bilinear",
            align_corners=False,
        ).view(b, leads, c, 80, 80)

        # Timestep trajectory
        timesteps_desc = get_sampling_timesteps(
            total_timesteps=self.timesteps,
            num_steps=num_steps,
            schedule_type=schedule_type,
        )

        # Initial noise: start from pure Gaussian noise unless refinement mode is active
        if init_residual is not None and init_noise_level is not None:
            t_init = torch.full((b,), init_noise_level, device=device, dtype=torch.long)
            r_cur, _ = self.q_sample(init_residual, t_init)
            start_idx = min(range(len(timesteps_desc)), key=lambda i: abs(timesteps_desc[i] - init_noise_level))
            timesteps_desc = timesteps_desc[start_idx:]
        else:
            r_cur = torch.randn(b, leads, c, 80, 80, device=device)

        def denoise_fn(r_in: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
            return self.denoiser(
                r_t=r_in,
                t=t_tensor,
                history=history,
                future_forecast=future_forecast,
                terrain=terrain,
            )

        sampler_clean = sampler.lower().replace("-", "_")
        if sampler_clean in ("dpm_solver", "dpm_solver_2m", "dpmsolver"):
            r_sampled = sample_dpm_solver_2m_trajectory(
                denoise_fn=denoise_fn,
                r_init=r_cur,
                timesteps_desc=timesteps_desc,
                alphas_cumprod=self.alphas_cumprod,
                prediction_type=self.prediction_type,
            )
        elif sampler_clean == "pndm":
            r_sampled = sample_pndm_trajectory(
                denoise_fn=denoise_fn,
                r_init=r_cur,
                timesteps_desc=timesteps_desc,
                alphas_cumprod=self.alphas_cumprod,
                prediction_type=self.prediction_type,
            )
        else:
            r_sampled = sample_ddim_trajectory(
                denoise_fn=denoise_fn,
                r_init=r_cur,
                timesteps_desc=timesteps_desc,
                alphas_cumprod=self.alphas_cumprod,
                prediction_type=self.prediction_type,
                eta=eta,
            )

        # Final reconstruction = baseline + predicted fine residual
        y_downscaled_norm = baseline_up + r_sampled
        return y_downscaled_norm

    @torch.no_grad()
    def sample_ddim(
        self,
        history: torch.Tensor,
        future_forecast: torch.Tensor,
        terrain: torch.Tensor,
        steps: int = 32,
        eta: float = 0.0,
        seed: Optional[int] = None,
        schedule_type: str = "standard",
        sampler: str = "ddim",
    ) -> torch.Tensor:
        """Alias for sampling across specified steps, schedule, and solver."""
        return self.sample(
            history=history,
            future_forecast=future_forecast,
            terrain=terrain,
            num_steps=steps,
            seed=seed,
            schedule_type=schedule_type,
            sampler=sampler,
            eta=eta,
        )
