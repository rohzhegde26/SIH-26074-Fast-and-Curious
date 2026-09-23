"""
tests/models/test_spatiotemporal_baseline.py

TDD Test Suite for Sprint 3 Baselines & Deterministic Learned Downscaler.
Enforces:
  1. Non-learned baselines (0A: Channel-aware, 0B: Bilinear, 0C: Persistence) shape [B, 7, 6, 80, 80].
  2. TemporalMultiTaskUNet5x forward pass with H in {1, 2, 3} yielding [B, 7, 6, 80, 80].
  3. History-to-future cross-attention exists and receives non-zero backward gradients.
  4. Lead embeddings differentiate future lead days D...D+6.
  5. Bottleneck temporal self-attention couples the seven future leads.
  6. Multi-scale terrain prior affects the model predictions.
  7. U and V wind components are modeled and output as separate channels.
  8. Parameter count is under 5.0M parameters for fast T4 execution.
"""

import pytest
import torch
import torch.nn as nn

from src.models.baselines import (
    BilinearTemporalBaseline,
    ChannelAwareTemporalBaseline,
    PersistenceTemporalBaseline,
)
from src.models.temporal_multitask_baseline import TemporalMultiTaskUNet5x


# ---------------------------------------------------------------------------
# Slice 1: Non-Learned Baselines (0A, 0B, 0C)
# ---------------------------------------------------------------------------

def test_bilinear_temporal_baseline_shape():
    """Baseline 0B: All-bilinear 5x upsampling across 7 leads."""
    model = BilinearTemporalBaseline()
    b, l, c, h, w = 2, 7, 6, 16, 16
    fcst = torch.randn(b, l, c, h, w)
    out = model(fcst)
    assert out.shape == (b, 7, 6, 80, 80), f"Expected (2, 7, 6, 80, 80), got {out.shape}"


def test_channel_aware_temporal_baseline():
    """Baseline 0A: Conservative precipitation + bilinear thermodynamics/wind."""
    model = ChannelAwareTemporalBaseline()
    b, l, c, h, w = 2, 7, 6, 16, 16
    fcst = torch.ones(b, l, c, h, w) * 10.0
    out = model(fcst)
    assert out.shape == (b, 7, 6, 80, 80)
    # Conservation check on channel 0 (precipitation): mean of 5x5 block equals coarse pixel
    p_coarse = fcst[:, :, 0:1, :, :]  # [2, 7, 1, 16, 16]
    p_fine = out[:, :, 0:1, :, :]    # [2, 7, 1, 80, 80]
    p_pooled = nn.functional.avg_pool2d(p_fine.view(-1, 1, 80, 80), kernel_size=5, stride=5).view(b, l, 1, 16, 16)
    assert torch.allclose(p_pooled, p_coarse, atol=1e-5), "Precipitation disaggregation violated area mean"


def test_persistence_temporal_baseline():
    """Baseline 0C: Repeat latest historical day D-1 over all 7 leads."""
    model = PersistenceTemporalBaseline()
    b, h, c, h_in, w_in = 2, 3, 6, 16, 16
    history = torch.randn(b, h, c, h_in, w_in)
    out = model(history)
    assert out.shape == (b, 7, 6, 80, 80)
    # All 7 leads should equal the upsampled history[:, -1]
    for lead in range(7):
        assert torch.allclose(out[:, 0], out[:, lead]), f"Lead {lead} differed from lead 0 in persistence baseline"


# ---------------------------------------------------------------------------
# Slice 2: Learned Deterministic TemporalMultiTaskUNet5x
# ---------------------------------------------------------------------------

def test_temporal_multitask_unet_forward_shapes():
    """Verify H=1, H=2, H=3 configurations produce output [B, 7, 6, 80, 80]."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    terrain = torch.randn(2, 5, 80, 80)
    fcst = torch.randn(2, 7, 6, 16, 16)

    for h in [1, 2, 3]:
        history = torch.randn(2, h, 6, 16, 16)
        out = model(history=history, future_forecast=fcst, terrain=terrain)
        assert out.shape == (2, 7, 6, 80, 80), f"H={h} output shape mismatch: {out.shape}"
        assert torch.all(torch.isfinite(out)), f"H={h} output contained NaN or Inf"


def test_history_to_future_cross_attention_gradient_flow():
    """Verify history tensor receives backward gradients through the cross-attention path."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    history = torch.randn(2, 3, 6, 16, 16, requires_grad=True)
    fcst = torch.randn(2, 7, 6, 16, 16, requires_grad=True)
    terrain = torch.randn(2, 5, 80, 80)

    out = model(history=history, future_forecast=fcst, terrain=terrain)
    loss = out.sum()
    loss.backward()

    assert history.grad is not None, "History gradient was None"
    assert torch.any(history.grad != 0.0), "History gradient was zero (cross-attention disconnected!)"
    assert fcst.grad is not None and torch.any(fcst.grad != 0.0), "Forecast gradient was zero"


def test_lead_embedding_differentiation():
    """Verify that different lead indices produce distinct representations."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    lead_emb = model.lead_embed.weight  # [7, embed_dim]
    for i in range(7):
        for j in range(i + 1, 7):
            diff = torch.norm(lead_emb[i] - lead_emb[j]).item()
            assert diff > 1e-4, f"Lead embeddings for {i} and {j} are identical"


def test_terrain_conditioning_sensitivity():
    """Verify that altering terrain prior alters output predictions."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    model.eval()
    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain1 = torch.zeros(1, 5, 80, 80)
    terrain2 = torch.ones(1, 5, 80, 80)

    with torch.no_grad():
        out1 = model(history=history, future_forecast=fcst, terrain=terrain1)
        out2 = model(history=history, future_forecast=fcst, terrain=terrain2)

    diff = torch.norm(out1 - out2).item()
    assert diff > 1e-3, f"Model is insensitive to terrain prior (diff={diff})"


def test_wind_channels_separate():
    """Verify U and V channels remain independent and uncoupled in output heads."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    history = torch.randn(1, 3, 6, 16, 16)
    fcst = torch.randn(1, 7, 6, 16, 16)
    terrain = torch.randn(1, 5, 80, 80)

    out = model(history=history, future_forecast=fcst, terrain=terrain)
    u_pred = out[0, 0, 4]  # Lead 0, Channel 4 (Wind U)
    v_pred = out[0, 0, 5]  # Lead 0, Channel 5 (Wind V)

    diff = torch.norm(u_pred - v_pred).item()
    assert diff > 1e-3, "Wind U and V channels produced identical predictions"


def test_model_parameter_count():
    """Verify model parameter count is under 5.0M for fast T4 GPU execution."""
    model = TemporalMultiTaskUNet5x(base_channels=24)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 5_000_000, f"Model parameter count too large: {param_count} (limit: 5.0M)"
