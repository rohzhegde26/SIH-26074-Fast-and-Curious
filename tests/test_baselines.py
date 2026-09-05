"""
tests/test_baselines.py

Unit tests for Sprint 2 baseline architectures and dataset pipelines:
1. BilinearInterpolationBaseline produces exact 5x upsampling (16x16 -> 80x80).
2. DeepSDBaseline conforms to specified input/output shapes and respects non-negativity.
3. Metric calculations honestly separate All-Day vs. Wet-Day (> 2.5 mm).
"""

import pytest
import numpy as np
import torch
from src.models.baselines import BilinearInterpolationBaseline, DeepSDBaseline
from src.models.train import compute_metrics


def test_bilinear_exact_5x_scaling():
    """Verify bilinear baseline strictly upsamples 16x16 to 80x80."""
    model = BilinearInterpolationBaseline(scale_factor=5.0)
    x = torch.rand(4, 1, 16, 16)
    out = model(x)
    assert out.shape == (4, 1, 80, 80), f"Expected (4, 1, 80, 80), got {out.shape}"
    assert torch.all(out >= 0.0), "Bilinear output contains negative precipitation"


def test_deepsd_architecture_forward():
    """Verify DeepSD baseline processes elevation and ensures non-negative output."""
    model = DeepSDBaseline(in_channels=2, hidden_channels=32)
    x_lr = torch.rand(2, 1, 16, 16)
    dem_hr = torch.rand(2, 1, 80, 80)

    out = model(x_lr, dem_hr)
    assert out.shape == (2, 1, 80, 80), f"Expected (2, 1, 80, 80), got {out.shape}"
    assert torch.all(out >= 0.0), "DeepSD output contains negative precipitation values"


def test_wet_day_metric_eliminates_zero_bias():
    """
    Verify that wet-day MAE correctly evaluates only days with Rain > 2.5 mm,
    preventing 90% dry day skew from artificially suppressing error metrics.
    """
    # 10 samples: 8 dry days (0 mm), 2 wet days (10 mm)
    true_phys = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
    # Predictions: perfectly predicts dry days (0 mm), but underestimates wet days by 5 mm (5 mm)
    pred_phys = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0])

    metrics = compute_metrics(pred_phys, true_phys, wet_threshold=2.5)

    # All-day MAE = (0*8 + 5*2)/10 = 1.0 mm (artificially deflated by 80% dry days)
    assert np.isclose(metrics["all_mae"], 1.0)

    # Wet-day MAE = (5*2)/2 = 5.0 mm (honest appraisal of real rainfall error)
    assert np.isclose(metrics["wet_mae"], 5.0)
    assert metrics["wet_mae"] > metrics["all_mae"]
