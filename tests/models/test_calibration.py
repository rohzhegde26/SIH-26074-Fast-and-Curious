"""
tests/models/test_calibration.py

TDD Unit Tests for Sprint 8.5 Calibration and Diagnostic Module:
  1. Ensemble spread rescaling (invariance of mean, variance scaling, non-negativity).
  2. Isotonic and logistic threshold probability calibration.
  3. Split conformal prediction interval calibration with non-negative lower bounds.
  4. Spatial sharpness metrics (Laplacian energy and 2D radial power spectral density).
"""

import math
import numpy as np
import pytest
import torch

from src.models.calibration import (
    rescale_ensemble_spread,
    IsotonicProbabilityCalibrator,
    LogisticProbabilityCalibrator,
    ConformalIntervalCalibrator,
    compute_laplacian_energy,
    compute_radial_psd,
)


class TestSpreadRescalingSeam:
    """Tests for rescale_ensemble_spread public interface."""

    def test_identity_at_alpha_one(self):
        """Alpha = 1.0 must return the exact input tensor."""
        x = torch.randn(8, 2, 7, 6, 16, 16)
        x_scaled = rescale_ensemble_spread(x, alpha=1.0, dim=0, enforce_non_negative=False)
        assert torch.allclose(x_scaled, x, atol=1e-6)

    def test_mean_invariance_before_clipping(self):
        """Ensemble mean must remain strictly invariant under linear spread rescaling."""
        x = torch.randn(8, 2, 7, 6, 16, 16) * 5.0 + 10.0
        orig_mean = x.mean(dim=0)
        
        for alpha in [1.25, 1.5, 2.0, 3.0]:
            x_scaled = rescale_ensemble_spread(x, alpha=alpha, dim=0, enforce_non_negative=False)
            scaled_mean = x_scaled.mean(dim=0)
            assert torch.allclose(scaled_mean, orig_mean, atol=1e-5), f"Mean shifted at alpha={alpha}"

    def test_variance_scaling_unclipped(self):
        """Variance must scale exactly as alpha^2 under linear spread rescaling."""
        x = torch.randn(16, 4, 7, 6, 16, 16) * 3.0
        alpha = 2.0
        x_scaled = rescale_ensemble_spread(x, alpha=alpha, dim=0, enforce_non_negative=False)
        
        orig_std = x.std(dim=0, unbiased=False)
        scaled_std = x_scaled.std(dim=0, unbiased=False)
        assert torch.allclose(scaled_std, orig_std * alpha, atol=1e-5)

    def test_non_negative_enforcement(self):
        """When enforce_non_negative=True, output must have no values below zero."""
        x = torch.randn(8, 2, 7, 6, 16, 16) * 5.0
        x_scaled = rescale_ensemble_spread(x, alpha=2.0, dim=0, enforce_non_negative=True)
        assert (x_scaled >= 0.0).all()

    def test_lead_dependent_alpha_vector(self):
        """Lead-dependent vector alpha [T_f] scales each lead time day independently."""
        T_f = 7
        x = torch.randn(8, 2, T_f, 6, 16, 16) * 4.0 + 5.0
        alphas = torch.tensor([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5])
        
        x_scaled = rescale_ensemble_spread(x, alpha=alphas, dim=0, enforce_non_negative=False)
        
        # Verify lead 0 is unchanged (alpha=1.0)
        assert torch.allclose(x_scaled[:, :, 0], x[:, :, 0], atol=1e-5)
        # Verify lead 5 scaled by 2.0
        orig_std_5 = x[:, :, 5].std(dim=0, unbiased=False)
        scaled_std_5 = x_scaled[:, :, 5].std(dim=0, unbiased=False)
        assert torch.allclose(scaled_std_5, orig_std_5 * 2.0, atol=1e-5)


class TestProbabilityCalibrationSeam:
    """Tests for threshold probability calibration interfaces."""

    def test_isotonic_monotonicity_and_bounds(self):
        """Isotonic regression predictions must be monotonic and bounded in [0, 1]."""
        np.random.seed(42)
        raw_p = np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
        # Over-confident raw probabilities vs targets
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1])
        
        calibrator = IsotonicProbabilityCalibrator()
        calibrator.fit(raw_p, y)
        
        test_p = np.linspace(0, 1, 50)
        calibrated_p = calibrator.predict(test_p)
        
        assert (calibrated_p >= 0.0).all() and (calibrated_p <= 1.0).all()
        # Verify monotonic non-decreasing
        assert (np.diff(calibrated_p) >= -1e-7).all()

    def test_logistic_calibration_bounds(self):
        """Logistic calibrator must produce smooth probabilities bounded in [0, 1]."""
        np.random.seed(42)
        n = 200
        raw_p = np.random.uniform(0.01, 0.99, size=n)
        true_prob = 1.0 / (1.0 + np.exp(-2.0 * (raw_p - 0.5)))
        y = (np.random.uniform(size=n) < true_prob).astype(int)
        
        calibrator = LogisticProbabilityCalibrator()
        calibrator.fit(raw_p, y)
        
        calibrated_p = calibrator.predict(raw_p)
        assert (calibrated_p >= 0.0).all() and (calibrated_p <= 1.0).all()


class TestConformalIntervalSeam:
    """Tests for split-conformal prediction interval calibration."""

    def test_conformal_coverage_and_non_negativity(self):
        """Conformal intervals must guarantee empirical coverage and respect P >= 0."""
        np.random.seed(42)
        n_cal = 500
        # Synthetic precipitation predictions and true observations
        true_mean = np.random.exponential(scale=5.0, size=n_cal)
        pred_mean = np.maximum(0.0, true_mean + np.random.normal(0, 2.0, size=n_cal))
        pred_std = np.maximum(0.5, pred_mean * 0.3)
        y_cal = np.maximum(0.0, true_mean + np.random.normal(0, 1.5, size=n_cal))
        
        calibrator = ConformalIntervalCalibrator()
        calibrator.fit(pred_mean, pred_std, y_cal, coverage=0.90)
        
        # Test out of sample
        n_test = 500
        true_test = np.random.exponential(scale=5.0, size=n_test)
        test_pred_mean = np.maximum(0.0, true_test + np.random.normal(0, 2.0, size=n_test))
        test_pred_std = np.maximum(0.5, test_pred_mean * 0.3)
        y_test = np.maximum(0.0, true_test + np.random.normal(0, 1.5, size=n_test))
        
        lower, upper = calibrator.predict_interval(test_pred_mean, test_pred_std, non_negative=True)
        
        # Invariant: lower bounds strictly non-negative
        assert (lower >= 0.0).all()
        assert (upper >= lower).all()
        
        # Check empirical coverage is close to nominal 90%
        empirical_coverage = np.mean((y_test >= lower) & (y_test <= upper))
        assert empirical_coverage >= 0.82, f"Coverage too low: {empirical_coverage}"


class TestSpatialSharpnessSeam:
    """Tests for spatial Laplacian energy and power spectral density."""

    def test_laplacian_energy_ordering(self):
        """Noisy high-frequency fields must have higher Laplacian energy than smooth fields."""
        smooth = torch.ones(1, 1, 32, 32) * 5.0
        # Add smooth variation
        for i in range(32):
            smooth[:, :, i, :] += math.sin(i / 10.0)
            
        noise = torch.randn(1, 1, 32, 32) * 2.0
        
        energy_smooth = compute_laplacian_energy(smooth)
        energy_noise = compute_laplacian_energy(noise)
        
        assert energy_noise > energy_smooth * 5.0

    def test_radial_psd_computation(self):
        """Radial PSD must return positive power spectrum over spatial frequencies."""
        field = np.random.randn(64, 64)
        freqs, power = compute_radial_psd(field)
        
        assert len(freqs) == len(power)
        assert len(freqs) > 0
        assert (power >= 0.0).all()
