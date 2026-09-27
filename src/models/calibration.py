"""
src/models/calibration.py

Sprint 8.5 Calibration and Diagnostic Module for Spatiotemporal Weather Downscaling:
  1. Multiplicative Ensemble Spread Rescaling (Scalar and Lead-Dependent alpha).
  2. Isotonic and Logistic Threshold Probability Calibration.
  3. Split Conformal Prediction Intervals with Non-Negative Physical Bounds.
  4. Spatial Sharpness and Texture Diagnostics (Laplacian Energy and 2D Radial PSD).
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def rescale_ensemble_spread(
    members: torch.Tensor,
    alpha: Union[float, torch.Tensor],
    dim: int = 0,
    enforce_non_negative: bool = False,
) -> torch.Tensor:
    """
    Applies multiplicative ensemble spread rescaling around the ensemble mean.
    x'_k = mu + alpha * (x_k - mu)

    Invariants:
      1. Unclipped ensemble mean is strictly invariant: mean(x') == mean(x).
      2. Unclipped variance scales by alpha^2: Var(x') == alpha^2 * Var(x).
      3. If enforce_non_negative is True, outputs satisfy x >= 0.

    Args:
        members: Tensor containing ensemble members, shape [K, ...] where dim is ensemble axis.
        alpha: Spread multiplier (scalar float, or 1D Tensor of shape [T_f] for lead-dependent scaling).
        dim: Dimension index of the ensemble members (default: 0).
        enforce_non_negative: If True, clamp values to >= 0.0 post-scaling.

    Returns:
        Rescaled ensemble tensor with identical shape and dtype.
    """
    mean = members.mean(dim=dim, keepdim=True)
    residuals = members - mean

    if isinstance(alpha, (int, float)):
        scaled_residuals = residuals * float(alpha)
    elif isinstance(alpha, torch.Tensor):
        if alpha.ndim == 0:
            scaled_residuals = residuals * alpha.item()
        elif alpha.ndim == 1:
            # Broadcast alpha [T_f] to members [K, B, T_f, C, H, W] or [K, T_f, ...]
            # Assume lead-time dimension is at index 2 if ndim >= 4, else index 1
            lead_dim = 2 if members.ndim >= 4 else 1
            shape = [1] * members.ndim
            shape[lead_dim] = len(alpha)
            alpha_bcast = alpha.view(*shape).to(device=members.device, dtype=members.dtype)
            scaled_residuals = residuals * alpha_bcast
        else:
            raise ValueError(f"alpha tensor must be 0D or 1D, got ndim={alpha.ndim}")
    else:
        raise TypeError(f"alpha must be float or torch.Tensor, got {type(alpha)}")

    scaled = mean + scaled_residuals

    if enforce_non_negative:
        scaled = torch.clamp(scaled, min=0.0)

    return scaled


class IsotonicProbabilityCalibrator:
    """
    Monotonic non-parametric probability calibrator mapping raw ensemble frequencies to calibrated probabilities.
    Ensures outputs are non-decreasing and strictly bounded in [0, 1].
    """

    def __init__(self):
        self.fitted = False
        self.calibrator = None
        self._fallback_bins: Optional[np.ndarray] = None
        self._fallback_vals: Optional[np.ndarray] = None

    def fit(self, raw_probs: np.ndarray, binary_targets: np.ndarray) -> "IsotonicProbabilityCalibrator":
        raw_flat = np.asarray(raw_probs, dtype=np.float64).ravel()
        tgt_flat = np.asarray(binary_targets, dtype=np.float64).ravel()

        if SKLEARN_AVAILABLE:
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator.fit(raw_flat, tgt_flat)
        else:
            # Simple binning fallback if sklearn is absent
            bins = np.linspace(0.0, 1.0, 11)
            bin_indices = np.digitize(raw_flat, bins) - 1
            vals = np.zeros(len(bins))
            for b in range(len(bins)):
                mask = bin_indices == b
                vals[b] = np.mean(tgt_flat[mask]) if np.any(mask) else bins[b]
            # Enforce monotonicity via cumulative maximum
            self._fallback_bins = bins
            self._fallback_vals = np.maximum.accumulate(vals)

        self.fitted = True
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predict.")
        raw_arr = np.asarray(raw_probs, dtype=np.float64)
        orig_shape = raw_arr.shape
        raw_flat = raw_arr.ravel()

        if SKLEARN_AVAILABLE and self.calibrator is not None:
            calibrated = self.calibrator.predict(raw_flat)
        else:
            calibrated = np.interp(raw_flat, self._fallback_bins, self._fallback_vals)

        calibrated = np.clip(calibrated, 0.0, 1.0)
        return calibrated.reshape(orig_shape)


class LogisticProbabilityCalibrator:
    """
    Parametric Platt scaling calibrator:
      logit(p) -> sigmoid(w0 + w1 * logit(p))
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.w0 = 0.0
        self.w1 = 1.0
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, binary_targets: np.ndarray) -> "LogisticProbabilityCalibrator":
        raw_flat = np.clip(np.asarray(raw_probs, dtype=np.float64).ravel(), self.eps, 1.0 - self.eps)
        tgt_flat = np.asarray(binary_targets, dtype=np.int64).ravel()

        if len(np.unique(tgt_flat)) < 2 or not SKLEARN_AVAILABLE:
            self.w0 = 0.0
            self.w1 = 1.0
            self.fitted = True
            return self

        logit_raw = np.log(raw_flat / (1.0 - raw_flat)).reshape(-1, 1)
        clf = LogisticRegression(solver="lbfgs", C=1.0)
        clf.fit(logit_raw, tgt_flat)
        self.w1 = float(clf.coef_[0, 0])
        self.w0 = float(clf.intercept_[0])

        self.fitted = True
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predict.")
        raw_arr = np.clip(np.asarray(raw_probs, dtype=np.float64), self.eps, 1.0 - self.eps)
        logit_raw = np.log(raw_arr / (1.0 - raw_arr))
        z = self.w0 + self.w1 * logit_raw
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return np.clip(calibrated, 0.0, 1.0)


class ConformalIntervalCalibrator:
    """
    Split conformal prediction interval calibrator with physical non-negative bounds:
      C(x) = [max(0, mu - q * sigma), mu + q * sigma]
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps
        self.q_hat = 1.645 # standard normal 90% default
        self.fitted = False

    def fit(
        self,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        targets: np.ndarray,
        coverage: float = 0.90,
    ) -> "ConformalIntervalCalibrator":
        mu = np.asarray(pred_mean, dtype=np.float64).ravel()
        sigma = np.maximum(np.asarray(pred_std, dtype=np.float64).ravel(), self.eps)
        y = np.asarray(targets, dtype=np.float64).ravel()

        # Studentized non-conformity score
        scores = np.abs(y - mu) / sigma

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - gamma)) / n
        n = len(scores)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        k_idx = int(math.ceil((n + 1) * coverage))
        k_idx = min(n - 1, max(0, k_idx - 1))
        sorted_scores = np.sort(scores)
        self.q_hat = float(sorted_scores[k_idx])
        self.fitted = True
        return self

    def predict_interval(
        self,
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        non_negative: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(pred_mean, dtype=np.float64)
        sigma = np.maximum(np.asarray(pred_std, dtype=np.float64), self.eps)

        lower = mu - self.q_hat * sigma
        upper = mu + self.q_hat * sigma

        if non_negative:
            lower = np.maximum(0.0, lower)
            upper = np.maximum(lower, upper)

        return lower, upper


def compute_laplacian_energy(field: torch.Tensor) -> float:
    """
    Computes spatial Laplacian energy: mean squared response of 2D discrete Laplacian filter.
    E_lap = mean( (nabla^2 X)^2 )

    Args:
        field: Tensor of shape [..., H, W].

    Returns:
        Scalar float representing Laplacian energy.
    """
    if field.ndim < 2:
        raise ValueError(f"field must have at least 2 dimensions, got {field.ndim}")

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=field.device,
        dtype=field.dtype,
    ).view(1, 1, 3, 3)

    orig_shape = field.shape
    h, w = orig_shape[-2], orig_shape[-1]
    field_2d = field.reshape(-1, 1, h, w)

    # Replicate padding for boundaries
    padded = F.pad(field_2d, (1, 1, 1, 1), mode="replicate")
    filtered = F.conv2d(padded, kernel)

    energy = torch.mean(filtered ** 2).item()
    return float(energy)


def compute_radial_psd(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D radially-averaged power spectral density (RAPSD).

    Args:
        field: 2D numpy array [H, W].

    Returns:
        Tuple of (radial_frequencies, radial_power).
    """
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {arr.shape}")

    h, w = arr.shape
    f_shift = np.fft.fftshift(np.fft.fft2(arr - np.mean(arr)))
    power_spectrum = np.abs(f_shift) ** 2 / (h * w)

    y, x = np.indices((h, w))
    center = (int(h / 2), int(w / 2))
    r = np.hypot(x - center[1], y - center[0]).astype(int)

    max_radius = int(min(center[0], center[1]))
    radial_power = np.zeros(max_radius)
    counts = np.zeros(max_radius)

    for i in range(max_radius):
        mask = (r == i)
        if np.any(mask):
            radial_power[i] = np.mean(power_spectrum[mask])
            counts[i] = np.sum(mask)

    freqs = np.arange(max_radius, dtype=np.float64) / max_radius
    return freqs, radial_power
