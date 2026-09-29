"""
src/eval/multivariate_diagnostics.py

Sprint 10: Multivariate Physical Consistency and Calibration Diagnostics.
Evaluates joint physical relationships across predicted variables:
  1. Thermodynamic Invariant: P(Tmax < Tmin) and maximum inversion gap.
  2. Moisture-Precipitation Coupling: Bivariate correlation r(P, RH).
  3. Physical Bounds and Mass Conservation: Negative precipitation mass shift.
  4. Probabilistic Threshold Verification: Brier Score (BS) and Brier Skill Score (BSS).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch


def audit_thermodynamic_bounds(
    tmax: torch.Tensor,
    tmin: torch.Tensor,
) -> Dict[str, float]:
    """
    Checks physical invariant: Tmax >= Tmin.
    
    Args:
        tmax: Maximum temperature tensor in deg C.
        tmin: Minimum temperature tensor in deg C.
        
    Returns:
        Dict with violation rate and maximum inversion gap in deg C.
    """
    inversion = tmin - tmax
    violations = inversion > 0.0
    violation_rate = float(torch.mean(violations.float()).item())
    max_gap = float(torch.max(torch.clamp(inversion, min=0.0)).item())

    return {
        "violation_rate": round(violation_rate, 5),
        "violation_percentage": round(violation_rate * 100.0, 3),
        "max_inversion_gap_degc": round(max_gap, 3),
    }


def compute_precipitation_rh_coupling(
    precip: torch.Tensor,
    rh: torch.Tensor,
    wet_threshold: float = 2.5,
) -> Dict[str, float]:
    """
    Computes spatial correlation between precipitation rate and relative humidity.
    Physical reality: Higher precipitation rates correlate positively with near-saturated air masses.
    """
    p_flat = precip.reshape(-1).float()
    rh_flat = rh.reshape(-1).float()

    # Mean RH in wet vs dry cells
    wet_mask = p_flat > wet_threshold
    dry_mask = p_flat <= wet_threshold

    mean_rh_wet = float(torch.mean(rh_flat[wet_mask]).item()) if torch.any(wet_mask) else 0.0
    mean_rh_dry = float(torch.mean(rh_flat[dry_mask]).item()) if torch.any(dry_mask) else 0.0

    # Pearson correlation
    p_centered = p_flat - torch.mean(p_flat)
    rh_centered = rh_flat - torch.mean(rh_flat)
    cov = torch.mean(p_centered * rh_centered)
    p_std = torch.std(p_flat)
    rh_std = torch.std(rh_flat)
    corr = float((cov / (p_std * rh_std + 1e-8)).item())

    return {
        "pearson_corr_p_rh": round(corr, 4),
        "mean_rh_wet_cells": round(mean_rh_wet, 2),
        "mean_rh_dry_cells": round(mean_rh_dry, 2),
        "rh_contrast_wet_minus_dry": round(mean_rh_wet - mean_rh_dry, 2),
    }


def compute_mass_shift_diagnostics(
    raw_precip: torch.Tensor,
    repaired_precip: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluates physical bounds repair impact on total precipitation mass.
    Verifies that negative precipitation zeroing shifts total mass by < 1.0%.
    """
    raw_total = float(torch.sum(torch.clamp(raw_precip, min=0.0)).item())
    raw_negative = float(torch.sum(torch.clamp(-raw_precip, min=0.0)).item())
    repaired_total = float(torch.sum(repaired_precip).item())

    shift_ratio = raw_negative / max(1e-6, raw_total)
    net_diff = abs(repaired_total - raw_total) / max(1e-6, raw_total)

    return {
        "raw_negative_mass_ratio": round(shift_ratio, 5),
        "negative_mass_percentage": round(shift_ratio * 100.0, 3),
        "net_repair_mass_shift": round(net_diff, 5),
        "conservation_compliant": shift_ratio < 0.01,
    }


def compute_brier_scores(
    ensemble_forecasts: torch.Tensor,
    observations: torch.Tensor,
    thresholds: List[float] = [2.5, 15.0, 30.0, 50.0],
) -> Dict[str, Dict[str, float]]:
    """
    Computes Brier Score (BS) and Brier Skill Score (BSS) against sample climatology.
    
    Args:
        ensemble_forecasts: Tensor of shape [K, ...] with K ensemble members.
        observations: Tensor of shape [...] with ground truth.
        thresholds: List of precipitation thresholds in mm/day.
    """
    results = {}
    for thr in thresholds:
        # Forecast probability: fraction of members exceeding threshold
        p_fc = torch.mean((ensemble_forecasts >= thr).float(), dim=0)
        # Binary event occurrence
        o_obs = (observations >= thr).float()

        # Brier Score = mean((P_fc - O_obs)^2)
        bs = float(torch.mean((p_fc - o_obs) ** 2).item())

        # Climatological baseline probability
        p_clim = float(torch.mean(o_obs).item())
        bs_clim = float(torch.mean((p_clim - o_obs) ** 2).item())

        # Brier Skill Score = 1 - BS / BS_clim
        bss = float(1.0 - (bs / max(1e-6, bs_clim)))

        results[f"threshold_{int(thr)}mm"] = {
            "threshold_mm": thr,
            "brier_score": round(bs, 4),
            "climatology_brier_score": round(bs_clim, 4),
            "brier_skill_score": round(bss, 4),
            "event_frequency": round(p_clim, 4),
        }

    return results
