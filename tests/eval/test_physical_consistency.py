"""
tests/eval/test_physical_consistency.py

Unit tests for Sprint 10 multivariate physical consistency diagnostics:
  1. Thermodynamic invariant checks (Tmax >= Tmin).
  2. Moisture-precipitation coupling and RH contrast.
  3. Physical bounds repair mass conservation (< 1.0% negative shift).
  4. Brier Skill Score calculation on ensemble exceedance.
  5. Zero em dashes in test files.
"""

from pathlib import Path
import pytest
import torch

from src.eval.multivariate_diagnostics import (
    audit_thermodynamic_bounds,
    compute_precipitation_rh_coupling,
    compute_mass_shift_diagnostics,
    compute_brier_scores,
)

ROOT = Path(__file__).resolve().parents[2]


def test_zero_em_dashes_in_diagnostics():
    p = ROOT / "src" / "eval" / "multivariate_diagnostics.py"
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "\u2014" not in content
    assert "\u2013" not in content


def test_thermodynamic_bounds():
    # Valid: Tmax > Tmin everywhere
    tmax = torch.tensor([32.0, 30.0, 28.0])
    tmin = torch.tensor([22.0, 20.0, 18.0])
    diag = audit_thermodynamic_bounds(tmax, tmin)
    assert diag["violation_rate"] == 0.0
    assert diag["max_inversion_gap_degc"] == 0.0

    # Inversion: 1 violation
    tmin_inv = torch.tensor([22.0, 31.0, 18.0])
    diag_inv = audit_thermodynamic_bounds(tmax, tmin_inv)
    assert diag_inv["violation_rate"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert diag_inv["max_inversion_gap_degc"] == 1.0


def test_precipitation_rh_coupling():
    precip = torch.tensor([0.0, 0.5, 10.0, 45.0])
    rh = torch.tensor([45.0, 50.0, 88.0, 95.0])
    diag = compute_precipitation_rh_coupling(precip, rh, wet_threshold=2.5)
    assert diag["pearson_corr_p_rh"] > 0.5
    assert diag["mean_rh_wet_cells"] > diag["mean_rh_dry_cells"]
    assert diag["rh_contrast_wet_minus_dry"] > 0.0


def test_mass_shift_diagnostics():
    # Minor negative noise (<1% mass)
    raw = torch.tensor([10.0, 20.0, -0.05, 15.0])
    repaired = torch.clamp(raw, min=0.0)
    diag = compute_mass_shift_diagnostics(raw, repaired)
    assert diag["conservation_compliant"] is True
    assert diag["raw_negative_mass_ratio"] < 0.01


def test_brier_scores():
    # 4 ensemble members predicting high rain
    # Event occurs in 3 of 4 spatial points
    fc = torch.tensor([
        [20.0, 35.0, 5.0, 40.0],
        [18.0, 32.0, 8.0, 42.0],
        [22.0, 38.0, 2.0, 36.0],
        [19.0, 31.0, 6.0, 39.0],
    ])
    obs = torch.tensor([21.0, 33.0, 4.0, 38.0])
    scores = compute_brier_scores(fc, obs, thresholds=[15.0, 30.0])

    assert "threshold_15mm" in scores
    assert "threshold_30mm" in scores
    assert scores["threshold_15mm"]["brier_skill_score"] > 0.5
