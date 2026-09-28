"""
tests/models/test_sprint9_provenance.py

Provenance, Zero-Em-Dash, and Baseline Invariant Verification for Sprint 9.
"""

from pathlib import Path
import pytest

from src.models.scalable_residual_diffusion import TIER_CHANNEL_CONFIGS

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CHECKPOINT_SHA256 = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"
EXPECTED_PARAM_COUNT = 15_685_478


def test_sprint9_baseline_invariants():
    """Verify Dense-S baseline invariant constants."""
    assert TIER_CHANNEL_CONFIGS["dense_s"]["exact_control_params"] == EXPECTED_PARAM_COUNT
    assert TIER_CHANNEL_CONFIGS["dense_s"]["base_channels"] == 96


def test_zero_em_dashes_in_sprint9_source_files():
    """Verify strict prohibition of em dashes in all Sprint 9 models and test files."""
    files_to_check = [
        ROOT / "src" / "models" / "moe.py",
        ROOT / "src" / "models" / "scalable_residual_diffusion.py",
        ROOT / "tests" / "models" / "test_capacity_scaling.py",
        ROOT / "tests" / "models" / "test_moe.py",
        ROOT / "tests" / "models" / "test_sprint9_provenance.py",
    ]

    for p in files_to_check:
        if p.exists():
            content = p.read_text(encoding="utf-8")
            assert "\u2014" not in content, f"Em dash (\\u2014) found in {p.name}"
            assert "\u2013" not in content, f"En dash (\\u2013) found in {p.name}"


def test_notebook_budget_and_eval_invariants():
    """Verify notebooks match 30-epoch Candidate 3 training budget and full 2022 validation evaluation."""
    notebooks = [
        ROOT / "notebooks" / "sprint_9_phase1_dense_capacity_scaling.ipynb",
        ROOT / "notebooks" / "sprint_9_model_capacity_scaling.ipynb",
    ]

    for nb in notebooks:
        assert nb.exists(), f"Notebook {nb.name} does not exist"
        txt = nb.read_text(encoding="utf-8")
        assert "\u2014" not in txt, f"Em dash in {nb.name}"
        assert "\u2013" not in txt, f"En dash in {nb.name}"
        assert "TRAIN_EPOCHS = 30" in txt, f"30-epoch training schedule missing in {nb.name}"
        assert "MAX_EVAL_CUBES = None" in txt, f"Full validation set evaluation missing in {nb.name}"
        assert "CosineAnnealingLR" in txt, f"CosineAnnealingLR scheduler missing in {nb.name}"
        assert "Ferro et al. 2008" in txt, f"Canonical Fair-CRPS missing in {nb.name}"


