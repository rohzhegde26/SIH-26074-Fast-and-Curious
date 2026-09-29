"""
tests/models/test_final_profiles.py

Tests for Sprint 10 Operational Profiles (ACCURATE and ENSEMBLE).
Verifies:
  1. Profile configuration integrity and channel counts.
  2. Zero em dashes in configs.
  3. Matched NFE budgets: ACCURATE (32 NFE), ENSEMBLE (64 NFE).
  4. Dense-L parameter footprint matches 51,997,958 exactly.
"""

from pathlib import Path
import pytest
import yaml

from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "final"


def test_accurate_profile_config():
    p = CONFIG_DIR / "profile_accurate.yaml"
    assert p.exists(), f"Missing {p}"
    content = p.read_text(encoding="utf-8")
    assert "\u2014" not in content
    assert "\u2013" not in content

    cfg = yaml.safe_load(content)
    assert cfg["profile_name"] == "ACCURATE"
    assert cfg["model_tier"] == "dense_l"
    assert cfg["num_members"] == 2
    assert cfg["denoising_steps"] == 16
    assert cfg["target_nfe"] == 32
    assert cfg["num_members"] * cfg["denoising_steps"] == 32
    assert cfg["base_channels"] == 176
    assert cfg["total_parameters"] == 51_997_958


def test_ensemble_profile_config():
    p = CONFIG_DIR / "profile_ensemble.yaml"
    assert p.exists(), f"Missing {p}"
    content = p.read_text(encoding="utf-8")
    assert "\u2014" not in content
    assert "\u2013" not in content

    cfg = yaml.safe_load(content)
    assert cfg["profile_name"] == "ENSEMBLE"
    assert cfg["model_tier"] == "dense_l"
    assert cfg["num_members"] == 8
    assert cfg["denoising_steps"] == 8
    assert cfg["target_nfe"] == 64
    assert cfg["num_members"] * cfg["denoising_steps"] == 64
    assert cfg["base_channels"] == 176
    assert cfg["total_parameters"] == 51_997_958


def test_model_instantiation_from_tier():
    model = create_scalable_residual_diffusion("dense_l")
    prof = model.profile_compute()
    assert prof["total_parameters"] == 51_997_958
    assert prof["active_parameters"] == 51_997_958
    assert prof["trainable_parameters"] == 51_997_958
