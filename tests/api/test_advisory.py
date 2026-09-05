import pytest

from src.advisory.engine import build_advisory, rainfall_band, VALID_CROPS, VALID_STAGES
from src.advisory.rules import DRY, LIGHT, MODERATE, HEAVY


def test_heavy_rain_advisory_avoids_fertilizer():
    advisory = build_advisory("ragi", "harvest", 70.0, 80.0)
    assert "fertilizer" in advisory.action_en.lower()
    assert advisory.action_kn
    assert "ಗೊಬ್ಬರ" in advisory.action_kn  # fertilizer in Kannada


def test_crop_and_stage_change_the_advice():
    ragi = build_advisory("ragi", "flowering", 8.0, 12.0)
    paddy = build_advisory("paddy", "flowering", 8.0, 12.0)
    assert ragi.action_en != paddy.action_en
    assert "24 hours" in ragi.action_en
    assert "excess water" in paddy.action_en


def test_all_crop_stage_combinations_produce_valid_bilingual_advisory():
    for crop in VALID_CROPS:
        for stage in VALID_STAGES:
            adv = build_advisory(crop, stage, expected_mm=10.0, likely_max_mm=18.0)
            assert adv.stage == stage.title()
            assert len(adv.action_en) > 20
            assert len(adv.action_kn) > 20
            assert adv.action_en != adv.action_kn


def test_rainfall_bands():
    assert rainfall_band(1.0) == DRY
    assert rainfall_band(5.0) == LIGHT
    assert rainfall_band(25.0) == MODERATE
    assert rainfall_band(75.0) == HEAVY


def test_irrigation_and_harvest_precautions():
    # likely_max > 5 should advise skipping irrigation
    adv_irr = build_advisory("ragi", "vegetative", 3.0, 6.0)
    assert "irrigation" in adv_irr.action_en.lower()

    # harvest with rain >= 15.5 should advise protecting produce
    adv_harv = build_advisory("paddy", "harvest", 16.0, 20.0)
    assert "protect" in adv_harv.action_en.lower()
    assert "ರಕ್ಷಿಸಿ" in adv_harv.action_kn


def test_invalid_crop_or_stage_raises_error():
    with pytest.raises(ValueError, match="Unsupported crop"):
        build_advisory("wheat", "sowing", 5.0, 10.0)

    with pytest.raises(ValueError, match="Unsupported crop stage"):
        build_advisory("ragi", "germination", 5.0, 10.0)
