import pytest

from src.advisory.engine import evaluate_lookahead_risk, rainfall_band_name


def test_dry_today_heavy_tomorrow_triggers_washoff_hazard():
    res = evaluate_lookahead_risk(today_expected_mm=1.8, tomorrow_expected_mm=22.4, tomorrow_likely_max_mm=30.0)
    assert res["has_washoff_hazard"] is True
    assert res["spray_window"] == "HOLD"
    assert res["irrigation_window"] == "POSTPONE"
    assert "Leaching Risk" in res["warning_en"]
    assert "ರಸಗೊಬ್ಬರ ಎಚ್ಚರಿಕೆ" in res["warning_kn"]


def test_dry_today_dry_tomorrow_is_safe():
    res = evaluate_lookahead_risk(today_expected_mm=0.5, tomorrow_expected_mm=1.0, tomorrow_likely_max_mm=3.0)
    assert res["has_washoff_hazard"] is False
    assert res["spray_window"] == "SAFE"
    assert res["harvest_window"] == "SAFE"
    assert res["irrigation_window"] == "IRRIGATE"
    assert res["warning_en"] is None
    assert res["warning_kn"] is None


def test_heavy_rain_today_holds_spray_and_drains():
    res = evaluate_lookahead_risk(today_expected_mm=30.4, tomorrow_expected_mm=5.0, tomorrow_likely_max_mm=8.0)
    assert res["has_washoff_hazard"] is False
    assert res["spray_window"] == "HOLD"
    assert res["harvest_window"] == "HOLD"
    assert res["irrigation_window"] == "DRAIN"


def test_rainfall_band_names():
    assert rainfall_band_name(1.0) == "dry"
    assert rainfall_band_name(5.0) == "light"
    assert rainfall_band_name(30.0) == "moderate"
    assert rainfall_band_name(70.0) == "heavy"
