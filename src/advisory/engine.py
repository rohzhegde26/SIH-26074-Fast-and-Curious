from __future__ import annotations

from dataclasses import dataclass

from .rules import (
    Advice,
    CROP_STAGE_ACTIONS,
    DRY,
    HEAVY,
    LIGHT,
    MODERATE,
    STAGE_PREFIX,
    FinancialRisk,
    compute_financial_risk,
)

VALID_CROPS = {"ragi", "paddy", "sugarcane"}
VALID_STAGES = {
    "sowing", "vegetative", "flowering", "harvest",
    "germination", "tillering", "grand_growth", "ripening"
}


@dataclass(frozen=True)
class Advisory:
    stage: str
    action_en: str
    action_kn: str
    financial_risk: FinancialRisk | None = None



def rainfall_band(expected_mm: float) -> Advice:
    if expected_mm < 2.5:
        return DRY
    if expected_mm <= 15.5:
        return LIGHT
    if expected_mm <= 64.4:
        return MODERATE
    return HEAVY


def build_advisory(
    crop: str,
    stage: str,
    expected_mm: float,
    likely_max_mm: float,
    wind_kph: float = 8.0,
    rh_pct: float = 65.0,
    tmax_c: float = 31.5,
    tmin_c: float = 21.0,
) -> Advisory:
    if crop not in VALID_CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    if stage not in CROP_STAGE_ACTIONS.get(crop, {}):
        raise ValueError(f"Unsupported crop stage: {stage}")

    base = rainfall_band(expected_mm)
    prefix_en, prefix_kn = STAGE_PREFIX[stage]
    crop_stage = CROP_STAGE_ACTIONS[crop][stage]
    notes_en: list[str] = [crop_stage.en, base.en]
    notes_kn: list[str] = [crop_stage.kn, base.kn]

    # Multi-variable DAMU/KVK agromet wind & humidity thresholds
    if wind_kph >= 15.0:
        notes_en.append(f"High wind speed ({wind_kph:.1f} km/h): Withhold foliar spraying due to chemical drift hazard.")
        notes_kn.append(f"ಹೆಚ್ಚಿನ ಗಾಳಿಯ ವೇಗ ({wind_kph:.1f} ಕಿಮೀ/ಗಂ): ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ, ರಾಸಾಯನಿಕ ಗಾಳಿಗೆ ಹರಡುವ ಅಪಾಯವಿದೆ.")
    elif expected_mm < 2.5 and wind_kph < 15.0 and rh_pct < 80.0:
        notes_en.append("Favorable weather (Rain < 2.5mm, Wind < 15 km/h): Safe window for scheduled spraying.")
        notes_kn.append("ಅನುಕೂಲಕರ ಹವಾಮಾನ (ಮಳೆ < 2.5 ಮಿಮೀ, ಗಾಳಿ < 15 ಕಿಮೀ/ಗಂ): ಸಿಂಪಡಣೆಗೆ ಸುರಕ್ಷಿತ ಸಮಯ.")

    if rh_pct >= 85.0 and 20.0 <= tmax_c <= 30.0:
        notes_en.append(f"Fungal disease risk (RH {rh_pct:.0f}%, Temp {tmax_c:.1f}°C): Favorable conditions for blast/blight pathogen multiplication. Monitor crop foliage closely.")
        notes_kn.append(f"ಶಿಲೀಂಧ್ರ ರೋಗದ ಅಪಾಯ (ಆರ್ದ್ರತೆ {rh_pct:.0f}%, ಉಷ್ಣಾಂಶ {tmax_c:.1f}°C): ಬೆಂಕಿ ರೋಗ ಹರಡುವ ಸಾಧ್ಯತೆ ಹೆಚ್ಚಾಗಿದೆ. ಬೆಳೆಗಳನ್ನು ಸೂಕ್ಷ್ಮವಾಗಿ ಪರಿಶೀಲಿಸಿ.")
    elif rh_pct >= 85.0 and expected_mm < 2.5:
        notes_en.append(f"High relative humidity ({rh_pct:.0f}%): Monitor crop for fungal blast or blight development.")
        notes_kn.append(f"ಹೆಚ್ಚಿನ ಸಾಪೇಕ್ಷ ಆರ್ದ್ರತೆ ({rh_pct:.0f}%): ಬೆಳೆಯ ಶಿಲೀಂಧ್ರ ರೋಗ ಬಾಧೆಯನ್ನು ಸೂಕ್ಷ್ಮವಾಗಿ ಗಮನಿಸಿ.")

    # Thermal heat stress threshold
    if tmax_c >= 35.0:
        notes_en.append(f"Extreme heat stress (Tmax {tmax_c:.1f}°C): High thermal load during {stage} stage can induce pollen sterility. Provide light frequent irrigations.")
        notes_kn.append(f"ಅಧಿಕ ತಾಪಮಾನದ ಒತ್ತಡ ({tmax_c:.1f}°C): {stage} ಹಂತದಲ್ಲಿ ಹೂವು ಉದುರುವಿಕೆ ತಡೆಯಲು ಲಘು ನೀರಾವರಿ ನೀಡಿ.")

    if likely_max_mm > 10:
        notes_en.append("Do not apply fertilizer before this rain event.")
        notes_kn.append("ಈ ಮಳೆ ಸಂದರ್ಭಕ್ಕೂ ಮುನ್ನ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.")
    if likely_max_mm > 5:
        notes_en.append("Skip scheduled irrigation if rainfall occurs.")
        notes_kn.append("ಮಳೆ ಬಂದರೆ ನಿಗದಿತ ನೀರಾವರಿ ತಪ್ಪಿಸಿ.")
    if stage in {"harvest", "ripening"} and likely_max_mm >= 15.5:
        notes_en.append("Protect harvested produce and drying areas from rain.")
        notes_kn.append("ಕೊಯ್ಲು ಮಾಡಿದ ಬೆಳೆ ಮತ್ತು ಒಣಗಿಸುವ ಸ್ಥಳಗಳನ್ನು ಮಳೆಯಿಂದ ರಕ್ಷಿಸಿ.")

    fin_risk = compute_financial_risk(crop, stage, expected_mm, likely_max_mm)

    return Advisory(
        stage=stage.title(),
        action_en=prefix_en + " ".join(notes_en),
        action_kn=prefix_kn + " ".join(notes_kn),
        financial_risk=fin_risk,
    )


def rainfall_band_name(expected_mm: float) -> str:
    if expected_mm < 2.5:
        return "dry"
    if expected_mm <= 15.5:
        return "light"
    if expected_mm <= 64.4:
        return "moderate"
    return "heavy"


def evaluate_lookahead_risk(
    today_expected_mm: float,
    tomorrow_expected_mm: float,
    tomorrow_likely_max_mm: float,
) -> dict:
    """Evaluates 48-hour chemical leaching and operational windows.
    
    Prevents the 'False Safe Today' trap where dry weather today is followed
    by heavy rain tomorrow that would wash away applied urea and pesticides.
    """
    has_washoff_hazard = (
        today_expected_mm < 2.5
        and (tomorrow_expected_mm >= 10.0 or tomorrow_likely_max_mm >= 15.0)
    )

    if has_washoff_hazard:
        spray_window = "HOLD"
        harvest_window = "HOLD" if tomorrow_likely_max_mm >= 15.0 else "SAFE"
        irrigation_window = "POSTPONE"
        warning_en = (
            f"⚠️ 48-Hour Leaching Risk: Today is dry ({today_expected_mm:.1f} mm), but tomorrow brings "
            f"{tomorrow_expected_mm:.1f} mm rain ({tomorrow_likely_max_mm:.1f} mm likely max). "
            "Withhold urea top-dressing and pesticide spraying today to prevent runoff waste."
        )
        warning_kn = (
            f"⚠️ 48 ಗಂಟೆಗಳ ರಸಗೊಬ್ಬರ ಎಚ್ಚರಿಕೆ: ಇಂದು ಒಣಹವೆ ({today_expected_mm:.1f} ಮಿಮೀ) ಇದ್ದರೂ, ನಾಳೆ "
            f"{tomorrow_expected_mm:.1f} ಮಿಮೀ ಮಳೆ ({tomorrow_likely_max_mm:.1f} ಮಿಮೀ ಸಂಭಾವ್ಯ ಗರಿಷ್ಠ) ಇದೆ. "
            "ಗೊಬ್ಬರ ಕೊಚ್ಚಿಹೋಗುವುದನ್ನು ತಪ್ಪಿಸಲು ಯೂರಿಯಾ ಹಾಗೂ ಕೀಟನಾಶಕ ಸಿಂಪಡಿಸಬೇಡಿ."
        )
    elif today_expected_mm >= 15.5:
        spray_window = "HOLD"
        harvest_window = "HOLD"
        irrigation_window = "DRAIN"
        warning_en = None
        warning_kn = None
    elif today_expected_mm >= 2.5:
        spray_window = "HOLD" if tomorrow_expected_mm >= 10.0 else "RISKY"
        harvest_window = "HOLD"
        irrigation_window = "POSTPONE"
        warning_en = None
        warning_kn = None
    else:
        spray_window = "SAFE"
        harvest_window = "SAFE"
        irrigation_window = "IRRIGATE"
        warning_en = None
        warning_kn = None

    return {
        "has_washoff_hazard": has_washoff_hazard,
        "spray_window": spray_window,
        "harvest_window": harvest_window,
        "irrigation_window": irrigation_window,
        "warning_en": warning_en,
        "warning_kn": warning_kn,
    }


from datetime import date, timedelta

DAY_NAMES_KN = ["ಸೋಮ", "ಮಂಗಳ", "ಬುಧ", "ಗುರು", "ಶುಕ್ರ", "ಶನಿ", "ಭಾನು"]


def build_7day_forecast(record: dict) -> list[dict]:
    """Generates an agro-meteorologically consistent 7-day forecast series with 48h lookahead."""
    if "multi_day_forecast" in record and record["multi_day_forecast"]:
        return record["multi_day_forecast"]

    base_date_str = str(record.get("forecast_date", "2023-07-01"))
    try:
        base_d = date.fromisoformat(base_date_str)
    except Exception:
        base_d = date(2023, 7, 1)

    exp_0 = float(record.get("expected_mm", 0.0))
    lmin_0 = float(record.get("likely_min_mm", 0.0))
    lmax_0 = float(record.get("likely_max_mm", 0.0))
    lgd = str(record.get("lgd_code", "0"))

    # Demo contrast fixtures
    if lgd == "215504":  # Banavasi: 1.8mm dry today -> 22.4mm wash-off hazard tomorrow!
        daily_rains = [
            (exp_0, lmin_0, lmax_0),
            (22.4, 14.0, 31.5),
            (7.8, 3.2, 12.0),
            (1.2, 0.0, 3.5),
            (0.0, 0.0, 1.5),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 2.0),
        ]
    elif lgd == "219388":  # Nalligere: 30.4mm Cloudburst today -> clearing out
        daily_rains = [
            (exp_0, lmin_0, lmax_0),
            (4.8, 1.2, 9.0),
            (1.5, 0.0, 3.5),
            (0.2, 0.0, 1.0),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
        ]
    elif lgd == "219431":  # Naguvanahalli: Dry window throughout
        daily_rains = [
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
            (1.2, 0.0, 2.5),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.5),
        ]
    else:
        seed_hash = sum(ord(c) for c in lgd)
        daily_rains = [(exp_0, lmin_0, lmax_0)]
        for day_i in range(1, 7):
            synoptic_factor = [0.0, 1.8, 0.9, 0.3, 0.1, 0.05, 0.1][day_i]
            local_variation = ((seed_hash * (day_i + 3) * 17) % 100) / 100.0 - 0.2
            day_exp = max(0.0, round(exp_0 * synoptic_factor + local_variation * 3.5, 1))
            day_lmin = max(0.0, round(day_exp * 0.5, 1))
            day_lmax = round(day_exp * 1.6 + 2.0, 1)
            daily_rains.append((day_exp, day_lmin, day_lmax))

    items = []
    for day_i in range(7):
        cur_d = base_d + timedelta(days=day_i)
        cur_exp, cur_lmin, cur_lmax = daily_rains[day_i]

        if day_i < 6:
            next_exp, _, next_lmax = daily_rains[day_i + 1]
        else:
            next_exp, next_lmax = 0.0, 0.0

        lookahead = evaluate_lookahead_risk(cur_exp, next_exp, next_lmax)
        band = rainfall_band_name(cur_exp)

        if day_i == 0:
            label_en = "Today"
            label_kn = "ಇಂದು"
        elif day_i == 1:
            label_en = "Tomorrow"
            label_kn = "ನಾಳೆ"
        else:
            w_idx = cur_d.weekday()
            label_en = cur_d.strftime("%a %d")
            label_kn = f"{DAY_NAMES_KN[w_idx]} {cur_d.strftime('%d')}"

        if band == "dry" and not lookahead["has_washoff_hazard"]:
            sum_en = "Safe weather window. Ideal for field labour, spraying, and weeding."
            sum_kn = "ಅನುಕೂಲಕರ ಒಣ ಹವೆ. ಕಳೆ ಕೀಳಲು ಮತ್ತು ಸಿಂಪರಣೆಗೆ ಸೂಕ್ತ ದಿನ."
        elif lookahead["has_washoff_hazard"]:
            sum_en = f"Clear today, but {next_exp:.1f} mm rain tomorrow! Withhold fertilizer top-dressing."
            sum_kn = f"ಇಂದು ಒಣಹವೆ, ಆದರೆ ನಾಳೆ {next_exp:.1f} ಮಿಮೀ ಮಳೆ! ರಸಗೊಬ್ಬರ ಹಾಕಬೇಡಿ."
        elif band == "light":
            sum_en = "Light showers expected. Field operations can safely proceed with care."
            sum_kn = "ಸಾಧಾರಣ ತುಂತುರು ಮಳೆ. ಎಚ್ಚರಿಕೆಯಿಂದ ಕೃಷಿ ಕೆಲಸ ಮುಂದುವರಿಸಿ."
        else:
            sum_en = f"Heavy rain risk ({cur_exp:.1f} mm). Postpone fertilizer and clear drainage."
            sum_kn = f"ಭಾರಿ ಮಳೆ ಸಂಭವ ({cur_exp:.1f} ಮಿಮೀ). ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ ಹಾಗೂ ಚರಂಡಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ."

        # Natural synoptic variation: rain cools Tmax and raises RH
        tmax_base = float(record.get("tmax_c", 31.5))
        tmin_base = float(record.get("tmin_c", 21.0))
        rh_base = float(record.get("rh_pct", 68.0))
        wind_base = float(record.get("wind_kph", 8.5))

        rain_cooling = min(3.5, cur_exp * 0.15)
        day_tmax = round(tmax_base - rain_cooling + (0.5 if cur_exp < 1.0 else 0.0), 1)
        day_tmin = round(tmin_base - rain_cooling * 0.4, 1)
        day_rh = min(100.0, round(rh_base + (15.0 if cur_exp >= 2.5 else 0.0), 1))
        day_wind = round(wind_base + (2.0 if cur_exp >= 10.0 else 0.0), 1)

        heat_stress = "NONE"
        if day_tmax >= 38.0:
            heat_stress = "SEVERE"
        elif day_tmax >= 35.0:
            heat_stress = "MODERATE"

        disease_flag = bool(day_rh >= 85.0 and 20.0 <= day_tmax <= 30.0)

        items.append({
            "date": cur_d.isoformat(),
            "day_offset": day_i,
            "day_label_en": label_en,
            "day_label_kn": label_kn,
            "expected_mm": float(cur_exp),
            "likely_min_mm": float(cur_lmin),
            "likely_max_mm": float(cur_lmax),
            "tmax_c": float(day_tmax),
            "tmin_c": float(day_tmin),
            "rh_pct": float(day_rh),
            "wind_kph": float(day_wind),
            "heat_stress_level": heat_stress,
            "disease_risk_flag": disease_flag,
            "rainfall_band": band,
            "spray_window": lookahead["spray_window"],
            "harvest_window": lookahead["harvest_window"],
            "irrigation_window": lookahead["irrigation_window"],
            "lookahead_warning_en": lookahead["warning_en"],
            "lookahead_warning_kn": lookahead["warning_kn"],
            "advisory_summary_en": sum_en,
            "advisory_summary_kn": sum_kn,
        })

    return items



