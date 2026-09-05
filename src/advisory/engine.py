from __future__ import annotations

from dataclasses import dataclass

from .rules import Advice, CROP_STAGE_ACTIONS, DRY, HEAVY, LIGHT, MODERATE, STAGE_PREFIX

VALID_CROPS = {"ragi", "paddy"}
VALID_STAGES = {"sowing", "vegetative", "flowering", "harvest"}


@dataclass(frozen=True)
class Advisory:
    stage: str
    action_en: str
    action_kn: str


def rainfall_band(expected_mm: float) -> Advice:
    if expected_mm < 2.5:
        return DRY
    if expected_mm <= 15.5:
        return LIGHT
    if expected_mm <= 64.4:
        return MODERATE
    return HEAVY


def build_advisory(crop: str, stage: str, expected_mm: float, likely_max_mm: float) -> Advisory:
    if crop not in VALID_CROPS:
        raise ValueError(f"Unsupported crop: {crop}")
    if stage not in VALID_STAGES:
        raise ValueError(f"Unsupported crop stage: {stage}")

    base = rainfall_band(expected_mm)
    prefix_en, prefix_kn = STAGE_PREFIX[stage]
    crop_stage = CROP_STAGE_ACTIONS[crop][stage]
    notes_en: list[str] = [crop_stage.en, base.en]
    notes_kn: list[str] = [crop_stage.kn, base.kn]

    if likely_max_mm > 10:
        notes_en.append("Do not apply fertilizer before this rain event.")
        notes_kn.append("ಈ ಮಳೆ ಸಂದರ್ಭಕ್ಕೂ ಮುನ್ನ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.")
    if likely_max_mm > 5:
        notes_en.append("Skip scheduled irrigation if rainfall occurs.")
        notes_kn.append("ಮಳೆ ಬಂದರೆ ನಿಗದಿತ ನೀರಾವರಿ ತಪ್ಪಿಸಿ.")
    if stage == "harvest" and likely_max_mm >= 15.5:
        notes_en.append("Protect harvested produce and drying areas from rain.")
        notes_kn.append("ಕೊಯ್ಲು ಮಾಡಿದ ಬೆಳೆ ಮತ್ತು ಒಣಗಿಸುವ ಸ್ಥಳಗಳನ್ನು ಮಳೆಯಿಂದ ರಕ್ಷಿಸಿ.")

    return Advisory(stage=stage.title(), action_en=prefix_en + " ".join(notes_en), action_kn=prefix_kn + " ".join(notes_kn))
