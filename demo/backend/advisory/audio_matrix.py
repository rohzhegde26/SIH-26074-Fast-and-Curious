"""
src/advisory/audio_matrix.py
Canonical matrix mapping (crop, growth_stage, rain_risk) to precached offline audio assets.
Ensures 100% deterministic resolution of spoken Mandya Kannada advisories without network calls.
"""
import os
from typing import List, Tuple

CROPS: List[str] = ["ragi", "paddy"]
STAGES: List[str] = ["sowing", "vegetative", "flowering", "harvest"]
RISKS: List[str] = ["dry", "light_rain", "heavy_rain"]

def resolve_audio_file(crop: str, stage: str, risk: str) -> str:
    """
    Resolve any (crop, stage, risk) combination to a canonical audio file in frontend/audio/.
    """
    c = crop.lower().strip()
    s = stage.lower().strip()
    r = risk.lower().strip()

    if r == "heavy_rain":
        if s == "harvest":
            return f"{c}_harvest_rot_kn.mp3"
        elif s == "vegetative":
            return f"{c}_veg_rain_kn.mp3"
        else:
            return "heavy_cloudburst_kn.mp3"
    elif r == "light_rain":
        if s == "harvest":
            return f"{c}_harvest_rot_kn.mp3"
        elif s == "sowing" and c == "ragi":
            return "ragi_sow_dry_kn.mp3"
        else:
            return "dry_window_safe_kn.mp3"
    else:  # dry
        if s == "sowing" and c == "ragi":
            return "ragi_sow_dry_kn.mp3"
        else:
            return "dry_window_safe_kn.mp3"

def get_all_matrix_combinations() -> List[Tuple[str, str, str]]:
    """Return all 24 (crop, stage, risk) combinations."""
    combos = []
    for c in CROPS:
        for s in STAGES:
            for r in RISKS:
                combos.append((c, s, r))
    return combos
