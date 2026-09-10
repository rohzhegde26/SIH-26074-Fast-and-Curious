"""
src/advisory/economics.py

Phenology-Weighted Economic Cost-of-Error Service.
Loads empirical INR loss benchmarks cited from UAS Bangalore agricultural extension data.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

STATIC_DIR = Path(__file__).resolve().parents[2] / "data" / "static"
CROP_ECONOMICS_PATH = STATIC_DIR / "crop_economics.json"

# Stage mapping fallbacks for normalized phenological categories
STAGE_CATEGORY_MAP: Dict[str, str] = {
    "harvest": "harvest",
    "ripening": "harvest",
    "vegetative": "vegetative",
    "tillering": "vegetative",
    "grand_growth": "vegetative",
    "flowering": "flowering",
    "sowing": "sowing",
    "germination": "sowing",
}


@lru_cache(maxsize=1)
def load_crop_economics() -> List[Dict[str, Any]]:
    """Loads and caches the empirical crop economic loss parameters."""
    if CROP_ECONOMICS_PATH.exists():
        with open(CROP_ECONOMICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


@lru_cache(maxsize=32)
def get_crop_risk_inr(crop: str, stage: str) -> int:
    """
    Looks up empirical risk estimate in INR per acre for a given crop and stage.
    Falls back to stage category default if crop-specific match is not found.
    """
    norm_crop = crop.lower()
    norm_stage = stage.lower()

    data = load_crop_economics()

    # Exact crop + stage match
    for entry in data:
        if entry.get("crop", "").lower() == norm_crop and entry.get("stage", "").lower() == norm_stage:
            return int(entry.get("risk_inr_per_acre", 0))

    # Fallback to category match (e.g. ripening -> harvest)
    cat_stage = STAGE_CATEGORY_MAP.get(norm_stage, norm_stage)
    for entry in data:
        if entry.get("crop", "").lower() == norm_crop and entry.get("stage", "").lower() == cat_stage:
            return int(entry.get("risk_inr_per_acre", 0))

    # Global stage category fallback across any crop
    for entry in data:
        if entry.get("stage", "").lower() == cat_stage:
            return int(entry.get("risk_inr_per_acre", 0))

    return 0


def get_crop_economics_provenance(crop: str, stage: str) -> Optional[Dict[str, Any]]:
    """Returns provenance metadata (source citation and agronomic assumption) for an advisory."""
    norm_crop = crop.lower()
    norm_stage = stage.lower()
    data = load_crop_economics()

    for entry in data:
        if entry.get("crop", "").lower() == norm_crop and entry.get("stage", "").lower() == norm_stage:
            return {
                "source": entry.get("source"),
                "assumption": entry.get("assumption"),
                "risk_inr_per_acre": entry.get("risk_inr_per_acre"),
            }

    cat_stage = STAGE_CATEGORY_MAP.get(norm_stage, norm_stage)
    for entry in data:
        if entry.get("stage", "").lower() == cat_stage:
            return {
                "source": entry.get("source"),
                "assumption": entry.get("assumption"),
                "risk_inr_per_acre": entry.get("risk_inr_per_acre"),
            }

    return None
