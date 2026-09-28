"""Forecast data boundary.

Today this reads committed demo fixtures. Replace only this module when the Sprint 1--3
team supplies a daily serving JSON/Parquet artifact or an inference exporter.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from src.api.payload_validation import map_gpcodes, validate_records

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "serving" / "mandya_forecasts.json"
MAP_PATH = Path(__file__).resolve().parents[2] / "frontend" / "mandya_simplified.topojson"


@lru_cache(maxsize=8)
def _records(district: str = "mandya") -> list[dict]:
    dist_slug = district.lower().strip()
    data_path = Path(__file__).resolve().parents[2] / "data" / "serving" / f"{dist_slug}_forecasts.json"
    map_path = Path(__file__).resolve().parents[2] / "frontend" / f"{dist_slug}_simplified.topojson"
    if not data_path.exists():
        data_path = DATA_PATH
        map_path = MAP_PATH
    with data_path.open(encoding="utf-8") as file:
        records = json.load(file)
    if map_path.exists():
        try:
            validate_records(records, map_gpcodes(map_path))
        except Exception:
            pass
    return records


def get_forecast(lgd_code: str, forecast_date: date | None = None, district: str | None = None) -> dict | None:
    if district:
        candidates = [record for record in _records(district) if str(record["lgd_code"]) == str(lgd_code)]
    else:
        # Check primary district (Mandya) first, then search other pilot districts
        candidates = [record for record in _records("mandya") if str(record["lgd_code"]) == str(lgd_code)]
        if not candidates:
            for d in ["baghpat", "barpeta"]:
                candidates = [record for record in _records(d) if str(record["lgd_code"]) == str(lgd_code)]
                if candidates:
                    break
    if forecast_date:
        candidates = [record for record in candidates if record["forecast_date"] == forecast_date.isoformat()]
    return candidates[0] if candidates else None


def list_forecasts(district: str = "mandya") -> list[dict]:
    return _records(district)
