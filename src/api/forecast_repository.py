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


@lru_cache(maxsize=1)
def _records() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as file:
        records = json.load(file)
    validate_records(records, map_gpcodes(MAP_PATH))
    return records


def get_forecast(lgd_code: str, forecast_date: date | None = None) -> dict | None:
    candidates = [record for record in _records() if str(record["lgd_code"]) == str(lgd_code)]
    if forecast_date:
        candidates = [record for record in candidates if record["forecast_date"] == forecast_date.isoformat()]
    return candidates[0] if candidates else None


def list_forecasts() -> list[dict]:
    return _records()
