"""Validation for the Sprint 1--3 daily serving artifact."""
from __future__ import annotations

import json
from pathlib import Path


def map_gpcodes(topojson_path: Path) -> set[str]:
    topology = json.loads(topojson_path.read_text(encoding="utf-8"))
    collections = [value for value in topology.get("objects", {}).values() if value.get("type") == "GeometryCollection"]
    if not collections:
        raise ValueError("TopoJSON has no GeometryCollection")
    return {
        str(geometry.get("properties", {}).get("gpcode", "")).strip()
        for collection in collections
        for geometry in collection.get("geometries", [])
    } - {""}


def validate_records(records: list[dict], valid_gpcodes: set[str]) -> None:
    if not records:
        raise ValueError("Forecast payload is empty")
    seen: set[tuple[str, str]] = set()
    for record in records:
        required = {"lgd_code", "panchayat_name", "district", "forecast_date", "timestamp_utc", "expected_mm", "likely_min_mm", "likely_max_mm"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Forecast record missing fields: {sorted(missing)}")
        code = str(record["lgd_code"])
        if code not in valid_gpcodes:
            raise ValueError(f"LGD/GPCODE {code} is absent from the simplified map")
        lower, expected, upper = (float(record["likely_min_mm"]), float(record["expected_mm"]), float(record["likely_max_mm"]))
        if min(lower, expected, upper) < 0 or lower > upper:
            raise ValueError(f"Invalid rainfall bounds for {code}")
        key = (code, str(record["forecast_date"]))
        if key in seen:
            raise ValueError(f"Duplicate forecast for {code} on {record['forecast_date']}")
        seen.add(key)
