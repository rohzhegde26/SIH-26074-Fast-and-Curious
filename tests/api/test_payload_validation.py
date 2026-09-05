import pytest

from src.api.payload_validation import validate_records


def test_rejects_code_absent_from_map():
    record = {"lgd_code": "missing", "panchayat_name": "x", "district": "MANDYA", "forecast_date": "2026-09-04", "timestamp_utc": "2026-09-03T18:00:00Z", "expected_mm": 1, "likely_min_mm": 0, "likely_max_mm": 2}
    with pytest.raises(ValueError, match="absent from the simplified map"):
        validate_records([record], {"215504"})
