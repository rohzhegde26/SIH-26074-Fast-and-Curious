# Sprint 1--3 to Sprint 4 handoff contract

Provide one JSON array at `data/serving/mandya_forecasts.json` after each forecast run.
The API reads this file through `src/api/forecast_repository.py`.

```json
[
  {
    "lgd_code": "215504",
    "panchayat_name": "BANAVASI",
    "district": "MANDYA",
    "forecast_date": "2026-09-04",
    "timestamp_utc": "2026-09-03T18:00:00Z",
    "expected_mm": 14.2,
    "likely_min_mm": 8.5,
    "likely_max_mm": 22.1,
    "ragi_stage": "vegetative",
    "paddy_stage": "vegetative"
  }
]
```

Rules:

- `lgd_code` is a string and exactly matches `gpcode` in `frontend/mandya_simplified.topojson`.
- `expected_mm`, `likely_min_mm`, and `likely_max_mm` are non-negative millimetres.
- `likely_min_mm <= likely_max_mm`.
- `timestamp_utc` is ISO-8601 UTC and ends in `Z`.
- Crop stages are one of `sowing`, `vegetative`, `flowering`, or `harvest`.
- The output must be quantile-mapped, CQR-bounded, and zonally aggregated before export.
- Backend aggregation uses `mandya_full.geojson`; the browser uses only the simplified TopoJSON.

Current GIS validation: the supplied simplified TopoJSON contains 235 geometries, of
which 234 have a non-empty unique `gpcode`. The canonical count is: 234 active Gram Panchayats in the current Mandya pilot dataset (filtered from the 258 source cadastral listing).
