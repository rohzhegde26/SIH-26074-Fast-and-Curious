# Sprint 5 — Backend API (Days 11–15, parallel with Sprint 4/6)

Reference: `00-CANONICAL-SPEC.md` and `shared/schemas.py` (from Sprint 0).

## Goal
A FastAPI service exposing the four core endpoints, backed by real model
output from Sprints 2–4 by the end of this sprint — not mocked JSON.

## Tasks

### Early parallelization (start Day 11, don't wait for Sprint 2–4 to finish)
- Build all four endpoints against the `shared/schemas.py` contracts using
  **mocked** responses first, so the Frontend Lead (Sprint 6) can start
  integrating immediately.
- Swap each endpoint's mock for the real model call as soon as the
  corresponding sprint (2, 3, or 4) produces a working function — track this
  explicitly as a checklist so nothing gets forgotten and left mocked.

### Endpoints (`backend/app/routes/`)
- `POST /forecast/ice` — calls Sprint 2's exported ONNX model, returns SIC
  probability grid + calibrated uncertainty for a requested date/AOI.
- `POST /iceberg/track` — calls Sprint 3's `predict_drift`, returns
  trajectory + confidence cones + force breakdown.
- `POST /route/optimize` — calls Sprint 4's `optimize_route`, returns path +
  RIO heatmap + fuel/time estimate.
- `GET /risk/polaris` — returns a RIO grid for a date/AOI independent of
  routing (used for the standalone heatmap view).

### Inference wiring (`backend/app/inference/`)
- Load the Sprint 2 ONNX model via ONNX Runtime. Measure actual inference
  latency on your target hardware and **record the real number** — don't
  assume the "<100ms" target from earlier drafts was hit; report what you
  measured, and if it's slower, say so honestly in the pitch and explain the
  mitigation (caching, pre-computed tiles, etc.).
- Add a Redis cache keyed on (endpoint, date, AOI) so repeated queries during
  the demo don't re-run inference every time.

### Storage wiring (`backend/app/db/`)
- PostGIS for iceberg vectors (reuse Sprint 1/3's tables directly, don't
  duplicate schema).
- TimescaleDB hypertable `sic_timeseries(time, lat, lon, sic, rio)` — write
  forecast/RIO output here as it's computed, so repeated requests for
  already-computed dates hit the database instead of re-running the model.

### Ingestion trigger
- One endpoint or CLI command that re-triggers Sprint 1's ingestion pipeline
  for new NRT data, so the "we can operate on live data" claim in the pitch
  is actually backed by a real trigger, not just a diagram.

### Tests (`backend/tests/`)
- `pytest` + `httpx` tests for all four endpoints against fixture data (a
  small saved cube/route so tests don't require live network calls).
- CI runs these on every push.

### API docs
- Confirm FastAPI's auto-generated OpenAPI docs at `/docs` reflect the real
  schemas — check this manually, it's a fast, visible thing a judge might
  actually click on.

## Definition of Done
- [ ] All four endpoints return real model output end-to-end, checklist of
      "mock → real" swaps fully complete
- [ ] Measured (not assumed) inference latency documented in
      `docs/backend_performance.md`
- [ ] Redis caching demonstrably reduces repeated-query latency (show a
      before/after number)
- [ ] `pytest backend/tests/` passes in CI
- [ ] `/docs` (OpenAPI) renders correctly and matches the real schemas
