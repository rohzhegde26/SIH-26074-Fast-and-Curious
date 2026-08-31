# Sprint 3 — Iceberg Detection & Drift Forecasting (Days 6–10)

Reference: `00-CANONICAL-SPEC.md`. Depends on Sprint 1's PostGIS iceberg
tracks and weak-label detections. Runs in parallel with Sprint 2.

## Goal
A working physics-prior + residual-learning drift forecaster (IDRIFTNET-lite,
grounded in the real published IDRIFTNET architecture), plus a detection/
tracking pipeline that uses the Sprint 1 pretrained-detector output — no
from-scratch labeling in this sprint.

## Tasks

### Physics prior (`models/iceberg/wagner_drift.py`)
```python
def wagner_drift(area, u10, v10, uo, vo, rho_air=1.225, rho_water=1025,
                  Ca=1.0, Cw=0.75, freeboard=45, draft=275):
    """
    Analytical steady-state drift velocity per Wagner (2017):
    V_ice = V_water + gamma * (V_air - V_water)
    gamma = sqrt((rho_air*Ca*A_sail) / (rho_water*Cw*A_keel))
    A_sail/A_keel derived from freeboard/draft and berg area.
    Returns (u_ice, v_ice).
    """
```
- Implement exactly this, with the scale-aware gamma — small bergs should
  come out near the classic ~2% wind rule, large tabular bergs (like A23A)
  should come out current-dominated (gamma much smaller). Write a unit test
  asserting both regimes come out in the right direction.

### Melt/decay coupling (`models/iceberg/melt_decay.py`)
- Wave erosion and basal melt terms per the canonical formulas, updating
  `area(t+Δt)` and therefore `gamma(t+Δt)` — this makes the physics prior
  non-linear over a multi-day forecast instead of static.

### IDRIFTNET-lite residual model (`models/iceberg/idriftnet_lite.py`)
- Grounded in the real IDRIFTNET paper (arXiv 2507.00036): sliding window of
  5+ timesteps, features = (lat, lon, area, u10, v10, uo, vo), predicting
  next-timestep lat/lon. For the internal round, implement the **residual
  physics + 2-layer LSTM** version (defer the full Rotate Block + Gabor-
  Spectral network to the Finals-phase roadmap — say so explicitly in the
  code comments and the pitch, don't claim the full architecture if you only
  built the lite version).
- Output: 5-day trajectory + Gaussian-mixture uncertainty → rendered as
  50%/90% confidence ellipses.
- Train on A23A history (2014–2024), validate on the most recent available
  year, per canonical spec.

### Evaluation (`models/iceberg/evaluate_drift.py`)
- Compute FDE (Final Displacement Error) and ADE (Average Displacement Error)
  for IDRIFTNET-lite vs. the pure physics prior alone (`wagner_drift` with no
  residual correction) — this comparison is your novelty proof, so make sure
  it actually runs on real held-out data, not a cherry-picked window.
- Save the numbers to `docs/iceberg_results.md`. Do not copy the published
  IDRIFTNET paper's numbers into your own results table — report what your
  implementation actually measured, even if it's a less dramatic
  improvement than the paper's.

### Detection & tracking (`models/iceberg/detect_and_track.py`)
- Load the Sprint 1 weak-label detections (from the pretrained AWI/ESA
  detector) from PostGIS.
- Link detections across time — use ByteTrack if you can integrate it
  cleanly in the timeline; otherwise implement a simple IoU/nearest-neighbor
  tracker as a documented fallback (a working simple tracker beats a broken
  advanced one for a live demo).
- Fuse tracked detections with BYU/NIC positions via a Kalman filter for
  smoothed position estimates.

### Explainability (`models/iceberg/force_breakdown.py`)
- `force_breakdown(iceberg_id, timestamp) -> {"wind_pct": ..., "current_pct":
  ..., "coriolis_pct": ...}` — derive this from the relative magnitude of
  each force term in the physics prior at that timestep. This feeds the
  "click iceberg → why is it moving this way" demo feature directly.

### Packaging
- `predict_drift(iceberg_id, horizon_days=5) -> {"trajectory": [...],
  "confidence_cone_50": [...], "confidence_cone_90": [...]}` as a clean
  importable function — this is exactly what Sprint 5's backend will call,
  so agree on this exact signature with the Backend Lead now.

## Definition of Done
- [ ] `wagner_drift` unit test confirms small-berg vs. large-berg gamma
      behaves in the physically correct direction
- [ ] IDRIFTNET-lite trained and produces FDE/ADE numbers that beat the pure
      physics baseline on held-out data (real numbers, not the paper's)
- [ ] Detection + tracking pipeline produces at least one real tracked
      trajectory from the Sentinel-1 scenes pulled in Sprint 1
- [ ] `force_breakdown` returns sane percentages summing to ~100%
- [ ] `predict_drift` function signature agreed with Backend Lead and
      documented in `shared/schemas.py`
