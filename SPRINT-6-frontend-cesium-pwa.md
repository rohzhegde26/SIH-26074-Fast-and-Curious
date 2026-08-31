# Sprint 6 — Frontend: Scientist View + Captain PWA (Days 11–16, parallel)

Reference: `00-CANONICAL-SPEC.md` and `shared/schemas.py`.

## Goal
A dual-view Next.js frontend — a rich 3D scientist view and a lightweight
offline-capable captain view — both wired to real backend data by sprint end.

## Tasks

### Early parallelization
- Start against Sprint 5's mocked endpoints immediately (Day 11), don't wait
  for real model output — swap to real data as Sprint 5 completes its
  mock→real checklist.

### Scientist view (`frontend/app/scientist/`)
- CesiumJS via Resium: 3D globe. Note — Cesium's base globe display uses
  geographic lat/lon (EPSG:4326) regardless of what CRS your analysis grid
  uses; that's normal, not a contradiction of the canonical spec's CRS rule.
- Daily SIC animation layer (from `/forecast/ice`).
- Time-dynamic iceberg entities with trajectory + 50%/90% confidence
  ellipses (from `/iceberg/track`).
- deck.gl density heatmap for iceberg clustering.
- RIO heatmap overlay with traffic-light coloring (green ≥0, yellow -10 to 0,
  red < -10), from `/risk/polaris`.
- Click-iceberg panel: force breakdown bar chart (Recharts) using the
  `force_breakdown` output from Sprint 3.

### Captain view (`frontend/app/captain/`)
- MapLibre GL JS, simplified 2D, large touch targets.
- Route + ETA + RIO color display, from `/route/optimize`.
- Offline-first PWA via `next-pwa`, pre-caching corridor tiles.
- **Use Dexie.js** for browser-side offline storage/sync queue — the
  canonical spec corrects an earlier draft that specified "Isar DB," which is
  a Flutter/Dart package and cannot run in this Next.js stack. Dexie is the
  correct web equivalent for the same job.
- Simple JWT auth stub — don't over-invest here for an internal round demo.

### Low-bandwidth validation
- Test both views under Chrome DevTools network throttling (simulate a slow/
  intermittent connection). If the "works at sea on low bandwidth" claim is
  going in the pitch, verify it actually holds up under throttling before you
  say it — this is an easy thing for a technically-minded judge to ask you to
  demonstrate live.

### Integration
- Remove all mocked/hardcoded JSON from the frontend once Sprint 5's real
  endpoints are live — do a final grep for mock fixtures before Sprint 6 is
  considered done.

## Definition of Done
- [ ] Scientist view renders real SIC animation, iceberg trajectories with
      confidence cones, RIO heatmap, and the force-breakdown panel
- [ ] Captain view renders a real optimized route with RIO coloring and ETA
- [ ] PWA installs and shows pre-cached corridor tiles with network disabled
      (verify this literally, don't just assume next-pwa "should" work)
- [ ] Both views tested under throttled network and the result is honestly
      reflected in what you claim in the pitch
- [ ] No hardcoded/mock JSON remains in the frontend codebase
