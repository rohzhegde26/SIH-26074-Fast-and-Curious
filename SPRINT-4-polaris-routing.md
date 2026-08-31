# Sprint 4 — POLARIS Risk & Route Optimization (Days 11–13)

Reference: `00-CANONICAL-SPEC.md`. Depends on Sprint 2's SIC forecast and
Sprint 3's iceberg positions/drift.

## Goal
A working risk index (RIO) computed from real forecast output, and a route
optimizer that actually avoids high-risk cells — this is the module that
answers the "navigation routes" third of the PS, and neither prior draft plan
had it fully working end-to-end, so treat it as first-class, not an
afterthought.

## Tasks

### Ice-type derivation (`routing/ice_type.py`)
- Derive ice type (nilas, grey, grey-white, first-year thin/medium/thick,
  second-year, multi-year, iceberg) from SIC + a thickness proxy.
- Be explicit and honest about what your thickness proxy actually is for the
  internal round — if you don't have SMOS/CryoSat-2 access working, use a
  documented reduced proxy (e.g. SIC + a climatological thickness lookup by
  ice type and season) and say so in code comments and the pitch. Don't claim
  a data source you didn't actually integrate.

### POLARIS RIO (`routing/polaris.py`)
```python
def compute_rio(ice_type_fractions: dict, vessel_ice_class: str) -> float:
    """
    RIO = sum(Ci * RIVi) where Ci is concentration in tenths of ice type i,
    RIVi is the Risk Index Value for the given vessel ice class, from the
    official IMO POLARIS RIV table.
    RIO >= 0: normal operation
    -10 <= RIO < 0: elevated risk, reduce speed
    RIO < -10: special consideration / avoid
    """
```
- Use the real RIV table values for the vessel class NCPOR actually charters
  (1A Super or PC5-equivalent) — don't invent placeholder RIV numbers, look
  up the actual IMO table.
- Unit test: a cell of 100% open water should give a high positive RIO; a
  cell of heavy multi-year ice should give a strongly negative RIO.

### Safe speed & fuel model (`routing/safe_speed.py`)
- `V_safe = V_open * (1 - exp(RIO/k))`, k≈15 (tune if your unit tests show
  unreasonable speeds at the boundary RIO values).
- `Fuel = a*V_safe**3 + b*IceResistance(h_ice, Ci)`,
  `IceResistance = k3 * h_ice**1.5 * Ci`.
- Use fuel-consumption figures from the NCPOR tender document referenced in
  `docs/references.md` (typical Ice Class 1A: ~20–30 tons/day, ~12 knots open
  water, ~3 knots in 60% ice) to fit/sanity-check `a`, `b`, `k3` — document
  how you derived these constants, since a judge may ask.

### Route optimization (`routing/hfs_astar.py`)
- Grid: 6.25 km, 8-connected, over the corridor from `shared/grid.py`.
- Cost function: `g(n) = w1*distance + w2*fuel(RIO) + w3*iceberg_proximity +
  w4*turn_penalty`, `iceberg_proximity = exp(-dist_to_nearest_berg / 10km)`
  using Sprint 3's tracked/predicted iceberg positions.
- Heuristic: great-circle distance / V_open.
- Constraints: land avoidance (from GEBCO/land mask), RIO threshold > -10,
  max SIC 70% for a PC5-class vessel.
- Implement plain A* first and get it working correctly — only invest in the
  literature's HFS-A* heuristic refinement if time remains after the basic
  version is demo-ready. A working plain A* beats a broken "advanced" one.
- This A* router is the internal-round replacement for the Goldilocks plan's
  RL/simulator-based routing (state/action/reward environment) — see the
  canonical spec's "Explicitly descoped" section. Mention this trade-off
  explicitly in the pitch roadmap slide as a deliberate scope choice, not an
  oversight.
- Multi-objective: implement a simple weighted-sum sweep across
  (fuel-weight, risk-weight, time-weight) to approximate a Pareto front for
  the demo slider. Full NSGA-II is a documented Finals-phase item, not
  internal-round scope — don't quietly drop it, write it into the roadmap
  slide instead.

### Validation (`routing/validate_route.py`)
- Generate one full demo route (Cape Town → Bharati, or Cape Town → Maitri)
  using real Sprint 2 forecast + Sprint 3 iceberg data for a specific date.
- Produce a before/after comparison: show the great-circle "naive" route vs.
  your optimized route, with the RIO improvement and estimated fuel/time
  difference — this is the "80–100s" beat of your demo script (see Sprint 7).

## Definition of Done
- [ ] `compute_rio` unit tests pass with sane behavior at both extremes
- [ ] Safe-speed/fuel constants documented with their derivation, not just
      hardcoded numbers
- [ ] `optimize_route(start, end, date)` runs end-to-end on real Sprint 2/3
      outputs and returns a path that respects the RIO/SIC constraints
- [ ] Before/after demo route generated and saved (path + RIO heatmap image)
      for use in Sprint 7's pitch deck and demo script
- [ ] NSGA-II / full multi-objective explicitly documented as Finals-phase,
      not silently dropped
