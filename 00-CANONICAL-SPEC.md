# SIH26059 — Canonical Technical Spec (single source of truth)

This file exists because earlier drafts of this plan disagreed with each other on
projection, grid size, channel count, and date ranges. Every sprint below
references **this file** for those numbers. If you need to change one of them,
change it here first, then update the sprint that's affected — never let a
sprint silently redefine a spec value.

## Problem statement (verbatim scope, SIH26059)
Org: Ministry of Earth Sciences (MoES) · Dept: NCPOR · Category: Software
> Build an AI/ML decision-support platform that forecasts Antarctic sea-ice
> concentration, predicts iceberg trajectories, and identifies safe,
> fuel-efficient navigation routes for research vessels, using satellite,
> oceanographic, and meteorological data.

Three deliverable modules, in priority order for the internal round:
1. Sea-ice concentration forecast (1–7 day)
2. Iceberg detection + trajectory forecast
3. Navigation risk (POLARIS) + route optimization

A system that only does (1) is not a complete answer to the PS. A system that
does all three badly beats a system that does one thing perfectly — judge for
completeness first, polish second.

## Coordinate reference system
- **Use EPSG:3412** — "NSIDC Sea Ice Polar Stereographic South" — for ALL
  analysis, reprojection, and grid math (ice cube, currents, winds, routing
  grid).
- **Do NOT use EPSG:4087.** That code is WGS84 World Equidistant Cylindrical
  (Plate Carrée) — a global cylindrical projection with severe area/direction
  distortion near the poles. It is unsuitable for polar grid analysis and was
  a factual error in an earlier draft of this plan. If you see EPSG:4087
  anywhere in ingestion or routing code, it's a bug — fix it.
- The CesiumJS **display** globe can still render in geographic lat/lon
  (EPSG:4326) for visualization — that's normal and fine. The CRS rule above
  applies to your analysis/routing grid, not to how Cesium draws the basemap.

## Area of interest (MVP)
- Corridor: **Cape Town → Bharati**, bounding box **30°E–90°E, 70°S–50°S**.
- Grid resolution: **6.25 km**.
- Grid dimensions: computed at ingestion time from the corridor bounding box
  projected into EPSG:3412 (don't hardcode a pixel count from memory — earlier
  drafts disagreed on 1264×1328 vs 1792×1792; compute and log the real value
  once your reprojection code runs, then paste the actual number here).
- Stretch goal (Finals, not internal round): full circumpolar south of 55°S.

## Temporal resolution
- **6-hourly** time steps: 00/06/12/18 UTC.
- Forward-fill ice fields between observations; linear-interpolate forcing
  fields (wind/current/SST).

## Dataset split (three-way, reconciled)
| Dataset | Window | Purpose |
|---|---|---|
| **Pretrain** | Widest available per-provider (self-supervised, no labels) | Learn Antarctic physics via masked reconstruction |
| **Train** | 2018–2023 | Supervised heads (SIC forecast, drift, weak-label detection) |
| **Test** | 2024–2025 (strict, unseen) | Time holdout. Also run sensor holdout and space holdout — see exact definitions below |

### OOD split definitions (concrete, not aspirational)
- **Sensor holdout:** condition the trained forecast head on OSI SAF-sourced
  SIC inputs, score its output against NSIDC CDR (or Bremen, whichever you
  did NOT train the target on) ground truth for the same days. Tests whether
  the model overfit to one sensor's quirks rather than the underlying ice
  edge.
- **Space holdout:** split the corridor itself in two —
  **train/validate on the western sub-corridor (30°E–60°E, Cape Town
  approach), test on the eastern sub-corridor (60°E–90°E, Bharati approach)**.
  Do **not** use "Atlantic sector (0–60°W) vs Ross Sea (~160°E–160°W)" as a
  holdout pair — that was an error carried over from an earlier draft.
  Neither sector falls inside this project's 30°E–90°E corridor at all, so a
  model "tested" on Ross Sea data was never actually evaluated against
  anything you ingested. Use the two sub-corridors above instead.
- **Time holdout:** the 2024–2025 test window, by construction.
- These three definitions must live in **one versioned manifest file**
  (`ingestion/splits_manifest.json`, produced in Sprint 1 — see that sprint's
  tasks), not be redecided ad hoc when Sprint 2 gets to evaluation.

## Feature cube (MVP = 10 channels)
1. SIC (14-day history)
2. u10 (ERA5 10 m wind, eastward)
3. v10 (ERA5 10 m wind, northward)
4. uo (ocean current, eastward — OSCAR/Copernicus PHY)
5. vo (ocean current, northward)
6. SST
7. Bathymetry (GEBCO, static)
8. SIC anomaly vs climatology
9. Land mask (static)
10. Distance to coast (static)

A 20-channel version (adding Bremen AMSR2, WAV Stokes drift, GFS, extra SIC
products for the sensor-swap trick) is a **Finals-phase stretch goal**, not an
internal-round requirement. Don't let scope creep here eat sprint time.

## Labeling philosophy (non-negotiable — this is the project's biggest risk if ignored)
- **Never hand-label SAR imagery from scratch for the internal round.**
  Bounding-box labeling 1000 Sentinel-1 tiles takes weeks, not the 2 days
  earlier drafts allocated to it.
- Ice concentration: the SIC product itself is the label (input == target for
  self-supervision; different-day SIC is the label for forecasting).
- Iceberg drift: the BYU/NIC track itself is the label.
- Iceberg detection: use a **pretrained** detector (AWI/ESA SAR iceberg
  detector, or a documented open substitute if that access falls through) +
  USNIC/BYU points as weak labels. From-scratch YOLO fine-tuning on
  hand-labeled tiles is explicitly a **Finals-phase** task, not internal-round.

## Explicitly descoped for the internal round (documented, not silently dropped)
These were part of earlier drafts' full scope. Cutting them for the internal
round is a deliberate choice to ship a working three-module system rather
than a partially-working four-or-five-module one — but the choice must stay
visible in the roadmap slide, not disappear quietly:
- **RL/simulator-based routing** (Goldilocks' 4th dataset: state/action/
  reward environment for route learning). Replaced for the internal round by
  the deterministic A* router in Sprint 4, which directly satisfies the PS's
  "identify safe routes" requirement with far less implementation risk than
  standing up a converging RL loop in the same window as two other ML
  models. Full RL-based routing is a named Finals-phase item.
- **Full NSGA-II multi-objective routing** (Sprint 4 uses a weighted-sum
  approximation instead).
- **From-scratch YOLO fine-tuning on hand-labeled SAR tiles** (Sprint 3 uses
  a pretrained detector instead).
- **Full IDRIFTNET (Rotate Block + Gabor-Spectral network)** (Sprint 3 builds
  the physics + 2-layer LSTM "lite" version instead).
- **Circumpolar coverage** (MVP is the Cape Town–Bharati corridor only).
- **WAV/GFS/INCOIS/HYCOM as extra-variance channels** (10-channel MVP cube
  only; add via `ingestion/sources/wav.py`, `gfs.py`, `incois_hycom.py`
  later, following the same pattern as the existing source scripts).

## Datasets (fully free, no manual labeling required)
**Use fully:** OSI SAF 401-d/408-a (via CMEMS), Bremen AMSR2 6.25 km,
Copernicus Marine PHY (uo/vo/SST/SSH), ERA5 (u10/v10), GEBCO bathymetry,
BYU/NIC iceberg tracks, USNIC SIGRID-3 (weak labels).

**Use for augmentation/validation only:** NSIDC CDR v6, NSIDC NRT, GFS,
WAV 001_027 — these are Finals-phase additions, wire the ingestion code to
support them but don't block the internal-round MVP on getting all of them.

**Explicitly out of scope for internal round:** raw Sentinel-1 used for
training a detector from scratch, ICESat-2, MODIS optical — heavy compute/
labeling for marginal MVP benefit.

## Tech stack (pinned, with one correction from an earlier draft)
- Env: conda, Python 3.10
- Ingestion: xarray, rioxarray, pyproj, cfgrib, copernicusmarine, cdsapi,
  sentinelsat
- Storage: PostgreSQL + PostGIS + TimescaleDB, Zarr, Redis
- Training: PyTorch + Lightning, ONNX export
- Inference: ONNX Runtime, FastAPI
- Frontend: Next.js, CesiumJS (via Resium), deck.gl, MapLibre GL JS, Recharts
- Offline/PWA: **Dexie.js** for browser-side offline storage — an earlier
  draft specified "Isar DB," which is a Flutter/Dart package and cannot run in
  a Next.js/JS frontend. Dexie is the correct web equivalent for the same
  offline-cache job.
- DevOps: Docker Compose, GitHub Actions CI

## Honesty rules for the pitch
- Any economic figure (charter savings, fuel %, Rs crore numbers) must be
  labeled **"modeled estimate"** in slides and speech — these are back-of-
  envelope calculations, not measured results, and claiming otherwise is a
  credibility risk under judge questioning.
- Any latency/performance number quoted in the demo (e.g. inference time) must
  be a number you actually measured in Sprint 5, not a target you assumed.
- If a data source listed above turns out to be inaccessible during the
  hackathon (rate limits, account approval delay, etc.), document the
  substitute you used — don't silently claim the original source in the
  slides.
