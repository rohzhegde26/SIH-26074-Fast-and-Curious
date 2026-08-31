# Sprint 1 — Data Pipeline & Ingestion (Days 2–5)

Reference: `00-CANONICAL-SPEC.md`. Depends on Sprint 0's provider credentials
and `shared/grid.py`.

## Goal
A queryable, correctly-reprojected 10-channel Zarr cube for the Cape
Town–Bharati corridor, plus a PostGIS table of iceberg tracks and weak-label
detections — re-runnable with one command, and validated against a known
reference point before anyone starts training on it.

## Tasks

### Reprojection module (`ingestion/reproject.py`)
- Implement `reproject_to_3412(dataset, target_res_km=6.25) -> xr.Dataset`
  using `rioxarray` + `pyproj`, importing the CRS/bbox/resolution constants
  from `shared/grid.py` — do not hardcode EPSG:3412 or the bbox anywhere
  else.
- Unit test: reproject a synthetic point at Bharati station's known
  coordinates (69°24'S, 76°11'E) and assert it lands inside the corridor grid
  at a sane pixel index. Do the same for Maitri (70°45'S, 11°38'E) — Maitri
  should land near the western edge of the 30°E–90°E corridor or just outside
  it; if it's wildly off, your reprojection has a bug, not your corridor
  choice.
- Once this test passes, compute the actual grid dimensions (rows × cols) for
  the corridor at 6.25 km and paste that number into `00-CANONICAL-SPEC.md`,
  replacing the placeholder note there.

### Per-source ingestion scripts (`ingestion/sources/`)
One script per source, each producing a standardized `xr.Dataset` with
consistent variable names, ready for the reprojection step:
- `osi_saf.py` — pull OSI SAF 401-d + 408-a via CMEMS, output SIC + ice type.
- `bremen_amsr2.py` — pull Bremen 6.25 km SIC.
- `copernicus_phy.py` — pull uo, vo, SST, SSH.
- `era5.py` — pull u10, v10 via cdsapi.
- `gebco.py` — one-time static bathymetry load.
- `byu_nic.py` — parse the BYU/NIC ASCII iceberg archive into a clean
  `(iceberg_id, timestamp, lat, lon, area)` table, write to PostGIS.
- `usnic_sigrid3.py` — parse SIGRID-3 ice charts into weak-label ice-type
  polygons for the corridor.

Each script should be independently re-runnable (idempotent — re-running for
a date range you already have shouldn't duplicate or corrupt data).

### Feature cube assembly (`ingestion/build_cube.py`)
- Assemble the 10 MVP channels listed in the canonical spec, at 6-hourly
  steps, forward-filling ice fields and linear-interpolating forcing fields.
- Compute SIC anomaly vs. climatology (build a simple per-pixel, per-day-of-
  year climatology from whatever historical SIC data you pulled, don't
  overengineer this for MVP).
- Compute static land mask and distance-to-coast from GEBCO.
- Write the assembled cube to Zarr with chunking sized for fast per-day,
  per-region xarray access (chunk along time and a moderate spatial tile,
  not one giant chunk).

### Sensor-swap augmentation utility (`ingestion/sensor_swap.py`)
- Implement the Goldilocks variance trick: given a date, randomly select
  which SIC product (OSI SAF / Bremen / NSIDC CDR if available) populates the
  "SIC" channel for that training sample, and log the choice per-sample so
  training runs are reproducible. This is used by Sprint 2's pretrain stage.

### Iceberg weak-label detection (`ingestion/detect_weak_labels.py`)
- Run a pretrained SAR iceberg detector (AWI/ESA, or a documented open
  substitute) over the 2 Sentinel-1 scenes covering A23A/A68 pulled in
  Sprint 0.
- Store detections (bounding polygon + confidence) in PostGIS.
- Cross-reference detections against BYU/USNIC known positions by nearest-
  neighbor distance — flag matches vs. new detections. This table feeds
  Sprint 3's detection/tracking pipeline directly; don't re-derive it there.

### Dataset split manifest (`ingestion/splits.py`)
- Generate `ingestion/splits_manifest.json`, the single versioned source of
  truth for every train/val/test partition Sprint 2 will use — do this here,
  not ad hoc during model evaluation:
  - `pretrain_window`: widest available range per provider
  - `train_window`: 2018-01-01 to 2023-12-31
  - `test_window`: 2024-01-01 to 2025-12-31 (strict, unseen)
  - `sensor_holdout`: `{"train_source": "OSI_SAF", "test_source": "NSIDC_CDR"}`
    (or Bremen — whichever you did NOT use as the training target)
  - `space_holdout`: `{"train_region": [30, 60, -70, -50], "test_region":
    [60, 90, -70, -50]}` (lon_min, lon_max, lat_min, lat_max — western vs
    eastern sub-corridor, both inside the actual AOI; see canonical spec)
- Write a leakage test (`ingestion/tests/test_splits.py`) that programmatically
  asserts: no date appears in both `train_window` and `test_window`, and no
  grid cell assigned to `space_holdout.test_region` overlaps
  `space_holdout.train_region`. This is cheap to write and catches the most
  common way an OOD claim quietly becomes false — an accidental overlap
  nobody noticed until a judge asks how it was validated.
- Sprint 2 must load partitions from this manifest, not redefine them.

### Validation notebook (`ingestion/validate.ipynb`)
- Plot the corridor extent on a map, confirm Maitri/Bharati fall where
  expected.
- Plot % non-null coverage per channel per day — flag any channel with
  suspiciously sparse coverage before training starts on it.
- Plot one full day's assembled 10-channel cube as a sanity-check figure —
  this figure is also useful later for the pitch deck's "data fusion" slide.

### Tests (`ingestion/tests/`)
- Schema test: assembled cube has exactly the 10 named channels with correct
  units/attrs.
- CRS test: cube's CRS is EPSG:3412, not anything else.
- Coverage test: fail if any channel's non-null coverage drops below an
  agreed threshold (pick something reasonable, e.g. 90%, and document why).

## Definition of Done
- [ ] `reproject_to_3412` passes the Bharati/Maitri sanity-check unit tests
- [ ] Actual grid dimensions computed and recorded in the canonical spec
- [ ] Zarr cube exists, loads via `xarray.open_zarr`, has all 10 channels
- [ ] `splits_manifest.json` exists with pretrain/train/test windows plus
      sensor-holdout and space-holdout definitions, and the leakage test in
      `ingestion/tests/test_splits.py` passes
- [ ] PostGIS has iceberg tracks (BYU/NIC) and weak-label detections
      (Sentinel-1 via pretrained detector)
- [ ] Validation notebook committed with the coverage and corridor-extent
      plots
- [ ] `pytest ingestion/tests/` passes in CI
- [ ] The whole pipeline is triggerable with one script/command, not a
      sequence of manual steps someone has to remember
