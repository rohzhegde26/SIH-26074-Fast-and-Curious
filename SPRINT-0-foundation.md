# Sprint 0 — Foundation, Environment, Scaffolding (Days 0–1)

Reference: `00-CANONICAL-SPEC.md`

## Goal
Everyone can run the (empty) stack locally, every data-provider credential is
verified with a real successful pull, and the repo has the shared scaffolding
every later sprint depends on — so no sprint wastes its first day on setup.

## Tasks

### Repo & CI (Data/DevOps Lead)
- Create the GitHub repo with this structure:
  ```
  /canonical-spec/          # copy of 00-CANONICAL-SPEC.md, treated as law
  /ingestion/                # Sprint 1
  /models/sea_ice/           # Sprint 2
  /models/iceberg/           # Sprint 3
  /routing/                  # Sprint 4
  /backend/                  # Sprint 5 (FastAPI app)
  /frontend/                 # Sprint 6 (Next.js app)
  /infra/docker-compose.yml
  /infra/.github/workflows/ci.yml
  /docs/                     # novelty matrix, references, pitch assets
  README.md
  environment.yml
  ```
- Set up branch strategy: `main` protected, `dev` integration branch,
  feature branches `feature/sea-ice`, `feature/iceberg`, `feature/backend`,
  `feature/frontend`, `feature/routing`.
- GitHub Actions CI: lint (ruff/flake8) + a placeholder test job that always
  passes for now — wire real tests in as each sprint adds them.
- Add `.gitignore` covering data caches, model checkpoints, `.env` files —
  credentials must never be committed.
- Create a GitHub Project board and add one issue per task listed in every
  sprint file in this repo (copy each task line as an issue title).

### Environment
- `environment.yml` — conda env `polar-dss`, Python 3.10, with the full
  package set from the canonical spec's tech stack section.
- Verify every team member can `conda env create -f environment.yml` and
  activate it without errors.

### Data provider accounts & connectivity test (Data/DevOps Lead)
For each provider, create an account if needed, then run one small test pull
and confirm non-empty, sane data before moving on:
- Copernicus Marine (CMEMS) — test pull of OSI SAF SIC for one day, one small
  bounding box.
- Copernicus Climate Data Store (CDS) — test `cdsapi` pull of one day of ERA5
  u10/v10.
- Copernicus Dataspace (Sentinel-1) — test `sentinelsat` query for one scene
  over the Bharati approach.
- University of Bremen AMSR2 FTP — test download of one day's 6.25 km grid.
- BYU/NIC iceberg database — download the ASCII archive, confirm it parses.
- GEBCO — download the 2024 bathymetry grid (static, one-time).
- USNIC SIGRID-3 — confirm access to at least a recent week's ice chart.

Document exactly which of these worked on the first try and which needed a
workaround — this becomes the "data gap" fallback answer for judge Q&A later,
so write it down now while it's fresh, don't try to reconstruct it in Sprint 7.

### Shared code contracts (write these now so Sprints 1–6 don't diverge)
- `shared/grid.py`: constants for EPSG:3412, corridor bbox, 6.25 km
  resolution, 6-hourly timestep — imported by ingestion, models, and routing.
  No sprint should hardcode these values separately.
- `shared/schemas.py`: Pydantic models for `SicForecastResponse`,
  `IcebergTrackResponse`, `RouteOptimizeResponse`, `PolarisRiskResponse` —
  Backend and Frontend leads agree on these field names *now*, in Sprint 0,
  so Sprint 5 and Sprint 6 don't have to renegotiate the API contract midway.

### Infra skeleton (Data/DevOps Lead)
- `docker-compose.yml` bringing up: PostgreSQL+PostGIS, TimescaleDB, Redis,
  a placeholder FastAPI container, a placeholder Next.js container.
- `docker compose up` should succeed (even with empty apps) by end of Sprint 0.

### Frontend hello-world (Frontend Lead)
- Next.js app scaffold with CesiumJS via Resium rendering a basic 3D globe
  (Cesium Ion default imagery is fine at this stage — no real data yet).

### Backend hello-world (Backend Lead)
- FastAPI app with the four endpoint routes stubbed to return fixed mock
  JSON matching the `shared/schemas.py` contracts.

### Research & Pitch Lead (runs in parallel from day 0)
- Start `/docs/novelty-matrix.md`: one row per differentiator (POLARIS,
  IDRIFTNET-style residual drift, sensor-invariant training, dual
  scientist/captain view) vs. what a "generic UNet + Leaflet" baseline team
  would submit.
- Start `/docs/references.md`: NCPOR tender document, Indian Antarctic Bill
  2022, IDRIFTNET paper (arXiv 2507.00036), POLARIS/IMO methodology
  reference, MT-IceNet / ANTSIC-UNet papers — one line each on what you're
  actually using from it.
- Confirm the real team size and actual internal-round deadline with your
  SPOC, and adjust the day numbers in `01-SPRINT-OVERVIEW.md` to match if
  they differ from the 21-day assumption.

## Definition of Done
- [ ] Repo pushed with the structure above
- [ ] `environment.yml` builds for every team member
- [ ] `docker compose up` succeeds
- [ ] Every provider above has one confirmed successful test pull, documented
- [ ] `shared/grid.py` and `shared/schemas.py` exist and are imported (not
      copy-pasted) by every later sprint's code
- [ ] Novelty matrix and references doc exist with at least a first draft
