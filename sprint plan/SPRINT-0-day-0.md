# Sprint 0 — Day 0: Foundation, Scaffolding, Registrations & District Pinning

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Day 0  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Establish environment scaffolding, verify cloud/data account credentials with live test pulls, pin geographic pilot parameters in JSON, validate Mandya panchayat geometries via DuckDB, kickoff national CHIRPS ingestion, enforce repository safety guardrails (CODEOWNERS, .gitignore, licenses), and pin the 11 "What We Will NOT Claim" boundaries so no sprint wastes time on setup or unapproved architecture.

---

## 2. Team Responsibilities & Ownership

| Role | Day 0 Deliverables |
|---|---|
| **Data/GIS Lead** | CDSE registration, GLO-30 DEM test script, CDS registration (ERA5 conditional), DuckDB panchayat validation, `pinned_district.json`, CHIRPS download kickoff, `check_registration.py` scaffold. |
| **ML Lead** | Environment definition (`requirements.txt`/`environment.yml`), verify PyTorch AMP on GPU (RTX 3060), review registration assertion math, CODEOWNERS rules for `/src/losses/`. |
| **ML/Eval Engineer** | Prepare train/val/cal/test split architecture (train 2010–2020, val 2021, cal 2022, test 2023), define patch index specifications (80×80 HR / 16×16 LR). |
| **Backend & Frontend** | Verify repository scaffolding, agree on API schemas and offline cache architecture (IndexedDB / ServiceWorker contracts). |
| **Domain/Product + Pitch Lead** | Mandya cropping calendar (Ragi + Paddy), pin Section 6 "11 items NOT to claim" in README and team channels, draft data-attribution block. |

---

## 3. Pinned Decisions Relevant to Day 0

* **Pilot Scope:** Karnataka, **MANDYA primary** (4,961 km² / 258 GPs = 19.2 km² avg = ~165 HR pixels at 0.05°; bbox ~20×26 pixels ≈ 520 HR pixels). Backup: **MYSURU**. **FORBIDDEN: Bangalore Urban / BBMP (0 GPs)**.
* **Spatial Holdout:** Mandya + 0.5° buffer (~50–100 km) strictly excluded from training; buffer excluded from patch index.
* **LR-HR Pair:** LR = IMD 0.25° native (centers 6.5 + 0.25k, ~752 km²/cell) = block/taluk scale. HR = CHIRPS 0.05° (centers +0.025° offset, ~30.1 km²/cell) = panchayat scale (5× downscaling, kernel=5).
* **DEM Choice:** Primary GLO-30 via CDSE S3 (`s3://copernicus-dem-30m/`, CC-BY 4.0, no quota). Fallback: SRTM via open-data bucket. **Bhuvan is dropped** (10 tiles/day quota blocks 10-day timeline).
* **Deleted Sources:** **NCMRWF IMDAA 12km is permanently deleted** (resolution inversion bug: 12km is finer than 27km IMD; NCMRWF SLA unacceptable).
* **Panchayat Ingestion:** `india-geodata` 319,287 LGD parquet (351 MB) queried via DuckDB with pushdown filter; never load whole parquet into memory via geopandas.

---

## 4. Day 0 Detailed Tasks

### A. Provider Accounts & Connectivity Testing
- [ ] **Register CDSE (GLO-30 DEM — Mandatory):**
  - Register at `dataspace.copernicus.eu`.
  - Generate API token and configure S3 access to `s3://copernicus-dem-30m/`.
  - Run `python scripts/test_cdse.py` to verify tile retrieval over Karnataka bbox.
  - Implement `scripts/download_glo30.py` for Mandya elevation data; provide fallback via `scripts/download_srtm.py`.
- [ ] **Register CDS (ERA5 — Conditional/Optional):**
  - Register at `cds.climate.copernicus.eu`, accept the 2 terms/licenses, obtain API key, and configure `~/.cdsapirc`.
  - Run `python scripts/test_cds.py` for a 1-day Karnataka bbox retrieval.
  - *Note:* DoD is conditional — ERA5 is optional; only pull if ahead of schedule.
- [ ] **Do NOT Register NCMRWF:** Confirm IMDAA is completely omitted from the data pipeline.

### B. Geographic & Pilot Pinning
- [ ] **Create `src/data/pinned_district.json`:**
  ```json
  {
    "primary": "MANDYA",
    "backup": "MYSURU",
    "forbidden": ["BANGALORE URBAN", "BBMP"],
    "expected_count": {
      "min": 80,
      "max": 300
    },
    "buffer_deg": 0.5,
    "area_km2": 4961,
    "hr_pixels_area": 165,
    "hr_pixels_bbox": [20, 26],
    "area_per_pixel_km2": 30.1
  }
  ```
- [ ] **DuckDB Panchayat Ingestion, Topology Validation & Dual-Export Rule:**
  - Implement `scripts/validate_panchayat.py --district MANDYA --buffer 0.5`.
  - Execute DuckDB pushdown: `SELECT * FROM read_parquet('panchayats.parquet') WHERE stname='KARNATAKA' AND dtname='MANDYA'` (<1 sec, <5 MB RAM).
  - Apply `shapely.validation.make_valid()`.
  - Assertions:
    - GP count is between 80 and 300 (Mandya has 258 GPs).
    - Valid geometries > 98%.
    - Coordinate system: EPSG:4326 for analysis, EPSG:7755 for display/viz only.
    - Generate spatial holdout mask: Mandya boundary + 0.5° buffer (~50–100 km).
    - Assert `patch_index ∩ buffer == ∅`.
  - **MANDATORY DUAL-EXPORT RULE (Prevents Broken Area Audits & Heavy PWA Loads):**
    1. `data/processed/mandya_full.geojson`: Full geodetic precision. Used strictly for DuckDB queries, `src/data/zonal_aggregation.py`, and `pyproj.Geod` area closure audits ($\le 10^{-3}$ relative error).
    2. `data/processed/mandya_simplified.topojson`: Coordinate-rounded / topology-preserved (tolerance $10^{-4}$ deg $\approx 10\text{ m}$ precision, payload $< 400\text{ KB}$). Used strictly for client-side mobile PWA rendering and IndexedDB caching.
    - *Critical Guardrail:* Never feed simplified geometry into area calculation or $w_i = f_i \cdot A_i$ zonal weights, which would immediately break the $10^{-3}$ audit gate.

### C. Ingestion Kickoff & Bandwidth Risk Mitigation
- [ ] **Download National CHIRPS Ingestion Kickoff:**
  - Execute `python scripts/download_chirps.py --years 2010-2023 --bbox 68,8,97,37` (India domain: ~620×580 pixels at 0.05°).
  - Target monsoon days (JJAS: 122 days/year × 14 years = 1,708 days).
  - Save as cropped Zarr store (3–6 GB raw).
  - **Bandwidth Risk Mitigation:** ~1,700 daily GeoTIFFs is ~10–25 GB. If bandwidth is constrained, fallback to **2014–2023** (~1,100 days, ~200k patches), which remains statistically sufficient.

### D. Grid Registration Assertion Setup
- [ ] **Implement `scripts/check_registration.py`:**
  - IMD 0.25° grid centers: `6.5 + 0.25k`.
  - CHIRPS 0.05° GeoTIFF origin: half-pixel offset `+0.025°`.
  - Day-0 test logic: Verify whether IMD cell boundaries strictly coincide with CHIRPS 5×5 pixel blocks. If offset by 0.025°, define one-time area-weighted remap of IMD onto CHIRPS-nested grid or shift CHIRPS by 0.025°, and freeze the transform in `loaders.py`.
  - *Warning:* Naive 5×5 reshape mis-registers pairs by ~2.7 km; constant-field tests pass regardless, so only an explicit registration assert catches this shift.

### E. Repository Scaffolding, Governance & Documentation
- [ ] **Scaffold Directory Structure:** Create directories per Section 8:
  - `data/raw/` (gitignored), `data/processed/`, `notebooks/`, `src/data/`, `src/losses/`, `src/eval/`, `src/api/`, `src/advisory/`, `scripts/`, `frontend/`, `docs/`, `tests/`.
  - Include placeholder for `scripts/run_pipeline.py` (dedicated $\le 2$-hour timeboxed thin orchestrator glue script for single-command live demo).
- [ ] **Environment & Ignore Files:**
  - Create `.gitignore` ignoring `data/raw/`, `*.parquet`, `*.nc`, `*.hgt`, `*.zarr`, `.env`.
  - Create `.env.example` with `CDSE_TOKEN=`, `CDS_API_KEY=` placeholders (no secrets committed).
  - Create `CODEOWNERS`:
    ```
    /src/losses/ @ML Lead
    /src/data/ @Data/GIS Lead
    docs/pitch_deck.pdf @Pitch Lead
    ```
- [ ] **Data Attribution Block in README:**
  - CHIRPS: Funk et al., UCSB Climate Hazards Center (CC-BY).
  - DEM: Copernicus GLO-30 (CC-BY 4.0).
  - IMD: India Meteorological Department, Ministry of Earth Sciences acknowledgement.
  - Panchayat Boundaries: india-geodata / Local Government Directory (LGD).
  - Operational Roadmap: NASA IMERG Early/Late.
- [ ] **Pin Section 6 "What We Will NOT Claim":** Add the 11 forbidden claims to the project README and team communications.

---

## 5. Definition of Done (Day 0)

- [ ] `CDSE` token active and `scripts/test_cdse.py` retrieves GLO-30 test tile from S3 without quota error.
- [ ] `CDS` test retrieve `test.nc` passes (conditional, only if ERA5 is used).
- [ ] `src/data/pinned_district.json` created with Mandya parameters and Bangalore Urban explicitly forbidden.
- [ ] `scripts/validate_panchayat.py` runs in <5 seconds using DuckDB, asserts 80–300 GPs (Mandya = 258), valid > 98%, exports spatial holdout buffer mask (0.5°), and strictly generates both `mandya_full.geojson` (math/audits) and `mandya_simplified.topojson` (PWA display <400KB).
- [ ] `scripts/check_registration.py` created to assert cell edge coincidence and catch 2.7 km offset.
- [ ] `scripts/download_chirps.py` executed and actively downloading monsoon 2010–2023 (or fallback 2014–2023).
- [ ] `.gitignore`, `.env.example`, `CODEOWNERS`, and data-attribution block committed to repository.
- [ ] Section 6 (11 items NOT to claim) pinned in README.
