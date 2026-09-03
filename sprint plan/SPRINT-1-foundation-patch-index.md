# Sprint 1 — Days 1–2: Foundation, Patch Index & Grid Registration

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Days 1–2  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Complete ingestion and spatial alignment of All-India CHIRPS, IMD 0.25°, GLO-30 DEM, and Mandya panchayat vector geometries. Build the all-India patch index store (Zarr/LMDB) enforcing land-fraction filtering and spatial holdout buffer exclusion. Implement and pass unit tests for grid registration (2.7 km shift assert) and physical conservation laws (kernel=5, HR cosine weights, clean polygon weighting) *in isolation before model training commences*.

---

## 2. Team Responsibilities & Ownership

| Role | Sprint 1 Deliverables |
|---|---|
| **Data/GIS Lead** | Ingest IMD 0.25° grid, verify national CHIRPS Zarr, download GLO-30 DEM, construct all-India patch index with buffer exclusion & land filter, run `scripts/check_registration.py`, implement `src/data/loaders.py`, `src/data/patch_extraction.py`, `src/data/zonal_aggregation.py`. |
| **ML Lead** | Write unit tests for physical conservation in `tests/test_conservation.py`, implement `src/losses/conservation.py` with `kernel=5`, `count_include_pad=False`, cosine weights at HR centers, verify analytic spherical area formulas. |
| **ML/Eval Engineer** | Author `tests/test_patch_geometry.py` and `tests/test_splits.py`, verify 4-way temporal split (train 2010–2020, val 2021, cal 2022, test 2023) and spatial holdout mask. |
| **Domain/Product Lead** | Document Mandya agro-climatic baseline and 2 dominant crops (Ragi + Paddy per RDPR calendar). |

---

## 3. Pinned Technical Specifications & Mathematics

### A. Resolution Pair & 5× Direct Scaling
* **LR:** IMD 0.25° native grid (~752 km²/cell at 13°N, 27.75 km × 27.12 km, 135×129 grid, origin 6.5°N 66.5°E).
* **HR:** CHIRPS 0.05° native grid (~30.1 km²/cell, 5.55 km × 5.42 km, standard GeoTIFF origin with +0.025° half-pixel centers).
* **Scale Factor:** Direct $5\times$ linear downscaling ($0.25 / 0.05 = 5$, kernel=5). Never write "3 blocks × 2x = 4x" or "4x".

### B. Grid Registration (Catching the 2.7 km Shift)
* Standard GeoTIFF half-pixel centers vs. IMD grid centers create a potential $\sim 2.7\text{ km}$ systematic shift.
* **Day-1 Assert:** Verify IMD cell boundaries strictly align with CHIRPS 5×5 blocks.
* If offset: perform a one-time area-weighted remap of IMD onto the CHIRPS-nested grid (or shift CHIRPS by 0.025°), and freeze the affine transform in `src/data/loaders.py`.
* *Note:* Constant-field conservation tests pass even with mis-registration; only `test_registration.py` catches this systematic shift.

### C. Physical Conservation Grid-to-Grid (Kernel=5)
$$\mathbf{w}_{\text{HR}} = \cos(\text{lat}_{\text{HR\_rad}})$$
$$\mathcal{C}[\text{HR}] = \frac{\text{avg\_pool2d}(\text{HR} \odot \mathbf{w}_{\text{HR}}, k=5, s=5, \text{count\_include\_pad}=\text{False})}{\text{avg\_pool2d}(\mathbf{w}_{\text{HR}}, k=5, s=5, \text{count\_include\_pad}=\text{False})}$$
$$\mathcal{L}_{\text{cons}} = \text{MSE}(\mathcal{C}[\text{HR}], \text{LR})$$
* Cosine variation over 1° at 12.5°N is $\approx 0.4\%$ (negligible locally, but $\sim 19\%$ across India $8^\circ\text{--}37^\circ\text{N}$, making it mandatory).
* Weights $\mathbf{w}$ must be computed at **HR centers**, not LR.

### D. Clean Grid-to-Polygon Conservation (No Double-Counting)
* For each panchayat $P$:
  1. Rasterize polygon to fractional coverage $f_i \in [0, 1]$ on the 0.05° HR grid.
  2. Compute analytic spherical cell area:
     $$A_i = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\text{lat}_i)$$
     where $R = 6371008.8\text{ m}$, and $\Delta\phi, \Delta\lambda$ are cell resolutions in radians.
  3. Effective cell weight: $w_i = f_i \cdot A_i$.
  4. Aggregate precipitation:
     $$\text{Rain}_P = \frac{\sum_i \text{HR}_i \cdot w_i}{\sum_i w_i}$$
* **CRS Policy:** EPSG:7755 is for map display only. Never multiply an equal-area CRS area by $\cos(\text{lat})$ (double-counting error). Use `pyproj.Geod` for absolute area audits.
* **Validation Gates:**
  1. Per polygon: $\sum_{\text{cells}} \text{inter\_area}(P, \text{cell}) == \text{area}(P)$ within relative tolerance $10^{-3} \cdot \text{area}(P)$.
  2. Per interior HR cell: $\sum_P \text{fraction}(P \cap \text{cell}) \approx 1.0$ (boundary cells reported, not failed).
  3. Reported audit statistic: $|\text{mean\_HR}(P) - \text{mean\_LR}(P_{\text{cell}})|$ with practical tolerance $\le 10\%$ or $5\text{ mm}$ (never assert $10^{-3}\text{ mm}$).

### E. Patch Geometry & All-India Scope
* **The Bug:** Mandya is 4,961 km² ($\approx 165$ HR pixels by area, bounding box $\sim 20\times 26$ pixels). A single $64\times 64$ HR patch is $123,000\text{ km}^2$ ($24.8\times$ larger than Mandya district). Training on a single district is mathematically impossible.
* **The Solution:** Train on All-India monsoon domain ($68^\circ\text{--}97^\circ\text{E}, 8^\circ\text{--}37^\circ\text{N}$, $\approx 620\times 580$ HR pixels).
* **Patch Specifications:**
  - $80\times 80$ HR patches with corresponding $16\times 16$ LR context.
  - Stride: 40 pixels.
  - Raw patch yield: $\sim 182$ patches/day $\times 1,708$ days $\approx 310,856$ raw patches.
  - Filtering: Retain only patches with **land fraction $\ge 70\%$** (removes ocean/desert waste).
  - Spatial Holdout: Strictly exclude any patch overlapping Mandya $+ 0.5^\circ$ buffer.
  - Usable patch yield: $\sim 120\text{--}140$ patches/day $\times 1,708$ days = **$\sim 200,000\text{--}240,000$ usable patches**.
  - Cache store: Pre-saved to Zarr/LMDB ($\approx 8.5\text{ GB}$ compressed).

---

## 4. Sprint 1 Detailed Tasks

### A. Data Pipelines & Loading
- [ ] **Ingest IMD 0.25° Daily Rainfall (1901–2024):**
  - Implement `scripts/download_imd.py` to fetch IMD 0.25° binary/gridded data.
  - Parse into xarray Dataset ($135\times 129$ grid, latitude $6.5^\circ\text{--}38.5^\circ\text{N}$, longitude $66.5^\circ\text{--}100.0^\circ\text{E}$).
- [ ] **Ingest National CHIRPS 0.05° & GLO-30 DEM:**
  - Complete `scripts/download_chirps.py` for JJAS monsoon (1,708 days, 2010–2023). If bandwidth constrained, apply fallback to 2014–2023 (1,100 days).
  - Crop GLO-30 DEM to India domain and compute slope/aspect channels.
- [ ] **Panchayat Filtering & Spatial Holdout Creation:**
  - Filter `india-geodata` to Mandya (258 GPs) and backup Mysuru using DuckDB.
  - Generate holdout polygon: `Mandya.buffer(0.5)`.
  - Save holdout mask in `src/data/pinned_district.json`.
- [ ] **Implement `src/data/loaders.py`:**
  - Standardized xarray loader with lazy windowed reads and float32 casting.
  - Apply frozen registration transform.

### B. Patch Extraction Pipeline
- [ ] **Implement `scripts/build_patch_index.py` & `src/data/patch_extraction.py`:**
  - Extract $80\times 80$ HR / $16\times 16$ LR patch coordinates across India monsoon domain.
  - Calculate land fraction per patch; filter out patches $< 70\%$ land.
  - Perform spatial intersection test against Mandya $+ 0.5^\circ$ buffer; drop any intersecting patches.
  - Store indexed patches in Zarr / LMDB ($8.5\text{ GB}$ raw compressed).
  - Verify patch count: assert usable patches between $200,000$ and $240,000$.

### C. Conservation & Losses Implementation
- [ ] **Implement `src/losses/conservation.py`:**
  - Function `conservation_loss_grid(hr_pred, lr_true, lats_deg)` implementing area-weighted average pooling with $\cos(\text{lat})$ at HR centers, $k=5, s=5$.
  - Function `zonal_polygon_aggregate(hr_grid, polygons, lats_deg)` implementing clean $w_i = f_i \cdot A_i$ with analytic spherical cell area.
- [ ] **Mandya Crop Documentation:**
  - Document Ragi and Paddy growth stages and rainfall sensitivities in `docs/agro_baseline.md`.

---

## 5. Test Suite & Verification Gates

The following automated test suite in `tests/` must pass by End of Day 2:

- [ ] **`tests/test_gis.py`:**
  - Assert Mandya GP count is $258$ (within $[80, 300]$).
  - Assert valid geometry fraction $> 98\%$.
  - Assert `patch_index ∩ Mandya_buffer_0.5 == ∅`.
  - Assert all indexed patches have land fraction $\ge 70\%$.
- [ ] **`tests/test_registration.py`:**
  - Verify IMD cell boundaries and CHIRPS $5\times 5$ blocks align exactly without the $2.7\text{ km}$ systematic shift.
- [ ] **`tests/test_conservation.py`:**
  - `test_conservation_exact`: On a uniform constant field, downscaled then coarsened field matches input within $10^{-6}$.
  - `test_conservation_detects_sum_bug`: Assert that naive summation fails unit test.
  - Test verifies kernel=5 and cosine weights computed at HR centers.
- [ ] **`tests/test_patch_geometry.py`:**
  - Assert single-district patch generation fails ($64\times 64 > \text{Mandya}$).
  - Assert all-India patch index yields $\approx 200,000\text{--}240,000$ usable patches.
- [ ] **Visual Sanity Check:**
  - Plot aligned LR IMD, HR CHIRPS, GLO-30 DEM, and Mandya vector overlay for a single monsoon date (`docs/alignment_sanity.png`).

---

## 6. Definition of Done (Sprint 1)

- [ ] All 5 unit tests (`test_gis.py`, `test_registration.py`, `test_conservation.py`, `test_patch_geometry.py`, `test_splits.py`) pass in CI.
- [ ] Pre-cached patch index store exists on disk ($\approx 200\text{k--}240\text{k}$ patches, Zarr/LMDB, $8.5\text{ GB}$).
- [ ] Grid registration assert passes, transform frozen in `loaders.py`.
- [ ] Mandya spatial holdout ($+0.5^\circ$ buffer) provably disjoint from training patch store (`patch_index ∩ buffer == ∅`).
- [ ] Constant-field conservation error $< 10^{-6}$.
- [ ] **Exit Rule:** If any test fails at EOD Day 2, team does NOT proceed to model training until resolved.
