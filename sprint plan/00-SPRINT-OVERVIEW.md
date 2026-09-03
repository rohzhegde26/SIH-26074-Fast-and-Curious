# SIH26074 — Master Sprint Overview, Technical Spec & Finals Playbook

**Master Specification:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Problem Statement:** SIH26074 — Downscaling of weather forecast from Block level to Panchayat level (Ministry of Earth Sciences / MoES)  
**Theme (Corrected):** Agriculture, FoodTech & Rural Development  
**Repository:** [`rohzhegde26/SIH-26074-Fast-and-Curious`](https://github.com/rohzhegde26/SIH-26074-Fast-and-Curious)  
**Status:** FINAL V2 (SIH Finals-Ready)

---

## 1. Timeline & Sprint Index

| Sprint | Days | Focus Area | Primary Owner(s) | Sprint Document |
|---|---|---|---|---|
| **Sprint 0** | Day 0 | Accounts, Scaffolding, District Pinning, DuckDB Validation & Bandwidth Strategy | Data/GIS + ML Lead | [SPRINT-0-day-0.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-0-day-0.md) |
| **Sprint 1** | Days 1–2 | Foundation, All-India Patch Index, Grid Registration Assert & Isolated Conservation Tests | Data/GIS + ML Lead | [SPRINT-1-foundation-patch-index.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-1-foundation-patch-index.md) |
| **Sprint 2** | Days 3–4 | Core Model 5× Direct Scaling, Baselines (Bilinear, DeepSD, RF) & Conservation Loop | ML Lead + ML/Eval | [SPRINT-2-core-model-5x.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-2-core-model-5x.md) |
| **Sprint 3** | Days 5–6 | Quantile Mapping Calibration, CQR Uncertainty (90% Test Coverage) & Zonal Polygon Stats | Data/GIS + ML/Eval | [SPRINT-3-calibration-cqr-aggregation.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-3-calibration-cqr-aggregation.md) |
| **Sprint 4** | Days 7–8 | FastAPI Services, Offline-First Mobile PWA & Bilingual Advisory (Ragi + Paddy) | Backend + Frontend | [SPRINT-4-api-pwa-advisory.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-4-api-pwa-advisory.md) |
| **Sprint 5** | Days 9–10 | Pitch Deck, 2-Min Video Storyboard, Automated Grep Audits & Finals QA Rehearsal | Pitch Lead + Team | [SPRINT-5-pitch-deck-finals.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-5-pitch-deck-finals.md) |

---

## 2. Team Roles & Ownership Matrix (Team of 6)

| Role | Core Responsibilities | Primary Sprints |
|---|---|---|
| **Data/GIS Lead** | Ingestion of IMD 0.25°, CHIRPS 0.05°, GLO-30 DEM; DuckDB panchayat geometry processing; patch index extraction ($\ge 70\%$ land filter); grid registration assert; clean zonal polygon aggregation weights ($w_i = f_i \cdot A_i$). | 0, 1, 3 |
| **ML Lead** | 5× direct U-Net/CNN downscaling architecture; physical mass conservation pooling loss ($k=5$, $\mathbf{w}$ at HR centers); PyTorch AMP fp16 training loop; integration of terrain channels. | 0, 1, 2 |
| **ML/Eval Engineer** | Fast patch loader (Zarr/LMDB); baselines (Bilinear, DeepSD-style CNN, optional RF); 4-way temporal splits; per-$0.25^\circ$-cell quantile mapping; Conformalized Quantile Regression (CQR); hill-vs-plains evaluation. | 1, 2, 3 |
| **Backend Engineer** | FastAPI endpoints (`/api/forecast/{lgd_code}`, `/api/egramswaraj/mock` labeled mock); serving quantile mapping and CQR predictions; OpenAPI spec documentation. | 0, 4 |
| **Frontend/PWA Engineer** | Offline-first mobile PWA; Service Worker caching (Network-First forecast, Cache-First tiles); IndexedDB storage; airplane-mode detection banner; interactive Mandya map. | 0, 4 |
| **Domain/Product + Pitch Lead** | Mandya crop calendars (Ragi + Paddy); bilingual agro-advisory rules in Kannada & English; pitch deck and 2-minute video; Section 6 enforcement (11 forbidden claims); honest ceiling defense. | Throughout, heavy in 4, 5 |

### Merge & Headcount Contingencies
* **Never merge Pitch Lead:** Dedicated narrative, scientific defense, and jury rehearsal ownership is required full-time.
* **If 5 members:** Merge Backend and Frontend roles; descope Frontend to static cached map tiles if needed.
* **If 4 members:** Require pre-filtered Mandya parquet, GLO-30 tiles, and pre-cached patch Zarr ready on Day 0.
* **CODEOWNERS Enforcement:**
  - `/src/losses/` $\to$ `@ML Lead`
  - `/src/data/` $\to$ `@Data/GIS Lead`
  - `docs/pitch_deck.pdf` $\to$ `@Pitch Lead`

---

## 3. High-Level Architecture & Pipeline Flow

```
+------------------------------------------------------------------------------------+
| 1. DATA SOURCES & INGESTION                                                        |
| - IMD 0.25° Daily Rainfall (Native LR coarse input, 1901-2024, 135x129 grid)       |
| - CHIRPS 0.05° Daily Rainfall (Native HR target, 1981-present, free no login)       |
| - Copernicus GLO-30 DEM (30m elevation via CDSE S3, slope & aspect derived)        |
| - India-Geodata Parquet (319k LGD panchayats -> DuckDB Mandya 258 GPs filtered)    |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 2. SPATIAL PRE-PROCESSING & REGISTRATION ASSERT                                    |
| - Grid Registration: Check IMD 6.5+0.25k vs CHIRPS +0.025° offset (catches 2.7km)  |
| - Spatial Holdout: Mandya + 0.5° buffer strictly excluded from patch index         |
| - Dual-Export Rule: mandya_full.geojson (math/audits) vs mandya_simplified.topojson|
| - Patch Extraction: All-India 80x80 HR / 16x16 LR (stride 40, land fraction >=70%) |
|   Yield: ~200k-240k usable patches stored in Zarr/LMDB (8.5 GB compressed)         |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 3. CORE 5x DOWNSCALING MODEL & LOSS CONSERVATION                                   |
| - Direct 5x Linear Downscaling (0.25° -> 0.05°, kernel=5)                          |
| - Terrain-Conditioned U-Net / CNN with DEM, Slope, Aspect                          |
| - Mass Conservation Loss: avg_pool2d(HR * cos(lat_hr), k=5) == LR                  |
| - 4 Splits: Train 2010-2020 | Val 2021 | Cal 2022 | Test 2023                      |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 4. POST-PROCESSING, CALIBRATION & UNCERTAINTY                                      |
| - Per-0.25°-Cell Quantile Mapping to IMD gauge gold (preserves intra-cell texture) |
| - Conformalized Quantile Regression (CQR): 90% empirical coverage on test 2023     |
| - Clean Zonal Polygon Aggregation: w_i = f_i * A_i (using full-precision geodata)  |
| - Bookkeeping Gates: Polygon area closed to 1e-3; interior cell partition sum ~ 1   |
| - Mandatory Hill-vs-Plains Error Stratification & Extreme Event CSI (R95, R99)     |
+-----------------------------------------+------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
| 5. DELIVERY & ADVISORY SYSTEM                                                      |
| - Thin Orchestrator: scripts/run_pipeline.py (<=2hr timebox, single-command demo)   |
| - FastAPI backend: /api/forecast/{lgd_code} & /api/egramswaraj/mock (labeled mock) |
| - Mobile PWA: Offline viewing with Airplane-Mode alert (<400KB TopoJSON display)   |
| - Bilingual Agro-Advisory: Stage-specific advice in Kannada & English (Ragi+Paddy) |
| - "Honest Ceiling": Panchayat-scale (~5.5km = 1 GP), never overclaimed sub-GP      |
+------------------------------------------------------------------------------------+
```

---

## 4. The 5 Known Historical Bugs (Fixed in V2)

1. **Conservation-as-Sum Bug:** Naive summation of HR pixels to match LR introduced a $16\times$ error at $4\times$ ($25\times$ at $5\times$). Fixed by area-weighted average pooling with $\cos(\text{lat})$ weights at HR centers ($k=5$). Unit-tested in isolation before training.
2. **$3 \text{ blocks} \times 2\times = 4\times$ Arithmetic Bug:** Multi-stage doubling ($2^3$) is $8\times$, not $4\times$ or $5\times$. Fixed by using direct, single-stage $5\times$ downscaling ($0.25^\circ \to 0.05^\circ$, kernel 5).
3. **Resolution Inversion Bug:** NCMRWF IMDAA 12 km is finer than IMD 27 km (0.25°), making it invalid as a coarse LR input without prior coarsening. Fixed by permanently deleting IMDAA and using native IMD 0.25° as the LR baseline.
4. **CRS Area Distortion & Cosine Double-Counting:** Degree² in EPSG:4326 distorts cell areas by $>10\%$ between Kerala and Punjab. Multiplying an equal-area projection by cosine double-corrects. Fixed by using analytic spherical cell areas ($A_i = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\text{lat})$) for weights, `pyproj.Geod` for absolute audits, and EPSG:7755 strictly for map visualization.
5. **Single-District Patch Geometry Bug:** Mandya (4,961 km²) is only $\sim 165$ HR pixels (bbox $\sim 20\times 26$ pixels). A single $64\times 64$ HR patch is $123,000\text{ km}^2$ ($24.8\times$ larger than Mandya). Training on a single district is mathematically impossible. Fixed by training across the all-India monsoon domain with Mandya strictly held out.

---

## 5. What We Will NOT Claim (The 11 Jury Guardrails)

* **Claim 1:** Will **NOT** say we integrated real-time BharatFS — we used IMD as LR and CHIRPS as HR training target; BharatFS is on the operational roadmap.
* **Claim 2:** Will **NOT** say the system generates forecasts offline — it views the last-synced forecast offline.
* **Claim 3:** Will **NOT** claim write access or direct push to e-GramSwaraj — integration-ready mock interface only.
* **Claim 4:** Will **NOT** present raw MC-dropout as calibrated probabilities — we present CQR calibrated intervals (*"Expected X mm, likely Y–Z mm, 90% empirical coverage on test"*).
* **Claim 5:** Will **NOT** claim architectural novelty for the downscaling neural network itself — novelty is system-level (all-India training with spatial holdout, clean polygon aggregation, physical conservation, registration assert, quantile calibration, and CQR uncertainty).
* **Claim 6:** Will **NOT** claim Tier 2 or Tier 3 operational status — roadmap only (IMERG Early/Late, BharatFS).
* **Claim 7:** Will **NOT** claim 30,416 Karnataka GPs — official active count is ~5,788–6,376; Mandya has 258 GPs.
* **Claim 8:** Will **NOT** claim statewide Cartosat DEM mosaicking in 10 days on Bhuvan — quota is 10/day; we use GLO-30 via CDSE S3 with zero quota.
* **Claim 9:** Will **NOT** claim IMD temperature exists at 0.25° — IMD gridded rainfall is 0.25°; temperature is 1.0°.
* **Claim 10:** Will **NOT** claim Bangalore Urban as a pilot district — Bangalore Urban has 0 GPs (BBMP wards).
* **Claim 11:** Will **NOT** claim sub-panchayat resolution detail — at 5.5 km, 1 pixel $\approx 1$ GP. We deliver panchayat-scale, not sub-panchayat. State this before the jury asks.

---

## 6. Risk Register & Mitigations (Section 7 of Master Spec)

| Risk | Likelihood | Impact | Mitigation Strategy | Owner |
|---|---|---|---|---|
| **1. 0 GPs in Bangalore Urban / Messy Mandya** | High | High (blocks differentiator) | Pin Mandya Day 0; DuckDB pushdown filter; assert count 80–300 (258 GPs), valid > 98%; exclude buffer from patch index; test_gis. | Data/GIS + Pitch Lead |
| **2. Re-adding Deleted NCMRWF IMDAA** | Low | High (reintroduces inversion bug) | IMDAA permanently deleted in Section 0; CODEOWNERS blocks changes to `/src/data/` without review. | ML Lead |
| **3. Bhuvan 10/day Quota Blocks DEM** | High if used | Medium (loses terrain) | Use GLO-30 via CDSE S3 primary (no quota); fallback to SRTM open-data bucket. | Data/GIS Lead |
| **4. CDSE/CDS 403 / Grid Mis-registration (~2.7 km)** | Medium | Medium (systematic shift) | Accept licenses Day 0; run `test_cdse.py` & `check_registration.py` Day 0; regrid once area-weighted or shift 0.025°; freeze transform in `loaders.py`. | Data/GIS Lead |
| **5. Patch Index OOM / Desert & Sea Waste** | High | Medium (wasted compute & time) | DuckDB pushdown; lazy xarray reads; filter land fraction $\ge 70\%$; assert `patch_index ∩ buffer == ∅`. | Data/GIS Lead |
| **6. Single-District Training Impossible** | High if not fixed | High (model never trains) | All-India training ($80\times 80$ HR / $16\times 16$ LR, ~200k–240k usable patches); Mandya $+ 0.5^\circ$ buffer strictly held out. | ML Lead |
| **7. Conservation Loss Fails to Converge** | Medium | High | Unit-test in isolation before model training; verify $k=5$, $\mathbf{w}$ at HR centers, `count_include_pad=False`; clean $w_i = f_i \cdot A_i$ polygon weights. | ML Lead |
| **8. Quantile Mapping per HR Cell Destroys Texture** | Medium | High (hallucinates gauge truth) | Mandate mapping per $0.25^\circ$ LR cell (not HR cell); report headline metrics on calibrated outputs, raw CHIRPS as diagnostics. | ML/Eval |
| **9. PWA Offline Exceeds Time Budget** | Medium | Medium | Timebox Sprint 4 to 1 day; ship robust cache-fallback (IndexedDB) if full background sync service worker balloons. | Frontend |
| **10. Team Overclaims Under Jury Pressure** | High | High (credibility risk) | Review Section 5 live before pitch; grep codebase and slides for banned phrases; Pitch Lead owns narrative full-time. | Pitch Lead |
| **11. Bandwidth Download Bottleneck (~10–25 GB)** | Medium | Medium (schedule delay) | If 2010–2023 (~1,700 days) download is too slow, fallback immediately to 2014–2023 (~1,100 days, ~200k patches still statistically valid). | Data/GIS Lead |

---

## 7. SIH Finals Readiness & Jury Defense FAQ

### Q1: "Isn't this just spatial interpolation / hindcasting? How does this work in real operations tomorrow morning at 06:00 IST?"
* **Defense:**  
  *"We deliberately built this as **Tier 1: Supervised Downscaling under Perfect-Model Conditions**. In meteorological literature (e.g., DeepSD, ClimDown), isolating the physical downscaling mapping using observation pairs (IMD 0.25° $\to$ CHIRPS 0.05°) is standard practice to prevent numerical forecast drift from contaminating the learned topographical transfer function. For real-time deployment (Tier 2/3), our pipeline is format-compatible with operational GFS 0.25° and NCUM grids, utilizing NASA IMERG Early (~4-hour latency) as the operational high-resolution verification reference."*

### Q2: "Did you account for the 03:00 UTC IMD rainfall cutoff?"
* **Defense:**  
  *"Yes. IMD 0.25° observations represent 08:30 IST to 08:30 IST (03:00 UTC) accumulation. Our per-$0.25^\circ$-cell quantile mapping explicitly aligns the cumulative distribution function of satellite accumulations to IMD gauge totals, effectively absorbing systematic temporal phase and calibration offsets while preserving high-resolution spatial gradients."*

### Q3: "What is your skill on actual rain events, excluding dry days?"
* **Defense:**  
  *"We do not hide behind dry-day inflated MAE numbers. In `docs/hill_vs_plains.png` and our evaluation suite, we explicitly report Critical Success Index (CSI) for heavy rainfall thresholds (R95 and R99), and evaluate wet-day MAE ($>2.5\text{ mm}$) separately from dry-day accuracy."*

---

## 8. Master Definition of Done (15 Measurable Items)

- [ ] `test_conservation_exact` passes: constant field coarsened equals input within $10^{-6}$ ($k=5$, $\mathbf{w}$ at HR centers).
- [ ] `test_conservation_detects_sum_bug` passes: naive summation fails unit test.
- [ ] `test_gis` passes: Mandya count 80–300 (258 GPs), valid $>98\%$, buffer $0.5^\circ$ excluded from train, `patch_index ∩ buffer == ∅`, land fraction $\ge 70\%$.
- [ ] `test_patch_geometry` passes: single-district patch bug caught, all-India $80\times 80$ HR / $16\times 16$ LR verified.
- [ ] `test_registration` passes: IMD vs. CHIRPS grid edge alignment verified, $2.7\text{ km}$ shift caught.
- [ ] `test_quantile_mapping` passes: mapping executed per $0.25^\circ$ LR cell, preserving intra-cell spatial variance.
- [ ] Downscale factor arithmetic verified: direct $5\times$ ($0.25^\circ \to 0.05^\circ$, kernel 5) used everywhere; grep for `3 blocks × 2x = 4x` returns 0.
- [ ] Panchayat-level output verified across 3 Mandya panchayats with polygon area closure error $\le 10^{-3}$ and interior cell partition $\approx 1.0$.
- [ ] Dual-export verified: `mandya_full.geojson` used strictly for backend aggregation and geodetic area closure, `mandya_simplified.topojson` used strictly for mobile PWA display ($<400\text{ KB}$).
- [ ] Single-command demo orchestrator `scripts/run_pipeline.py` ($\le 2$-hour timebox) executes end-to-end in $< 5\text{ seconds}$ on a single date.
- [ ] Calibration curve artifact generated: `docs/calibration_curve.png`.
- [ ] CQR coverage report generated: `docs/cqr_coverage.png` ($90\% \pm 2\%$ on unseen 2023 test data).
- [ ] Hill-vs-plains stratified error breakdown generated: `docs/hill_vs_plains.png`.
- [ ] Wet-Day MAE ($>2.5\text{ mm}$) and Extreme Event CSI (R95/R99) reported separately from aggregate dry-day metrics.
- [ ] Uncertainty reported as calibrated range: *"Expected X mm, likely Y–Z mm ($90\%$ coverage)"*, never bare uncalibrated percentage.
- [ ] Offline PWA operational: displays *"Viewing cached forecast from {timestamp}"* with functional airplane-mode demonstration.
- [ ] Mock government integration `/api/egramswaraj/mock` explicitly labeled as mock prototype in code, Swagger, and headers.
- [ ] README and pitch deck strictly audited: all 11 forbidden claims absent via grep, data-attribution block present, honest ceiling clearly acknowledged.
