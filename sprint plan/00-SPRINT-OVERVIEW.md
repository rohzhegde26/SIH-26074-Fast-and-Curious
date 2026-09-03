# SIH26074 — Sprint Plan Overview & Roadmap

**Master Specification:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Problem Statement:** SIH26074 — Downscaling of weather forecast from Block level to Panchayat level (Ministry of Earth Sciences / MoES)  
**Theme:** Agriculture, FoodTech & Rural Development  
**Repository:** [`rohzhegde26/SIH-26074-Fast-and-Curious`](https://github.com/rohzhegde26/SIH-26074-Fast-and-Curious)  
**Status:** FINAL V2 (Commit-Ready)

---

## 1. Timeline & Sprint Index

| Sprint | Days | Focus Area | Primary Owner(s) | Sprint Document |
|---|---|---|---|---|
| **Sprint 0** | Day 0 | Accounts, Scaffolding, District Pinning & Ingestion Prep | Data/GIS + ML Lead | [SPRINT-0-day-0.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-0-day-0.md) |
| **Sprint 1** | Days 1–2 | Foundation, All-India Patch Index & Grid Registration Assert | Data/GIS + ML Lead | [SPRINT-1-foundation-patch-index.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-1-foundation-patch-index.md) |
| **Sprint 2** | Days 3–4 | Core Model 5× Direct Scaling, Baselines & Conservation Loop | ML Lead + ML/Eval | [SPRINT-2-core-model-5x.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-2-core-model-5x.md) |
| **Sprint 3** | Days 5–6 | Quantile Calibration, CQR Uncertainty & Zonal Polygon Stats | Data/GIS + ML/Eval | [SPRINT-3-calibration-cqr-aggregation.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-3-calibration-cqr-aggregation.md) |
| **Sprint 4** | Days 7–8 | FastAPI Services, Offline-First PWA & Bilingual Advisory | Backend + Frontend | [SPRINT-4-api-pwa-advisory.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-4-api-pwa-advisory.md) |
| **Sprint 5** | Days 9–10 | Pitch Deck, 2-Min Video, Jury QA Rehearsal & Finals Readiness | Pitch Lead + Team | [SPRINT-5-pitch-deck-finals.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SPRINT-5-pitch-deck-finals.md) |

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
* **Never merge Pitch Lead:** Dedicated narrative and jury defense ownership is required throughout.
* **If 5 members:** Merge Backend and Frontend roles; descope Frontend to static cached map tiles if needed.
* **If 4 members:** Require pre-filtered Mandya parquet, GLO-30 tiles, and pre-cached patch Zarr ready on Day 0.
* **CODEOWNERS Enforcement:**
  - `/src/losses/` $\to$ `@ML Lead`
  - `/src/data/` $\to$ `@Data/GIS Lead`
  - `docs/pitch_deck.pdf` $\to$ `@Pitch Lead`

---

## 3. High-Level Architecture & Pipeline Flow

```
+-----------------------------------------------------------------------------------+
| 1. DATA SOURCES & INGESTION                                                       |
| - IMD 0.25° Daily Rainfall (Native LR coarse input, 1901-2024)                    |
| - CHIRPS 0.05° Daily Rainfall (Native HR target, 1981-present, free no login)      |
| - Copernicus GLO-30 DEM (30m elevation via CDSE S3, slope & aspect derived)       |
| - India-Geodata Parquet (319k LGD panchayats -> DuckDB Mandya 258 GPs filtered)   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. SPATIAL PRE-PROCESSING & REGISTRATION ASSERT                                   |
| - Grid Registration: Check IMD 6.5+0.25k vs CHIRPS +0.025° offset (catches 2.7km) |
| - Spatial Holdout: Mandya + 0.5° buffer strictly excluded from patch index        |
| - Patch Extraction: All-India 80x80 HR / 16x16 LR (stride 40, land fraction >=70%)|
|   Yield: ~200k-240k usable patches stored in Zarr/LMDB (8.5 GB compressed)        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. CORE 5x DOWNSCALING MODEL & LOSS CONSERVATION                                  |
| - Direct 5x Linear Downscaling (0.25° -> 0.05°, kernel=5)                         |
| - Terrain-Conditioned U-Net / CNN with DEM, Slope, Aspect                         |
| - Mass Conservation Loss: avg_pool2d(HR * cos(lat_hr), k=5) == LR                 |
| - 4 Splits: Train 2010-2020 | Val 2021 | Cal 2022 | Test 2023                     |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. POST-PROCESSING, CALIBRATION & UNCERTAINTY                                     |
| - Per-0.25°-Cell Quantile Mapping to IMD gauge gold (preserves intra-cell texture)|
| - Conformalized Quantile Regression (CQR): 90% empirical coverage on test 2023    |
| - Clean Zonal Polygon Aggregation: w_i = f_i * A_i (analytic spherical cell area) |
| - Bookkeeping Gates: Polygon area closed to 1e-3; interior cell partition sum ~ 1  |
| - Mandatory Hill-vs-Plains Error Stratification                                   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 5. DELIVERY & ADVISORY SYSTEM                                                     |
| - FastAPI backend: /api/forecast/{lgd_code} & /api/egramswaraj/mock (labeled mock)|
| - Mobile PWA: Offline viewing of cached forecasts with Airplane-Mode alert banner |
| - Bilingual Agro-Advisory: Stage-specific advice in Kannada & English (Ragi+Paddy)|
| - "Honest Ceiling": Panchayat-scale (~5.5km = 1 GP), never overclaimed sub-GP     |
+-----------------------------------------------------------------------------------+
```

---

## 4. Master Definition of Done (15 Measurable Items)

- [ ] `test_conservation_exact` passes: constant field coarsened equals input within $10^{-6}$ ($k=5$, $\mathbf{w}$ at HR centers).
- [ ] `test_conservation_detects_sum_bug` passes: naive summation fails unit test.
- [ ] `test_gis` passes: Mandya count 80–300 (258 GPs), valid $>98\%$, buffer $0.5^\circ$ excluded from train, `patch_index ∩ buffer == ∅`, land fraction $\ge 70\%$.
- [ ] `test_patch_geometry` passes: single-district patch bug caught, all-India $80\times 80$ HR / $16\times 16$ LR verified.
- [ ] `test_registration` passes: IMD vs. CHIRPS grid edge alignment verified, $2.7\text{ km}$ shift caught.
- [ ] `test_quantile_mapping` passes: mapping executed per $0.25^\circ$ LR cell, preserving intra-cell spatial variance.
- [ ] Downscale factor arithmetic verified: direct $5\times$ ($0.25^\circ \to 0.05^\circ$, kernel 5) used everywhere; grep for `3 blocks × 2x = 4x` returns 0.
- [ ] Panchayat-level output verified across 3 Mandya panchayats with polygon area closure error $\le 10^{-3}$ and interior cell partition $\approx 1.0$.
- [ ] Calibration curve artifact generated: `docs/calibration_curve.png`.
- [ ] CQR coverage report generated: `docs/cqr_coverage.png` ($90\% \pm 2\%$ on unseen 2023 test data).
- [ ] Hill-vs-plains stratified error breakdown generated: `docs/hill_vs_plains.png`.
- [ ] Uncertainty reported as calibrated range: *"Expected X mm, likely Y–Z mm ($90\%$ coverage)"*, never bare uncalibrated percentage.
- [ ] Offline PWA operational: displays *"Viewing cached forecast from {timestamp}"* with functional airplane-mode demonstration.
- [ ] Mock government integration `/api/egramswaraj/mock` explicitly labeled as mock prototype in code, Swagger, and headers.
- [ ] README and pitch deck strictly audited: all 11 forbidden claims absent via grep, data-attribution block present, honest ceiling clearly acknowledged.
