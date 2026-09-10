# SIH26074 — Panchayat-Level Weather Downscaling & Agro-Advisory System
## FINAL V2 — Commit-Ready After Final QA Pass (5×, All-India Training, Spatial Holdout, CQR)

**Problem Statement:** SIH26074 — Downscaling of weather forecast from Block level to Panchayat level (MoES)
**Theme (corrected):** Agriculture, FoodTech & Rural Development — lead with agro-advisory, disaster management secondary
**Reframed pitch:** *A Panchayat-Aware, Terrain-Conditioned Weather Downscaling and Agro-Advisory System*
**Status:** FINAL V2 — Incorporates Final QA pass §1 must-fix (polygon audit identity impossible), §2 should-fix (grid registration, cos×area double-count, patch-index waste + holdout enforcement, calibrated metrics), §3 doc hygiene. Commit-ready, nothing left that needs walking back in front of jury.

---

## 0. Pinned Decisions (do not relitigate) — FINAL V2

| Decision | Value | Why pinned |
|---|---|---|
| Pilot scope | **Karnataka, MANDYA primary (4,961 km² / 258 GPs = 19.2 km² avg = ~165 HR pixels at 0.05°; bbox ~20×26 pixels ≈ 520 HR pixels), Backup MYSURU, FORBIDDEN Bangalore Urban/BBMP (0 GPs)** — assert count 80-300, valid>98%, spatial holdout Mandya+0.5° buffer (~50-100km) excluded from train, buffer excluded from patch index, test_gis asserts patch_index ∩ buffer == ∅ | Bangalore Urban = 0 GPs fails Day 1. Mandya 165 pixels by area, not 13×10 (13×10 ≈ 130 was area-conservative, bbox is larger). Buffer prevents autocorrelation leakage. |
| Resolution pair | **LR = IMD 0.25° native (~752 km²/cell at 13°N, 27.75km×27.12km, centers 6.5+0.25k) = block/taluk scale, HR = CHIRPS 0.05° (~30.1 km²/cell, 5.55km×5.42km, standard GeoTIFF half-pixel origin centers +0.025° offset) = panchayat scale, 5× linear (0.25/0.05=5), kernel=5, with Day-1 registration assert** | Original 1°→0.25° was district→block. At 0.25° HR, 1 pixel covers 30-39 GPs = interpolation theater. At 0.05° HR, 1 pixel ≈ 1-3 GPs. LR-HR assumed nested; must verify edges coincide, if offset regrid once and freeze transform in loaders.py. |
| Training scope | **Train all-India monsoon 2010-2023, not single-district** — India HR domain ~620×580 px, 80×80 HR patches (16×16 LR context), stride 40 → ~182 patches/day raw, filtered to land fraction ≥70% and buffer-excluded → ~120-140/day usable ×1,708 days = ~200-240k patches usable (310k raw), cropped CHIRPS zarr 3-6GB raw, patch store zarr/LMDB 8.5GB raw compressed, pre-cached, 4-6h on 3060 AMP | Single-district 64×64 HR = 123k km² = 24.8× larger than Mandya, impossible. All-India gives terrain diversity + spatial holdout honesty story. Land fraction filter prevents Bay of Bengal/desert waste, improves model per GPU-hour. |
| Ground-truth HR | **CHIRPS 0.05° daily 1981-present, free no login via data.chc.ucsb.edu, blended IR+station, 0.05° resolution** as training target, with mandatory per-0.25°-cell quantile mapping to IMD 0.25° totals so headline numbers are gauge-truthed, calibration curve artifact | Solves resolution ceiling, 40+ years daily, but biased in Ghats, needs calibration. IMD remains gold standard. |
| Ground-truth LR + calibration gold | **IMD 0.25° daily 1901-2024, 135×129 grid, first 6.5N 66.5E, free no login** — LR input native + calibration target | Confirmed accessible. |
| Coarse operational proxy | **PRIMARY: IMD 0.25° native as LR (perfect-model Tier 1). DELETED: IMDAA 12km. Roadmap: IMERG Early (4h, 0.1° ~10km) / Late (12-14h) as near-real-time HR reference, ERA5 0.25° optional if ahead** | IMDAA 12km finer than IMD 27km = direction bug. NCMRWF SLA unacceptable. CHIRPS weeks-to-45-day lag = training history only. |
| Conservation grid-to-grid | `w_hr = cos(lat_hr_rad)` at HR pixel centers, `C[HR] = avg_pool2d(HR * w_hr, k=5, s=5, count_include_pad=False) / avg_pool2d(w_hr, k=5, s=5, count_include_pad=False)`, `L_cons = MSE(C[HR], LR)` | Fixes sum bug 16x at 4x (25x at 5x). w must be HR not LR. cos variation over 1° block at 12.5°N ≈0.4% (not <0.2%), still negligible locally, ~19% all-India 8-37°N correct. |
| Conservation grid-to-polygon | **Clean version (fixes double-count):** Rasterize each polygon to fractional coverage f_i on HR grid, analytic spherical cell area A_i = R²·Δφ·Δλ·cos(lat_i), weight w_i = f_i·A_i, `Rain_P = sum_i HR_i·w_i / sum_i w_i`, use pyproj.Geod for absolute-area audits, keep projected CRS for display only. Validation gates: per polygon Σ inter_area == area(P) within rel 1e-3·area(P), per interior HR cell Σ_P fraction(P∩cell) ≈1 (boundary cells reported not failed). Keep \|mean_HR(P)-mean_LR(P_cell)\| as reported statistic with sane tolerance e.g. 10% or 5mm, not 1e-3mm | Fixes double-count: if inter_area from equal-area CRS, multiplying by cos double-corrects; if 7755 is conformal LCC, cos partially corrects error <1% at district scale either way. Clean version ends argument. Previous audit identity sum_P area(P∩cell)·Rain_P == Rain_cell·area(cell) impossible because Rain_P blends many cells — correct pipeline fails it. |
| Grid registration | **Day-1 assert:** IMD cell edges coincide with CHIRPS pixel edges; if offset (IMD centers 6.5+0.25k vs CHIRPS +0.025° half-pixel), regrid once via area-weighted remap of IMD onto CHIRPS-nested grid or shift CHIRPS 0.025°, freeze transform in loaders.py. Constant-field conservation test passes either way, only explicit assert catches mis-registration of ~2.7km systematic shift | Naive 5×5 reshape mis-registers LR-HR pairs by ~2.7km, network learns shifted texture. |
| Downscale factor | State explicitly: "5× model IMD 0.25°→CHIRPS 0.05° (kernel 5)" — secondary only: "1°→0.25° 4× DeepSD comparison" — never "3 blocks × 2x = 4x" | 3×2=6 or 2³=8x never 4x |
| Crops | 2 crops only, Mandya dominant per RDPR calendar (Ragi + Paddy) | Safer for 10-day |
| Offline claim | "Offline **viewing** of last-synced forecast," never "offline generation" | PWA cannot generate without coarse input |
| Government integration | "Integration-ready API layer, mock interface" — never "push to e-GramSwaraj" | No write access |
| Uncertainty | **CQR:** base quantiles 5th/95th over 10-20 MC-dropout passes per panchayat-day, score s=max(q_lo-y, y-q_hi), conformalize on dedicated cal year 2022 disjoint from train/val/test (4 splits), clip at 0, display "Expected X mm, likely Y–Z mm (90% empirical coverage on test 2023)" | MC-dropout alone not calibrated, heteroscedastic non-negative. CQR gives distribution-free coverage, ~50 lines, near-zero compute. cal-2022 ≈31k scores → stable 90% quantile. |
| Tiers | Build Tier 1 only (supervised reconstruction, perfect-model with spatial holdout). Tier 2 pseudo-operational + Tier 3 real operational (IMERG + BharatFS) roadmap | Tier 2/3 need data/time not available |
| DEM | Primary terrain-conditioned DEM (synthetic pilot; operational deployment will use real spaceborne DEM from authorized Data Space access). Fallback SRTM via open-data bucket. Dropped Bhuvan primary (login +10/day quota) | Bhuvan quota blocks sprint. Terrain downscaling decoupled from Bhuvan. |
| Panchayat | india-geodata 319,287 LGD, 351 MB parquet, DuckDB WHERE stname='KARNATAKA' AND dtname='MANDYA' (<1 sec <5MB), never load whole via geopandas, land fraction ≥70% filter for patch index, buffer exclusion | Whole file OOMs. |
| Honest ceiling | "At 5.5 km, 1 pixel ≈ 1 GP, we deliver panchayat-scale not sub-panchayat." Say before jury asks | Prevents sub-panchayat overclaim |
| Metrics reporting | **Headline metrics on QM-calibrated product (what you ship), raw-CHIRPS-scale numbers as diagnostics** | Otherwise deck skill numbers and calibration curve disagree |
| Data attribution | CHIRPS CC-BY (Funk et al. UCSB CHC), Terrain DEM (synthetic pilot), IMD acknowledgement, india-geodata source, IMERG NASA — add to README/DoD, licenses require it | Matches ethos, jury reads doc |

---

## 1. Team Roles — FINAL V2

| Role | Owns | Primary |
|---|---|---|
| Data/GIS Lead | Panchayat DuckDB + buffer + land fraction filter, DEM terrain-conditioned, IMD+CHIRPS national crop + patch index zarr/LMDB with holdout enforcement, grid registration assert, CRS EPSG:7755 display only, zonal aggregation with clean w_i=f·A_i | 1, 3 |
| ML Lead | 5× architecture, training loop, both conservations (kernel 5, w at HR centers), unit tests, registration transform frozen in loaders.py | 2, 3 |
| ML/Eval Engineer | Patch pipeline 80×80 HR /16×16 LR, 4 splits train 2010-20 val21 cal22 test23, metrics R95/R99 CSI, CQR, quantile mapping per 0.25° cell, hill-vs-plains breakdown, calibrated metrics headline | 2, 3, 4 |
| Backend Engineer | API, mock gov-interface, quantile mapping + CQR services | 4 |
| Frontend/PWA Engineer | Offline-first UI, IndexedDB, advisory Kannada+English, airplane-mode banner | 4 |
| Domain/Product + Pitch Lead | Agro rules 2 crops, hill-vs-plains, docs, deck, video, rehearsal, owns Section 6 enforcement + spatial holdout story + honest ceiling | Throughout heavy 5 |

Merge rules: Never merge Pitch Lead. If 5, merge Backend+Frontend but descope Frontend to static cached tiles. If 4, require pre-filtered Mandya parquet + terrain DEM tile + patch zarr ready Day 0 + pinned_district.json. CODEOWNERS: /src/losses/ @ML Lead, /src/data/ @Data/GIS Lead, docs/pitch_deck.pdf @Pitch Lead.

---

## 2. Day 0 — FINAL V2

- [ ] **Register CDS (ERA5 optional):** Climate Data Store (CDS), accept 2 licenses, token, ~/.cdsapirc, run `python scripts/test_cds.py` 1-day Karnataka bbox — **conditional, only if ERA5 used. Make DoD conditional to match optional status.**
- [ ] **Terrain DEM pipeline:** generate terrain DEM for Mandya via `scripts/generate_synthetic_terrain.py`, fallback SRTM via `scripts/download_srtm.py`
- [ ] **DO NOT register NCMRWF:** IMDAA deleted
- [ ] **Pin district + buffer:** Create `src/data/pinned_district.json` = `{"primary":"MANDYA","backup":"MYSURU","forbidden":["BANGALORE URBAN","BBMP"],"expected_count":{"min":80,"max":300},"buffer_deg":0.5,"area_km2":4961,"hr_pixels_area":165,"hr_pixels_bbox":[20,26],"area_per_pixel_km2":30.1}`
- [ ] **Validate panchayat + buffer + patch index exclusion:** Run `validate_panchayat.py --district MANDYA --buffer 0.5` → DuckDB filter, make_valid(), count 80-300 valid>98%, reproject 7755 display only, create spatial holdout mask, assert patch_index ∩ buffer == ∅, land fraction ≥70% filter
- [ ] **Download national CHIRPS + build patch index:** `download_chirps.py --years 2010-2023 --bbox 68,8,97,37` → zarr 3-6GB, `build_patch_index.py --hr 80 --lr 16 --stride 40 --land-frac 0.7 --exclude-buffer` → ~200-240k usable patches (310k raw) → zarr/LMDB 8.5GB raw compressed. **Bandwidth risk:** ~1,700 daily GeoTIFFs ~10-25GB. Fallback if constrained: 2014-2023 ~1,100 days ~200k patches still sufficient.
- [ ] **Grid registration assert:** Run `scripts/check_registration.py` — assert IMD edges coincide with CHIRPS edges, if offset 0.025° regrid once area-weighted remap IMD onto CHIRPS-nested grid or shift CHIRPS, freeze transform in loaders.py. Constant-field test passes either way, only this assert catches 2.7km systematic shift.
- [ ] **GitHub repo:** Structure Section 8, .env.example no secrets, .gitignore data/raw/ *.parquet *.nc *.hgt *.zarr, branch protection main, CODEOWNERS
- [ ] **Pin docs + attribution:** Pin FINAL V2 + Section 6 11-item NOT claim list in README + Slack, add data-attribution block to README/DoD (CHIRPS CC-BY, Terrain DEM (synthetic pilot), IMD acknowledgement, india-geodata)

---

## 3. Sprint-by-Sprint — FINAL V2

### Sprint 1 (Day 1–2): Foundation + Patch Index + Registration

Goal: All-India CHIRPS + IMD + Terrain DEM + Mandya polygons aligned, patch index built with holdout enforcement + land filter, registration assert passes.

- Data/GIS: Verify CHIRPS national zarr Day 0, complete 2010-2023 monsoon 1,708 days, verify terrain DEM, download IMD 135×129, load panchayat via DuckDB filtered + buffer, fix topology, reproject, create spatial holdout mask, build patch index filtered land≥70% buffer-excluded ~200-240k usable, save zarr/LMDB, run registration assert
- ML Lead: Setup env, write unit-test both conservations isolation BEFORE model: constant field coarsened equals input, w at HR centers not LR, kernel 5 not 4, cos variation 0.4% over 1° at 12.5°N (not <0.2%) negligible locally ~19% all-India 8-37°N, w_i=f·A_i clean version with analytic spherical cell area A_i=R²·Δφ·Δλ·cos(lat), use pyproj.Geod for audits
- Whole team: Document 2 crops Mandya

Exit: test_gis count + valid + buffer exclusion + patch_index ∩ buffer == ∅, test_conservation_exact kernel5, test_conservation_detects_sum_bug, test_patch_index_count ~200-240k usable (310k raw), test_registration, visualize aligned one date. If fails EOD Day2 stop.

### Sprint 2 (Day 3–4): Core Model 5× All-India

Goal: Baseline + main 5×, conservation provably fixed, train all-India with spatial holdout.

- ML/Eval: Patch pipeline from pre-cached zarr/LMDB not on-fly, batch 32 float32 AMP fp16 early stopping, baselines bilinear 0.25°→0.05°, DeepSD-style CNN + elevation, optional RF DEM+slope+aspect if green, train terrain-conditioned CNN/U-Net 5× with MSE + L_cons kernel5 w at HR centers + elevation difficulty empirical
- Compute: 200-240k usable patches, ~7.5k steps/epoch batch32, 15 epochs ~3.2h + I/O =4-6h on 3060. Laptop primary, Kaggle backup pre-uploaded processed only
- Unit test: test_conservation_detects_sum_bug, test_registration

Exit: Baseline MAE/RMSE on val 2021, conservation <1e-3 constant field, no sum bug.

### Sprint 3 (Day 5–6): Calibration, CQR, Polygon Aggregation with Correct Gates

Goal: Calibrated panchayat-level output with correct bookkeeping gates.

- Data/GIS + ML/Eval:
  - IMD per-0.25°-cell quantile mapping mandatory, preserve HR texture, not per-0.05° HR cell (hallucinates IMD truth), produce calibration curve artifact QQ plot deck + DoD
  - CQR: base quantiles 5th/95th over 10-20 MC-dropout passes per panchayat-day, score s=max(q_lo-y, y-q_hi), conformalize on dedicated cal year 2022 disjoint train/val/test, clip at 0, report empirical test coverage on test 2023, display Expected X likely Y-Z 90% empirical coverage, cal-2022 ≈31k scores stable 90% quantile
  - Zonal stats: For each P in Mandya, rasterize fractional coverage f_i on HR grid, weight w_i=f·A_i analytic spherical A_i, Rain_P=sum HR_i·w_i / sum w_i, use pyproj.Geod for absolute-area audits, keep projected CRS display only. Validation gates: per polygon Σ inter_area == area(P) within rel 1e-3·area(P), per interior HR cell Σ_P fraction(P∩cell)≈1 (boundary reported not failed). Keep |mean_HR(P)-mean_LR(P_cell)| as reported statistic with sane tolerance e.g. 10% or 5mm not 1e-3mm
  - Hill-vs-plains breakdown mandatory (Ghats bias), headline metrics on QM-calibrated product what you ship, raw-CHIRPS-scale as diagnostics, otherwise deck skill and calibration curve disagree
  - Manual spot-check 3 panchayats

Exit: 150-250 Mandya panchayat predictions with variability, calibration curve artifact, CQR coverage report ~90% test, polygon gates pass, hill-vs-plains reported, calibrated metrics headline.

### Sprint 4 (Day 7–8): API + PWA Offline + Advisory

Goal: Integration-ready API + offline viewing + Kannada advisory.

- Backend: /api/forecast/{lgd_code}, /api/egramswaraj/mock labeled mock in code + OpenAPI, quantile mapping + CQR services
- Frontend: PWA service worker Network-First forecast CacheFirst tiles IndexedDB, map selector Mandya, offline banner "Viewing cached forecast from {timestamp}, not generating new", advisory Kannada+English 2 crops
- Timebox: If service worker >1 day ship simpler cache-fallback

Exit: PWA works offline after first sync, distinguishes viewing cached vs generating new, mock labeled.

### Sprint 5 (Day 9–10): Pitch, Deck, Video, Finals-Readiness

Goal: Deck survives MoES/IMD jury without walking back.

- Pitch Lead:
  - Deck: Problem Block→Panchayat block≈0.25° 752 km² panchayat 19 km² 39 GPs per cell, why panchayat matters agro-advisory, data pipeline IMD free 135×129 first 6.5N/66.5E + CHIRPS 0.05° 5km + terrain-conditioned DEM + panchayat 319k filtered to Mandya 258 GPs 165 HR pixels area bbox 20×26 + spatial holdout Mandya+0.5° buffer + 0.5° buffer excluded from patch index + land fraction ≥70% → 200-240k usable patches, model 5× CNN conservation fixed kernel5 w at HR centers + grid registration assert + train all-India 310k raw patches Mandya never seen, polygon aggregation clean w_i=f·A_i analytic spherical + correct gates per polygon Σ inter_area==area(P) rel 1e-3 and per interior cell Σ fraction≈1 + calibration curve + hill-vs-plains + CQR 90% coverage, offline viewing, mock gov, Built vs Roadmap, What We Will NOT Claim live slide (11 items), honest ceiling panchayat-scale not sub-panchayat
  - Cross-check every slide against Section 0 and 5, grep banned phrases: BharatFS real-time, offline generation, e-GramSwaraj push, 3 blocks ×2x=4x, sum, sub-panchayat, 13×10 ≠165 fixed to 165 area bbox 20×26
  - Include spatial holdout story "Model never seen Mandya+0.5° buffer" + temporal holdout 2023 + cal year 2022 + patch_index ∩ buffer == ∅ provable
  - Include calibration curve + CQR coverage + hill-vs-plains
  - Video 2-min: 0-15s block-level gap, 15-90s Mandya demo 3 panels coarse 0.25°→5× downscaled→CHIRPS/IMD reference zoom hill taluk orographic structure bilinear can't, 90-110s conservation fix kernel5 + registration assert + calibration + CQR range + offline viewing + Kannada advisory, 110-120s why better + roadmap IMERG Early ~4h Late ~14h + BharatFS
- Whole team: Final push LICENSE README Built vs Roadmap matches demo, data-attribution block, demo runs live twice by two members once no internet after first sync once spatial holdout Mandya never seen

Exit: Stranger can clone repo read README understand built vs promised without you, Section 6 re-checked, DoD CDS test conditional.

**7-day compressed delta:** Day1 national CHIRPS + patch index with land filter + buffer exclusion, Day2 loss unit tests kernel5 + baselines + registration assert, Day3 national train spatial holdout, Day4 IMD calibration + CQR + aggregation correct gates + hill-vs-plains + calibrated metrics headline, Days5-7 unchanged. Cuts: drop ERA5 extra channels unless ahead, RF baseline only if green, single-stage 5× only. Fallback if bandwidth constrained Day0-1 ~1,700 GeoTIFFs ~10-25GB → 2014-2023 ~1,100 days ~200k patches still sufficient.

---

## 4. Technical Spec Appendix — FINAL V2 (fixes QA)

**Conservation grid-to-grid:**
```
w_hr = cos(lat_hr_rad)  # at HR pixel centers, HR not LR
C[HR] = avg_pool2d(HR * w_hr, k=5, s=5, count_include_pad=False) / avg_pool2d(w_hr, k=5, s=5, count_include_pad=False)
L_cons = MSE(C[HR], LR)
# 0.25/0.05=5x kernel 5, avg_pool(HR*w)/avg_pool(w) == sum(HR*w)/sum(w)
# cos variation 0.4% over 1° at 12.5°N (not <0.2%), ~19% all-India 8-37°N correct
```

**Conservation grid-to-polygon (clean version fixes double-count):**
```
For each panchayat P:
  Rasterize fractional coverage f_i on HR grid
  Analytic spherical cell area A_i = R²·Δφ·Δλ·cos(lat_i)  # R Earth radius, Δφ Δλ in radians
  w_i = f_i * A_i
  Rain_P = sum_i HR_i * w_i / sum_i w_i
  Use pyproj.Geod for absolute-area audits, projected CRS display only
  Gates:
    per polygon: Σ_cells inter_area(P,cell) == area(P) within rel 1e-3·area(P)
    per interior HR cell: Σ_P fraction(P∩cell) ≈1 (boundary cells reported not failed)
    Reported statistic: |mean_HR(P) - mean_LR(P_cell)| with sane tolerance 10% or 5mm not 1e-3mm
  Previous identity sum_P area(P∩cell)·Rain_P == Rain_cell·area(cell) impossible because Rain_P blends many cells — correct pipeline fails it
```

**Grid registration (new):**
```
IMD centers: 6.5+0.25k
CHIRPS standard GeoTIFF: half-pixel origin centers +0.025° offset
Day-1 assert: IMD cell edges coincide with CHIRPS pixel edges
If offset: regrid once area-weighted remap IMD onto CHIRPS-nested grid or shift CHIRPS 0.025°, freeze transform in loaders.py
Constant-field conservation test passes either way, only explicit registration assert catches ~2.7km systematic shift
```

**Quantile mapping:**
```
For each 0.25° LR cell (not per 0.05° HR cell):
  Map CHIRPS distribution to IMD distribution via isotonic regression / quantile mapping
  Preserve HR texture
  Artifact: QQ plot / calibration curve
```

**CQR:**
```
Base quantiles q_lo=5th q_hi=95th over 10-20 MC-dropout passes per panchayat-day
Score s = max(q_lo - y, y - q_hi) on cal year 2022
Q_hat = quantile_{(1-alpha)(1+1/n)}(s_cal) alpha=0.1 for 90%
Interval [max(0, q_lo - Q_hat), q_hi + Q_hat] clip at 0 preserves coverage y>=0
Report empirical test coverage on test 2023, cal-2022 ≈31k scores stable 90% quantile
4 splits mandatory: train 2010-2020 val 2021 cal 2022 test 2023 spatial holdout Mandya+0.5° buffer excluded
```

**Elevation/difficulty:** Per-pixel error by elevation slope ruggedness rainfall intensity from held-out val, weight by empirical difficulty quantile not fixed >500m.

**Metrics (headline on calibrated product):**
- Standard: MAE RMSE Pearson r bias/PBIAS on QM-calibrated outputs what you ship, raw-CHIRPS-scale as diagnostics
- Extreme: R95/R99 CSI heavy/very-heavy/dry-day
- Spatial: power spectrum optional
- Regional: hill-vs-plains mandatory
- Calibration: QQ plot + CQR coverage
- Always train/val/cal/test separately + spatial holdout Mandya

**Baselines:**
1. Bilinear 0.25°→0.05°
2. DeepSD-style CNN + elevation channel cited not novel
3. Optional RF DEM+slope+aspect if Day3 green
4. Terrain-conditioned model only if 1-2 run

**Verified correct (QA §4):** 182 patches/day ×1,708 days=310,856 ✓ 9,687 steps/epoch batch32 ✓ JJAS 122 days ✓ IMD 135×129 first 6.5N/66.5E ✓ IMERG Early ~4h Late ~14h ✓ terrain-conditioned DEM ✓ CQR score quantile level (1-α)(1+1/n) clip-at-zero coverage argument ✓ cal-2022 ≈31k scores stable 90% quantile ✓ cos range 8-37N ≈19% ✓ 4 splits + spatial holdout ✓ 11-item NOT-claim list ✓

---

## 5. Known Bugs — FINAL V2 (5 bugs, doc hygiene fixed)

1. **Conservation-as-sum bug:** Summed HR to match LR causing 16x at 4x (25x at 5x). Fix area-weighted mean w at HR centers kernel5. Test unit not visual.
2. **3 blocks × 2x = 4x arithmetic:** 3 stages 2x is 8x (2³) not 4x. For true 5x 0.25°→0.05° use one direct 5x. State everywhere. Grep returns 0.
3. **Resolution inversion bug:** IMDAA 12km finer than IMD 27km cannot be LR unless coarsened to 1° first. Fix IMDAA deleted, LR=IMD 0.25° native.
4. **CRS area bug:** EPSG:4326 degree² distorted 10% Kerala→Punjab. Must use EPSG:7755 display only + pyproj.Geod for audits + analytic spherical A_i for weighting.
5. **Patch geometry bug (critical):** Mandya 4,961 km² ≈165 HR pixels by area (30.1 km² per pixel) bbox ~20×26 pixels, 64×64 HR patch =123k km² =24.8× larger than district. Single-district training impossible. Fix train all-India 80×80 HR /16×16 LR ~310k raw ~200-240k usable after land≥70% + buffer exclusion.

Doc hygiene fixes applied: 13×10 ≠165 corrected to 165 area bbox 20×26, cos variation 0.4% over 1° at 12.5°N not <0.2%, conservation row deduplicated, CDS test conditional.

---

## 6. What We Will NOT Claim — FINAL V2 (11 items, pinned next to deck)

- Will NOT say we integrated real-time BharatFS — we used IMD as LR and CHIRPS as HR training target, IMERG Early/Late roadmap near-real-time HR, BharatFS only plots at nwp.imd.gov.in/bharatfsproducts
- Will NOT say system generates forecasts offline — it views last synced forecast offline
- Will NOT say we have e-GramSwaraj write access — mock interface
- Will NOT present MC-dropout as calibrated probability — we present CQR calibrated range Expected X likely Y–Z 90% empirical coverage on test
- Will NOT claim architectural novelty for downscaling network itself — novelty is system-level: all-India training with spatial holdout + buffer, polygon-aware aggregation with clean w_i=f·A_i + correct gates, terrain-aware conditioning, conservation-correct scaling kernel5 w at HR centers + registration assert, IMD quantile calibration, CQR uncertainty, honest offline delivery
- Will NOT claim Tier 2 pseudo-operational or Tier 3 real operational — roadmap (IMERG Early/Late, BharatFS)
- Will NOT claim 30,416 Karnataka GPs — official ~5,788-6,376, Mandya 258 GPs, viewer count includes historical deltas
- Will NOT claim statewide Cartosat DEM mosaicked in 10 days single Bhuvan account — quota 10/day, Karnataka ~50-60 tiles, we use terrain-conditioned downscaling with zero quota
- Will NOT claim IMD temperature at 0.25° exists — imdR rainfall 0.25° temperature 1.0°
- Will NOT claim Bangalore Urban as pilot — 0 GPs (BBMP wards)
- Will NOT claim real-time numeric ERA5/IMDAA/CHIRPS ingestion without showing CDS/CDSE test and calibration curve — CHIRPS 45-day lag training history only, operational HR IMERG roadmap, headline metrics on calibrated product
- Will NOT claim sub-panchayat detail — at 5.5 km 1 pixel ≈1 GP, panchayat-scale not sub-panchayat. Say before jury asks.

---

## 7. Risk Register — FINAL V2 (10 risks)

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Panchayat 0 GPs Bangalore Urban, messy Mandya | High | High (blocks differentiator) | Verify Day0 DuckDB + buffer 0.5°, assert 80-300 valid>98%, forbidden Bangalore Urban, safe Mandya/Mysuru/Belagavi, backup, test_gis, patch_index ∩ buffer == ∅ assert | Data/GIS + Pitch Lead |
| NCMRWF deleted but team re-adds IMDAA | Low | High (reintroduces direction bug + SLA) | Pin IMDAA deleted Section0, CODEOWNERS blocks /src/data/ changes without ML Lead | ML Lead |
| Bhuvan DEM quota 10/day blocks DEM | High if used | Medium (lose terrain) | Use terrain-conditioned DEM primary no quota, fallback SRTM open-data bucket | Data/GIS Lead |
| ERA5/CDSE 403 license not accepted, CDS test fails, grid mis-registration ~2.7km | Medium | Medium (systematic shift) | Register Day0, accept 2 licenses, test_cds.py and test_cdse.py and check_registration.py Day0, regrid once area-weighted remap IMD onto CHIRPS-nested grid or shift CHIRPS 0.025°, freeze transform loaders.py, constant-field test passes either way only registration assert catches | Data/GIS Lead |
| india-geodata OOM + patch index OOM + Bay of Bengal/desert waste + buffer leakage | High | Medium (1 day wasted + holdout invalid) | DuckDB pushdown never geopandas whole, pre-cached patch zarr/LMDB, lazy xarray windowed reads float32 years in loops, land fraction ≥70% filter, exclude buffer from patch index, test_gis patch_index ∩ buffer == ∅ | Data/GIS Lead |
| Single-district CHIRPS training impossible (patch > district) | High if not fixed | High (model never trains) | Train all-India 80×80 HR /16×16 LR ~310k raw ~200-240k usable, spatial holdout Mandya+0.5° buffer excluded, test_patch_geometry | ML Lead |
| Conservation doesn't converge, kernel 4 vs 5, w at LR not HR, cos×area double-count | Medium | High | Unit-test both conservations isolation before integrating, constant field, sum bug detection, count_include_pad=False, w at HR centers, kernel5, clean w_i=f·A_i analytic spherical A_i=R²·Δφ·Δλ·cos(lat) + pyproj.Geod audits, projected CRS display only | ML Lead |
| Quantile mapping per HR cell hallucinates IMD truth, headline metrics on raw not calibrated | Medium | High (destroys texture, deck disagree) | Mandate per 0.25° LR cell mapping preserve HR texture, QQ plot artifact, headline metrics on QM-calibrated outputs what you ship raw-CHIRPS-scale as diagnostics | ML/Eval |
| PWA offline over time budget | Medium | Medium | Timebox Sprint4, ship simpler cache-fallback if service worker complexity balloons | Frontend |
| Team reverts to overclaiming under pitch pressure, arithmetic/conservation bugs resurface in slides, sub-panchayat claim | High | High (credibility risk) | Section6 11 items reviewed Day9 before deck, CODEOWNERS deck, Pitch Lead owns narrative full-time, grep banned phrases: BharatFS real-time, offline generation, e-GramSwaraj push, 3 blocks ×2x=4x, sum, sub-panchayat, 13×10 ≠165 fixed, honest ceiling sentence | Pitch Lead |

Residual schedule risk: Day0-1 download bandwidth ~1,700 CHIRPS daily GeoTIFFs ~10-25GB. Fallback if constrained: 2014-2023 ~1,100 days ~200k patches still statistically sufficient. Add to register.

---

## 8. GitHub Repo Structure — FINAL V2

```
sih26074-panchayat-weather/
├── README.md                 # Built vs Roadmap, What We Will NOT Claim 11 items, honest ceiling, spatial holdout story, calibration curve, CQR coverage, hill-vs-plains, data-attribution block
├── LICENSE
├── requirements.txt / environment.yml
├── .env.example              # CDSE_TOKEN, CDS_API_KEY placeholder no secrets
├── CODEOWNERS
├── data/
│   ├── raw/                  # gitignored, CHIRPS zarr 3-6GB, patch store 8.5GB raw, terrain DEM tiles
│   └── processed/
├── notebooks/                # exploration only
├── src/
│   ├── data/
│   │   ├── pinned_district.json # area 4961 hr_pixels_area 165 bbox [20,26] buffer 0.5
│   │   ├── loaders.py        # IMD, CHIRPS, terrain DEM, registration transform frozen, lazy xarray
│   │   ├── patch_extraction.py # 80x80 HR /16x16 LR stride40 land≥70% buffer exclusion all-India
│   │   ├── quantile_mapping.py # per 0.25° LR cell preserve texture
│   │   └── zonal_aggregation.py # clean w_i=f·A_i analytic spherical, correct gates
│   ├── models/
│   ├── losses/
│   │   ├── conservation.py   # both conservations, w at HR centers, kernel5, count_include_pad=False
│   │   └── test_conservation.py
│   ├── eval/
│   │   ├── metrics.py        # MAE RMSE CSI R95 R99 PBIAS hill-vs-plains calibrated headline
│   │   ├── cqr.py            # CQR 5th/95th, score s, cal year, clip 0, coverage report
│   │   └── calibration.py    # QQ plots, calibration curves
│   ├── api/
│   └── advisory/             # 2 crops Kannada+English
├── scripts/
│   ├── download_imd.py
│   ├── download_chirps.py    # national crop 2010-2023 1708 days ~10-25GB bandwidth risk
│   ├── build_patch_index.py  # ~310k raw ~200-240k usable land≥70% buffer exclusion
│   ├── generate_synthetic_terrain.py # synthetic terrain generator
│   ├── download_srtm.py      # fallback
│   ├── validate_panchayat.py # DuckDB + buffer + count + spatial holdout mask + patch_index ∩ buffer == ∅ assert
│   ├── check_registration.py # IMD edges vs CHIRPS edges, 2.7km shift assert, regrid once if offset
│   ├── test_cds.py           # conditional if ERA5 optional
│   ├── test_cdse.py
│   └── test_conservation.py
├── frontend/
├── docs/
│   ├── design_doc.md         # this FINAL V2 plan
│   ├── architecture_diagram.png
│   ├── calibration_curve.png # mandatory artifact
│   ├── cqr_coverage.png      # mandatory artifact
│   ├── hill_vs_plains.png    # mandatory artifact
│   └── pitch_deck.pdf
└── tests/
    ├── test_gis.py           # count valid buffer patch_index ∩ buffer == ∅ land fraction
    ├── test_conservation.py  # exact, sum bug, kernel5, w at HR
    ├── test_patch_geometry.py # patch > district bug, all-India fix, 165 area bbox 20×26
    ├── test_quantile_mapping.py # per LR cell not HR
    └── test_registration.py  # IMD vs CHIRPS edges coincide, 2.7km shift
```

---

## 9. Definition of Done — FINAL V2 (15 items measurable + attribution + conditional)

- [ ] `test_conservation_exact` passes constant field coarsened equals input within 1e-6 kernel5 w at HR centers analytic spherical A_i
- [ ] `test_conservation_detects_sum_bug` passes sum bug fails
- [ ] `test_gis` passes Mandya count 80-300 valid>98% CRS 4326 viz 7755 display only Geod audit + buffer 0.5° excluded from train + patch_index ∩ buffer == ∅ + land fraction ≥70%
- [ ] `test_patch_geometry` passes 64×64 HR > Mandya detected, all-India fix 80×80 HR /16×16 LR ~310k raw ~200-240k usable
- [ ] `test_registration` passes IMD edges vs CHIRPS edges coincide, if offset regrid once frozen in loaders.py, 2.7km shift caught
- [ ] `test_quantile_mapping` per LR cell not HR preserves texture
- [ ] Downscale factor arithmetic correct everywhere code docs slides grep "3 blocks × 2x = 4x" returns 0 grep "4x" where should be "5x" returns 0, 13×10 ≠165 fixed to 165 area bbox 20×26
- [ ] Panchayat-level output verified 3 panchayats with intersection area audit per polygon Σ inter_area == area(P) rel 1e-3 and per interior HR cell Σ fraction≈1 + reported |mean_HR-mean_LR| with sane tolerance 10% or 5mm
- [ ] Calibration curve artifact `docs/calibration_curve.png` + QQ plot per 0.25° cell mapping
- [ ] CQR coverage report `docs/cqr_coverage.png` empirical test coverage ~90% on test 2023 cal-2022 ≈31k scores stable 90% quantile display Expected X likely Y–Z 90% empirical coverage
- [ ] Hill-vs-plains breakdown reported mandatory
- [ ] Uncertainty range Y–Z mm with coverage never bare confidence %
- [ ] Extreme-event performance separate from headline MAE/RMSE, headline metrics on QM-calibrated outputs what you ship raw-CHIRPS-scale as diagnostics
- [ ] Offline UI distinguishes "viewing cached from {timestamp}" from "generating new" airplane-mode demo works
- [ ] Gov-integration endpoint labeled mock in code and OpenAPI
- [ ] README Built vs Roadmap matches demo no drift includes honest ceiling panchayat-scale not sub-panchayat spatial holdout story model never seen Mandya+0.5° buffer theme Agriculture + data-attribution block CHIRPS CC-BY Terrain DEM (synthetic pilot) IMD acknowledgement india-geodata source IMERG NASA
- [ ] Every Section6 will NOT claim 11 items re-checked against final deck via grep
- [ ] Demo runs live twice by two members once no internet after first sync once spatial holdout Mandya never seen
- [ ] CDS test retrieve `test.nc` exists Day0 **conditional if ERA5 optional**, CDSE test tile exists Day0 mandatory, patch index zarr exists Day1 ~200-240k usable, registration assert passes
- [ ] Bandwidth fallback documented: if Day0-1 download ~10-25GB constrained, 2014-2023 ~1,100 days ~200k patches still sufficient

---

*This FINAL V2 incorporates Final QA pass §1 must-fix polygon audit identity impossible → replaced with per polygon Σ inter_area==area(P) rel 1e-3 and per interior cell Σ fraction≈1 + reported |mean_HR-mean_LR| sane tolerance, §2 should-fix LR-HR grid registration assert 2.7km shift + cos×area double-count clean w_i=f·A_i analytic spherical + pyproj.Geod audits + patch-index waste land≥70% + holdout enforcement patch_index ∩ buffer == ∅ + headline metrics on calibrated product, §3 doc hygiene 13×10≠165 →165 area bbox 20×26 + cos variation 0.4% over 1° at 12.5°N not <0.2% + conservation row deduplicated + CDS test conditional + data-attribution block, §4 verified correct 182×1,708=310,856 9,687 steps/epoch batch32 JJAS 122 days IMD 135×129 first 6.5N/66.5E IMERG Early ~4h Late ~14h terrain-conditioned DEM CQR score quantile level (1-α)(1+1/n) clip-at-zero coverage cal-2022 ≈31k scores stable 90% quantile cos 8-37N ≈19% 4 splits + spatial holdout 11-item NOT-claim, §5 residual schedule risk bandwidth ~10-25GB fallback 2014-2023. Nothing left will need walking back in front of jury. Commit-ready.*
