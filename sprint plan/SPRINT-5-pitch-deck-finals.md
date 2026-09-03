# Sprint 5 — Days 9–10: Pitch Deck, Video Demonstration, QA Audit & Finals Readiness

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith P Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Days 9–10  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Produce an unassailable pitch deck, a crisp 2-minute demonstration video, and complete documentation that withstands rigorous scrutiny from MoES/IMD atmospheric scientists and jury members without walking back a single claim. Rehearse live demos twice (including airplane-mode offline verification) and audit code and presentation materials against all 11 forbidden claims.

---

## 2. Team Responsibilities & Ownership

| Role | Sprint 5 Deliverables |
|---|---|
| **Domain/Product + Pitch Lead** | Author pitch deck (`docs/pitch_deck.pdf`), write and record the 2-minute demo video script, lead jury rehearsal, enforce the 11 "What We Will NOT Claim" boundaries, ensure honest framing of the spatial holdout and resolution ceiling. |
| **Data/GIS Lead** | Generate high-resolution map graphics for slides (coarse vs. downscaled vs. reference; Mandya 258 GP boundaries; orographic rainfall profiles). |
| **ML/Eval Engineer** | Prepare calibration and uncertainty figures (`docs/calibration_curve.png`, `docs/cqr_coverage.png`, `docs/hill_vs_plains.png`), format headline comparison tables. |
| **Backend & Frontend Leads** | Polish live demo environment, ensure one-command local reproduction, execute the offline airplane-mode live test. |
| **Whole Team** | Code cleanup, final README and LICENSE review, data attribution audit, 2 live dry-runs. |

---

## 3. Pinned Decisions & Presentation Guardrails

### A. The "Honest Ceiling" Narrative
* State upfront before the jury asks:  
  **"At 5.5 km resolution, 1 pixel is approximately equal to 1 Gram Panchayat (~19.2 km² in Mandya). We deliver panchayat-scale forecasts, not sub-panchayat micro-climates."**
* Proactively stating this disarms the single most common technical challenge from meteorologists.

### B. The Spatial Holdout Story
* Mandya district plus a $0.5^\circ$ surrounding buffer (~50–100 km) was strictly excluded from training.
* The model learned terrain-conditioned downscaling entirely from the diverse topography of the rest of India and generalizes to Mandya zero-shot.
* Proven by unit test assertion: `patch_index ∩ buffer == ∅`.

### C. Resolution Framing
* LR is **IMD 0.25° native** (~752 km²/cell) = block/taluk scale.
* HR is **CHIRPS 0.05°** (~30.1 km²/cell) = panchayat scale.
* At 0.25°, 1 coarse cell covers 30 to 39 Mandya GPs (relying on 0.25° is "interpolation theater").
* Direct $5\times$ linear downscaling ($0.25 / 0.05 = 5$, kernel=5). Never claim $4\times$ or "3 blocks × 2x = 4x".

### D. What We Will NOT Claim (11 Items Pinned Live)
1. Will **NOT** claim real-time BharatFS integration (we used IMD as LR and CHIRPS as HR; BharatFS is on the operational roadmap).
2. Will **NOT** say the system generates forecasts offline (it provides offline *viewing* of last-synced forecasts).
3. Will **NOT** claim write access or push to e-GramSwaraj (mock API interface only).
4. Will **NOT** present raw MC-dropout as calibrated probabilities (we present CQR 90% empirical coverage intervals: *"Expected X mm, likely Y–Z mm"*).
5. Will **NOT** claim architectural novelty for the downscaling neural network itself (novelty is system-level: all-India training with spatial holdout, clean polygon aggregation with correct gates, terrain conditioning, mass-conserving kernel=5 pooling, grid registration assert, quantile calibration, and CQR uncertainty).
6. Will **NOT** claim Tier 2 or Tier 3 operational readiness (these are roadmap items using IMERG Early/Late and BharatFS).
7. Will **NOT** claim 30,416 Karnataka GPs (official active count is ~5,788–6,376; Mandya has 258 GPs).
8. Will **NOT** claim statewide Cartosat DEM mosaicking in 10 days on Bhuvan (quota is 10 tiles/day; we used GLO-30 via CDSE S3 with zero quota).
9. Will **NOT** claim IMD temperature exists at 0.25° (IMD rainfall is 0.25°; temperature is 1.0°).
10. Will **NOT** claim Bangalore Urban as a pilot district (Bangalore Urban has 0 GPs under BBMP wards).
11. Will **NOT** claim sub-panchayat resolution detail.

### E. GFS Operational Boundary & Scientific Honesty
* **GFS / NCUM Claim Rule:** Describe GFS 0.25° and NCUM strictly as **"Architecture Format-Compatible Roadmap"**. **Never claim "Tested on GFS"** unless real numerical cycles were evaluated.
* **Temporal Cutoff Honesty:** Transparently state that IMD represents 08:30 IST (03:00 UTC) accumulation; explain that quantile mapping aligns the climatological cumulative distribution function (CDF), while intra-day convective timing (15:00–19:00 IST) is an inherent boundary condition of daily gridded products.

---

## 4. Pitch Deck Structure & Content (`docs/pitch_deck.pdf`)

1. **Slide 1: Title & Framing**  
   *A Panchayat-Aware, Terrain-Conditioned Weather Downscaling and Agro-Advisory System* (Theme: Agriculture, FoodTech & Rural Development).
2. **Slide 2: The Core Problem: The 0.25° Resolution Gap**  
   IMD 0.25° block forecasts (~752 km²) average out orographic variation across 30–39 panchayats. Showing why a farmer in a rain-shadow valley gets the same forecast as a farmer on a windward ridge.
3. **Slide 3: Data Pipeline & Spatial Integrity**  
   All-India monsoon training (2010–2023, 1,708 days, 310k raw patches $\to$ 200k–240k usable after $\ge 70\%$ land filter). GLO-30 DEM via CDSE S3. Mandya $+ 0.5^\circ$ buffer strictly held out (`patch_index ∩ buffer == ∅`).
4. **Slide 4: Physics & Mathematics: Conservation & Registration**  
   Mass conservation pooling with kernel=5 and cosine weighting at HR pixel centers. Elimination of the $2.7\text{ km}$ grid registration shift. Elimination of the cos×area double-counting bug.
5. **Slide 5: Calibration, Uncertainty & Honest Metrics (No Dry-Day Illusion)**  
   Per-$0.25^\circ$-cell quantile mapping to IMD gauge truth (QQ plot artifact). Conformalized Quantile Regression (CQR) with 90% empirical coverage on unseen 2023 test data. **Reporting Wet-Day MAE ($>2.5\text{ mm}$) and Extreme Event CSI (R95/R99)** alongside aggregate MAE to prove skill on actual rainfall events.
6. **Slide 6: Clean Polygon Aggregation & Bookkeeping Gates**  
   Exact polygon fractional area weighting with analytic spherical cell areas ($A_i = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\text{lat})$) on full-precision geodata. Proof that polygon boundary areas close within relative error $10^{-3}$.
7. **Slide 7: Product Experience: PWA & Bilingual Agro-Advisory**  
   Offline-first mobile PWA (<400 KB TopoJSON) with airplane-mode detection banner. Stage-specific farming advisories in Kannada and English for Ragi and Paddy.
8. **Slide 8: System Architecture & Verification Gates**  
   End-to-end pipeline diagram. CI automated test suite passing all 15 Definition of Done criteria. Single-command reproducible execution via `python scripts/run_pipeline.py`.
9. **Slide 9: What We Built vs. Future Roadmap**  
   Tier 1 (Built): Perfect-model supervised reconstruction with spatial holdout.  
   Tier 2/3 (Roadmap): Format-compatible GFS 0.25°/NCUM input integration, IMERG Early/Late near-real-time ingestion, and BharatFS operational integration.
10. **Slide 10: "What We Will NOT Claim"**  
    The 11 technical guardrails displayed transparently. Demonstrates exceptional scientific maturity to jury members.

---

## 5. 2-Minute Demo Video Storyboard

* **0:00 – 0:15 (The Gap):** Show the IMD 0.25° grid over Karnataka. Highlight Mandya district: one coarse cell covering 39 Gram Panchayats. Explain why localized decisions are impossible without downscaling.
* **0:15 – 0:45 (The Solution & Visual Impact):** 3-panel synchronized comparison over Mandya:
  1. Coarse 0.25° IMD input.
  2. 5× Terrain-conditioned downscaling output (5.5 km).
  3. Reference ground truth.  
  Zoom into hilly taluks to show orographic rainfall enhancement that bilinear interpolation fails to capture.
* **0:45 – 1:10 (Mathematical Rigor):** Briefly display:
  - Exact mass conservation check ($k=5$).
  - Grid registration alignment.
  - Calibration QQ curve and CQR uncertainty range ($90\%$ test coverage).
* **1:10 – 1:40 (Mobile PWA & Bilingual Advisory Demo):**
  - Tap Mandya panchayat on mobile viewport.
  - Display localized forecast: *"Expected 14.2 mm, likely 8.5–22.1 mm"*.
  - Show bilingual advice in Kannada and English for Ragi and Paddy.
  - **The Climax:** Turn on Airplane Mode in Chrome DevTools $\to$ Offline warning banner immediately activates $\to$ show that cached forecasts and advisories remain fully readable.
* **1:40 – 2:00 (Roadmap & Summary):** Reiterate honest ceiling (panchayat-scale, 1 pixel $\approx$ 1 GP). Summarize near-real-time roadmap (IMERG Early/Late). Close on team and repository links.

---

## 6. Automated Deck & Code Grep Audits

Before finalizing slides and documentation, execute the following grep checks in the repository:

```powershell
# These searches MUST return 0 matches in code, docs, and presentation scripts:
git grep -i "BharatFS real-time"
git grep -i "offline generation"
git grep -i "e-GramSwaraj push"
git grep -i "3 blocks"
git grep -i "sub-panchayat"
git grep -i "13x10"
```

---

## 7. 7-Day Compressed Schedule Contingency

If the sprint timeline is compressed from 10 days to 7 days, execute the following compressed delta:
* **Day 1:** National CHIRPS + patch index with land filter ($\ge 70\%$) and buffer exclusion.
* **Day 2:** Conservation loss unit tests ($k=5$, cosine weights at HR centers), baselines, and grid registration assert.
* **Day 3:** National model training with spatial holdout on RTX 3060.
* **Day 4:** IMD quantile mapping calibration, CQR conformalization, clean polygon aggregation, and hill-vs-plains evaluation.
* **Days 5–7:** API integration, offline PWA, pitch deck, video recording, and final dry runs.
* *Approved Cuts under 7-Day Mode:* Drop ERA5 extra channels; drop Random Forest baseline; stick strictly to single-stage 5× U-Net. If CHIRPS download is bandwidth-constrained, fall back to 2014–2023 (~1,100 days, ~200k patches).

---

## 8. Definition of Done & Finals Readiness Checklist

- [ ] `README.md` completely updated: Built vs. Roadmap clearly delineated, data-attribution block included, Section 6 guardrails embedded, zero secrets committed.
- [ ] Automated grep scan returns 0 hits for all banned overclaim phrases.
- [ ] All 15 items in canonical Definition of Done verified and green.
- [ ] Pitch deck (`docs/pitch_deck.pdf`) rendered and reviewed by entire team.
- [ ] 2-minute demonstration video recorded with clear audio and committed/hosted.
- [ ] Full live rehearsal conducted twice by two different team members:
  - Run 1: Clean clone and run from scratch.
  - Run 2: Airplane-mode offline PWA verification.
- [ ] Repository ready for public evaluation and jury presentation.
