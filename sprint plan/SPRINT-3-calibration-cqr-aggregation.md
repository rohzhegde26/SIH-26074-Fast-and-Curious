# Sprint 3 — Days 5–6: Calibration, CQR Uncertainty & Polygon Aggregation

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith P Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Days 5–6  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Convert raw 5× model gridded outputs into calibrated, uncertainty-bounded, panchayat-level operational forecasts. Implement per-$0.25^\circ$-cell quantile mapping to ground outputs in IMD gauge truth, conformalized quantile regression (CQR) with rigorous 90% empirical coverage, clean area-weighted polygon aggregation with bug-free geometric validation gates, and mandatory hill-vs-plains orographic error decomposition.

---

## 2. Team Responsibilities & Ownership

| Role | Sprint 3 Deliverables |
|---|---|
| **Data/GIS Lead** | Implement clean polygon zonal stats in `src/data/zonal_aggregation.py`, run geometric audit gates using `pyproj.Geod`, compute analytic spherical cell areas $A_i = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\text{lat}_i)$, generate Mandya GP fractional coverage weights. |
| **ML/Eval Engineer** | Build quantile mapping service in `src/eval/calibration.py` (per $0.25^\circ$ cell), implement CQR module in `src/eval/cqr.py` on calibration split 2022, log test 2023 coverage, compute headline calibrated metrics in `src/eval/metrics.py` (MAE, RMSE, CSI R95/R99, PBIAS). |
| **ML Lead** | Support MC-dropout passes (10–20 forward passes per panchayat-day) for heteroscedastic base quantile estimation ($q_{0.05}, q_{0.95}$). |
| **Domain/Pitch Lead** | Spot-check 3 Mandya panchayats with local farming context, inspect orographic contrast in Western Ghats vs. plains, draft calibration slides. |

---

## 3. Pinned Technical Specifications & Mathematics

### A. Per-0.25°-Cell Quantile Mapping (Calibration to Gauge Gold)
* **The Rule:** Quantile mapping must be executed per $0.25^\circ$ LR cell (using IMD gauge distribution), **never** per $0.05^\circ$ HR cell.
* **Why:** Mapping per $0.05^\circ$ cell falsely hallucinates that IMD ground truth exists at 5 km resolution and destroys high-resolution terrain texture. Mapping per $0.25^\circ$ cell preserves intra-cell orographic gradients while correcting regional bias against the IMD gauge gold standard.
* **Artifact:** QQ plot / calibration curve saved to `docs/calibration_curve.png`.

### B. Conformalized Quantile Regression (CQR)
1. **Base Quantile Estimation:** Run 10–20 Monte Carlo dropout passes per panchayat-day to extract base lower and upper quantiles ($q_{\text{lo}} = 5\text{th}$, $q_{\text{hi}} = 95\text{th}$).
2. **Dedicated Calibration Split:** Conformalize strictly on **monsoon year 2022** ($\approx 31,000$ panchayat-day scores, guaranteeing stable $90\%$ quantile calculation).
3. **Conformal Nonconformity Score:**
   $$s_i = \max(q_{\text{lo}}(x_i) - y_i, \; y_i - q_{\text{hi}}(x_i))$$
4. **Finite-Sample Correction:**
   $$\hat{Q} = \text{Quantile}_{\frac{\lceil (n+1)(1-\alpha) \rceil}{n}}\left(\{s_i\}_{i=1}^n\right) \quad (\text{with } \alpha = 0.10 \text{ for } 90\% \text{ coverage})$$
5. **Calibrated Prediction Interval (Clip-at-Zero):**
   $$\mathcal{I}(x) = \left[\max(0, \; q_{\text{lo}}(x) - \hat{Q}), \quad q_{\text{hi}}(x) + \hat{Q}\right]$$
   *Clipping at 0 preserves coverage integrity because physical rainfall $y \ge 0$.*
6. **Reporting Standard:** Test on unseen monsoon 2023. UI and pitch must state:  
   *"Expected $X\text{ mm}$, likely $Y\text{--}Z\text{ mm}$ ($90\%$ empirical coverage on test 2023)"*. Never report an uncalibrated bare confidence percentage.

### C. Clean Grid-to-Polygon Aggregation & Fixed Audit Gates
* **Clean Weighting Formulation:**
  For each panchayat polygon $P$:
  $$A_i = R^2 \cdot \Delta\phi \cdot \Delta\lambda \cdot \cos(\text{lat}_i) \quad (R = 6371008.8\text{ m})$$
  $$w_i = f_i \cdot A_i \quad (f_i = \text{fractional intersection area of cell } i \text{ with } P)$$
  $$\text{Rain}_P = \frac{\sum_i \text{HR}_i \cdot w_i}{\sum_i w_i}$$
* **Why the Old Audit Identity Failed:**  
  The old check $\sum_P \text{area}(P \cap \text{cell}) \cdot \text{Rain}_P == \text{Rain}_{\text{cell}} \cdot \text{area}(\text{cell})$ is mathematically impossible because $\text{Rain}_P$ blends rainfall from multiple surrounding cells. A correct aggregation pipeline will naturally fail that identity.
* **New Validation Gates:**
  1. **Polygon Area Completeness:**
     $$\left|\sum_{\text{cells}} \text{inter\_area}(P, \text{cell}) - \text{area}(P)\right| \le 10^{-3} \cdot \text{area}(P)$$
  2. **Interior Cell Partition:**
     For every cell strictly interior to the district: $\sum_P f_i \approx 1.0$ (boundary cells reported, not failed).
  3. **Discrepancy Sanity Check:**
     $$|\text{mean\_HR}(P) - \text{mean\_LR}(P_{\text{cell}})| \le 10\% \text{ or } 5\text{ mm (reported statistic, not zero-tolerance)}$$
* **CRS Rule:** Use `pyproj.Geod` for absolute geodetic area audits. EPSG:7755 is for visual presentation only. Never multiply an equal-area projection by cosine of latitude.

### D. Mandatory Reporting: Calibrated vs. Raw & Regional Breakdown
* **Headline Metrics:** Must be computed and reported on the **QM-calibrated product** (what is actually shipped to the user): MAE, RMSE, Pearson $r$, PBIAS.
* **Raw CHIRPS Metrics:** Retained as internal diagnostics only. (Reporting raw metrics in the deck causes a direct contradiction with the calibration curve).
* **Hill-vs-Plains Breakdown:** Mandatory separate error reporting for Ghats/mountainous terrain vs. plains to address known satellite retrieval biases.

---

## 4. Sprint 3 Detailed Tasks

### A. Quantile Mapping Service (`src/eval/calibration.py`)
- [ ] Implement `fit_quantile_mapper(imd_lr_history, chirps_hr_coarsened_history)` per $0.25^\circ$ cell using isotonic regression.
- [ ] Apply mapping to raw 5× model predictions to enforce IMD gauge climatology while preserving intra-cell HR spatial structure.
- [ ] Document in module that QM aligns the climatological cumulative distribution function (CDF) to IMD gauge totals, absorbing bulk offsets, but does not claim to alter daily convective storm timing (15:00–19:00 IST).
- [ ] Generate and commit `docs/calibration_curve.png` (QQ plot showing pre- and post-calibration distributions).

### B. CQR Pipeline (`src/eval/cqr.py`)
- [ ] Implement MC-dropout inference mode in `src/models/unet_5x.py` (dropout active at test time, 10–20 forward passes).
- [ ] Compute conformal nonconformity scores $s$ over all days of monsoon 2022 ($\approx 31,000$ panchayat observations).
- [ ] Extract empirical threshold $\hat{Q}$ at $90\%$ quantile level.
- [ ] Evaluate empirical coverage on test split (monsoon 2023): assert empirical test coverage is between $88\%$ and $92\%$.
- [ ] Generate and commit `docs/cqr_coverage.png` (coverage reliability diagram and interval width distributions).

### C. Zonal Polygon Aggregation (`src/data/zonal_aggregation.py`)
- [ ] Pre-compute intersection weights $w_i = f_i \cdot A_i$ for all 258 Mandya panchayats loading strictly from `data/processed/mandya_full.geojson` (never the simplified display TopoJSON).
- [ ] Run validation gate 1: verify polygon area completeness within $10^{-3}$ relative error against `pyproj.Geod`.
- [ ] Run validation gate 2: verify interior HR cell partition $\sum_P f_i \approx 1.0$.
- [ ] Execute daily panchayat aggregations for Mandya across test year 2023.
- [ ] Log difference statistic $|\text{mean\_HR}(P) - \text{mean\_LR}(P_{\text{cell}})|$.

### D. Comprehensive Metric Suite (`src/eval/metrics.py`)
- [ ] Compute standard headline metrics on QM-calibrated output: MAE, RMSE, Pearson $r$, PBIAS.
- [ ] **Dry-Day Separation & Categorical Skill:**
  - Compute **Wet-Day MAE** strictly conditioned on $\text{Rain} > 2.5\text{ mm}$ (eliminates artificial dry-day inflation).
  - Compute Critical Success Index (CSI), Probability of Detection (POD), and False Alarm Ratio (FAR) across standard IMD rainfall categories:
    * Light rain: $2.5\text{--}15.5\text{ mm}$
    * Moderate rain: $15.5\text{--}64.4\text{ mm}$
    * Heavy / Extreme: $>64.5\text{ mm}$ (R95 / R99 percentiles)
- [ ] Compute hill-vs-plains stratified error breakdown. Save artifact: `docs/hill_vs_plains.png`.
- [ ] Perform manual spot-check audit across 3 diverse Mandya panchayats (e.g., hill boundary vs. canal irrigated plains).

---

## 5. Verification Gates & Definition of Done

- [ ] `tests/test_quantile_mapping.py` passes: verifies mapping is executed per $0.25^\circ$ cell and preserves intra-cell spatial variance.
- [ ] Zonal aggregation validation gates pass on all 258 Mandya panchayats:
  - Relative area difference $\le 10^{-3}$.
  - Interior cell sum $\approx 1.0$.
- [ ] CQR empirical test coverage on unseen monsoon 2023 satisfies $90\% \pm 2\%$.
- [ ] Calibrated prediction output generated for all Mandya panchayats (150–258 valid predictions with realistic spatial variability).
- [ ] All three mandatory evaluation artifacts committed:
  - `docs/calibration_curve.png`
  - `docs/cqr_coverage.png`
  - `docs/hill_vs_plains.png`
- [ ] Headline metrics table updated in documentation using calibrated numbers.
