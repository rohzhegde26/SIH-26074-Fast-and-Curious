# Sprint 8.5 Champion 2023 Holdout Test Report

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Evaluation Protocol:** Strict single-pass holdout evaluation on 2023 test set using frozen calibration parameters selected on 2022. Zero post-hoc tuning.  

---

## 1. Holdout Metadata & Provenance

- **Checkpoint SHA-256:** `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`
- **Trainable Parameters:** 15,685,478
- **Evaluation Dataset Split:** `test` (Calendar Year 2023, 122 cubes, 854 slices)
- **Frozen Alpha:** 3.0
- **Inference Execution Time:** 117.3 seconds

---

## 2. 6-Channel CRPS Performance on 2023 Holdout

**Overall 6-Channel Mean CRPS:** **0.6371**

| Variable | Channel Name | CRPS | Unit | Spread-Skill Ratio |
|---|---|---:|---|---:|
| **precipitation** | `precipitation` | 2.0810 | mm/day | 0.000 |
| **tmax** | `tmax` | 0.2663 | degC | 0.000 |
| **tmin** | `tmin` | 0.2287 | degC | 0.000 |
| **rh** | `rh` | 0.3164 | % | 0.000 |
| **wind_u** | `wind_u` | 0.7014 | m/s | 0.000 |
| **wind_v** | `wind_v` | 0.2288 | m/s | 0.000 |

---

## 3. Precipitation Calibration: Uncalibrated vs Frozen Calibrated

| Metric | Uncalibrated (alpha=1.0) | Calibrated (Frozen alpha*) | Delta Impact |
|---|---:|---:|---:|
| **Precipitation CRPS (mm/day)** | 2.0810 | **1.7806** | -0.3004 |
| **Ensemble Spread (mm)** | 1.261 | **3.403** | +2.142 |
| **RMSE (mm)** | 7.487 | 7.649 | +0.162 |
| **Spread-Skill Ratio** | 0.168 | **0.445** | +0.276 |
| **50% Interval Coverage** | 5.4% | **12.3%** | +7.0% |
| **80% Interval Coverage** | 9.2% | **17.2%** | +8.0% |
| **90% Interval Coverage** | 10.6% | **18.3%** | +7.7% |
| **90% Sharpness Width (mm)** | 3.12 | 8.23 | +5.11 |

---

## 4. Probability Calibration on 2023 Holdout

| Event Threshold | Raw Ensemble Brier | Isotonic Calibrated Brier | Raw BSS | Isotonic Calibrated BSS |
|---|---:|---:|---:|---:|
| **P > 15 mm** | 0.03471 | **0.02999** | 0.4906 | **0.5599** |
| **P > 30 mm** | 0.01741 | **0.01616** | 0.5256 | **0.5595** |

---

## 5. Conformal Prediction Intervals on 2023 Holdout

| Nominal Target Coverage | Empirical 2023 Coverage | Conformal Quantile (q_hat) | Mean Interval Width | Physical Bound |
|---|---:|---:|---:|:---:|
| **50% Target** | **7.6%** | 0.837 | 2.07 mm | P >= 0.0 strictly verified |
| **80% Target** | **13.2%** | 1.752 | 4.08 mm | P >= 0.0 strictly verified |
| **90% Target** | **16.6%** | 2.804 | 6.17 mm | P >= 0.0 strictly verified |

---

## 6. Deterministic Point Metrics and Physical Integrity on 2023 Holdout

- **Wet MAE (P >= 1.0 mm):** 7.611 mm
- **CSI @ 30 mm:** 0.5812
- **Raw Negative Pixel Fraction:** 36.64%
- **Precipitation Mass Shift:** 3.31%
- **Thermodynamic Inversion Violations (Tmin > Tmax):** 0.00%
- **Single Member Laplacian Retention Ratio:** 18.4%
- **Ensemble Mean Laplacian Retention Ratio:** 6.1%