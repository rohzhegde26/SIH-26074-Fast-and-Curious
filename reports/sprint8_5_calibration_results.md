# Sprint 8.5 Calibration Results Report

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Status:** Empirical Post-Hoc Calibration Results  

---

## 1. Experiment 2A: Multiplicative Spread Rescaling (Precipitation)

### Protocol:
- Calibration Fit Split: Cubes 0..60 (first 61 cubes of 2022 validation set).
- Calibration Evaluation Split: Cubes 61..121 (remaining 61 cubes evaluated out-of-sample).
- The spread scaling factor alpha* was chosen strictly on the fit split to minimize spread-skill deficit, then evaluated frozen on the evaluation split.

| Spread Factor (alpha) | Precip CRPS (mm/day) | Ensemble Spread (mm) | RMSE (mm) | Spread-Skill Ratio | 50% Coverage | 80% Coverage | 90% Coverage | 90% Sharpness (mm) | Evaluation Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| **alpha = 1.00** | 1.7995 | 1.211 | 5.658 | **0.214** | 9.1% | 15.0% | **17.1%** | 3.01 | Grid Reference |
| **alpha = 1.25** | 1.7308 | 1.495 | 5.662 | **0.264** | 10.8% | 17.4% | **19.6%** | 3.71 | Grid Reference |
| **alpha = 1.50** | 1.6780 | 1.774 | 5.668 | **0.313** | 12.4% | 19.5% | **21.6%** | 4.39 | Grid Reference |
| **alpha = 2.00** | 1.6097 | 2.319 | 5.690 | **0.408** | 15.2% | 22.7% | **24.7%** | 5.72 | Grid Reference |
| **alpha = 3.00** | 1.5711 | 3.362 | 5.770 | **0.583** | 19.4% | 26.7% | **28.2%** | 8.23 | **Selected alpha*** |

---

## 2. Experiment 2B: Threshold Probability Recalibration (P > 15 mm, P > 30 mm)

Evaluating raw ensemble exceedance frequencies against Isotonic Regression and Logistic Platt Scaling on the held-out 2022 validation block.
Climatological reference: 2015-2021 training climatology (P > 15 rate: 11.0322%, P > 30 rate: 5.8157%).

| Threshold | Method | Brier Score (lower is better) | Brier Skill Score (BSS) (higher is better) | Status |
|---|---|---:|---:|:---:|
| **P > 15 mm** | Raw Ensemble Frequency | 0.03096 | 0.7129 | Uncalibrated Reference |
| **P > 15 mm** | Isotonic Calibration | **0.02847** | **0.7360** | Calibrated Non-Parametric (Champion) |
| **P > 15 mm** | Logistic Platt Scaling | 0.02929 | 0.7284 | Calibrated Parametric |
| **P > 30 mm** | Raw Ensemble Frequency | 0.01527 | 0.7129 | Uncalibrated Reference |
| **P > 30 mm** | Isotonic Calibration | **0.01449** | **0.7276** | Calibrated Non-Parametric (Champion) |
| **P > 30 mm** | Logistic Platt Scaling | 0.01502 | 0.7176 | Calibrated Parametric |

---

## 3. Experiment 2C: Split-Conformal Prediction Intervals (P >= 0)

Conformal calibration fitted on first 61 cases of 2022; evaluated strictly out-of-sample on remaining 61 cases.

| Target Nominal Coverage | Empirical Test Coverage | Conformal Quantile (q_hat) | Mean Interval Width (mm) | Physical Bound Enforced |
|---|---:|---:|---:|:---:|
| **50% Nominal** | **12.8%** | 0.837 | 1.99 mm | P >= 0.0 strictly verified |
| **80% Nominal** | **20.8%** | 1.752 | 4.00 mm | P >= 0.0 strictly verified |
| **90% Nominal** | **25.8%** | 2.804 | 6.15 mm | P >= 0.0 strictly verified |