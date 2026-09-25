# Sprint 7 Benchmark: Diffusion-Step & Sampler Frontier

**Program**: Smart India Hackathon (SIH) 2026 - Problem Statement 26074  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Date**: September 25, 2026  

---

## Table A: Sampler Accuracy & Computational Efficiency Frontier (2022 Validation Season)

| Condition ID | Sampler Family | Steps ($S$) | NFE | Latency (ms) | Speedup | CMVS (Val) | Wet-MAE (mm) | CSI@15 | CSI@30 | Tmax MAE (°C) | Wind RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GATE_LEGACY_32** | DDIM | 32 | 32 | 756.6 | **1.05x** | 2.2582 | 14.28 | 0.284 | 0.344 | 1.93 | 8.34 |
| **STEP_32_REF** | DDIM | 32 | 32 | 790.7 | **1.00x** | **2.2566** | 14.27 | 0.284 | 0.344 | 1.93 | 8.33 |
| **STEP_04** | DDIM | 4 | 4 | 103.7 | **7.63x** | 1.8895 | 12.70 | 0.322 | 0.426 | 1.61 | 6.90 |
| **STEP_08** | DDIM | 8 | 8 | 202.2 | **3.91x** | 2.1174 | 13.67 | 0.297 | 0.372 | 1.81 | 7.78 |
| **STEP_16** | DDIM | 16 | 16 | 398.2 | **1.99x** | 2.2127 | 14.08 | 0.288 | 0.352 | 1.89 | 8.16 |
| **STEP_64** | DDIM | 64 | 64 | 1578.2 | **0.50x** | 2.2764 | 14.36 | 0.282 | 0.340 | 1.95 | 8.41 |
| **DPM_04** | DPM_SOLVER | 4 | 4 | 103.9 | **7.61x** | 2.0235 | 13.26 | 0.307 | 0.394 | 1.73 | 7.42 |
| **DPM_08** | DPM_SOLVER | 8 | 8 | 203.2 | **3.89x** | 2.2379 | 14.19 | 0.286 | 0.347 | 1.92 | 8.26 |
| **DPM_16** | DPM_SOLVER | 16 | 16 | 399.5 | **1.98x** | **2.2834** | 14.39 | 0.282 | 0.339 | 1.96 | 8.44 |
| **DPM_32** | DPM_SOLVER | 32 | 32 | 791.6 | **1.00x** | **2.2952** | 14.44 | 0.281 | 0.336 | 1.97 | 8.49 |
| **PNDM_08** | PNDM | 8 | 8 | 203.0 | **3.90x** | 2.2863 | 14.40 | 0.282 | 0.339 | 1.96 | 8.45 |
| **PNDM_16** | PNDM | 16 | 16 | 400.5 | **1.97x** | 2.2935 | 14.43 | 0.281 | 0.337 | 1.97 | 8.48 |

> [!NOTE]
> **Composite Meteorological Validation Score (CMVS)**:
> $$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}_{\text{val}}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}_{\text{val}}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}_{\text{val}}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}_{\text{val}}}{1.69}\right)$$

---

## Table B: Confirmatory Holdout Test Set Performance (2023 Season - Quarantined)

| Sampler Champion | Split | Wet-MAE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Tmin MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|
| **STEP_04** | 2023 Holdout Test | **13.49** | **0.188** | **0.293** | **1.67** | **1.70** | **7.64** |

---

## Research Hypotheses Falsification & Validation Summary

- **H1_corrected_discretization_integrity**: **CONFIRMED**
  - `legacy_cmvs`: 2.2582251116011167
  - `corrected_cmvs`: 2.2565927535491674
  - `cmvs_delta_pct`: 0.07228500132973507
  - `legacy_wet_mae`: 14.276803834097725
  - `corrected_wet_mae`: 14.270671572004046
- **H2_ddim_step_pareto_frontier**: **CONFIRMED**
  - `ddim32_cmvs`: 2.2565927535491674
  - `ddim16_cmvs`: 2.2126702464998607
  - `cmvs_degradation_pct`: -1.946408228964907
  - `latency_reduction_pct`: 49.645542010809464
- **H3_extreme_precipitation_step_sensitivity**: **FALSIFIED**
  - `csi30_drop_pct`: -23.98812042451075
  - `wet_mae_increase_pct`: -11.032932187791666
- **H4_high_order_ode_efficiency**: **FALSIFIED**
  - `dpm16_cmvs`: 2.2834454931750265
  - `ddim32_cmvs`: 2.2565927535491674
  - `speedup_factor`: 1.9792376103393385
- **H5_lead_time_trajectory_robustness**: **FALSIFIED**
  - `d0_error_ratio`: 0.9585042260359912
  - `d6_error_ratio`: 0.9565973877071032
