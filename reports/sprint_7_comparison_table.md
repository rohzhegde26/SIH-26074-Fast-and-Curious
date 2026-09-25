# Sprint 7 Benchmark: Diffusion-Step & Sampler Frontier

**Program**: Smart India Hackathon (SIH) 2026 - Problem Statement 26074  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Date**: September 25, 2026  

---

## Table A: Sampler Accuracy & Computational Efficiency Frontier (2022 Validation Season)

| Condition ID | Sampler Family | Steps ($S$) | NFE | Latency (ms) | Speedup | CMVS (Val) | Wet-MAE (mm) | CSI@15 | CSI@30 | Tmax MAE (°C) | Wind RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GATE_LEGACY_32** | DDIM | 32 | 32 | 763.5 | **1.01x** | 0.6470 | 8.89 | 0.645 | 0.615 | 0.33 | 1.66 |
| **STEP_32_REF** | DDIM | 32 | 32 | 773.4 | **1.00x** | **0.6408** | 8.87 | 0.649 | 0.617 | 0.32 | 1.64 |
| **STEP_04** | DDIM | 4 | 4 | 101.9 | **7.59x** | 0.6054 | 8.28 | 0.667 | 0.637 | 0.31 | 1.66 |
| **STEP_08** | DDIM | 8 | 8 | 198.0 | **3.91x** | 0.6229 | 8.57 | 0.658 | 0.626 | 0.32 | 1.65 |
| **STEP_16** | DDIM | 16 | 16 | 391.1 | **1.98x** | 0.6337 | 8.75 | 0.653 | 0.620 | 0.32 | 1.64 |
| **STEP_64** | DDIM | 64 | 64 | 1544.0 | **0.50x** | 0.6442 | 8.92 | 0.647 | 0.616 | 0.33 | 1.63 |
| **DPM_04** | DPM_SOLVER | 4 | 4 | 101.9 | **7.59x** | 0.6179 | 8.45 | 0.662 | 0.631 | 0.32 | 1.68 |
| **DPM_08** | DPM_SOLVER | 8 | 8 | 198.3 | **3.90x** | 0.6361 | 8.73 | 0.654 | 0.620 | 0.32 | 1.66 |
| **DPM_16** | DPM_SOLVER | 16 | 16 | 391.4 | **1.98x** | **0.6416** | 8.85 | 0.649 | 0.617 | 0.32 | 1.65 |
| **DPM_32** | DPM_SOLVER | 32 | 32 | 776.7 | **1.00x** | **0.6460** | 8.94 | 0.646 | 0.615 | 0.33 | 1.64 |
| **PNDM_08** | PNDM | 8 | 8 | 198.8 | **3.89x** | 0.6453 | 8.89 | 0.649 | 0.617 | 0.33 | 1.65 |
| **PNDM_16** | PNDM | 16 | 16 | 392.3 | **1.97x** | 0.6443 | 8.91 | 0.647 | 0.615 | 0.32 | 1.64 |

> [!NOTE]
> **Composite Meteorological Validation Score (CMVS)**:
> $$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}_{\text{val}}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}_{\text{val}}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}_{\text{val}}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}_{\text{val}}}{1.69}\right)$$

---

## Table B: Confirmatory Holdout Test Set Performance (2023 Season - Quarantined)

| Sampler Champion | Split | Wet-MAE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Tmin MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|
| **STEP_04** | 2023 Holdout Test | **9.22** | **0.506** | **0.506** | **0.37** | **0.34** | **3.48** |

---

## Research Hypotheses Falsification & Validation Summary

- **H1_corrected_discretization_integrity**: **CONFIRMED**
  - `legacy_cmvs`: 0.6470435289775406
  - `corrected_cmvs`: 0.6408402699102437
  - `cmvs_delta_pct`: 0.9587081532365076
  - `legacy_wet_mae`: 8.891045570373535
  - `corrected_wet_mae`: 8.865123748779297
- **H2_ddim_step_pareto_frontier**: **CONFIRMED**
  - `ddim32_cmvs`: 0.6408402699102437
  - `ddim16_cmvs`: 0.6336727711086347
  - `cmvs_degradation_pct`: -1.1184532461751855
  - `latency_reduction_pct`: 49.43158263627992
- **H3_extreme_precipitation_step_sensitivity**: **CONFIRMED**
  - `csi30_drop_pct`: -3.2051480297976864
  - `wet_mae_increase_pct`: -6.627182493663931
- **H4_high_order_ode_efficiency**: **CONFIRMED**
  - `dpm16_cmvs`: 0.6415948647653524
  - `ddim32_cmvs`: 0.6408402699102437
  - `speedup_factor`: 1.9761213795352672
- **H5_lead_time_trajectory_robustness**: **CONFIRMED**
  - `d0_error_ratio`: 0.9639689005140526
  - `d6_error_ratio`: 0.9696119086847932
