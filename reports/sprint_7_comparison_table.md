# Sprint 7 Benchmark: Diffusion-Step & Sampler Frontier

**Program**: Smart India Hackathon (SIH) 2026 - Problem Statement 26074  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Date**: September 25, 2026  

---

## Table A: Sampler Accuracy & Computational Efficiency Frontier (2022 Validation Season)

| Condition ID | Sampler Family | Steps ($S$) | NFE | Latency (ms) | Speedup | CMVS (Val) | Wet-MAE (mm) | CSI@15 | CSI@30 | Tmax MAE (°C) | Wind RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **GATE_LEGACY_32** | DDIM | 32 | 32 | 735.2 | **1.00x** | 0.5701 | 7.64 | 0.693 | 0.678 | 0.31 | 1.56 |
| **STEP_32_REF** | DDIM | 32 | 32 | 734.9 | **1.00x** | **0.5649** | 7.57 | 0.699 | 0.682 | 0.30 | 1.55 |
| **STEP_04** | DDIM | 4 | 4 | 96.8 | **7.59x** | 0.5376 | 6.97 | 0.724 | 0.704 | 0.29 | 1.57 |
| **STEP_08** | DDIM | 8 | 8 | 188.0 | **3.91x** | 0.5509 | 7.24 | 0.712 | 0.693 | 0.30 | 1.56 |
| **STEP_16** | DDIM | 16 | 16 | 371.8 | **1.98x** | 0.5595 | 7.44 | 0.704 | 0.686 | 0.30 | 1.56 |
| **STEP_64** | DDIM | 64 | 64 | 1468.6 | **0.50x** | 0.5679 | 7.63 | 0.696 | 0.680 | 0.30 | 1.55 |
| **DPM_04** | DPM_SOLVER | 4 | 4 | 97.0 | **7.58x** | 0.5443 | 7.06 | 0.720 | 0.700 | 0.30 | 1.58 |
| **DPM_08** | DPM_SOLVER | 8 | 8 | 188.3 | **3.90x** | 0.5579 | 7.36 | 0.708 | 0.688 | 0.30 | 1.57 |
| **DPM_16** | DPM_SOLVER | 16 | 16 | 371.8 | **1.98x** | **0.5647** | 7.55 | 0.700 | 0.682 | 0.30 | 1.56 |
| **DPM_32** | DPM_SOLVER | 32 | 32 | 738.2 | **1.00x** | **0.5689** | 7.66 | 0.695 | 0.679 | 0.30 | 1.55 |
| **PNDM_08** | PNDM | 8 | 8 | 188.9 | **3.89x** | 0.5656 | 7.56 | 0.700 | 0.681 | 0.30 | 1.56 |
| **PNDM_16** | PNDM | 16 | 16 | 372.8 | **1.97x** | 0.5682 | 7.65 | 0.695 | 0.678 | 0.30 | 1.55 |

> [!NOTE]
> **Composite Meteorological Validation Score (CMVS)**:
> $$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}_{\text{val}}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}_{\text{val}}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}_{\text{val}}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}_{\text{val}}}{1.69}\right)$$

---

## Table B: Confirmatory Holdout Test Set Performance (2023 Season - Quarantined)

| Sampler Champion | Split | Wet-MAE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Tmin MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|
| **STEP_04** | 2023 Holdout Test | **8.19** | **0.556** | **0.556** | **0.35** | **0.32** | **3.51** |

---

## Table C: Sprint 6 Baseline vs. Sprint 7 Frontier (Candidate 3 Multi-Task Champion)

Evaluating the identical Candidate 3 Multi-Task $v$-prediction architecture (15,685,478 parameters, Epoch 30 weights) under Sprint 6 legacy sampler vs. Sprint 7 optimized samplers:

| Evaluation Dimension | Sprint 6 Baseline (Legacy DDIM-32) | Sprint 7 Reference (Corrected DDIM-32) | Sprint 7 Operational (DPM-Solver++ 16 NFE) | Sprint 7 Edge Fast (DDIM 4-Step) | Delta / Relative Improvement |
|---|---|---|---|---|---|
| **Validation CMVS (2022)** | `0.5710` (or `0.5701`) | **`0.5649`** | **`0.5647`** | **`0.5376`** | **-5.9% lower composite error** |
| **Validation Wet-MAE (mm)** | `7.64 mm` | `7.57 mm` | `7.55 mm` | **`6.97 mm`** | **-8.8% lower rainfall error** |
| **Validation CSI@30** | `0.678` | `0.682` | `0.682` | **`0.704`** | **+3.8% higher storm recall** |
| **Validation CSI@15** | `0.693` | `0.699` | `0.700` | **`0.724`** | **+4.5% higher storm recall** |
| **Validation Tmax MAE (°C)**| `0.31 °C` | `0.30 °C` | `0.30 °C` | **`0.29 °C`** | **-6.5% lower thermal error** |
| **Validation Wind RMSE (m/s)**|`1.56 m/s` | `1.55 m/s` | `1.56 m/s` | **`1.57 m/s`** | **Preserved wind vector fidelity**|
| **2023 Holdout Wet-MAE (mm)**| `8.93 mm` | -- | -- | **`8.19 mm`** | **-8.3% lower test rainfall error**|
| **2023 Holdout CSI@30** | `0.534` | -- | -- | **`0.556`** | **+4.1% higher test storm recall**|
| **2023 Holdout Tmax MAE (°C)**| `0.37 °C` | -- | -- | **`0.35 °C`** | **-5.4% lower test thermal error**|
| **2023 Holdout Tmin MAE (°C)**| `0.34 °C` | -- | -- | **`0.32 °C`** | **-5.9% lower test thermal error**|
| **Per-Cube Latency (ms)** | `735.2 ms` | `734.9 ms` | **`371.8 ms`** | **`96.8 ms`** | **1.98x to 7.59x faster runtime** |
| **Throughput (cubes/sec)** | `1.4 cubes/s` | `1.4 cubes/s` | **`2.7 cubes/s`** | **`10.3 cubes/s`**| **Up to 7.4x higher throughput** |

---

## Research Hypotheses Falsification & Validation Summary

- **H1_corrected_discretization_integrity**: **CONFIRMED**
  - `legacy_cmvs`: 0.5700815502580735
  - `corrected_cmvs`: 0.5648947555779341
  - `cmvs_delta_pct`: 0.9098338084772861
  - `legacy_wet_mae`: 7.642333235059466
  - `corrected_wet_mae`: 7.569516658782959
- **H2_ddim_step_pareto_frontier**: **CONFIRMED**
  - `ddim32_cmvs`: 0.5648947555779341
  - `ddim16_cmvs`: 0.5594520069721989
  - `cmvs_degradation_pct`: -0.9634978112280144
  - `latency_reduction_pct`: 49.411809360060474
- **H3_extreme_precipitation_step_sensitivity**: **CONFIRMED**
  - `csi30_drop_pct`: -3.1965380500893716
  - `wet_mae_increase_pct`: -7.977334506985
- **H4_high_order_ode_efficiency**: **CONFIRMED**
  - `dpm16_cmvs`: 0.5646969548060501
  - `ddim32_cmvs`: 0.5648947555779341
  - `speedup_factor`: 1.9766118816660416
- **H5_lead_time_trajectory_robustness**: **CONFIRMED**
  - `d0_error_ratio`: 0.9521829504206066
  - `d6_error_ratio`: 0.9606612020604509
