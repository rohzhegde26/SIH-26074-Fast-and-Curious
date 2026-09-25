# Sprint 6 Benchmark: Residual Diffusion Formulation & Meteorological Refinement

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Date**: September 25, 2026  

## Invariants Maintained Across All Formulations
- **Antecedent History**: Frozen $H^* = 14$ days
- **Spatial Context**: Frozen $N^* = 24$ coarse cells ($N/M = 1.50$, $660 \times 660$ km domain)
- **Target Grid**: $M = 16$ coarse cells downscaled $5\times$ to $80 \times 80$ fine cells ($0.05^\circ$)
- **Model Capacity**: Strictly locked at **15,685,478 parameters**
- **Sampler**: Deterministic DDIM-32 ($\eta = 0.0$)

---

## Table A: Multi-Task Validation Objective and Convergence Table (2022 Validation Season)

| Candidate ID | Formulation | Parameterization | Loss Objective | Best Epoch | Val Loss | Val CMVS | Wet-MAE (mm) | CSI@30 | Tmax MAE (°C) | Wind RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **EXP-01** | Candidate 1 (Control) | `epsilon` | `uniform` | Epoch 25 | 0.0364 | **0.6490** | 8.70 | 0.631 | 0.37 | 1.69 |
| **EXP-02** | Candidate 2 (v-pred) | `v_prediction` | `uniform` | Epoch 15 | 0.0653 | **0.6493** | 8.95 | 0.615 | 0.33 | 1.66 |
| **EXP-03** | Candidate 3 (Multi-Task Tail) | `v_prediction` | `group_tail` | Epoch 30 | 0.3726 | **0.5710** | 7.67 | 0.679 | 0.30 | 1.57 |

> [!NOTE]
> **CMVS Calculation**: $\text{CMVS} = 0.35 \left(\frac{\text{WetMAE}}{8.70}\right) + 0.35 \left(1.0 - \frac{\text{CSI@30}}{0.631}\right) + 0.15 \left(\frac{\text{TmaxMAE}}{0.37}\right) + 0.15 \left(\frac{\text{WindRMSE}}{1.69}\right)$. Lower is better.

---

## Table B: Confirmatory Holdout Test Performance (2023 Season - Quarantined)

| Candidate ID | Formulation | Precip Wet-MAE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Tmin MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|
| **EXP-01** | Candidate 1 (Control) | **9.90** | 0.464 | 0.489 | 0.46 | 0.44 | 3.56 |
| **EXP-02** | Candidate 2 (v-pred) | **9.77** | 0.487 | 0.489 | 0.39 | 0.36 | 3.50 |
| **EXP-03** | Candidate 3 (Multi-Task Tail) | **8.93** | 0.524 | 0.534 | 0.37 | 0.33 | 3.49 |

---

## Scientific Hypothesis Evaluation

### H1_vpred_stability: **FALSIFIED**
- **val_cmvs_control**: 0.6489777853044069
- **val_cmvs_vpred**: 0.649252936216254
- **delta_cmvs_pct**: -0.042397585568833294

### H2_tail_heteroscedasticity: **CONFIRMED**
- **csi30_vpred**: 0.6150046746855004
- **csi30_multitask**: 0.6788611780304984
- **delta_csi30_pct**: 10.383092352534202
- **tmax_mae_diff_pct**: -7.2225976870979185
