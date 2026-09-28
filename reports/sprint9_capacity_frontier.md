# Sprint 9: Model Capacity Scaling Pareto Frontier Report

## 1. Executive Capacity Frontier Overview
This report establishes the baseline evaluation of the frozen control model (**Dense-S / Candidate 3**) and documents the experimental protocol and target hypotheses for the dense scaling ladder (**Dense-M** and **Dense-L**) and **MoE-4**.

All evaluations are conducted under matched inference compute (32 NFE) on the authentic 2022 validation dataset:
- **Point Forecast Mode**: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE).
- **Distribution Mode**: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE).

### Multi-Dimensional Pareto Table

| Model Tier | Total Params | Active Params | Param Ratio | Status | Wet-MAE (mm) | CSI@30 | Fair-CRPS | Raw SSR | Calibrated SSR | Cov@90 | Lap Retention ($R_{\text{Lap}}$) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Candidate 3 Control)** | 15,685,478 | 15,685,478 | 1.00x | **VALIDATED BASELINE** | 8.15 | 0.5812 | 1.842 | 0.284 | 0.582 | 0.284 | 0.215 |
| **Dense-M (Target [25M, 35M])** | 31,198,518 | 31,198,518 | 1.99x | PENDING_PHASE1_KAGGLE | Target < 7.80 | Target > 0.6000 | Target < 1.650 | Target > 0.350 | Target > 0.700 | Target > 0.350 | Target > 0.250 |
| **Dense-L (Target [45M, 65M])** | 51,997,958 | 51,997,958 | 3.31x | PENDING_PHASE1_KAGGLE | Target < 7.40 | Target > 0.6250 | Target < 1.500 | Target > 0.450 | Target > 0.800 | Target > 0.450 | Target > 0.300 |
| **MoE-4 (Sparse Bottleneck)** | 35,765,990 | 15,685,478 | 1.00x (active) | PENDING_PHASE2_KAGGLE | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M |

---

## 2. Invariant Scientific Verification
The following operational constants are strictly frozen across all capacity tiers to ensure model capacity is the sole independent variable:
- History Length: H = 14 days
- Coarse Spatial Context: N = 24 (0.25 degree resolution)
- Target Central Region: M = 16 (0.05 degree downscaled resolution, 80x80 grid)
- Diffusion Steps: T = 100
- Diffusion Parameterization: v-prediction (Salimans & Ho, 2022)
- Diffusion Schedule: Linear beta schedule (beta_start = 1e-4, beta_end = 0.035)
- Loss Formulation: Multi-task group-tail loss:
  - Convective tail focal weight: 3.0 on extreme cells (> 15 mm)
  - Variable group weights: lambda_precip = 1.0, lambda_thermo = 1.2, lambda_wind = 1.1

---

## 3. Phased Execution Roadmap
1. **Phase 1 (Dense Scaling Ladder)**:
   Execute `sprint_9_phase1_dense_capacity_scaling.ipynb` on Kaggle interactive GPU. Train Dense-M (31.20M) and Dense-L (52.00M) on 2015-2021 training set and evaluate matched 32 NFE against Dense-S on the 2022 validation set.
2. **Phase 2 (MoE Sparse Routing)**:
   Evaluate MoE-4 (35.77M total / 15.69M active) against the winning dense tier to determine whether sparse routing decouples quality from active latency.
