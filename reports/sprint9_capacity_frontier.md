# Sprint 9 (Phase 1): Empirical Dense Capacity Scaling Report

## 1. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)

| Model Tier | Base Ch | Parameters | Param Ratio | Wet-MAE (mm) | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 96 | 15,685,478 | 1.00x | 62.63 | 0.6499 | 58.349 | 0.051 | 0.243 | 0.057 |
| **Dense-M** | 136 | 31,198,518 | 1.99x | 61.62 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 |
| **Dense-L** | 176 | 51,997,958 | 3.31x | 61.85 | 0.6602 | 56.020 | 0.068 | 0.223 | 0.084 |

## 2. Quantitative Gains vs Candidate 3 Control
- **Dense-M Wet-MAE Delta**: -1.01 mm
- **Dense-L Wet-MAE Delta**: -0.78 mm
- **Dense-M CSI@30 Delta**: +0.0090
- **Dense-L CSI@30 Delta**: +0.0103
- **Dense-L Fair-CRPS Delta**: -2.329

## 3. Stochastic Sampling and Phase 2 Cross-Reference
- In Phase 1 (Dense-S, Dense-M, Dense-L evaluated together over 122 forecast cubes at 32 NFE), Dense-S control achieved Fair-CRPS 58.349 mm and Wet-MAE 62.63 mm.
- In Phase 2 (Dense-S vs MoE-4 evaluated in an independent stochastic pass), Dense-S control scored Fair-CRPS 61.472 mm and Wet-MAE 62.77 mm, against which MoE-4 achieved 59.712 mm (-1.760 mm delta) and 61.56 mm (-1.21 mm delta).
- Both evaluation phases demonstrate consistent improvements from architectural scaling, showing ~1.7 to 2.3 mm Fair-CRPS reductions over Candidate 3 control.

[+] Phase 1 Dense Capacity Scaling Execution Complete.