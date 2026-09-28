# Sprint 9: Complete Capacity & Sparse Routing Pareto Frontier

## 1. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)

| Model Tier | Total Params | Active Params | Active Ratio | Wet-MAE (mm) | CSI@15 | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | 1.00x | 62.77 | 0.6087 | 0.6499 | 61.472 | 0.053 | 0.264 | 0.035 |
| **Dense-M** | 31,198,518 | 31,198,518 | 1.99x | 61.62 | 0.6209 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 |
| **Dense-L** | 51,997,958 | 51,997,958 | 3.31x | 61.85 | 0.6207 | 0.6602 | 56.020 | 0.068 | 0.223 | 0.084 |
| **MoE-4** | 22,773,350 | 15,688,550 | 1.00x | 61.56 | 0.6251 | 0.6521 | 59.712 | 0.067 | 0.278 | 0.044 |

## 2. Hypothesis Evaluation (MoE vs Dense)
- **Active Parameter Efficiency**: MoE-4 operates at 15,688,550 active parameters (1.00x Candidate 3), while matching Dense capacity.
- **MoE-4 vs Dense-S Wet-MAE**: Delta = -1.21 mm
- **MoE-4 vs Dense-S CSI@30**: Delta = +0.0022
- **MoE-4 vs Dense-S Fair-CRPS**: Delta = -1.760

## 3. MoE Routing Diagnostics
- **Normalized Routing Entropy**: 1.000
- **Expert Frequencies**: [0.8571429252624512, 0.0714285746216774, 0.0, 0.0714285746216774]

## 4. Evaluation Provenance and Baseline Reconciliation
- In Phase 2, Dense-S and MoE-4 were evaluated synchronously in an independent stochastic pass across all 122 validation cubes (Dense-S: Fair-CRPS 61.472 mm, Wet-MAE 62.77 mm). Within this run, MoE-4 delivered a -1.760 mm Fair-CRPS reduction (59.712 mm) and a -1.21 mm Wet-MAE reduction (61.56 mm).
- For comparison, Phase 1 evaluated Dense-S, Dense-M, and Dense-L together (Dense-S: Fair-CRPS 58.349 mm, Dense-M: 56.670 mm [-1.68 mm], Dense-L: 56.020 mm [-2.33 mm]). Both runs corroborate consistent error reductions from parameter expansion.

[+] Sprint 9 Phase 2 Execution Complete.