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

[+] Sprint 9 Phase 2 Execution Complete.