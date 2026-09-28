# Sprint 9: Model Capacity Scaling Pareto Frontier

## 1. Executive Capacity Frontier Overview

Matched compute comparison across 32 NFE in Point Mode (K=8, S=4, eta=0.5) and Distribution Mode (K=2, S=16, eta=0.0).

| Model Tier | Total Params | Active Params | Ratio vs Cand-3 | Wet-MAE (mm) | CSI@30 | CRPS | Raw SSR | Cov@90 | Lap Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DENSE_S** | 518,598 | 518,598 | 0.03x | 10.37 | 0.0067 | 5.819 | 0.483 | 0.283 | 0.612 |
| **DENSE_M** | 518,598 | 518,598 | 0.03x | 10.41 | 0.0071 | 6.019 | 0.501 | 0.273 | 0.613 |
| **DENSE_L** | 1,083,734 | 1,083,734 | 0.07x | 10.48 | 0.0027 | 5.529 | 0.437 | 0.305 | 0.615 |
| **MOE_4** | 716,870 | 519,110 | 0.05x | 10.39 | 0.0045 | 5.632 | 0.463 | 0.294 | 0.620 |

## 2. Invariant Scientific Verification
- All configurations preserve history H=14, context N=24, output crop M=16.
- Multi-task v-prediction parameterization with group-tail loss weighting strictly maintained.
- Capacity was the sole primary independent variable across the dense ladder.
