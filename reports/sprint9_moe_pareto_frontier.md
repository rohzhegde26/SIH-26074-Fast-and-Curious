# Sprint 9: Complete Capacity & Sparse Routing Pareto Frontier

## 1. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)

| Model Tier | Total Params | Active Params | Active Ratio | Wet-MAE (mm)* | CSI@15 | CSI@30 | Fair-CRPS | Raw SSR | Range Cov (K=2) | Lap Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | 1.00x | 62.77 | 0.6087 | 0.6499 | 61.472 | 0.053 | 0.264 | 0.035 |
| **Dense-M** | 31,198,518 | 31,198,518 | 1.99x | 61.62 | 0.6209 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 |
| **Dense-L** | 51,997,958 | 51,997,958 | 3.31x | 61.85 | 0.6207 | 0.6602 | 56.020 | 0.068 | 0.223 | 0.084 |
| **MoE-4** | 22,773,350 | 15,688,550 | 1.00x | 61.56 | 0.6251 | 0.6521 | 59.712 | 0.067 | 0.278 | 0.044 |

*Note on Metric Protocol: The Wet-MAE figures above reflect the Sprint 9 exploratory notebook protocol (linear un-normalization with wet-mask threshold > 1.0 mm, without Sprint 8 member-wise physical bounds repair). While internally consistent for cross-model ranking, they are not directly comparable to Sprint 8's 6.51 mm physical repair metric.*

---

## 2. Hypothesis Evaluation (MoE vs Dense)
- **Active Parameter Efficiency**: MoE-4 operates at 15,688,550 active parameters (1.0002x Candidate 3), while expanding total capacity to 22.77M.
- **MoE-4 vs Dense-S Wet-MAE**: Delta = -1.21 mm (61.56 vs 62.77)
- **MoE-4 vs Dense-S CSI@15**: Delta = +0.0164 (0.6251 vs 0.6087)
- **MoE-4 vs Dense-S Fair-CRPS**: Delta = -1.760 (59.712 vs 61.472)
- **Multidimensional Trade-Off**: While MoE-4 matches or exceeds Dense-M on Wet-MAE and CSI@15 at 15.69M active compute, Dense-M and Dense-L retain advantages on extreme-event CSI@30 (0.6589 and 0.6602) and Fair-CRPS (56.67 and 56.02 mm).

---

## 3. MoE Routing Diagnostics
- **Soft Routing Entropy**: 1.000 (computed across continuous router probability distributions P_e).
- **Hard Top-1 Dispatch Frequencies (Final Validation Batch)**: `[85.7%, 7.1%, 0.0%, 7.1%]`.
- **Assignment Distribution**: On the final validation batch (14 tokens across 2 cubes x 7 lead days), 12 tokens routed to Expert 0 (85.7%), 1 token to Expert 1 (7.1%), 0 to Expert 2 (0.0%), and 1 to Expert 3 (7.1%). Expert 0 acts as primary backbone. Correlating experts to physical meteorological regimes across all cubes requires future conditioned clustering.


---

## 4. Evaluation Provenance and Baseline Reconciliation
- In Phase 2, Dense-S and MoE-4 were evaluated synchronously in an independent stochastic pass across all 122 validation cubes (Dense-S: Fair-CRPS 61.472 mm, Wet-MAE 62.77 mm). Within this run, MoE-4 delivered a -1.760 mm Fair-CRPS reduction (59.712 mm) and a -1.21 mm Wet-MAE reduction (61.56 mm).
- For comparison, Phase 1 evaluated Dense-S, Dense-M, and Dense-L together (Dense-S: Fair-CRPS 58.349 mm, Dense-M: 56.670 mm [-1.68 mm], Dense-L: 56.020 mm [-2.33 mm]). Both runs corroborate consistent error reductions from parameter expansion.
- With K=2 stochastic members, the interval coverage metric represents ensemble range coverage (min to max), not a 5th to 95th empirical percentile interval.

[+] Sprint 9 Phase 2 Execution Complete.