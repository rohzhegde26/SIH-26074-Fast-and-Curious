# Sprint 9 Walkthrough: Model Capacity Scaling (Dense -> Larger Dense -> MoE)

## 1. Context and Objective
Following Sprint 8.5's bottleneck diagnosis, which proved that post-hoc calibration alone cannot fully overcome conditional under-dispersion, Sprint 9 evaluates whether neural model capacity scaling moves the accuracy-uncertainty-sharpness Pareto frontier outward.

## 2. Capacity Ladder Design
The capacity ladder isolates base channel width while keeping all other architectural and physical invariants locked:
- **Dense-S (Control)**: 15.69M parameters (Candidate 3 baseline).
- **Dense-M**: 31.20M parameters (1.99x scale, base_channels = 136).
- **Dense-L**: 52.00M parameters (3.31x scale, base_channels = 176).
- **MoE-4**: 22.77M total parameters with 15.69M active parameters (sparse Top-1 routing over 4 bottleneck experts).

## 3. Matched 32 NFE Inference Regimes
Each model tier is audited under matched inference compute:
1. **Point Forecast Mode**: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE). Focuses on deterministic Wet-MAE, extreme convective storm recall (CSI@15, CSI@30), and spatial textures.
2. **Distribution Mode**: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE). Focuses on probabilistic Fair-CRPS, Spread-Skill Ratio (SSR), 90% prediction interval coverage, and Brier Skill Scores.

## 4. Empirical Scaling Results (2022 Validation Dataset)

| Model Tier | Total Params | Active Params | Active Ratio | Wet-MAE (mm) | CSI@15 | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Retention ($R_{\text{Lap}}$) | Latency (s/cube) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | 1.00x | 62.77 | 0.6087 | 0.6499 | 61.472 | 0.053 | 0.264 | 0.035 | 1.208 |
| **Dense-M** | 31,198,518 | 31,198,518 | 1.99x | 61.62 | 0.6209 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 | 1.482 |
| **Dense-L** | 51,997,958 | 51,997,958 | 3.31x | 61.85 | 0.6207 | **0.6602** | **56.020** | **0.068** | 0.223 | **0.084** | 1.845 |
| **MoE-4 (Top-1)** | 22,773,350 | **15,688,550** | **1.00x** | **61.56** | **0.6251** | 0.6521 | 59.712 | 0.067 | **0.278** | 0.044 | 1.374 |

*Note: Wet-MAE and Fair-CRPS reflect the Sprint 9 exploratory notebook protocol (linear un-normalization with wet-mask > 1.0 mm, without Sprint 8 member-wise physical bounds repair). They are internally self-consistent across tiers, but distinct from Sprint 8's 6.51 mm physical repair benchmark. Coverage reflects K=2 ensemble range coverage.*

## 5. Artifacts and Checkpoints
- `models/checkpoints/sprint9_dense_m_weights.pt` (124.8 MB)
- `models/checkpoints/sprint9_dense_l_weights.pt` (208.0 MB)
- `models/checkpoints/sprint9_moe4_weights.pt` (91.2 MB)
- `reports/sprint9_moe_pareto_frontier.md`
- `reports/sprint9_moe_scaling_results.json`
- `reports/sprint9_capacity_frontier.md`
- `reports/sprint9_dense_scaling_results.json`
