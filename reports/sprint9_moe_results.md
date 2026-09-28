# Sprint 9 Sparse Mixture of Experts (MoE) Scaling Report

## 1. Executive Summary
This report analyzes the empirical routing behavior, quantitative skill, and efficiency tradeoffs of integrating sparse Mixture of Experts (MoE) into the spatiotemporal residual diffusion denoiser for Sprint 9:
- **Total Parameters**: 22,773,350 (MoE-4)
- **Active Parameters at Inference**: 15,688,550 (1.0002x Candidate 3 control active compute)
- **Architecture**: 4 Bottleneck Experts with Top-1 sparse routing at the 10x10 bottleneck stage
- **Auxiliary Loss**: Switch / GShard load balancing with coefficient $\alpha_{\text{aux}} = 0.01$
- **Validation Dataset**: Full authentic 2022 validation season (122 forecast cubes, 427 daily slices)
- **Inference Regimes**: Matched 32 NFE (Point Mode: K=8, S=4, eta=0.5; Distribution Mode: K=2, S=16, eta=0.0)

---

## 2. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)

| Model Configuration | Total Params | Active Params | Active Ratio | Wet-MAE (mm)* | CSI@15 | CSI@30 | Fair-CRPS | Raw SSR | Range Cov (K=2) | Lap Retention ($R_{\text{Lap}}$) | Profiled Latency (s/cube) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | 1.00x | 62.77 | 0.6087 | 0.6499 | 61.472 | 0.053 | 0.264 | 0.035 | 1.208 |
| **Dense-M** | 31,198,518 | 31,198,518 | 1.99x | 61.62 | 0.6209 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 | 1.482 |
| **Dense-L** | 51,997,958 | 51,997,958 | 3.31x | 61.85 | 0.6207 | **0.6602** | **56.020** | **0.068** | 0.223 | **0.084** | 1.845 |
| **MoE-4 (Top-1)** | 22,773,350 | **15,688,550** | **1.00x** | **61.56** | **0.6251** | 0.6521 | 59.712 | 0.067 | **0.278** | 0.044 | 1.374 |

*Important Evaluation Protocol Note: The Wet-MAE figures reported above (61.56 to 62.77 mm) were computed under the Sprint 9 exploratory notebook protocol (linear un-normalization with wet-mask threshold > 1.0 mm, without Sprint 8 member-wise physical bounds repair). While internally consistent for ranking Dense-S, Dense-M, Dense-L, and MoE-4, these values are NOT directly comparable to Sprint 8's physical repair benchmark (Candidate 3 Wet-MAE ~6.51 mm under p_tgt > 2.5 mm).*

---

## 3. Quantitative Analysis & Hypothesis Verification

### H6: MoE Efficiency Trade-Off (Partially Supported - Selective Efficiency Win)
- **Point Accuracy**: MoE-4 achieves the lowest Wet-MAE (**61.56 mm**) across all models, outperforming both Dense-S (62.77 mm, -1.21 mm delta) and dense scaled models (Dense-M 61.62 mm, Dense-L 61.85 mm).
- **Moderate Precipitation Recall**: MoE-4 delivers the highest CSI@15 (**0.6251**), outperforming Dense-S (0.6087, +0.0164 delta) and Dense-M (0.6209).
- **Active Compute Decoupling**: MoE-4 executes with only 15.69M active parameters during inference (1.0002x ratio vs Candidate 3), preserving low latency (1.37s vs 1.85s for Dense-L).
- **Trade-Off Boundary**: MoE-4 does not uniformly dominate Dense-M or Dense-L across all axes. Dense-M retains better Fair-CRPS (56.67 vs 59.71) and CSI@30 (0.6589 vs 0.6521), while Dense-L achieves the highest extreme storm recall (CSI@30 = 0.6602) and spatial texture (0.084). MoE-4 represents a specialized active-compute efficiency strategy rather than a universal quality replacement.

### Spatial Detail & Convective Extremes Tradeoff
- While MoE-4 dominates point accuracy and efficiency, **Dense-L** achieves superior high-frequency spatial Laplacian retention (**0.084**, +140% over control) and highest CSI@30 (**0.6602**), showing that full-width dense representations excel at resolving sharp convective storm boundaries.

---

## 4. Empirical Routing Diagnostics

- **Soft Routing Entropy**: The router outputs a normalized entropy of 1.000 over the continuous softmax gating distribution $P_e$, confirming that the gating network assigns continuous non-zero probability mass across all 4 expert pathways prior to selection.
- **Hard Top-1 Dispatch Frequencies (Final Validation Batch)**: Measured on the final validation batch (14 spatiotemporal tokens across 2 cubes and 7 lead days), hard Top-1 expert assignments were:
  $$\mathbf{f} = [0.857, 0.071, 0.000, 0.071]$$
- **Dispatch Interpretation**:
  - Expert 0 serves as the primary backbone denoiser, receiving 12 of 14 token assignments (85.7%).
  - Experts 1 and 3 receive 1 token assignment each (7.1%), capturing non-modal patterns.
  - Expert 2 received zero Top-1 assignments in this batch, indicating that discrete selection concentrated on 3 of the 4 available experts.
  - Note: Attributing specific experts to named meteorological phenomena (such as stratiform flow or orographic shear) requires conditioned clustering analysis across all validation cubes and remains an area for future empirical investigation.


---

## 5. Architectural Recommendation for Edge Deployment
- **Edge / Panchayat Serving**: Deploy **MoE-4** as the primary serving engine. It delivers the lowest Wet-MAE (61.56 mm) and highest CSI@15 (0.6251) while executing within the tight 15.7M active parameter budget required for low-latency block/panchayat inference.
- **High-Performance Cluster / Severe Storm Warnings**: Deploy **Dense-L** for specialized severe storm workflows where maximum Laplacian texture sharpness (0.084) and extreme CSI@30 recall (0.6602) justify higher compute.
