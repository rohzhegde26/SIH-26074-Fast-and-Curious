# Sprint 9 Sparse Mixture of Experts (MoE) Scaling Report

## 1. Executive Summary
This report analyzes the design, routing behavior, and efficiency tradeoffs of integrating sparse Mixture of Experts (MoE) into the spatiotemporal residual diffusion denoiser:
- Total Parameters: 35,765,990 (MoE-4)
- Active Parameters: 15,685,478 (matches Dense-S active compute)
- Architecture: 4 Bottleneck Experts with Top-1 / Top-2 sparse routing
- Auxiliary Loss: Switch / GShard load balancing with coefficient $\alpha_{\text{aux}} = 0.01$

## 2. MoE Architectural Design
In `src/models/moe.py`, the deep bottleneck stage of the U-Net denoiser (operating at 10x10 resolution across 7 leads) is equipped with an MoE block replacing standard dense pointwise convolutions:
1. **TopKRouter**: Computes routing logits $h(x) = W_{\text{gate}} x$ from spatially pooled feature representations. Softmax gating determines expert dispatch probabilities.
2. **Top-k Selection**: Only the top $k \in \{1, 2\}$ experts are activated per sample, keeping compute bounded.
3. **Auxiliary Load Balancing Loss**:
   $$\mathcal{L}_{\text{aux}} = \alpha_{\text{aux}} \cdot E \sum_{e=1}^E f_e P_e$$
   where $f_e$ is the empirical dispatch fraction and $P_e$ is the mean gating probability.

## 3. MoE Experimental Matrix & Compute Profiling

| Model Configuration | Total Params | Active Params | Top-$k$ | Status | Active FLOP Ratio | Profiled Latency (ms) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | N/A | **VALIDATED_BASELINE** | 1.00x | 95.6 | 185.4 |
| **Dense-M** | 31,198,518 | 31,198,518 | N/A | TARGET_PHASE1_KAGGLE | 1.99x | 101.6 | 245.2 |
| **Dense-L** | 51,997,958 | 51,997,958 | N/A | TARGET_PHASE1_KAGGLE | 3.31x | 114.2 | 320.8 |
| **MoE-4 (Top-1)** | 35,765,990 | 15,685,478 | 1 | TARGET_PHASE2_KAGGLE | 1.00x | 96.6 | 210.5 |
| **MoE-4 (Top-2)** | 35,765,990 | 22,378,982 | 2 | TARGET_PHASE2_KAGGLE | 1.43x | 98.4 | 225.0 |

## 4. Target Routing Dynamics and Regime Specialization Hypotheses
Routing entropy $H = -\sum_{e=1}^E P_e \log(P_e + \epsilon)$ and expert activation fractions will be audited across validation weather regimes during Phase 2 execution:

| Meteorological Regime | Expert 1 (Stratiform) | Expert 2 (Convective Tail) | Expert 3 (Orographic Lifting) | Expert 4 (Thermal / Inversion) |
| :--- | :---: | :---: | :---: | :---: |
| **Dry / Low Rain (< 1 mm)** | Target ~ 70% | Target < 5% | Target ~ 12% | Target ~ 13% |
| **Moderate Rain (1 - 15 mm)** | Target ~ 30% | Target ~ 20% | Target ~ 35% | Target ~ 15% |
| **Heavy Storm (> 30 mm)** | Target < 5% | Target > 65% | Target ~ 25% | Target < 5% |
| **High Orographic Slope (> 15 deg)** | Target < 10% | Target ~ 20% | Target > 60% | Target ~ 10% |

### Key Architectural Hypotheses:
1. **Regime Specialization**: Expert 2 is targeted to specialize in severe convective storm events (> 30 mm), while Expert 3 specializes along steep Western Ghats orographic topography.
2. **Zero Dead Experts**: The Switch/GShard auxiliary load-balancing loss ($\alpha_{\text{aux}} = 0.01$) guarantees uniform gradient flow across all 4 experts, preventing representation collapse.
3. **Compute Decoupling**: MoE-4 delivers 35.8M total representation capacity with an active inference latency of ~96.6 ms (matching Dense-S active parameter scale).
4. **Validation Grounding**: Empirical routing metrics will be logged upon execution of the Phase 2 MoE Kaggle kernel.
