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

| Model Configuration | Total Params | Active Params | Top-$k$ | Active FLOP Ratio | 4-Step Latency (ms) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 15,685,478 | 15,685,478 | N/A | 1.00x | 95.6 | 185.4 |
| **Dense-M** | 31,198,518 | 31,198,518 | N/A | 1.99x | 101.6 | 245.2 |
| **Dense-L** | 51,997,958 | 51,997,958 | N/A | 3.31x | 114.2 | 320.8 |
| **MoE-4 (Top-1)** | 35,765,990 | 15,685,478 | 1 | 1.00x | 96.6 | 210.5 |
| **MoE-4 (Top-2)** | 35,765,990 | 22,378,982 | 2 | 1.43x | 98.4 | 225.0 |

## 4. Routing Dynamics and Regime Specialization
Routing entropy $H = -\sum_{e=1}^E P_e \log(P_e + \epsilon)$ and expert activation fractions were tracked across validation weather regimes:

| Meteorological Regime | Expert 1 (Stratiform) | Expert 2 (Convective Tail) | Expert 3 (Orographic Lifting) | Expert 4 (Thermal / Inversion) |
| :--- | :---: | :---: | :---: | :---: |
| **Dry / Low Rain (< 1 mm)** | 72.4% | 3.1% | 11.2% | 13.3% |
| **Moderate Rain (1 - 15 mm)** | 28.5% | 18.2% | 36.8% | 16.5% |
| **Heavy Storm (> 30 mm)** | 4.2% | 68.9% | 22.4% | 4.5% |
| **High Orographic Slope (> 15 deg)** | 8.1% | 21.3% | 61.2% | 9.4% |

### Key Diagnostic Observations:
1. **Regime Specialization Emerges Spontaneously**: Expert 2 activates primarily during severe convective storm events (> 30 mm, 68.9% load), while Expert 3 activates predominantly along steep mountain terrain (61.2% load).
2. **Zero Dead Experts**: Due to the Switch/GShard auxiliary loss, all four experts maintain minimum dispatch fractions $> 15\%$ across the global validation set, preventing expert collapse.
3. **Compute Decoupling**: MoE-4 delivers 35.8M total representation capacity with an active inference latency of 96.6 ms (virtually indistinguishable from Dense-S at 95.6 ms).
