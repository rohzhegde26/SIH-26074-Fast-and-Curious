# Sprint 9: Computational and Memory Scaling Efficiency

## 1. Architectural Parameter and Checkpoint Footprint
Parameters and FP16 model weight footprints measured directly from compiled PyTorch modules:

| Model Tier | Base Channels | Total Parameters | Active Parameters | Param Ratio | FP16 Checkpoint (MB) | Est. Peak VRAM (GB, BS=2) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense-S (Candidate 3)** | 96 | 15,685,478 | 15,685,478 | 1.00x | 29.92 | ~1.20 | Validated Baseline |
| **Dense-M** | 136 | 31,198,518 | 31,198,518 | 1.99x | 59.51 | ~1.95 | Phase 1 Target |
| **Dense-L** | 176 | 51,997,958 | 51,997,958 | 3.31x | 99.18 | ~2.85 | Phase 1 Target |
| **MoE-4 (Top-1)** | 96 | 35,765,990 | 15,685,478 | 2.28x | 68.22 | ~1.65 | Phase 2 Target |

---

## 2. Invariant Hardware Safety Check
- **Dual Tesla T4 VRAM Ceiling**: 16,160 MB (15.78 GB usable) per GPU.
- **Max Expected Allocation**: Dense-L at batch size 2 requires ~2.85 GB VRAM during AMP autocast training, leaving a 12.93 GB safety margin (> 80% headroom).
- **Inference Latency Profile**: In Candidate 3 (Dense-S), DDIM-4 takes 96.8 ms (24.2 ms/step). Dense-M is estimated at ~115 ms, Dense-L at ~145 ms, and MoE-4 at ~102 ms on Dual Tesla T4 accelerators.