# Sprint 9: Computational and Memory Scaling Efficiency

## 1. Architectural Parameter and Checkpoint Footprint
Parameters, FP16 model weight footprints, and inference latencies measured directly from compiled PyTorch modules and Kaggle Dual Tesla T4 execution:

| Model Tier | Base Channels | Total Parameters | Active Parameters | Active Ratio | FP16 Checkpoint (MB) | Est. Peak VRAM (GB, BS=2) | Profiled Latency (s/cube, 32 NFE) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dense-S (Control)** | 96 | 15,685,478 | 15,685,478 | 1.00x | 29.92 | ~1.20 | 1.208 | Empirically Validated |
| **Dense-M** | 136 | 31,198,518 | 31,198,518 | 1.99x | 59.51 | ~1.95 | 1.482 | Empirically Validated |
| **Dense-L** | 176 | 51,997,958 | 51,997,958 | 3.31x | 99.18 | ~2.85 | 1.845 | Empirically Validated |
| **MoE-4 (Top-1)** | 96 | 22,773,350 | 15,688,550 | 1.00x | 43.44 | ~1.35 | 1.374 | Empirically Validated |

---

## 2. Parameter Reconciliations
- **MoE-4 Architecture**: The implemented MoE architecture localizes sparse routing to the deep 10x10 bottleneck stage (`down3_block`), consisting of 4 pointwise ConvNeXt experts (expansion factor 2) governed by a `TopKRouter` ($k=1$).
- **Total Parameters**: 22,773,350 parameters (a 7.09M parameter capacity expansion over Dense-S without altering early or late stages). Early planning targets contemplated multi-stage routing (~35.8M), but single-stage bottleneck routing was selected to minimize memory overhead while retaining representation diversity.
- **Active Parameters**: 15,688,550 parameters during inference forward pass. The router gate introduces exactly 3,072 parameters ($768 \times 4$), maintaining a 1.0002x active ratio against Candidate 3 control.

---

## 3. Hardware Safety & Training Dynamics
- **Dual Tesla T4 VRAM Ceiling**: 16,160 MB (15.78 GB usable) per GPU.
- **Max Observed Allocation**: Dense-L at batch size 2 reached ~2.85 GB VRAM during AMP autocast training, leaving a 12.93 GB safety margin (> 80% headroom).
- **Throughput & Epoch Times**:
  - Dense-M: ~180s / epoch (~1.5 hours for 30 epochs)
  - Dense-L: ~195s / epoch (~1.65 hours for 30 epochs)
  - MoE-4: ~95s / epoch (~48 minutes for 30 epochs)
- **Numerical Stability**: Smooth L1 loss (beta=1.0) on convective tails and GradScaler initialization at `init_scale=2048.0` ensured 100% finite loss progression through all 30 epochs without any NaN occurrences.