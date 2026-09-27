# Sprint 8 Matched-Compute Pareto Frontier

**Core Identity**: $\text{Total NFE} = K \times S$

| Budget | Configuration | Members ($K$) | Steps ($S$) | $\eta$ | Nominal NFE | CRPS Type | CRPS ↓ | Wet-MAE (mm) ↓ | CSI@30 ↑ | Latency (ms) | Speedup vs Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **4 NFE** | `PHASE0_GATE_DDIM4` | 1 | 4 | 0.00 | 4 | CRPS_det | **0.6310** | 4.94 | 0.739 | 4541.2 | 0.2x |
| **8 NFE** | `B8_K2_S4` | 2 | 4 | 0.00 | 8 | Fair-CRPS | **0.3540** | 4.47 | 0.758 | 8552.1 | 0.1x |
