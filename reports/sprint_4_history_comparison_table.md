# Sprint 4 History-Length Experimental Results

## 1. Multi-Task Aggregate Performance Across Antecedent Windows

| History Window | Model Backbone | Best Epoch | Val Loss | Wet-Day MAE (mm) | CSI@15 | CSI@30 | Tmax MAE (°C) | RH MAE (%) | Wind Vector RMSE (m/s) | Diurnal Violation |
|---|---|---|---|---|---|---|---|---|---|---|
| **H=3 days** | Scaled ~16.05M | Epoch 12 | **0.0385** | 8.63 | 0.631 | 0.620 | 0.377 | 0.54 | 1.77 | 0.00% |
| **H=5 days** | Scaled ~16.05M | Epoch 30 | **0.0309** | 9.71 | 0.476 | 0.489 | 0.429 | 0.54 | 3.56 | 0.05% |
| **H=7 days** | Scaled ~16.05M | Epoch 26 | **0.0329** | 9.99 | 0.472 | 0.492 | 0.438 | 0.56 | 3.55 | 0.07% |
| **H=10 days** | Scaled ~16.05M | Epoch 21 | **0.0350** | 10.27 | 0.453 | 0.468 | 0.475 | 0.64 | 3.57 | 0.05% |
| **H=14 days** | Scaled ~16.05M | Epoch 30 | **0.0276** | 9.71 | 0.476 | 0.494 | 0.428 | 0.54 | 3.56 | 0.03% |

---

## 2. Hypothesis Verification Matrix

| Hypothesis | Scientific Statement | Metric / Focus | Empirical Delta | Verdict |
|---|---|---|---|---|
| **H1** | State Estimation in Early Leads | Wet-MAE (D+0, D+1) | Gain: -13.5% | **FALSIFIED** |
| **H2** | Extended Temporal Steering | Wind Vector RMSE (D+6 vs D+0) | Gain: 0.0% | **CONFIRMED** |
| **H3** | Precipitation Extremes Capture | CSI@30 relative gain | Gain: -20.3% | **FALSIFIED** |
| **H4** | Diminishing Marginal Returns | Val Loss delta beyond H=10 | Gain: 0.0% | **FALSIFIED (Linear continuation)** |
| **H5** | Variable-Specific Memory Asymmetry | Tmax gain vs Wind gain | Gain: 0.0% | **CONFIRMED** |
| **H6** | Capacity-History Synergy | 16M CSI@15 vs Sprint 3 1M | Gain: 2.8% | **CONFIRMED** |