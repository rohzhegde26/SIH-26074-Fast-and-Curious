# Sprint 5 Implementation Plan: Spatial-Context N/M Experiments for Spatiotemporal Weather Downscaling

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 5 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 24, 2026  

---

## 1. Executive Summary & Scientific Objective

### Core Scientific Question
**How much surrounding coarse-grid spatial context should the downscaler see when predicting the fixed target region, and what is the accuracy/compute trade-off as spatial context increases?**

### Principle of Scientific Isolation
Sprint 4 systematically evaluated the temporal dimension and identified $H^* = 14$ days as the optimal antecedent memory configuration via the validation objective ($\mathcal{L}_{\text{val}} = 0.0276$). 

Sprint 5 now **holds temporal context strictly frozen at $H^* = 14$** and isolates the **spatial context ratio ($N/M$)** as the sole independent variable. 
Model capacity (~15.69M parameters), training objective, optimizer, loss formulation, and the DDIM-32 sampling protocol are kept invariant.

---

## 2. Precise Mathematical Definition of $N$ and $M$

To eliminate all ambiguity between linear resolution, bounding boxes, and area scaling:

- **$M$ (Target Coarse Footprint)**:
  - Fixed coarse-grid linear footprint corresponding to the prediction area: **$M = 16$ coarse cells** ($16 \times 16$).
  - Native coarse resolution: $0.25^\circ \times 0.25^\circ$ ($\approx 27.75\text{ km} \times 27.75\text{ km}$).
  - Canonical geographical bounds: $[11.0^\circ\text{N}, 15.0^\circ\text{N}] \times [74.0^\circ\text{E}, 78.0^\circ\text{E}]$ ($4.0^\circ \times 4.0^\circ$ Peninsular domain covering Mandya and the Southern Karnataka Western Ghats).
  - Downscaling factor: Exactly **$5\times$** spatial scale.
  - Fine reference target: **$80 \times 80$ fine cells** ($0.05^\circ \times 0.05^\circ$, $[11.025^\circ\text{N}, 14.975^\circ\text{N}] \times [74.025^\circ\text{E}, 77.975^\circ\text{E}]$).
- **$N$ (Input Coarse Spatial Context Footprint)**:
  - Input coarse-grid linear dimension encompassing the target footprint: **$N \ge M$**.
  - The target region $M \times M$ is always located strictly at the spatial center of the $N \times N$ field:
    $$\text{Padding per side} = p = \frac{N - M}{2} \text{ coarse cells} = p \times 0.25^\circ$$
- **Linear Spatial Context Ratio**:
  $$\text{Linear Ratio} = \frac{N}{M}$$
- **Context Area Ratio**:
  $$\text{Area Ratio} = \left(\frac{N}{M}\right)^2$$

### Table 1: Spatial Context Candidate Matrix

| Configuration | Coarse Grid ($N \times N$) | Linear Ratio ($N/M$) | Area Ratio $(N/M)^2$ | Coarse Cell Padding ($p$) | Angular Bounding Box | Physical Extent ($\approx \text{km}$) | Synoptic / Meteorological Scale |
|---|---|---|---|---|---|---|---|
| **$N_{16}$ (Control)** | $16 \times 16$ | **1.00** | **1.00x** | 0 cells ($0.0^\circ$) | $[11.0^\circ, 15.0^\circ]\text{N} \times [74.0^\circ, 78.0^\circ]\text{E}$ | $440 \times 440\text{ km}$ | Local Target Basin only |
| **$N_{20}$** | $20 \times 20$ | **1.25** | **1.56x** | 2 cells ($0.5^\circ$) | $[10.5^\circ, 15.5^\circ]\text{N} \times [73.5^\circ, 78.5^\circ]\text{E}$ | $550 \times 550\text{ km}$ | Coastal Arabian Sea margin |
| **$N_{24}$** | $24 \times 24$ | **1.50** | **2.25x** | 4 cells ($1.0^\circ$) | $[10.0^\circ, 16.0^\circ]\text{N} \times [73.0^\circ, 79.0^\circ]\text{E}$ | $660 \times 660\text{ km}$ | Offshore Marine Boundary Layer |
| **$N_{32}$** | $32 \times 32$ | **2.00** | **4.00x** | 8 cells ($2.0^\circ$) | $[9.0^\circ, 17.0^\circ]\text{N} \times [72.0^\circ, 80.0^\circ]\text{E}$ | $880 \times 880\text{ km}$ | Cross-Peninsular Synoptic Wave |
| **$N_{40}$ (Optional)** | $40 \times 40$ | **2.50** | **6.25x** | 12 cells ($3.0^\circ$) | $[8.0^\circ, 18.0^\circ]\text{N} \times [71.0^\circ, 81.0^\circ]\text{E}$ | $1100 \times 1100\text{ km}$ | Sub-Continental Monsoonal Jet |

---

## 3. Deep Research: Meteorological Justification for Context Scales

### Why Spatial Context Matters in Indian Monsoon Downscaling
In standard super-resolution (computer vision), image features are largely self-contained. 
In atmospheric downscaling, however, the atmosphere is an advective, non-local dynamical system governed by the Navier-Stokes and continuity equations:
1. **Upstream Moisture Flux**: Monsoon precipitation in Karnataka is driven by the Southwesterly Low-Level Jet (Findlater Jet) transporting moisture from the Arabian Sea across $71^\circ-74^\circ\text{E}$. A $16 \times 16$ domain bounded at $74^\circ\text{E}$ truncates this upstream marine boundary layer at the coastline, forcing the model to infer precipitation without observing the approaching moisture column.
2. **Orographic Blocking & Hydraulic Jumps**: The Western Ghats barrier ($\approx 75.0^\circ-75.5^\circ\text{E}$) generates windward updrafts and leeward rain-shadow subsidence. Accurately modeling rain shadow intensity in Mandya requires observing the full windward-leeward pressure gradient and upstream Froude number across at least $1.0^\circ-2.0^\circ$ upstream of the barrier.
3. **Synoptic Vorticity & Monsoon Depressions**: Mesoscale convective systems (MCS) have spatial scales of $200-500\text{ km}$. Capturing incoming convective organization requires a linear context of at least $N=24$ ($660\text{ km}$).

---

## 4. Input Stream Spatial Conditioning Specification

Which streams receive wider spatial context?

1. **Future Forecast Conditioning**: **Receives $N \times N$**.
   - Input: NOAA GFS $0.25^\circ$ forecasts spanning leads $D$ through $D+6$.
   - Shape: $[B, 7, 6, N, N]$.
   - Justification: Synoptic steering and incoming frontal/convective boundaries must be observed in the coarse forecast prior to crossing into the target boundary.
2. **Historical Context ($H=14$)**: **Receives $N \times N$**.
   - Input: Antecedent observations spanning $D-14$ through $D-1$.
   - Shape: $[B, 14, 6, N, N]$.
   - Justification: Antecedent soil moisture anomalies and regional thermal cold pools upwind of the target influence convective triggering over the target region.
3. **Fine Terrain Prior**: **Maintains Fixed $80 \times 80$ Target Footprint**.
   - Shape: $[5, 80, 80]$.
   - Justification: The prediction objective is strictly the central $80 \times 80$ fine target. Injecting fine terrain over $N \times N$ would require a massive $5N \times 5N$ grid (e.g. $160 \times 160$ for $N=32$), exploding memory without providing additional target supervision.
4. **Supervision Targets**: **Strictly Fixed $80 \times 80$ Central Footprint**.
   - Shape: $[B, 7, 6, 80, 80]$.
   - Zero future observations or non-target regions enter the loss objective.

---

## 5. Architectural Mechanism: Halo-Context Encoder with Central RoI Cropping

### The Architectural Pitfall to Avoid
Sprint 5 **must NOT** resize or interpolate the $N \times N$ input field to $80 \times 80$. 
Interpolating a $24 \times 24$ or $32 \times 32$ field to $80 \times 80$ would physically alter the spatial scale factor (from $5\times$ to $2.5\times$), distorting cell resolution and destroying the meaning of spatial context.

### Proposed Architecture: Capacity-Matched Halo Encoder
The network consists of three functional stages:

```
[History: B, 14, 6, N, N]  [Forecast: B, 7, 6, N, N]
            \                 /
             \               /
     Spatiotemporal Halo-Context Encoder
           (3 ConvNeXt Blocks over N x N)
                       |
               [B, 96, N, N]
                       |
         Central RoI Spatial Cropping
         offset = (N - 16) // 2
         slice[offset : offset + 16, offset : offset + 16]
                       |
               [B, 96, 16, 16]
                       |
     Multi-Task 5x Spatial Upsampler
                       |
               [B, 96, 80, 80]  <--- Concatenate Fine Terrain [B, 5, 80, 80]
                       |
     Residual Diffusion U-Net Backbone
           (Operates strictly on 80 x 80)
                       |
           Output Target: [B, 7, 6, 80, 80]
```

### Mathematical Guarantee of Parameter-Count Fairness
- In 2D convolutions (kernel size $k$, in-channels $C_{\text{in}}$, out-channels $C_{\text{out}}$), the number of parameters is:
  $$\text{Params} = C_{\text{out}} \times C_{\text{in}} \times k \times k + C_{\text{out}}$$
- Parameter counts are **completely independent of the spatial dimension $N$**.
- Normalization layers (GroupNorm) and activation layers (GELU) have zero spatial parameter dependence.
- Therefore:
  $$\text{Params}(N=16) = \text{Params}(N=20) = \text{Params}(N=24) = \text{Params}(N=32) = \mathbf{15,685,478}$$
- This guarantees **zero capacity confounding**: any difference in validation loss or CSI@30 is 100% attributable to the spatial receptive field context, not model scaling.

### Computational Complexity Distinction
While parameters remain identical, spatial scaling introduces predictable computational trade-offs:
- **Encoder FLOPs**: Scale strictly as $\mathcal{O}(N^2)$.
- **Backbone FLOPs**: Scale as $\mathcal{O}(80^2)$ (invariant across all $N$, because diffusion operates strictly on the cropped central $80 \times 80$ space).
- **Peak Activation Memory**: Increases moderately in the shallow encoder stage ($\mathcal{O}(N^2)$), but remains dominated by the 8-lead diffusion reverse sampling passes on $80 \times 80$.

---

## 6. Dataset Extension Architecture: `multitask_temporal_v3_spatial_context.zarr`

### Versioning Rule
To prevent data corruption, `datasets/multitask_temporal_v2_h14.zarr` remains **frozen and immutable**.
A new versioned store is created: `datasets/multitask_temporal_v3_spatial_context.zarr`.

### Efficient Unified Store Design ($N_{\text{max}} = 32$)
Instead of generating four separate redundant Zarr stores for $N \in \{16, 20, 24, 32\}$, we materialize one master Zarr store at the maximum context dimension **$N_{\text{max}} = 32$** ($8.0^\circ \times 8.0^\circ$):

```
datasets/multitask_temporal_v3_spatial_context.zarr/
  ├── history:          [1098, 14, 6, 32, 32]  (float32, chunked [16, 14, 6, 32, 32])
  ├── future_forecast:  [1098, 7, 6, 32, 32]   (float32, chunked [16, 7, 6, 32, 32])
  ├── target:           [1098, 7, 6, 80, 80]   (float32, chunked [16, 7, 6, 80, 80])
  ├── terrain:          [5, 80, 80]            (float32)
  ├── dates:            [1098]                 (string)
  ├── splits:           [1098]                 (string: 854 train, 122 val, 122 test)
  └── .zattrs:
        ├── center_lat_bounds: [11.0, 15.0]
        ├── center_lon_bounds: [74.0, 78.0]
        ├── max_context_lat_bounds: [9.0, 17.0]
        ├── max_context_lon_bounds: [72.0, 80.0]
        ├── coarse_resolution_deg: 0.25
        ├── fine_resolution_deg: 0.05
        └── target_downscaling_factor: 5
```

### Dynamic Slicing Interface in PyTorch Dataset
The updated PyTorch Dataset `SpatiotemporalSpatialContextDataset` accepts an argument `context_size: int = 16` (where $16 \le N \le 32$).
During loading, it crops the central $N \times N$ region on the fly:
```python
crop_offset = (32 - self.context_size) // 2
history_N = history_32[:, :, crop_offset:crop_offset+N, crop_offset:crop_offset+N]
forecast_N = forecast_32[:, :, crop_offset:crop_offset+N, crop_offset:crop_offset+N]
```
This enables running all spatial context experiments ($N=16, 20, 24, 32$) from **one single validated dataset artifact** without redundant disk storage.

---

## 7. Explicit Scientific Hypotheses & Falsification Matrix

| Hypothesis | Description | Quantitative Metric | Expected Observation | Falsification Criterion |
|---|---|---|---|---|
| **H1: Upstream Moisture Capture** | Surrounding spatial context improves precipitation accuracy by capturing upstream advection from the Arabian Sea. | Precipitation Wet-MAE & RMSE | Wet-MAE decreases by $\ge 5.0\%$ from $N=16$ to $N=24$. | Wet-MAE increases or changes by $< 1.0\%$. |
| **H2: Nonlocal Dynamic Steering** | Surface wind fields benefit more from wide spatial context than thermodynamic fields due to synoptic pressure gradient resolution. | Relative gain in Wind RMSE vs Tmax MAE | $\Delta_{\text{Wind}} > 1.5 \times \Delta_{\text{Tmax}}$ from $N=16$ to $N=32$. | $\Delta_{\text{Wind}} \le \Delta_{\text{Tmax}}$. |
| **H3: Convective Extreme Recall** | Wider spatial context improves extreme cloudburst recall (CSI@30) by identifying mesoscale convective organizations upwind. | Precipitation CSI@30 | CSI@30 increases monotonically from $N=16$ to $N=24$. | CSI@30 drops or saturates before $N=20$. |
| **H4: Context Saturation Boundary** | Beyond a critical spatial scale ($N \approx 24-32$), marginal gains diminish as distant boundary noise dilutes local physical constraints. | Marginal Validation Loss $\Delta \mathcal{L}_{\text{val}}$ | $\Delta \mathcal{L}(24 \to 32) < 0.25 \times \Delta \mathcal{L}(16 \to 24)$. | $\Delta \mathcal{L}$ continues linear improvement beyond $N=32$. |
| **H5: Accuracy-Compute Pareto Frontier** | The relationship between downscaling skill and computational cost forms an optimal Pareto inflection point at $N/M = 1.50$ ($N=24$). | Validation Loss / TFLOPs efficiency | $N=24$ achieves the lowest loss-per-compute cost ratio. | $N=32$ or $N=16$ yields superior Pareto efficiency. |
| **H6: Lead-Time Sensitivity** | Later forecast horizons ($D+4$ to $D+6$) benefit more from wider spatial context than early leads ($D+0, D+1$) due to advective transit times. | Error reduction $\Delta(D+6)$ vs $\Delta(D+0)$ | $\frac{\Delta\text{RMSE}_{D+6}}{\text{RMSE}_{D+6}} > \frac{\Delta\text{RMSE}_{D+0}}{\text{RMSE}_{D+0}}$ | Early leads gain equal or greater relative benefit than late leads. |

---

## 8. Pre-Declared Model Selection & Evaluation Protocol

To prevent test-set leakage or subjective post-hoc cherry-picking:

1. **Pre-Declared Champion Selection Rule**:
   - Primary Criterion: **Lowest Multi-Task Validation Loss ($\mathcal{L}_{\text{val}}$)** on the 2022 validation season.
   - Tie-Breaker 1: Highest Validation Precipitation CSI@30.
   - Tie-Breaker 2: Lowest Validation Wind Vector RMSE.
   - Tie-Breaker 3: Lowest Training Wall-Clock Time / Compute Cost.
2. **Confirmatory Holdout Evaluation**:
   - The 2023 holdout season is evaluated strictly *after* the champion $N^*$ configuration is selected.
   - Reported as a post-selection confirmatory diagnostic across all meteorological variables.
3. **Fixed Sampler Protocol**:
   - Sampling Protocol: **DDIM with 32 deterministic reverse steps ($\eta=0.0$)**.
   - No sampler tuning or step sweeps are permitted in Sprint 5.

---

## 9. Phased Execution Gates & Quota Safety

### Compute Budget Constraints
- Current Kaggle GPU quota: **1.92 hours (115.4 minutes)**.
- Quota floor invariant: Dispatches halt immediately if quota drops below **0.50 hours (30 minutes)**.

### Phased Roadmap

```
PHASE 1: Data Acquisition & Materialization (Local CPU, 0 GPU hours)
  ├── 1.1 Ingest authentic ERA5/GFS for N=32 bounding box ([9°N, 17°N] x [72°E, 80°E])
  ├── 1.2 Build datasets/multitask_temporal_v3_spatial_context.zarr
  └── 1.3 Validate shapes, anti-leakage invariants, and coordinate registration

PHASE 2: Architecture & Local TDD Verification (Local CPU, 0 GPU hours)
  ├── 2.1 Implement HaloContextEncoder with Central RoI Cropping
  ├── 2.2 Verify parameter count is exactly 15,685,478 across all N in {16, 20, 24, 32}
  └── 2.3 Unit tests: tests/models/test_spatial_context_encoder.py (100% green)

PHASE 3: Remote Kaggle Execution (Dual Tesla T4 GPUs)
  ├── 3.1 Timing Probe: 1-epoch benchmark on N=24 to establish exact GPU runtime
  ├── 3.2 EXP-N16: Baseline control (N/M = 1.00)
  ├── 3.3 EXP-N24: Mesoscale context (N/M = 1.50)
  └── 3.4 EXP-N32: Synoptic context (N/M = 2.00)

PHASE 4: Benchmarking & Reporting (Local CPU, 0 GPU hours)
  ├── 4.1 Compile reports/sprint_5_spatial_benchmark_summary.json
  ├── 4.2 Generate reports/sprint_5_spatial_comparison_table.md
  └── 4.3 Evaluate hypotheses H1 through H6 and establish Sprint 6 Handoff
```

---

## 10. Definition of Done & No-Go Conditions

### Definition of Done (DoD)
- [ ] Master spatial context Zarr store materialized and verified with zero NaNs.
- [ ] Parameter count mathematically verified as exactly capacity-matched across all $N$.
- [ ] Unit test suite passing 100% locally.
- [ ] All spatial context experiments completed on Kaggle Dual Tesla T4 GPUs within compute budget.
- [ ] Champion checkpoints ($N^*$) and comprehensive reports retrieved into repo canonical locations.
- [ ] Pre-declared validation rule selects the winning spatial configuration $N^*$.
- [ ] Final comparison tables and hypothesis verification report committed and pushed to git.

### No-Go Conditions
- **DO NOT** resize the $N \times N$ input field to $80 \times 80$.
- **DO NOT** alter the historical context length from $H^* = 14$.
- **DO NOT** expand the prediction target footprint beyond the central $16 \times 16$ coarse ($80 \times 80$ fine) region.
- **DO NOT** evaluate or select $N^*$ based on 2023 holdout test metrics.
- **DO NOT** exceed the Kaggle GPU quota safety floor.
