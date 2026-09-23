# Sprint 5 Model Training Audit: Sprint 4 Handoff & Spatial-Context Readiness

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Author**: Antigravity Research Agent  
**Date**: September 24, 2026  

---

## 1. Executive Audit Summary

This audit critically evaluates the experimental artifacts, methodologies, and data contracts produced in **Sprint 4 (History-Length Experiments: $H \in \{3, 5, 7, 10, 14\}$)** to establish the scientific foundation for **Sprint 5 (Spatial-Context $N/M$ Experiments)**.

### Key Audit Findings
1. **Validation Selection is Legitimate**: The selection of $H^* = 14$ as the champion temporal configuration is strictly grounded in the pre-declared multi-task validation objective ($\mathcal{L}_{\text{val}} = 0.0276$, a 28.3% reduction over $H=3$'s $0.0385$). It was not selected via test-set snooping.
2. **Reporting Inconsistency in Summary Table**: In `reports/sprint_4_history_comparison_table.md`, row $H=3$ reported 2022 validation metrics (due to lack of a test evaluation block in the reference run), whereas rows $H \in \{5, 7, 10, 14\}$ reported 2023 holdout test metrics. This created an apples-to-oranges presentation that must be separated into distinct Validation and Test tables.
3. **Training Loss Zero-Filling Identified**: The insertion of `train_loss = 0.0` in the summary table was caused by `scripts/benchmark_history_experiments.py` populating tabular records from the post-hoc `test_metrics` block rather than the epoch history. Real training loss values exist in all five individual JSON reports.
4. **Antecedent Observational Source Transition**: For $H=14$, dates prior to May 28 (May 18-27) use pure ECMWF ERA5 reanalysis via Open-Meteo, whereas the post-May 28 series combines CHIRPS (precipitation), ERA5-Land (thermodynamics), and ERA5 (winds). For Sprint 5, scientific internal control requires using the exact same provenance stack across all spatial context conditions ($N/M$).
5. **Overfitting Assessment**: Generalization gaps for $H=14$ remained narrow ($\Delta = 0.0058$ between train loss $0.0334$ and val loss $0.0276$), indicating healthy regularization rather than unconstrained memorization.
6. **Remaining Compute Headroom**: Post-sweep Kaggle GPU quota is **1.92 hours** (115.4 minutes). Sprint 5 must use local CPU data materialization, local unit tests, and selective GPU execution to stay within quota before the weekly refresh.

---

## 2. Sprint 4 Result Audit: Epoch-by-Epoch Metric Verification

An inspection of the five raw JSON artifacts (`reports/training_diffusion_h03_history.json` through `h14_history.json`) reveals the true convergence trajectory of the ~16.05M parameter backbone on Kaggle Dual Tesla T4 GPUs:

### Table 1: Raw Empirical Metric Audit (Validation Split: 2022 Season)

| Experiment | History Window | Total Epochs | Best Val Epoch | Training Loss (at Best Val) | Best Val Loss ($\mathcal{L}_{\text{val}}$) | Precip Wet-MAE (mm) | Precip CSI@30 | Tmax MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|---|
| `EXP-H03-REF` | $H = 3$ days | 19 (early stop) | Epoch 12 | 0.0508 | 0.0385 | 8.63 | 0.620 | 0.38 | 1.77 |
| `EXP-H05` | $H = 5$ days | 30 (full pass) | Epoch 23 | 0.0402 | 0.0309 | 8.99 | 0.621 | 0.36 | 1.67 |
| `EXP-H07` | $H = 7$ days | 26 (early stop) | Epoch 19 | 0.0377 | 0.0329 | 8.69 | 0.636 | 0.44 | 1.69 |
| `EXP-H10` | $H = 10$ days | 21 (early stop) | Epoch 14 | 0.0421 | 0.0350 | 8.78 | 0.618 | 0.47 | 1.71 |
| `EXP-H14` | $H = 14$ days | 30 (full pass) | Epoch 23 | **0.0334** | **0.0276** | **8.72** | **0.634** | **0.36** | **1.64** |

### Table 2: Confirmatory Holdout Test Evaluation (2023 Season, 122 Sequences)

*Evaluated post-hoc on the selected champion checkpoint (`models/checkpoints/temporal_diffusion_hXX_champion.pt`) using fixed DDIM-32 ($\eta=0.0$).*

| Experiment | History Window | Precip MAE (mm) | Precip Wet-MAE (mm) | Precip RMSE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Wind Vector RMSE (m/s) |
|---|---|---|---|---|---|---|---|---|
| `EXP-H03-REF` | $H = 3$ days | *Not evaluated* | *Not evaluated* | *Not evaluated* | *Not evaluated* | *Not evaluated* | *Not evaluated* | *Not evaluated* |
| `EXP-H05` | $H = 5$ days | 4.08 | 9.71 | 10.91 | 0.476 | 0.489 | 0.429 | 3.56 |
| `EXP-H07` | $H = 7$ days | 4.46 | 9.99 | 10.87 | 0.472 | 0.492 | 0.438 | 3.55 |
| `EXP-H10` | $H = 10$ days | 4.47 | 10.27 | 11.39 | 0.453 | 0.468 | 0.475 | 3.57 |
| `EXP-H14` | $H = 14$ days | **4.08** | **9.71** | **10.86** | **0.476** | **0.494** | **0.428** | **3.56** |

---

## 3. Methodological Caveats & Resolutions

### A. Test-Set Contamination / Model Selection Audit
- **Inquiry**: Did the selection of $H=14$ depend on the 2023 test set?
- **Finding**: No. In `scripts/train_temporal_downscaler.py`, line 457 explicitly drives checkpoint serialization:
  ```python
  if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      torch.save(state_dict, ckpt_path)
  ```
  Checkpoint saving is strictly driven by `avg_val_loss` evaluated on the 2022 validation split.
  The test-set evaluation occurs in lines 496-545 strictly *after* all 30 training epochs and early-stopping logic have concluded.
- **Resolution for Sprint 5**: The Sprint 5 handoff must explicitly state: **"Selected temporal configuration: $H^* = 14$, selected purely via the multi-task validation loss objective."** Test metrics are strictly post-selection confirmatory diagnostics.

### B. Reporting Artifacts in Summary Compiler
- **Inquiry**: Why did the comparison table in Sprint 4 report `0.0` for training loss and mix validation/test metrics?
- **Finding**: In `scripts/benchmark_history_experiments.py`, lines 54-73 prioritized `test_metrics` when building the summary record. If `test_metrics` existed, the compiler extracted test numbers and defaulted `train_loss: 0.0`. For $H=3$, `test_metrics` was absent, so it fell back to lines 75-100, which extracted validation metrics.
- **Resolution for Sprint 5**: `scripts/benchmark_history_experiments.py` is refactored into two distinct tables:
  1. Table A: Multi-Task Validation Objective and Convergence Table (Train Loss, Val Loss, Val Wet-MAE, Val CSI@30, Convergence Epoch).
  2. Table B: Confirmatory Holdout Test Performance (2023 Season).

### C. History-Source Consistency
- **Inquiry**: Does the antecedent extension (May 18-27) create an unfair advantage for $H=14$?
- **Finding**: For dates May 18-27, all atmospheric variables are retrieved from official ECMWF ERA5 via Open-Meteo. For dates May 28 onwards, precipitation is from CHIRPS v2.0, thermodynamics from ERA5-Land, and wind from ERA5. Because $H=14$ is the only model accessing the pre-May 28 window across all training samples, there is an observational source shift in the earliest antecedent days.
- **Resolution for Sprint 5**: 
  1. Sprint 5 is an internal spatial-context experiment where $H=14$ is held **completely fixed**.
  2. Every spatial context ratio ($N/M \in \{1.00, 1.25, 1.50, 2.00\}$) will use the exact same data provenance stack.
  3. Consequently, the observational source transition is held strictly invariant across all spatial conditions, ensuring zero confounding of spatial context findings.

### D. Overfitting and Generalization Claims
- **Inquiry**: Does a continuously dropping training loss prove absence of overfitting?
- **Finding**: No. In classical deep learning, a model can overfit while training loss decreases.
  However, for $H=14$, validation loss reached its minimum of `0.0276` at epoch 23 (with training loss `0.0334`), and at epoch 30 validation loss remained `0.0289` (training loss `0.0341`).
  The training-validation loss delta is minimal ($\approx 0.005$), and validation loss remained lower than training loss due to the standard Dropout/Noise augmentation present during training passes.
  This confirms generalization without catastrophic divergence, but does not imply zero generalization error.

---

## 4. Hardware & Quota Audit

### Current Kaggle Accelerator Resource Status
- **Kaggle API User**: `rohitajitbharadwaj`
- **Total Weekly GPU Quota**: 6.00 hours (Dual Tesla T4 16GB)
- **Used GPU Quota**: 4.08 hours
- **Remaining GPU Quota**: **1.92 hours (115.4 minutes)**
- **Next Quota Refresh**: Saturday, September 26, 2026, 00:00:00 UTC

### Compute Allocation Strategy for Sprint 5
With 115 minutes remaining, running four 30-epoch runs at 40 minutes each would exceed quota ($4 \times 40 = 160$ minutes).
Therefore, Sprint 5 adopts a strict, phased compute strategy:
1. **Phase 1: Local Ingestion & Validation (0 GPU minutes)**:
   All raw spatial slicing, Zarr store generation (`multitask_temporal_v3_spatial_context.zarr`), and PyTorch unit tests execute locally on CPU.
2. **Phase 2: Local 1-Batch Shape Verification (0 GPU minutes)**:
   Verify that the spatial halo crop encoder runs locally on CPU with zero dimension mismatch.
3. **Phase 3: Screen 3 Core Ratios on Kaggle**:
   - $N=16$ ($N/M = 1.00$, baseline control): 1 checkpoint / report already exists or can be matched.
   - $N=24$ ($N/M = 1.50$, mesoscale capture): 1 Kaggle GPU run (~30-35 mins).
   - $N=32$ ($N/M = 2.00$, synoptic cross-peninsular capture): 1 Kaggle GPU run (~35-40 mins).
   Total estimated Kaggle GPU consumption: ~70 minutes, leaving >45 minutes of safety buffer.
