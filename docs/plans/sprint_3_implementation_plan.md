# Implementation Plan: Sprint 3 Multivariate Spatiotemporal Downscaler & Kaggle Training Architecture

This plan establishes the model training architecture and remote accelerator dispatch workflow for Sprint 3 of the SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level).

---

## 1. Executive Summary & Data Freeze Verification

Sprint 2 data engineering and validation are certified complete:
- **Frozen Zarr Store**: `datasets/multitask_temporal_v1.zarr` (1,098 total samples: 854 train, 122 val, 122 test).
- **Hard Quality Gates**: All 8 Tier 1 Quality Gates verified passed (100% pass rate in `data/dataset_qa_report.md`).
- **Strict Anti-Leakage & Provenance**: GFS forecast extraction verified with authentic GRIB magic headers and zero synthetic NWP fallback.
- **Normalization Fitted on Train**: Train-only parameters persisted in `data/normalization_stats.yaml` with invertible round-trip error < 1e-5.

Sprint 3 transitions the project from data engineering to predictive model architecture and accelerator execution on Kaggle.

```text
========================================================================================
                               SPRINT 3 DATAFLOW PIPELINE
========================================================================================

Antecedent History (D-3, D-2, D-1)        Coarse Forecast NWP (D ... D+6)
       [B, 3, 6, 16, 16]                         [B, 7, 6, 16, 16]
               │                                         │
               ▼                                         ▼
   1D Temporal Convolution                  Bilinear Upsampling to 80x80
   & Multi-Lead Feature Projection          + High-Res Static Terrain [B, 5, 80, 80]
               │                                         │
               └───────────────────┬─────────────────────┘
                                   │
                                   ▼
                   Spatiotemporal Backbone (U-Net 5x)
                     - Multi-Scale ConvNeXt Stages
                     - Temporal Cross-Attention across 7 Leads
                     - Inverted Bottleneck Latent Fusion
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
       [Phase 1 Baseline Mode]         [Phase 2 Diffusion Mode]
       Deterministic Prediction        Noise Prediction Residual Head
         y_det: [B, 7, 6, 80, 80]        eps_theta(r_t, t, conditioning)
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                  Physics-Informed Composite Loss
                    - Area-Weighted Mass Conservation (Cosine Lat)
                    - Diurnal Temperature Spread (Tmax >= Tmin)
                    - Environmental Lapse Rate Bounds
                    - Psychrometric RH Bounds
                                   │
                                   ▼
                  Kaggle Dual Tesla T4 Remote Training
                    - 6.0 hr GPU weekly quota management
                    - Automatic Mixed Precision (AMP)
                    - Automated artifact retrieval to models/checkpoints/
========================================================================================
```

---

## 2. User Review Required

> [!IMPORTANT]
> **Strict Remote Kaggle Execution**: As established in project directives, all multi-epoch model training routines execute exclusively on remote Kaggle accelerators (Dual Tesla T4 16GB GPUs or TPU VM v3-8). Local execution on CPU is strictly restricted to shape verification, single-batch forward/backward checks, and test suite execution.

> [!IMPORTANT]
> **Two-Phase Architectural Strategy**:
> - **Phase 1 (Sprint 3 Primary Baseline)**: Train the 7-day deterministic multivariate spatiotemporal baseline (`TemporalMultiTaskUNet5x`). This provides the benchmark metrics (MAE, RMSE, Wet-Day MAE, CSI@15, CSI@30, mass conservation error) required by the research roadmap before introducing stochastic diffusion complexity.
> - **Phase 2 (Continuous Residual Diffusion)**: Construct the conditional residual diffusion model (`SpatiotemporalDiffusionDownscaler`) which takes the deterministic forecast / bilinearly upsampled NWP as its conditioning base and denoises stochastic micro-scale residual fields.

---

## 3. Open Questions & Design Decisions

- **Temporal Encoder Choice**:
  - *Decision*: Use 1D Temporal Convolutions + Temporal Cross-Attention across the 7 lead days. This keeps parameter count under 4.5M while capturing cross-lead atmospheric advection without the high memory footprint of full 3D spatiotemporal self-attention.
- **Precipitation Non-Negativity**:
  - *Decision*: Enforce strict non-negativity using `StraightThroughNonNegative` autograd function, preserving leaky gradients for dry pixels while preventing negative precipitation values.
- **Diurnal Spread Invariant**:
  - *Decision*: Model $T_{\min}$ directly, and parameterize $T_{\max} = T_{\min} + \text{Softplus}(w_{\Delta T})$. This mathematically guarantees $T_{\max} \ge T_{\min}$ across every spatial grid cell and lead day.
- **Kaggle Dataset Mounting**:
  - *Decision*: Package `multitask_temporal_v1.zarr` into a Kaggle dataset artifact `rohitajitbharadwaj/sih26074-multitask-temporal-v1` so remote kernels mount it directly at `/kaggle/input/` with zero startup transfer latency.

---

## 4. Proposed Changes

### Component 1: Spatiotemporal Model Architecture (`src/models/`)

#### [NEW] [`src/models/temporal_multitask_baseline.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/src/models/temporal_multitask_baseline.py)
- Input signatures:
  - `history`: `[B, 3, 6, 16, 16]` (antecedent observations)
  - `forecast`: `[B, 7, 6, 16, 16]` (coarse NWP forecast)
  - `terrain`: `[B, 5, 80, 80]` (GLO-30 topography)
- Submodules:
  - `TemporalConditioningEncoder`: Maps `[B, 3, 6, 16, 16]` to coarse temporal context tokens.
  - `BilinearUpsampler5x`: Upsamples coarse forecast `[B, 7, 6, 16, 16]` to `[B, 7, 6, 80, 80]`.
  - `MultiScaleConvNeXtUNet`: Downsamples concatenated features (upsampled NWP + terrain + temporal tokens) through 3 stages (80x80 -> 40x40 -> 20x20 -> 10x10), applies temporal cross-attention across the 7 lead days at the bottleneck, and upsamples back to 80x80 with skip connections.
  - `PhysicsOutputHeads`:
    - Rain head: `StraightThroughNonNegative`
    - Temperature heads: $T_{\min}$ and $T_{\max} = T_{\min} + \text{Softplus}(\dots)$
    - RH head: Sigmoid bounded $[0, 100]\%$
    - Wind heads: Linear projection for $U, V$ components
- Output shape: `[B, 7, 6, 80, 80]`

#### [NEW] [`src/models/residual_diffusion.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/src/models/residual_diffusion.py)
- Implements continuous Gaussian diffusion (DDPM / EDM formulation).
- Forward noise schedule: $\mathbf{r}_t = \sqrt{\bar{\alpha}_t} \mathbf{r}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{\epsilon}$.
- Conditioning mechanism:
  - Sinusoidal timestep embedding $t \to \mathbb{R}^{d_{\text{time}}}$ injected into ConvNeXt blocks via Adaptive Group Normalization (AdaGN).
  - Conditioning tensor $\mathbf{c}$ composed of upsampled NWP forecast, antecedent history features, and high-resolution terrain.
- Denoising target: Predicts fine residual $\mathbf{r} = \mathbf{y}_{\text{fine}} - \mathbf{y}_{\text{coarse\_interp}}$.
- Reverse sampling: Fast deterministic DDIM sampler (4 to 16 steps) for rapid inference.

---

### Component 2: Physics-Constrained Loss Engine (`src/losses/`)

#### [NEW] [`src/losses/spatiotemporal_multitask_loss.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/src/losses/spatiotemporal_multitask_loss.py)
- Multi-task regression loss:
  - Precipitation: Log-cosh loss + Quantile Pinball loss ($\tau = 0.90$) for heavy convective tails.
  - Temperatures ($T_{\max}, T_{\min}$): Huber loss ($\beta = 1.0$).
  - Relative Humidity: L1 loss.
  - Wind ($U, V$): Vector magnitude log-cosh loss.
  - Homoscedastic uncertainty weighting across tasks using learnable log-variance parameters.
- Physical mass conservation constraint:
  - Uses `coarsen_hr_to_lr_torch` with cosine latitude weighting to coarsen predicted precipitation from 80x80 to 16x16.
  - Computes MSE penalty against coarse NWP precipitation in physical mm space (post-expm1 inversion):
    $$\mathcal{L}_{\text{mass}} = \frac{1}{7} \sum_{l=0}^{6} \|\mathcal{C}[\text{expm1}(\hat{P}_l)] - P_{\text{coarse}, l}\|_2^2$$
- Thermodynamic constraints:
  - Diurnal ordering penalty: $\text{mean}(\text{ReLU}(T_{\min} - T_{\max})^2)$.
  - Orographic lapse rate bounds: Penalizes temperature lapse rates outside $[4.0, 9.8]\,\text{K/km}$ over terrain gradients.

---

### Component 3: Kaggle Remote Training Pipeline (`scripts/kaggle/`)

#### [NEW] [`scripts/train_temporal_downscaler.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/scripts/train_temporal_downscaler.py)
- Complete self-contained remote training script:
  - Configurable for Phase 1 (deterministic baseline) and Phase 2 (diffusion).
  - Uses `SpatiotemporalDownscalingDataset` streaming directly from Zarr.
  - Automatic Mixed Precision (`torch.amp.autocast`) for maximal throughput on Nvidia T4 GPUs.
  - Cosine annealing learning rate scheduler with linear warmup.
  - Evaluation hook logging All-Day MAE, Wet-Day MAE, RMSE, CSI@15, CSI@30, and Mass Conservation Error per lead day.
  - Saves best checkpoint to `models/checkpoints/` and metrics history to `reports/`.

#### [NEW] [`scripts/kaggle/dispatch_temporal_baseline.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/scripts/kaggle/dispatch_temporal_baseline.py)
- Automated Kaggle packaging and dispatch orchestrator:
  - Pre-flight quota verification (checks 6.0h allowance).
  - Packages source code and attaches dataset `rohitajitbharadwaj/sih26074-multitask-temporal-v1`.
  - Generates kernel metadata with `"enable_gpu": true` and `"machine_shape": "NvidiaTeslaT4"`.
  - Pushes kernel via Kaggle API, monitors execution in real-time, and retrieves checkpoints and reports upon completion.
  - Post-run quota accounting update.

---

### Component 4: Test Suite & Verification (`tests/`)

#### [NEW] [`tests/models/test_spatiotemporal_baseline.py`](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/tests/models/test_spatiotemporal_baseline.py)
- Local smoke test suite executing strictly on CPU:
  - Validates forward pass shapes: history `[2, 3, 6, 16, 16]`, forecast `[2, 7, 6, 16, 16]`, terrain `[2, 5, 80, 80]` -> output `[2, 7, 6, 80, 80]`.
  - Validates physical output head invariants ($T_{\max} \ge T_{\min}$, $P \ge 0$, $0 \le \text{RH} \le 100$).
  - Validates backward pass and gradient flow through all parameters.
  - Validates loss computation and area-weighted mass conservation coarsening.

---

## 5. Verification Plan

### Automated Tests
1. **Model & Loss Unit Tests (Local CPU)**:
   ```powershell
   pytest tests/models/test_spatiotemporal_baseline.py -v
   ```
   Ensures zero shape errors, complete parameter gradient flow, and physical invariant enforcement.

2. **Full Repository Regression (Local CPU)**:
   ```powershell
   pytest tests/data/test_sprint2_dataset.py -v
   ```
   Confirms all Sprint 2 data engineering contracts remain intact.

3. **Kaggle Connection & Quota Pre-Check**:
   ```powershell
   python scripts/kaggle/test_kaggle_connection.py
   python scripts/kaggle/dispatch_kaggle.py --check-quota
   ```
   Confirms API connectivity and available GPU quota.

### Remote Training Verification
1. **Staging & Packaging Dry Run**:
   - Verify kernel bundle creation and dataset attachment without pushing.
2. **Kaggle Dual T4 Training Run**:
   ```powershell
   python scripts/kaggle/dispatch_temporal_baseline.py --epochs 25 --batch_size 8
   ```
   - Monitor remote training execution via Kaggle API.
   - Verify checkpoint `models/checkpoints/temporal_multitask_champion.pt` and metrics report download.
