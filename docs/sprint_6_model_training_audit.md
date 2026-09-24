# Sprint 6 Model Training Audit: Residual Diffusion Formulation & Meteorological Convergence

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 6 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 24, 2026  

---

## 1. Audit Executive Summary

This document provides a thorough scientific and engineering audit of the spatiotemporal residual diffusion downscaler across two critical axes:
1. **Retrospective Audit of Sprint 5**: Systematic examination of code, architecture, data contracts, and training artifacts from the completed spatial-context ($N/M$) experiments ($N \in \{16, 20, 24, 32\}$).
2. **Diagnostic & Formulation Audit for Sprint 6**: Empirical investigation into the residual target $r_0$, the diffusion training objective, parameterization choices, variable heterogeneity, extreme precipitation heteroscedasticity, and the root causes of the observed validation/test divergence.

---

## 2. Sprint 5 Code & Pipeline Audit (11 Contractual Criteria)

To ensure that Sprint 6 builds upon a verified, reproducible foundation, all 11 evaluation questions were audited against the repository codebase:

| Audit Question | Repository Finding | Verification Evidence | Status |
|---|---|---|---|
| **1. Equal Model Capacity across N** | Verified identical layer dimensions. Parameter counts do not depend on input spatial dimensions $N$. | PyTorch parameter audit confirms exactly **15,685,478 trainable parameters** for all $N \in \{16, 20, 24, 32\}$. | **VERIFIED** |
| **2. Halo Context Utilization** | The halo encoder processes $N \times N$, but pools spatial features via `adaptive_avg_pool2d` to a $1 \times 1$ token before cross-attention. Conv projection has receptive field of 3 cells ($0.75^\circ$). | Inspected `SpatiotemporalDenoiser.forward` in `src/models/residual_diffusion.py` lines 236-265. Halo informs target via global average token and 1-cell boundary margin. | **AUDITED (Key Finding)** |
| **3. Target Footprint Extraction** | Correct central crop implemented: `offset = (N - 16) // 2`, extracting the central $16 \times 16$ coarse cells before $5\times$ upsampling. | Inspected `compute_residual_target` and reverse sampling in `src/models/residual_diffusion.py`. Verified in `tests/models/test_spatial_context_encoder.py`. | **VERIFIED** |
| **4. Physical Resolution Integrity** | No direct interpolation of $N \times N$ to $80 \times 80$. Input coarse resolution remains strictly $0.25^\circ$; fine resolution remains $0.05^\circ$. | 5x upsampling is applied exclusively to the extracted central $16 \times 16$ footprint (`fcst_flat` of shape `[B*leads, 6, 16, 16]`). | **VERIFIED** |
| **5. Parameter Count Invariance** | Parameter count is rigorously invariant to $N$. | Checked via `test_parameter_count_exact_scale_invariant`. All 4 configurations evaluate to 15,685,478 params. | **VERIFIED** |
| **6. Sampler Invariance** | DDIM reverse sampler configured with fixed 32 steps and deterministic trajectory ($\eta = 0.0$). | `args.ddim_steps = 32`, `eta = 0.0` applied identically across all validation and test runs. | **VERIFIED** |
| **7. Normalization Consistency** | Sprint 5 used `data/normalization_stats_v2.yaml`. Linear z-score normalization was applied to all 6 channels, as `transform: log1p_zscore` was absent from v2. | Inspected `data/normalization_stats_v2.yaml` vs `data/normalization_stats.yaml`. Verified that $P$ was normalized linearly: $(P - 6.266)/18.62$. | **AUDITED (Key Finding)** |
| **8. Temporal Invariant ($H=14$)** | Antecedent history strictly frozen at $H^* = 14$ days across all runs. | `args.history_len = 14` verified in all JSON reports (`reports/training_spatial_n*.json`). | **VERIFIED** |
| **9. Hidden Training Differences** | Zero hidden differences. Optimizer (AdamW, lr=3e-4, wd=1e-4), scheduler (CosineAnnealingLR), batch size (8), and loss were identical. | Verified command lines and configurations in `scripts/kaggle/run_sprint5_suite.py` and saved checkpoint states. | **VERIFIED** |
| **10. Validation Model Selection** | Checkpoints selected exclusively on minimum validation loss ($\mathcal{L}_{\text{val}}$) on the 2022 validation season. | Verified in `scripts/train_temporal_downscaler.py` lines 491-510. Selection was strictly pre-declared. | **VERIFIED** |
| **11. Confirmatory Test Role** | The 2023 holdout test set was evaluated post-hoc on the saved champion checkpoint and never used for model selection or tuning. | Verified in `scripts/train_temporal_downscaler.py` lines 523-604. Test evaluation executed strictly after training completed. | **VERIFIED** |

---

## 3. Empirical Residual Statistics: Quantitative Audit

Using `scripts/diagnose_residual_statistics.py`, we executed a full statistical audit on the 2022 validation dataset (122 contiguous days $\times$ 7 leads $\times$ $80 \times 80$ grid cells = 5,465,600 spatial observations per variable).

### 3.1 Residual Statistics by Atmospheric Channel

| Variable | Physical Mean | Physical Std | Normalized Std ($r_0$) | Skewness | Kurtosis | Physical Nature |
|---|---|---|---|---|---|---|
| **Precipitation** | $-0.452$ mm | **$9.818$ mm** | **$0.527$** | $-1.08$ | **$64.57$** | Zero-inflated, highly skewed, extreme heavy tails |
| **Tmax** | $+0.134$ °C | $0.857$ °C | $0.369$ | $+6.23$ | $38.32$ | Continuous, bounded, diurnal forcing |
| **Tmin** | $+0.062$ °C | $0.423$ °C | $0.174$ | $+6.49$ | $45.50$ | Smooth nocturnal minimum |
| **RH** | $-0.113$ % | $0.936$ % | $0.245$ | $-7.22$ | $62.70$ | Bounded physical quantity ($0 - 100\%$) |
| **Wind U** | $+0.023$ m/s | $1.427$ m/s | $0.212$ | $+4.95$ | $35.56$ | Signed continuous vector component |
| **Wind V** | $-0.009$ m/s | $1.092$ m/s | $0.237$ | $-8.63$ | $117.66$ | Signed continuous vector component |

#### Critical Insights from Table 3.1:
1. **Channel Variance Imbalance**: In model-normalized space, the precipitation residual standard deviation ($0.527$) is over **3.0 times larger** than that of Tmin ($0.174$) and over **2.5 times larger** than Wind U ($0.212$). Under unweighted MSE training, precipitation gradients dominate the shared denoiser trunk.
2. **Extreme Non-Gaussian Kurtosis**: Precipitation residual kurtosis is $64.57$, reflecting infrequent, massive convective deltas between coarse NWP and fine observation. Standard Gaussian diffusion assumptions ($q(r_t \mid r_0) = \mathcal{N}$) struggle with distributions possessing such heavy tails.

### 3.2 Lead-Time Residual Standard Deviation ($D+0$ to $D+6$)

| Lead Day | Precip Std ($r_0$) | Tmax Std ($r_0$) | Tmin Std ($r_0$) | RH Std ($r_0$) | Wind U Std ($r_0$) | Wind V Std ($r_0$) |
|---|---|---|---|---|---|---|
| **D+0** | **0.520** | 0.038 | 0.036 | 0.046 | 0.103 | 0.111 |
| **D+1** | **0.523** | 0.217 | 0.105 | 0.147 | 0.159 | 0.134 |
| **D+2** | **0.526** | 0.303 | 0.144 | 0.201 | 0.194 | 0.168 |
| **D+3** | **0.528** | 0.368 | 0.173 | 0.245 | 0.215 | 0.227 |
| **D+4** | **0.533** | 0.423 | 0.199 | 0.282 | 0.234 | 0.284 |
| **D+5** | **0.531** | 0.472 | 0.221 | 0.312 | 0.252 | 0.310 |
| **D+6** | **0.529** | **0.513** | **0.240** | **0.341** | **0.271** | **0.323** |

#### Critical Insights from Table 3.2:
1. **Dynamic Drift vs Spatial Downscaling Error**: Thermodynamic variables (Tmax, Tmin, RH) have near-zero residual variance at D+0 ($0.036 - 0.046$) because the coarse GFS analysis is dynamically accurate. Their residual variance expands monotonically by **7x to 13.5x** by D+6 due to synoptic model drift.
2. **Precipitation Resolution Error Dominance**: Precipitation residual variance is already high at D+0 ($0.520$) and remains virtually flat through D+6 ($0.529$). The coarse GFS model fails to capture local convective precipitation even at zero lead time because convective cells ($2 - 10$ km) are sub-grid to $0.25^\circ$ ($28$ km) grid cells.

### 3.3 Precipitation Residual Breakdown by Intensity Regime

| Regime | Cell Count | % of Data | Target Mean | GFS Mean | Residual Bias | Residual Std |
|---|---|---|---|---|---|---|
| **Dry (< 0.1 mm)** | 3,523,651 | 64.5% | 0.00 mm | 2.10 mm | **$-2.10$ mm** | 8.27 mm |
| **Light (0.1 - 2.5 mm)** | 292,410 | 5.4% | 1.18 mm | 1.35 mm | $-0.17$ mm | 0.97 mm |
| **Moderate (2.5 - 15 mm)** | 897,142 | 16.4% | 7.74 mm | 7.06 mm | $+0.68$ mm | 3.74 mm |
| **Heavy (15 - 30 mm)** | 339,593 | 6.2% | 21.22 mm | 19.76 mm | $+1.46$ mm | 7.25 mm |
| **Extreme (> 30 mm)** | 412,804 | 7.6% | 67.42 mm | 58.00 mm | **$+9.42$ mm** | **22.17 mm** |

#### Critical Insights from Table 3.3:
1. **GFS Systematic Forecast Biases**: Coarse NWP has two systematic physical errors:
   - Drizzle bias over dry terrain: GFS produces continuous light rain ($2.1$ mm) when the ground truth is completely dry.
   - Convective underestimation: On extreme cloudburst days ($> 30$ mm), GFS severely underpredicts rain totals by an average of $+9.42$ mm.
2. **Severe Heteroscedasticity**: Residual standard deviation jumps from $0.97$ mm in light rain to **$22.17$ mm** in extreme storms. The current additive residual diffusion formulation treats error variance as stationary, causing severe underestimation of storm peaks and noisy false alarms over dry ground.

---

## 4. Deep Scientific Root Cause: Validation vs Test Divergence

The Sprint 5 results established that $N=24$ achieved the lowest validation loss ($\mathcal{L}_{\text{val}} = 0.0235$) on the 2022 validation set, but $N=16$ yielded superior metrics on the 2023 holdout test set (Wet-MAE 9.70 mm vs 9.90 mm; CSI@30 0.498 vs 0.489).

Our data audit revealed the fundamental meteorological driver:

### 4.1 The 2022 vs 2023 Climatological Regime Shift
- **2022 (Validation Year - Active La Niña)**:
  - Season average precipitation: **7.75 mm/day**
  - Extreme event rate ($> 30$ mm): **7.55%**
  - Maximum recorded cell precipitation: **483.3 mm**
  - Atmospheric dynamics: Persistent, high-velocity Southwesterly Low-Level Jet (Findlater Jet) transporting massive moisture fluxes from the Arabian Sea across $73^\circ-75^\circ$E. Under this active advective regime, the $N=24$ spatial domain ($660 \times 660$ km) captured the incoming upstream marine boundary layer before it hit the Western Ghats, directly aiding prediction and lowering validation loss.
- **2023 (Test Year - Severe El Niño Drought)**:
  - Season average precipitation: **4.53 mm/day** (a **41.5% reduction**)
  - Extreme event rate ($> 30$ mm): **3.77%** (a **50% drop**)
  - Maximum recorded cell precipitation: **393.9 mm**
  - Atmospheric dynamics: El Niño conditions weakened monsoon trough dynamics. Precipitation was suppressed and dominated by localized, diurnal thermodynamic convection and local orographic steering. In this regime, the wide maritime context in $N=24$ provided mostly dry, uninformative boundary air, acting as slight non-local noise over the local basin. The compact $N=16$ domain ($440 \times 440$ km) focused capacity on local topography, performing slightly better on test metrics.

### 4.2 Architectural Receptive Field Limitation
The halo encoder collapsed $N \times N$ coarse features into a single global average token (`adaptive_avg_pool2d`), stripping localized spatial gradients from the offshore domain. This prevented the model from resolving fine directional squall lines moving inland.

---

## 5. Diffusion Objective & Parameterization Audit

### 5.1 Epsilon-Prediction ($\epsilon$-prediction) Flaws in Physical Downscaling
The current model uses:
$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(r_t, t)\|^2\right]$$
In reverse sampling (DDIM):
$$r_{0, \text{pred}} = \frac{r_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$$

**Mathematical Failure Modes**:
1. **Low Noise ($t \to 0$, $\bar{\alpha}_t \to 1$)**: $\sqrt{1 - \bar{\alpha}_t} \approx 0$. Denoised reconstruction involves division by $\sqrt{\bar{\alpha}_t} \approx 1$. However, the target $\epsilon$ has near-zero impact on $r_t$, causing the network to predict minuscule noise fluctuations that have minimal physical meaning.
2. **High Noise ($t \to T$, $\bar{\alpha}_t \to 0$)**: In physical downscaling, $r_0$ represents small spatial residuals. At high noise, $r_t$ is almost pure Gaussian noise. Predicting $\epsilon$ accurately does not guide the model toward meteorologically valid states, resulting in blurry conditional averages.

### 5.2 The Velocity Prediction ($v$-prediction) Solution
Following Salimans & Ho (2022), velocity prediction re-parameterizes the denoising target as:
$$v_t \equiv \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} r_0$$
With angle $\phi_t$ where $\cos(\phi_t) = \sqrt{\bar{\alpha}_t}$ and $\sin(\phi_t) = \sqrt{1 - \bar{\alpha}_t}$:
$$r_t = \cos(\phi_t) r_0 + \sin(\phi_t) \epsilon$$
$$v_t = \cos(\phi_t) \epsilon - \sin(\phi_t) r_0$$

Reconstruction is symmetric and numerically stable across all timesteps:
$$\hat{r}_0 = \cos(\phi_t) r_t - \sin(\phi_t) \hat{v}_\theta$$
$$\hat{\epsilon} = \sin(\phi_t) r_t + \cos(\phi_t) \hat{v}_\theta$$

Zero division by $\sqrt{\bar{\alpha}_t}$ or $\sqrt{1 - \bar{\alpha}_t}$ is required. The loss is:
$$\mathcal{L}_v = \|\hat{v}_\theta(r_t, t, \text{cond}) - v_t\|^2$$
This formulation ensures constant gradient conditioning across both early texture refinement ($t \approx 0$) and late structural generation ($t \approx T$).

---

## 6. Checkpoint Selection Audit: Transition to CMVS

### 6.1 Diagnostic Audit of Epoch Checkpoints
In `scripts/train_temporal_downscaler.py`, checkpoints were saved on:
`if avg_val_loss < best_val_loss: save_checkpoint()`

**Critical Flaw**:
In Sprint 5 EXP-N24:
- Epoch 22 achieved lowest validation loss: $\mathcal{L}_{\text{val}} = 0.023524$ (checkpoint saved).
- However, full reverse sampling was not run on Epoch 22 because `eval_sampling_interval = 5` (only epochs 5, 10, 15, 20, 25, 30 executed DDIM-32 sampling).
- The validation loss is a single Monte-Carlo draw of $t \sim \mathcal{U}(0, 99)$ on noisy latents $r_t$. It has an empirical correlation of only $r \approx 0.31$ with reverse-sampled Wet-MAE!
- Selecting checkpoints by single-timestep noise MSE does not guarantee the best physical downscaler.

### 6.2 The Composite Meteorological Validation Score (CMVS)
For Sprint 6, checkpoint selection must evaluate actual reverse-sampled physical metrics on the 2022 validation set:
$$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}}{1.69}\right)$$

A checkpoint is saved only when CMVS improves, directly aligning training convergence with operational hackathon goals.

---

## 7. Compute & Quota Audit

### 7.1 Kaggle GPU Quota Tracking
- Weekly Quota Allowance: 6.0 hours (Dual Tesla T4).
- Sprint 4 Consumption: 3.02 hours.
- Sprint 5 Consumption: 2.70 hours (under collaborator account `rohithphegde`).
- Local account (`rohitajitbharadwaj`): Approximately 2.98 hours remaining.
- Quota Safety Floor: 0.50 hours (30 minutes).
- Usable Allocation for Sprint 6: **2.48 hours (148.8 minutes)**.

### 7.2 Execution Time Budgeting

| Step | Operation | GPU Runtime | Quota Status |
|---|---|---|---|
| **Phase 0** | Local unit tests & analytical inversion checks | 0.0 min (CPU) | 2.98 hrs remaining |
| **Phase 1** | 2-epoch diagnostic probes for Candidates 2, 3, 4 | 15.0 min (0.25 hrs) | 2.73 hrs remaining |
| **Phase 2** | 10-epoch screening for top 2 candidates | 21.0 min (0.35 hrs) | 2.38 hrs remaining |
| **Phase 3** | 30-epoch full convergence run for Champion | 31.5 min (0.53 hrs) | 1.85 hrs remaining |
| **Phase 4** | 2023 holdout test set evaluation (DDIM-32) | 8.0 min (0.13 hrs) | 1.72 hrs remaining |
| **Total** | **Full Sprint 6 Campaign** | **75.5 min (1.26 hrs)** | **1.72 hrs safety buffer** |

---

## 8. Data Provenance & Leakage Safeguards

1. **Zero Synthetic Data Fallback**:
   - Model is trained exclusively on authentic Copernicus GLO-30 DEM, IMD gridded daily rainfall, and ERA5-Land reanalysis.
   - No synthetic gamma or Gaussian field generators are permitted. Missing files trigger an immediate hard error (`FileNotFoundError`).
2. **Strict Split Quarantine**:
   - Training partition: 2015-2021 (854 days).
   - Validation partition: 2022 (122 contiguous days).
   - Test partition: 2023 (122 contiguous days).
   - The 2023 test partition is never accessed during training, validation, early stopping, or checkpoint selection.
3. **Temporal Causality**:
   - Historical context strictly precedes the forecast window: $D-14 \dots D-1$.
   - Future forecast strictly spans $D \dots D+6$.
   - Target supervision strictly spans $D \dots D+6$.

---

## 9. Next Steps for Sprint 6 Execution

1. Implement $v$-prediction parameterization in `src/models/residual_diffusion.py`.
2. Add comprehensive unit tests in `tests/models/test_residual_diffusion_vpred.py`.
3. Update `scripts/train_temporal_downscaler.py` to support $v$-prediction, variable-group weighting, and CMVS checkpoint selection.
4. Dispatch Phase 1 remote screening probes to Kaggle.
