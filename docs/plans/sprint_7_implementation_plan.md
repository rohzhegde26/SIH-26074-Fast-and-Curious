# Sprint 7 Implementation Plan: Diffusion-Step and Sampler Frontier for Spatiotemporal Weather Downscaling

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 7 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 25, 2026  

---

## 1. Executive Summary & Sprint 6 Scientific Handoff

### 1.1 The Established Baseline Handoff
Sprint 6 completed the fundamental reformulation of the spatiotemporal residual diffusion process on Kaggle Dual Tesla T4 accelerators. By replacing standard $\epsilon$-prediction with velocity prediction ($v$-prediction) and introducing multi-task variable-aware noise weighting with a focal convective-tail loss on precipitation ($w_p = 3.0$ on $> 15$ mm/day), Candidate 3 achieved a major leap in downscaling accuracy across both validation and quarantined holdout test splits:

| Metric | Sprint 5 Baseline (Candidate 1 Control) | Sprint 6 Champion (Candidate 3 Multi-Task Tail) | Relative Net Improvement |
|---|---|---|---|
| **Validation CMVS (2022)** | 0.6490 | **0.5710** | **-12.0%** (lower is better) |
| **Validation Wet-MAE (mm)** | 8.70 | **7.67** | **-11.8%** |
| **Validation CSI@30** | 0.631 | **0.679** | **+7.6%** |
| **Validation Tmax MAE (°C)** | 0.37 | **0.30** | **-18.9%** |
| **Validation Wind RMSE (m/s)** | 1.69 | **1.57** | **-7.1%** |
| **Holdout Test Wet-MAE (mm, 2023)** | 9.90 | **8.93** | **-9.8%** |
| **Holdout Test CSI@15 (2023)** | 0.464 | **0.524** | **+12.9%** |
| **Holdout Test CSI@30 (2023)** | 0.489 | **0.534** | **+9.2%** |
| **Holdout Test Tmax MAE (°C)** | 0.46 | **0.37** | **-19.6%** |
| **Holdout Test Tmin MAE (°C)** | 0.44 | **0.33** | **-25.0%** |
| **Holdout Test Wind RMSE (m/s)** | 3.56 | **3.49** | **-2.0%** |

### 1.2 Frozen Invariants for Sprint 7
Sprint 7 is strictly an **inference-only research sprint**. No model retraining or architectural re-scaling is performed. All investigations use the frozen Sprint 6 Candidate 3 champion checkpoint (`sprint6_candidate3_multitask_champion.pt`):
- **Antecedent Memory**: Frozen $H^* = 14$ days
- **Spatial Context**: Frozen $N/M = 24 / 16 = 1.50$ ($660 \times 660$ km domain)
- **Target Resolution**: $M = 16$ coarse cells downscaled $5\times$ to $80 \times 80$ fine cells ($0.05^\circ$)
- **Model Parameter Invariant**: Strictly locked at **15,685,478 trainable parameters**
- **Model Weights**: Frozen Candidate 3 checkpoint (`sprint6_candidate3_multitask_champion.pt`)
- **Denoising Formulation**: Velocity prediction ($v$-prediction) with group-tail weighting
- **Hardware Acceleration**: 6.0 hours of 2*Tesla T4 Kaggle student-tier GPUs available

---

## 2. Scientific Objective & The Core Research Question

In Sprints 1 through 6, diffusion inference was locked to a fixed 32-step DDIM schedule. Having established that the $v$-prediction residual formulation learns high-fidelity meteorological structures, Sprint 7 addresses the operational efficiency frontier:

> **How much test-time diffusion computation (denoising steps / neural function evaluations) is strictly necessary, and which ODE/SDE numerical sampler achieves the optimal accuracy, extreme-storm recall, spatial sharpness, and latency trade-off for operational edge deployment?**

Sprint 7 does NOT perform:
- Model size scaling (assigned to Sprint 9).
- History length sweeps (assigned to Sprint 4).
- Spatial context $N/M$ sweeps (assigned to Sprint 5).
- Loss function or parameterization sweeps (assigned to Sprint 6).
- Mixture-of-Experts routing (assigned to Sprint 9).
- Ensemble member scaling (assigned to Sprint 8).

---

## 3. Critical Sampler Correctness Audit: The Legacy DDIM Discretization Bug

### 3.1 Mathematical Audit of Current Implementation
In `src/models/residual_diffusion.py` lines 460-464:
```python
step_stride = self.timesteps // num_steps
time_seq = list(range(0, self.timesteps, step_stride))
time_seq = time_seq[:num_steps]
```
In the reverse sampling loop:
```python
for i in reversed(range(len(time_seq))):
    t_curr = time_seq[i]
    ...
```

For total training timesteps $T = 100$:

| Requested Steps ($S$) | Integer Stride (`100 // S`) | Generated `time_seq` | Initial Reverse Timestep ($t_{\max}$) | Final Timestep ($t_{\min}$) | Unreached Diffusion Range |
|---|---|---|---|---|---|
| **$S = 4$** | $25$ | `[0, 25, 50, 75]` | **$t = 75$** | $t = 0$ | $t \in [76, 99]$ (top 24% missing) |
| **$S = 8$** | $12$ | `[0, 12, 24, 36, 48, 60, 72, 84]` | **$t = 84$** | $t = 0$ | $t \in [85, 99]$ (top 15% missing) |
| **$S = 16$** | $6$ | `[0, 6, 12, ..., 90]` | **$t = 90$** | $t = 0$ | $t \in [91, 99]$ (top 9% missing) |
| **$S = 32$** | $3$ | `[0, 3, 6, ..., 93]` | **$t = 93$** | $t = 0$ | $t \in [94, 99]$ (top 6% missing) |
| **$S = 64$** | $1$ | `[0, 1, 2, ..., 63]` | **$t = 63$** | $t = 0$ | **$t \in [64, 99]$ (top 36% missing!)** |

### 3.2 Root Cause Analysis: Why This Is an Implementation Bug
1. **Initial Noise Distribution Mismatch**: Reverse diffusion initializes with pure standard Gaussian noise $r_T \sim \mathcal{N}(0, \mathbf{I})$. This assumes the process begins at $t \approx T - 1 = 99$, where $\bar{\alpha}_{99} \approx 0$ and the forward process has fully destroyed the data signal. Starting at $t = 93$ (or $t = 63$ for $S=64$) assumes that a pure Gaussian noise tensor represents the diffused latent at a much lower noise level, introducing an artificial distribution shift into the first reverse denoiser call.
2. **Severe Upper-Range Truncation at Higher Step Counts**: For $S = 64$, integer division `100 // 64` evaluates to $1$. Slicing `[:64]` halts at $t = 63$. The top 36 timesteps are completely omitted. Increasing steps from 32 to 64 actually lowers the starting timestep from 93 to 63, reversing the intended noise scale.
3. **Non-Uniform Discretization**: Across different step counts, the starting noise level changes arbitrarily ($75 \to 84 \to 90 \to 93 \to 63$), confounding step-count comparisons.

### 3.3 The Mathematically Correct Standard DDIM Discretization
Following the canonical formulation of Song et al. (ICLR 2021) and Nichol & Dhariwal (ICML 2021):
For $S$ steps spanning continuous time $[0, T-1]$, the discrete trajectory $\tau = [\tau_0, \tau_1, \dots, \tau_{S-1}]$ is defined such that:
$$\tau_k = \text{round}\left(k \cdot \frac{T - 1}{S - 1}\right), \quad k \in \{0, 1, \dots, S-1\}$$
Reversing $\tau$ yields a strictly descending sequence:
$$\tau_{\text{desc}} = [\tau_{S-1}, \tau_{S-2}, \dots, \tau_1, \tau_0]$$
where $\tau_{S-1} = T - 1 = 99$ and $\tau_0 = 0$.

For all step counts $S \in \{4, 8, 16, 32, 64\}$:
- Terminal timestep is always **$t = 99$** (where pure Gaussian noise $r_{99} \sim \mathcal{N}(0, \mathbf{I})$ is mathematically calibrated).
- Final timestep is always **$t = 0$** (where residual $r_0$ is fully reconstructed).
- Intermediate intervals are uniformly spaced across the entire cumulative variance schedule $\bar{\alpha}_t$.

---

## 4. Phase 0 Gate: Legacy DDIM-32 vs Corrected DDIM-32 Baseline

Before benchmarking alternative step counts or samplers, Sprint 7 executes a mandatory verification gate:

```
+-------------------------------------------------------------------------------------------------+
| PHASE 0 CORRECTNESS GATE: Candidate 3 Checkpoint on 2022 Validation                             |
|                                                                                                 |
|   Condition A: Legacy DDIM-32 (starts at t=93, step stride=3)                                   |
|   Condition B: Corrected Standard DDIM-32 (starts at t=99, uniform linspace to t=0)             |
|                                                                                                 |
|   Comparison Axes:                                                                              |
|   1. Numerical equivalence and finite output verification                                       |
|   2. CMVS, Wet-MAE, CSI@30, Tmax MAE, Wind Vector RMSE                                          |
|   3. Document whether earlier Sprint 5/6 metrics were affected by the t=93 truncation           |
|   4. Establish Condition B as the official reference frontier for Sprint 7                      |
+-------------------------------------------------------------------------------------------------+
```

If Condition B produces a measurable shift in validation metrics, Condition B is adopted as the true ground-truth baseline, and all subsequent sweeps ($4, 8, 16, 32, 64$ steps) are evaluated relative to Condition B.

---

## 5. Core Step-Count Experiment Matrix (DDIM Frontier)

Holding the model checkpoint strictly frozen, evaluate the corrected deterministic DDIM sampler ($\eta = 0.0$) across five discrete step budgets on the 2022 validation dataset:

| Configuration ID | Steps ($S$) | NFE | Timestep Trajectory ($\tau_k$) | Theoretical Latency Ratio | Target Question |
|---|---|---|---|---|---|
| **STEP-04** | 4 | 4 | `[99, 66, 33, 0]` | **$0.125\times$** (8x faster) | Ultra-fast emergency early warning |
| **STEP-08** | 8 | 8 | `[99, 85, 71, 57, 42, 28, 14, 0]` | **$0.250\times$** (4x faster) | Edge Panchayat node budget |
| **STEP-16** | 16 | 16 | Uniform 16 steps from 99 to 0 | **$0.500\times$** (2x faster) | Balanced operational target |
| **STEP-32 (Ref)**| 32 | 32 | Uniform 32 steps from 99 to 0 | **$1.000\times$** (Reference) | Full-quality Sprint 6 benchmark |
| **STEP-64** | 64 | 64 | Uniform 64 steps from 99 to 0 | **$2.000\times$** (2x compute) | Quality saturation boundary |

### 5.1 Scientific Evaluation of 128 Steps
Literature consensus (Karras et al., 2022; Lu et al., 2022) and empirical diffusion dynamics show that for low-dimensional conditional residual fields ($80 \times 80$ with strong conditioning priors), first-order DDIM exhibits asymptotic truncation error saturation past 32 to 64 steps. Furthermore, running 128 steps requires 128 neural function evaluations per sample, exceeding sensible operational latency ceilings for 7-day weather rollouts. 

**Decision**: 128 steps will NOT be run by default. It will only be probed on a single validation batch if 64 steps demonstrates a statistically significant improvement ($> 3.0\%$ gain in CMVS) over 32 steps.

---

## 6. Alternative Sampler Family Experiment Matrix

Higher-order ODE solvers can achieve lower local truncation errors per step than first-order DDIM, potentially matching 32-step quality at 8 to 16 steps.

### 6.1 Sampler Candidates for $v$-Prediction Models

1. **DDIM (1st Order Pseudo-Euler ODE)**:
   - Reference baseline. Non-stochastic trajectory ($\eta = 0.0$).
   - NFE per step: 1.
2. **DPM-Solver++ (2nd Order Multi-Step ODE)**:
   - Lu et al. (NeurIPS 2022): Specifically designed for fast ODE sampling of diffusion models in data space or velocity space.
   - DPM-Solver++(2M) multi-step uses two previous model outputs to construct a second-order polynomial trajectory, doubling convergence rate without extra NFE per step (except step 1).
   - NFE for $S$ steps: $S$.
3. **PNDM (Pseudo Numerical Methods for Diffusion Models)**:
   - Liu et al. (ICLR 2022): Fourth-order linear multi-step method (Adams-Bashforth) with Runge-Kutta warmup.
   - Solves the diffusion ODE with high-order numerical gradient integration.
   - NFE for $S$ steps: $S + 2$ (due to Runge-Kutta initialization).
4. **UniPC (Unified Predictor-Corrector, Optional Diagnostic)**:
   - Zhao et al. (NeurIPS 2023): Unified multi-step predictor-corrector framework for fast diffusion sampling.

### 6.2 Mathematical $v$-Prediction Inversion for ODE Solvers
Modern solvers typically expect either noise $\epsilon$ or reconstructed data $x_0$. For our $v$-prediction model, the denoiser outputs $\hat{v}_\theta(r_t, t, c)$. The interface must strictly apply the exact analytical identity:
$$\hat{r}_0 = \sqrt{\bar{\alpha}_t} r_t - \sqrt{1 - \bar{\alpha}_t} \hat{v}_\theta$$
$$\hat{\epsilon} = \sqrt{1 - \bar{\alpha}_t} r_t + \sqrt{\bar{\alpha}_t} \hat{v}_\theta$$
Solvers that operate on data-prediction (such as DPM-Solver++2M) will be provided with $\hat{r}_0$, while solvers expecting noise (such as PNDM) will receive $\hat{\epsilon}$. Passing raw $\hat{v}$ directly into an $\epsilon$-solver without transformation is strictly prohibited.

### 6.3 Matched-NFE Comparison Matrix

| Evaluation Tier | Target NFE Budget | DDIM Configuration | DPM-Solver++(2M) | PNDM Configuration | Target Operational Persona |
|---|---|---|---|---|---|
| **Ultra-Low Compute** | **~4 NFE** | DDIM (4 steps) | DPM-Solver++ (4 steps) | PNDM (4 steps, RK-2) | Mobile / Solar Panchayat Edge |
| **Low Compute** | **~8 NFE** | DDIM (8 steps) | DPM-Solver++ (8 steps) | PNDM (8 steps) | Local Block Server |
| **Medium Compute** | **~16 NFE** | DDIM (16 steps) | DPM-Solver++ (16 steps) | PNDM (16 steps) | District Center Operational |
| **High Compute (Ref)**| **~32 NFE** | DDIM (32 steps) | DPM-Solver++ (32 steps) | PNDM (32 steps) | State Meteorological Hub |

---

## 7. Comprehensive Meteorological & Physical Evaluation

Every sampler/step configuration is evaluated across the full suite of physical, spatial, and meteorological metrics on the 2022 validation dataset.

### 7.1 Multi-Task Meteorological Metrics
1. **Precipitation**:
   - All-Day MAE, Wet-Day MAE ($> 2.5$ mm), RMSE.
   - Categorical Skill: CSI@15 (heavy rain, $15$ mm/day), CSI@30 (extreme storm, $30$ mm/day).
   - Fraction of Skill Score (FSS) at neighborhood scales of 3, 5, and 9 cells ($0.15^\circ, 0.25^\circ, 0.45^\circ$).
   - Extreme event recall and false alarm rate.
2. **Thermodynamic Fields (Tmax, Tmin, RH)**:
   - Tmax MAE, Tmin MAE, Diurnal spread violation rate ($T_{\min} > T_{\max}$).
   - RH MAE, physical out-of-bounds rate ($\text{RH} < 0\%$ or $\text{RH} > 100\%$).
3. **Dynamic Wind Vector (U, V)**:
   - U MAE, V MAE, Vector Wind RMSE: $\sqrt{(U - U^*)^2 + (V - V^*)^2}$.
4. **Composite Score**:
   - Composite Meteorological Validation Score (CMVS):
     $$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}_{\text{val}}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}_{\text{val}}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}_{\text{val}}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}_{\text{val}}}{1.69}\right)$$

### 7.2 Lead-Time Drift Analysis ($D+0 \dots D+6$)
All metrics are decomposed and reported across each individual forecast lead day. A sampler that preserves $D+0$ skill but experiences catastrophic trajectory divergence at $D+5$ or $D+6$ will be penalized.

### 7.3 Extreme-Precipitation Heteroscedasticity Analysis
Sprint 6 proved that precipitation residual variance scales from $0.97$ mm in light rain to **$22.17$ mm** in storms $> 30$ mm/day. 
Sprint 7 explicitly tests the hypothesis:
> **Does reducing denoising steps degrade extreme precipitation recall (CSI@30) significantly faster than average precipitation error (All-Day MAE)?**

If lower step counts (e.g. 4 or 8 steps) truncate the heavy tail while preserving mean metrics, this trade-off must be explicitly documented for disaster management authorities.

### 7.4 Spatial and Spectral Power Analysis
To ensure that low-NFE samplers do not achieve low MAE merely by predicting oversmoothed, blurred fields:
- Compute 2D radially averaged FFT power spectrum of the predicted precipitation and wind residuals.
- Evaluate the high-frequency spectral slope ($k > 0.2\text{ km}^{-1}$).
- Calculate spatial variance ratio: $\text{Var}(\hat{y}) / \text{Var}(y^*)$.

---

## 8. Computational Efficiency & Hardware Profiling Protocol

All timing benchmarks are measured directly on Kaggle Dual Tesla T4 GPUs (and locally on CPU) with strict hardware invariants:
- **Precision**: Automatic Mixed Precision (`torch.amp.autocast('cuda')`).
- **Warmup**: 3 complete batches discarded before recording wall-clock time.
- **Timing Primitive**: `torch.cuda.Event(enable_timing=True)` for GPU-synchronized execution timing.
- **Metrics Logged**:
  1. Denoiser calls per 7-day forecast cube (NFE).
  2. Latency per 7-day forecast cube (milliseconds).
  3. Batch throughput (samples per second).
  4. Peak allocated CUDA VRAM (MB via `torch.cuda.max_memory_allocated`).
  5. Speedup factor relative to DDIM-32.

---

## 9. Compute Budget: 6.0 Hours Kaggle 2*Tesla T4 Allocation

The user has explicitly authorized the full student-tier compute envelope:
- **Available Compute**: **6.0 hours (360.0 minutes) of 2*Tesla T4 GPU accelerators**.
- **Execution Mode**: 100% inference. Zero training epochs.
- **Throughput Calibration**: One forward denoiser evaluation takes ~1.8 ms per sample. A 32-step reverse sampling pass over the entire 2022 validation set (122 samples) requires approximately **7.2 minutes**.

### 9.1 Phased Inference Budget Allocation

| Phase | Description | Scope | GPU Runtime | Cumulative Quota |
|---|---|---|---|---|
| **Phase 0** | Local CPU unit tests & schedule correctness verification | CPU only | 0.0 min | 6.00 hrs remaining |
| **Phase 1** | Gate: Legacy DDIM-32 vs Corrected DDIM-32 | Full 2022 Val (122 samples) $\times 2$ runs | 14.5 min | 5.76 hrs remaining |
| **Phase 2** | DDIM Step-Count Sweep ($S \in \{4, 8, 16, 64\}$) | Full 2022 Val (122 samples) $\times 4$ runs | 21.0 min | 5.41 hrs remaining |
| **Phase 3** | Alternative Samplers (DPM-Solver++ & PNDM at 4, 8, 16, 32 NFE) | Full 2022 Val (122 samples) $\times 6$ runs | 31.5 min | 4.88 hrs remaining |
| **Phase 4** | Pareto Frontier Selection & Latency Profiling | Batch throughput & VRAM profiling | 10.0 min | 4.71 hrs remaining |
| **Phase 5** | Confirmatory Holdout Test Run (Champion Sampler on 2023) | Full 2023 Test (122 samples) $\times 1$ run | 7.5 min | 4.59 hrs remaining |
| **Total** | **Full Sprint 7 Campaign** | **15 conditions evaluated** | **~84.5 min (1.41 hrs)** | **4.59 hrs safety buffer** |

Even with a comprehensive 15-condition evaluation matrix, the entire campaign consumes less than 1.5 hours of GPU time, preserving over **4.5 hours of safety margin**.

---

## 10. Confirmatory 2023 Holdout Test Set Protocol

To guarantee zero test-set leakage or multiple-hypothesis testing bias:
1. All step-count comparisons, sampler rankings, and Pareto frontier curves are established **strictly on the 2022 validation season**.
2. One single optimal configuration (the **Sprint 7 Champion Inference Configuration**) is selected based on the accuracy/latency Pareto frontier.
3. The quarantined 2023 holdout test set is evaluated **exactly once** using the champion configuration.
4. Results are reported in Table B of the Sprint 7 Benchmark Report.

---

## 11. Research Hypotheses & Falsification Criteria

### Hypothesis 1 (Corrected Discretization Integrity)
- **Statement**: Correcting the DDIM timestep schedule to span the full interval $[99 \to 0]$ eliminates noise initialization mismatch and lowers validation Wet-MAE and CMVS compared to the legacy schedule.
- **Control**: Legacy DDIM-32 (starting at $t=93$).
- **Treatment**: Corrected DDIM-32 (starting at $t=99$).
- **Metric**: Validation CMVS and Wet-MAE.
- **Falsification Criterion**: Corrected DDIM-32 increases CMVS by $> 1.0\%$ relative to legacy DDIM-32.

### Hypothesis 2 (DDIM Step-Count Pareto Frontier)
- **Statement**: Reducing DDIM steps from 32 to 16 retains at least $97.0\%$ of meteorological downscaling accuracy (CMVS increase $< 3.0\%$) while cutting inference latency by $50\%$.
- **Control**: Corrected DDIM-32.
- **Treatment**: Corrected DDIM-16.
- **Metric**: Validation CMVS and measured inference latency.
- **Falsification Criterion**: DDIM-16 degrades CMVS by $> 3.0\%$ or achieves latency reduction $< 40\%$.

### Hypothesis 3 (Extreme-Precipitation Step Sensitivity)
- **Statement**: Extreme storm recall (CSI@30) degrades significantly faster than average error metrics (All-Day MAE) as DDIM steps are reduced to 8 or 4 steps.
- **Control**: Corrected DDIM-32.
- **Treatment**: DDIM-8 and DDIM-4.
- **Metric**: Relative percentage drop in CSI@30 vs relative percentage increase in All-Day MAE.
- **Falsification Criterion**: The relative drop in CSI@30 is less than or equal to the relative increase in MAE at 4 steps.

### Hypothesis 4 (High-Order ODE Sampler Efficiency)
- **Statement**: Second-order DPM-Solver++(2M) achieves equivalent or lower CMVS at 16 NFE than DDIM achieves at 32 NFE.
- **Control**: Corrected DDIM-32 (32 NFE).
- **Treatment**: DPM-Solver++(2M) (16 NFE).
- **Metric**: Validation CMVS and Wet-MAE.
- **Falsification Criterion**: DPM-Solver++ at 16 NFE yields higher CMVS than DDIM at 32 NFE.

### Hypothesis 5 (Lead-Time Trajectory Robustness)
- **Statement**: Fast samplers ($4 - 8$ NFE) maintain parity on short leads ($D+0, D+1$) but exhibit larger relative degradation on extended leads ($D+5, D+6$) due to compounding discretization errors.
- **Control**: DDIM-32 per-lead metrics.
- **Treatment**: DDIM-8 per-lead metrics.
- **Metric**: Ratio of error degradation at $D+6$ vs $D+0$.
- **Falsification Criterion**: The percentage degradation at $D+6$ is less than or equal to that at $D+0$.

### Hypothesis 6 (Spectral Sharpness Retention)
- **Statement**: DPM-Solver++ at 16 NFE preserves high-frequency residual spatial power ($k > 0.2\text{ km}^{-1}$) within $5\%$ of the 32-step DDIM reference, avoiding the oversmoothing typical of low-step Euler methods.
- **Control**: DDIM-32 radially averaged power spectrum.
- **Treatment**: DPM-Solver++(2M) 16-step power spectrum.
- **Metric**: High-frequency spectral energy ratio: $E_{\text{high}}(\text{sampler}) / E_{\text{high}}(\text{DDIM-32})$.
- **Falsification Criterion**: Spectral energy ratio falls below $0.90$.

---

## 12. Test-Driven Development (TDD) Verification Plan

Create `tests/models/test_sampler_frontier.py` covering:
1. `test_corrected_ddim_schedule_bounds`: Verify that for all $S \in \{4, 8, 16, 32, 64\}$, trajectory starts at exactly 99 and ends at exactly 0.
2. `test_schedule_strictly_monotonic`: Verify strictly descending order without duplicate timesteps.
3. `test_vpred_sampler_roundtrip_math`: Verify exact analytical reconstruction of $r_0$ and $\epsilon$ from $v$ across random tensors.
4. `test_deterministic_reproducibility`: Verify that identical seed produces identical output tensors bit-for-bit with $\eta = 0.0$.
5. `test_dpm_solver_vpred_conversion`: Verify that DPM-Solver++ accurately receives data-prediction $\hat{r}_0$ converted from $v$.
6. `test_physical_clipping_invariants`: Verify that precipitation outputs are non-negative and RH is strictly bounded within $[0, 100]\%$.

---

## 13. Research Gates

- **Gate 1 (Mathematical Verification)**: 100% green tests in `tests/models/test_sampler_frontier.py` on local CPU before any Kaggle dispatch.
- **Gate 2 (Baseline Audit)**: Phase 1 comparison completed; establish whether legacy DDIM-32 had truncation bias.
- **Gate 3 (Pareto Selection)**: All 15 inference runs completed and plotted on Accuracy vs Latency frontier before touching the 2023 holdout test set.

---

## 14. Definition of Done

1. Sampler schedule corrected and verified in `src/models/residual_diffusion.py`.
2. DPM-Solver++ and PNDM inference interfaces implemented with exact $v$-prediction conversion.
3. Full step-count sweep ($S \in \{4, 8, 16, 32, 64\}$) completed on 2022 validation set.
4. High-order sampler sweep completed at matched NFE budgets.
5. Accuracy vs Latency Pareto frontier plotted and documented.
6. Quarantined 2023 test set evaluated once on the champion sampler configuration.
7. Synthesis reports produced:
   - `reports/sprint_7_sampler_frontier_summary.json`
   - `reports/sprint_7_comparison_table.md`
   - `docs/walkthrough_sprint_7.md`

---

## 15. No-Go Conditions

1. **Retraining Prohibition**: Abort if any script initiates backpropagation or optimizer updates. Sprint 7 is strictly inference.
2. **Quota Floor**: Abort if Kaggle GPU quota falls below 0.50 hours (30 minutes).
3. **Test Leakage**: Abort if the 2023 holdout test set is used to select step counts or tune samplers.
4. **Capacity Drift**: Abort if model architecture or parameter count (15,685,478) is modified.

---

## 16. Sprint 8 Handoff Specification

Sprint 7 delivers the single-sample accuracy/latency Pareto frontier, identifying the most efficient deterministic sampler (e.g. DPM-Solver++ at 16 NFE or DDIM at 16 NFE). 

This configuration will be handed directly to **Sprint 8: Ensemble and Test-Time Scaling**, which will investigate:
- Multi-member ensemble generation ($K \in \{2, 4, 8, 16, 32\}$ members) using stochastic sampling ($\eta > 0.0$).
- Continuous Ranked Probability Score (CRPS), Brier Score, and spread-skill reliability.
- The compute trade-off between **more denoising steps per member vs more ensemble members at fewer steps**.
