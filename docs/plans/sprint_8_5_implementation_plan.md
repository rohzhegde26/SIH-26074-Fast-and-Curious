# Sprint 8.5 Implementation Plan: Predictive Uncertainty, Calibration and Output-Quality Diagnostic

**Project:** SIH 2026 Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Sprint:** 8.5 of 10  
**Phase:** Deep Research and Diagnostic Specification  
**Status:** Pre-Implementation Specification (Awaiting User Review)  

---

## 1. Executive Summary and Problem Statement

### 1.1 Context and Motivation
Sprint 8 demonstrated that diffusion inference compute can be strategically distributed across denoising depth ($S$) and ensemble breadth ($K$). Under a compute budget matched at approximately 32 NFEs, an objective-dependent Pareto frontier emerged:
- **Deeper low-member sampling** ($K=2, S=16, \eta=0.0$) achieves the best continuous distribution scores: Fair-CRPS = 0.5401, Multivariate Energy Score ≈ 0.2769.
- **Broader shallow sampling** ($K=8, S=4, \eta=0.5$) achieves the best point-forecast accuracy and extreme storm recall: Wet-MAE = 6.51 mm, CSI@30 = 0.728.

However, Sprint 8 also uncovered two critical diagnostic vulnerabilities:
1. **Severe Precipitation Under-Dispersion:** The champion configuration achieves a precipitation spread-skill ratio (SSR) of only **0.437**, and its nominal 90% prediction interval covers only **21.0%** of true rainfall outcomes on the 2022 validation set. Increasing ensemble size $K$ from 8 to 16 yields diminishing returns and fails to resolve this spread deficit.
2. **Physical Repair Burden:** Between 31% and 37% of raw precipitation predictions fall below 0 and require non-negativity clipping ($P = \max(0, P)$), shifting approximately 7% to 8% of the total rainfall mass.

### 1.2 The Core Research Question
Before embarking on structural capacity expansion in Sprint 9, Sprint 8.5 addresses a foundational diagnostic question:
> **Is the remaining uncertainty deficit primarily a consequence of post-hoc calibration scale, reverse-diffusion sampler dynamics, or a fundamental representation bottleneck in the neural backbone?**

Can we attain the benefits of both sides of the Sprint 8 frontier (high point accuracy, high storm recall, and well-calibrated probabilistic bounds) without retraining or expanding the model?

---

## 2. Invariants and Guardrails

### 2.1 Frozen Architecture and Checkpoint Invariants
Sprint 8.5 is strictly an inference and post-processing diagnostic sprint. No training backpropagation is conducted.

- **Model Checkpoint:** `models/checkpoints/sprint6_candidate3_multitask_champion.pt`
- **Checkpoint SHA256:** `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`
- **Parameter Count:** 15,685,478 parameters
- **Architecture:** Spatio-temporal UNet with $H=14$ history frames, $N=24$ spatial context padding, $M=16$ central Panchayat output crop, $T=100$ diffusion training steps, linear noise schedule ($\beta_1 = 10^{-4}, \beta_T = 0.035$), and $v$-prediction objective.

### 2.2 Data Partitioning and Quarantines
- **Training Provenance (2015-2021):** Climatological reference only ($P > 15$ mm rate = 11.0322%, $P > 30$ mm rate = 5.8157% across 38.26M grid points).
- **Validation Dataset (2022):** 122 spatial-temporal forecast cubes (7 lead days each, 6 channels, 80x80 grid).
- **Internal 2022 Calibration Split:** Chronological split of the 122 cubes:
  - **Calibration Fit Set:** Cubes 1 to 61 (early monsoon / pre-monsoon).
  - **Calibration Eval Set:** Cubes 62 to 122 (late monsoon / peak monsoon).
  This prevents fitting calibration parameters on the exact cases used to evaluate them.
- **Holdout Dataset (2023):** 122 cubes strictly quarantined. Evaluated exactly once on the single chosen configuration at the conclusion of Sprint 8.5.

### 2.3 Operational Guardrails
Any post-hoc calibration or sampling adjustment must satisfy:
1. **Point-Forecast Guardrail:** Wet-MAE must not degrade by more than 1.0% relative to the uncalibrated reference ($K=8, S=4, \eta=0.5$ Wet-MAE reference = 6.51 mm, threshold $\le 6.57$ mm).
2. **Extreme-Recall Guardrail:** CSI@30 must not degrade by more than 0.010 absolute (reference = 0.728, threshold $\ge 0.718$).
3. **Physical Bound Guardrail:** Precipitation lower bounds must satisfy $P \ge 0.0$ mm/day. No negative precipitation values permitted.
4. **Latency Guardrail:** Calibration computation must add negligible inference latency ($< 5$ ms per cube).

---

## 3. Deep Research Literature Synthesis

### 3.1 Ensemble Model Output Statistics (EMOS) and Spread Rescaling
In operational numerical weather prediction (NWP), raw ensemble forecasts from dynamical models consistently exhibit under-dispersion due to unmodeled subgrid processes and finite ensemble size (Gneiting et al., 2005; Wilks, 2011). 

EMOS addresses this by fitting a parametric predictive distribution $Y \sim \mathcal{D}(\mu, \sigma)$ where:
$$\mu = a + b \bar{x}_{\text{ens}}, \quad \sigma^2 = c + d s^2_{\text{ens}}$$
For zero-bounded precipitation, Scheuerer (2014) introduced censored shifted Gamma and Generalized Extreme Value (GEV) EMOS models optimized via minimum Continuous Ranked Probability Score (CRPS) estimation.

In the non-parametric context of diffusion ensembles, a direct analog is **multiplicative ensemble spread inflation**:
$$x'_k = \bar{x} + \alpha (x_k - \bar{x})$$
where $\bar{x} = \frac{1}{K} \sum_{k=1}^K x_k$ is the ensemble mean, $x_k$ is the $k$-th member, and $\alpha \ge 1.0$ is the spread inflation factor. 

Crucially:
- When applied linearly, spread inflation **leaves the ensemble mean $\bar{x}$ completely invariant** ($\frac{1}{K}\sum x'_k = \bar{x}$).
- Consequently, deterministic point metrics (MAE, RMSE) of the ensemble mean are preserved before non-negative clipping.
- When followed by non-negativity enforcement $x''_k = \max(0, x'_k)$, asymmetric clipping can induce a positive mean shift for zero-bounded fields. The magnitude of this shift must be strictly monitored.

### 3.2 Probability Calibration for Extreme Heavy Rainfall
Evaluating threshold exceedance probabilities $p = P(Y > \tau)$ from raw ensemble frequencies $\hat{p} = \frac{1}{K} \sum \mathbb{I}(x_k > \tau)$ suffers from:
1. **Quantile coarseness at low $K$:** For $K=8$, probability resolution is restricted to increments of $1/8 = 0.125$.
2. **Reliability distortion:** Raw frequencies frequently over-predict mid-range probabilities while under-predicting extreme events.

Two standard post-processing techniques are evaluated:
1. **Isotonic Regression:** Non-parametric monotonic step-function mapping $\hat{p} \mapsto \tilde{p}$ minimizing squared error (Zadrozny & Elkan, 2002). Isotonic regression is strictly non-decreasing and adapts flexibly to empirical reliability curves.
2. **Logistic Calibration (Platt Scaling):** Parametric sigmoid transformation $\tilde{p} = \sigma(w_0 + w_1 \text{logit}(\hat{p}))$, which regularizes extreme probability tails and avoids overfitting on small calibration samples (Platt, 1999; Niculescu-Mizil & Caruana, 2005).

Both methods are fitted on the internal 2022 calibration split and evaluated on the held-out validation segment using Brier Score (BS), Brier Skill Score (BSS), and reliability diagrams.

### 3.3 Conformal Prediction with Non-Negative Bounds
Conformal prediction provides distribution-free, finite-sample prediction intervals with guaranteed marginal coverage $1 - \gamma$ (Vovk et al., 2005; Angelopoulos & Bates, 2021). 

In **Split Conformal Prediction**:
1. On the calibration split, compute non-conformity scores:
   $$R_i = |y_i - \hat{\mu}(x_i)| \quad \text{or normalized} \quad R_i = \frac{|y_i - \hat{\mu}(x_i)|}{\hat{\sigma}(x_i) + \epsilon}$$
2. Set the conformal threshold $\hat{q}$ as the $\lceil (N_{\text{cal}} + 1)(1 - \gamma) \rceil / N_{\text{cal}}$ empirical quantile of $R_i$.
3. On unseen cases, construct the prediction interval:
   $$C(x) = [\max(0, \hat{\mu}(x) - \hat{q} \hat{\sigma}(x)), \; \hat{\mu}(x) + \hat{q} \hat{\sigma}(x)]$$
Under the relevant exchangeability assumptions, split conformal provides finite-sample marginal coverage; Sprint 8.5 will empirically test coverage under the chronological weather split while enforcing the physical reality of non-negative precipitation.

### 3.4 Spatial Sharpness and Texture Metrics
Ensemble averaging acts as a low-pass spatial filter, reducing random high-frequency variance. While this improves pixel-wise MSE/MAE, it can create unphysically smooth precipitation fields lacking realistic convective storm cores (Ebert, 2008).

To verify that ensemble benefits do not stem from artificial blurring, we introduce two spatial sharpness diagnostics:
1. **Laplacian Energy ($E_{\text{Lap}}$):**
   $$E_{\text{Lap}}(x) = \frac{1}{HW} \sum_{i,j} (\nabla^2 x)_{i,j}^2$$
   where $\nabla^2 x$ is computed via a discrete $3 \times 3$ Laplacian kernel.
2. **Radially Averaged Power Spectral Density (RAPSD):**
   $$P(k) = \int_{|\mathbf{k}|=k} |\hat{X}(\mathbf{k})|^2 d\theta$$
   where $\hat{X}(\mathbf{k})$ is the 2D spatial discrete Fourier transform. RAPSD quantifies the preservation of meso-$\beta$ and meso-$\gamma$ convective energy ($k \ge 0.1 \text{ km}^{-1}$).

---

## 4. Detailed Experimental Plan (Phases 0 through 5)

### Phase 0: Metric and Provenance Reconciliation (Completed)
- **Objective:** Resolve condition-history CRPS vs bootstrap point estimate discrepancy.
- **Result:** Successfully traced to dataloader batch-weighting artifact on incomplete batch 31 (2 cubes).
- **Deliverables:** `reports/sprint8_5_metric_reconciliation.md` and `reports/sprint8_5_metric_reconciliation.json`.
- **Status:** Complete, reconciled, and documented.

### Phase 1: Uncertainty Bottleneck Diagnosis
- **Objective:** Quantify per-variable under-dispersion, spread-skill ratios, and coverage across leads $D+0$ to $D+6$.
- **Reference Conditions:**
  - Ref A: $K=2, S=16, \eta=0.0$
  - Ref B: $K=4, S=8, \eta=0.0$
  - Ref C: $K=8, S=4, \eta=0.5$
- **Diagnostic Metrics:**
  - Fair-CRPS (overall and per-variable: Precip, Tmax, Tmin, RH, U, V)
  - Spread-Skill Ratio ($SSR = \text{spread} / \text{skill}$)
  - Interval Coverage (50%, 80%, 90% nominal)
  - Prediction Interval Sharpness (average interval width)
  - Pairwise Ensemble Diversity (inter-member variance and correlation)
  - For precipitation's zero-inflated/discrete support, use rank histograms; use randomized PIT only when ties/zero mass are handled explicitly.
- **Key Questions Answered:**
  - Is under-dispersion isolated to precipitation or present across all thermodynamic variables?
  - Does uncertainty expand monotonically with lead time from Day 0 to Day 6?
  - Does increasing $K$ or $\eta$ materially reduce the spread deficit?

### Phase 2: Post-Hoc Calibration Experiments
- **Objective:** Evaluate lightweight post-processing adjustments to restore calibrated uncertainty without degrading point metrics.
- **Sub-Experiments:**
  1. **Experiment 2A: Multiplicative Spread Rescaling:**
     - Test grid: $\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$.
     - Evaluate global $\alpha$ vs lead-dependent $\alpha_d$ for $d \in \{0, \dots, 6\}$.
     - Enforce non-negativity $P \ge 0$ post-scaling.
  2. **Experiment 2B: Threshold Probability Recalibration:**
     - Fit isotonic regression and Platt logistic scaling on $P > 15$ mm and $P > 30$ mm on the internal 2022 calibration split.
     - Evaluate Brier score, Brier Skill Score (relative to 2015-2021 climatology), reliability, and resolution on the evaluation split.
  3. **Experiment 2C: Conformal Prediction Interval Adjustment:**
     - Fit split-conformal calibration on precipitation intervals.
     - Enforce non-negative lower bound $P \ge 0$.
     - Measure empirical coverage vs nominal (50%, 80%, 90%) and resulting interval sharpness.
  4. **Experiment 2D: Sampler vs Calibration Attribution:**
     - Compare: (i) Sampler-only changes, (ii) Calibration-only changes, (iii) Combined sampler + calibration.
     - Isolate whether the bottleneck is inference dynamics or post-hoc scale.

### Phase 3: Lead-Time Uncertainty Dynamics
- **Objective:** Profile forecast skill and spread progression across each lead day ($D+0, D+1, \dots, D+6$).
- **Metrics Tracked per Lead:**
  - Wet-MAE, CSI@15, CSI@30
  - Fair-CRPS and Precipitation CRPS
  - Spread-Skill Ratio and 90% coverage
  - Ensemble standard deviation vs root mean square error
- **Physical Criterion:** Spread must grow monotonically ($U_{D+6} > U_{D+0}$) matching the growth of forecast error, avoiding both under-dispersion at long leads and unphysical variance explosion.

### Phase 4: Precipitation Repair-Burden Attribution
- **Objective:** Test whether non-negativity clipping ($P = \max(0, P)$) is a material contributor to the observed precipitation under-dispersion or tail distortion. Treat clipping as a candidate mechanism to be tested, not as a proven causal explanation.
- **Conditions Compared:**
  - Raw unclipped output (diagnostic only)
  - Standard physical repair ($P = \max(0, P)$)
  - Calibrated spread + physical repair
- **Metrics Evaluated:**
  - Clipping fraction (% of pixels with $P_{\text{raw}} < 0$)
  - Precipitation mass shift ($\Delta \text{Mass} / \text{Mass}_{\text{total}}$)
  - Effect of clipping on extreme convective tails ($P > 30$ mm)

### Phase 5: Spatial Sharpness and Texture Preservation
- **Objective:** Verify that ensemble averaging does not oversmooth local convective topography.
- **Comparison Cohort:**
  - Sprint 6 Candidate 3 deterministic single sample ($K=1$)
  - $K=2, S=16$ ensemble mean
  - $K=4, S=8$ ensemble mean
  - $K=8, S=4$ ensemble mean
  - Single stochastic ensemble member ($k=1$ of $K=8$)
- **Metrics Evaluated:**
  - 2D Laplacian Energy ($E_{\text{Lap}}$)
  - Radial Power Spectral Density high-frequency retention ($k \ge 0.1 \text{ km}^{-1}$)
  - Fine-scale gradient energy

---

## 5. Statistical Rigor and Evaluation Protocol

1. **Case-Preserving Paired Bootstrap:**
   - $B = 1,000$ bootstrap iterations.
   - Resampling unit: Entire 7-day spatial-temporal cube (preserving intra-cube temporal and inter-channel dependencies).
   - Paired metric differences $\Delta = \text{Calibrated} - \text{Baseline}$.
   - 95% Percentile Confidence Intervals $[q_{0.025}, q_{0.975}]$.
2. **Single Holdout Evaluation:**
   - The 2023 holdout dataset (122 cubes) is evaluated exactly once after calibration hyperparameters are frozen.
   - Zero parameter tuning on holdout.

---

## 6. Decision Gate Criteria for Sprint 9

At the conclusion of Sprint 8.5, an evidence-based recommendation will be selected from three architectural pathways:

| Outcome | Diagnostic Evidence | Recommendation for Sprint 9 |
|---|---|---|
| **Path A: Calibration Sufficient** | Spread rescaling ($\alpha$) or conformal adjustment achieves SSR $\ge 0.85$ and 90% coverage $\ge 75\%$ while preserving Wet-MAE within 1% and CSI@30 within 0.01. | Keep Candidate 3 architecture. Sprint 9 focuses entirely on model-capacity scaling to advance the deterministic point-accuracy frontier. |
| **Path B: Sampler Bottleneck** | Post-hoc calibration fails or distorts tails, but modifying sampler dynamics (noise injection, stochastic schedule) improves dispersion without capacity increase. | Sprint 9 introduces advanced stochastic samplers (e.g. predictor-corrector, annealed Langevin) alongside architecture scaling. |
| **Path C: Representation Bottleneck** | Neither post-hoc calibration nor sampler tuning can achieve acceptable spread without destroying point accuracy or physical consistency. | Sprint 9 must increase model capacity and introduce explicit dispersion-promoting training objectives (e.g. CRPS loss, ensemble distillation, or variance-regularized score heads). |
