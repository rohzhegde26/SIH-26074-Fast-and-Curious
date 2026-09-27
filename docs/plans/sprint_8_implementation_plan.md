# Sprint 8 Implementation Plan: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 27, 2026  

---

## 1. Executive Summary and Sprint 7 Scientific Handoff

### 1.1 The Established Baseline Handoff
Sprint 6 completed the fundamental reformulation of the spatiotemporal residual diffusion model on remote Kaggle accelerators. By pairing velocity prediction ($v$-prediction) with variable-aware group loss and a focal precipitation-tail weighting ($w_p = 3.0$ on $> 15$ mm/day), Candidate 3 achieved a decisive breakthrough over standard $\epsilon$-prediction models.

Sprint 7 subsequently explored the deterministic numerical sampler and diffusion-step frontier, discovering that the learned continuous reverse diffusion trajectory can be traversed in as few as 4 DDIM steps without sacrificing downscaling accuracy or extreme precipitation recall:

| Metric | Sprint 6 Baseline (Candidate 3, DDIM-32 Legacy) | Sprint 7 Deterministic Champion (Candidate 3, DDIM-4 Standard) | Relative Net Improvement |
|---|---|---|---|
| **Validation CMVS (2022)** | 0.5710 | **0.5376** | **-5.9%** (lower is better) |
| **Validation Wet-MAE (mm)** | 7.67 | **6.97** | **-9.1%** |
| **Validation CSI@30** | 0.679 | **0.704** | **+3.7%** |
| **Validation Tmax MAE (°C)** | 0.305 | **0.301** | **-1.3%** |
| **Validation Wind RMSE (m/s)** | 1.57 | **1.56** | **-0.6%** |
| **Inference Latency (ms/cube)** | 734.9 | **96.8** | **-86.8% (7.6x faster)** |
| **Holdout Test Wet-MAE (mm, 2023)** | 8.93 | **8.19** | **-8.3%** |
| **Holdout Test CSI@30 (2023)** | 0.534 | **0.556** | **+4.1%** |
| **Holdout Test Tmax MAE (°C)** | 0.369 | **0.354** | **-4.1%** |
| **Holdout Test Wind RMSE (m/s)** | 3.49 | **3.51** | **+0.6%** |

### 1.2 Frozen Model Architecture and Checkpoint Invariants
Sprint 8 is strictly an **inference-only research sprint**. No neural network parameters are trained or fine-tuned. All experiments evaluate the frozen Sprint 6 Candidate 3 champion model (`sprint6_candidate3_multitask_champion.pt`):
- **Model Architecture**: Multi-Task Conditional UNet with Spatiotemporal Self-Attention and Cross-Attention
- **Trainable Parameters**: Strictly locked at **15,685,478 parameters**
- **Temporal Dimensions**: Antecedent history $H = 14$ days, forecast horizon $T_f = 7$ days
- **Spatial Dimensions**: Coarse context $N = 24$ ($660 \times 660$ km), target crop $M = 16$, fine resolution $80 \times 80$ ($0.05^\circ \approx 5.5$ km)
- **Diffusion Target**: $v$-prediction residual formulation ($v_t \equiv \alpha_t \epsilon - \sigma_t x_0$)
- **Checkpoint Object ID**: Git LFS hash `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`
- **Hardware Budget**: 6.0 hours of 2x Tesla T4 Kaggle student-tier GPU accelerators

### 1.3 Scope and Non-Goals
Sprint 8 investigates the test-time scaling frontier between denoising steps and ensemble sample counts.
- **In-Scope**:
  - Initial-noise stochasticity ($\eta = 0, K > 1$) versus trajectory stochasticity ($\eta > 0$).
  - Matched-compute budget allocations ($B = K \times S \in \{8, 16, 32, 64\}$).
  - Probabilistic verification (CRPS, Brier score, reliability diagrams, coverage, sharpness, spread-skill ratio, rank histograms, energy score).
  - Extreme precipitation tail calibration ($P > 15$ mm, $P > 30$ mm).
  - Lead-time uncertainty evolution ($D+0$ to $D+6$).
  - Sequential versus batched wall-clock latency profiling.
- **Out-of-Scope (Strict Non-Goals)**:
  - Model retraining, loss adjustments, or gradient backpropagation (frozen).
  - History length sweeps (settled in Sprint 4).
  - Spatial domain context sweeps (settled in Sprint 5).
  - Network parameter capacity expansion or MoE routing (assigned to Sprint 9).
  - Modifying the quarantined 2023 holdout test set (only evaluated once at sprint conclusion).

---

## 2. Core Scientific Objective and The Test-Time Compute Allocation Problem

### 2.1 The Test-Time Compute Identity
In inference execution of diffusion models, the primary algorithmic compute budget is defined by the total number of Neural Function Evaluations (NFE):

$$\text{Total NFE} = K \times S$$

where:
- $K \ge 1$ is the number of independent stochastic ensemble members.
- $S \ge 1$ is the number of reverse denoising steps per ensemble member.

Sprint 7 demonstrated that DDIM-4 ($S = 4$) reduces latency from 734.9 ms to 96.8 ms while outperforming DDIM-32 ($S = 32$) on deterministic error metrics. This massive computational surplus unlocks a fundamental question in generative weather forecasting:

> **At matched total test-time compute ($B = K \times S$), is extra computation better invested in deeper trajectory denoising (increasing $S$) or in broader ensemble diversity (increasing $K$)?**

### 2.2 Decoupling Initial-Noise Diversity from Trajectory Stochasticity
A critical methodological requirement of Sprint 8 is separating two distinct mechanisms of stochastic diversity:

1. **Initial-Noise Diversity ($\eta = 0, K > 1$)**:
   The standard DDIM update with $\eta = 0$ is completely deterministic along the ODE trajectory given an initial latent sample:
   $$x_0^{(k)} = \mathcal{G}_\theta\left(z_T^{(k)}, c\right), \quad z_T^{(k)} \sim \mathcal{N}(0, \mathbf{I})$$
   By drawing $K$ independent initial noise vectors $z_T^{(1)}, \dots, z_T^{(K)}$, we generate an ensemble representing different plausible basins of the conditional data distribution. This is pure initial-noise diversity.

2. **Trajectory Stochasticity ($\eta > 0$)**:
   When $\eta > 0$, stochastic noise is actively injected at each intermediate reverse diffusion transition step:
   $$x_{\tau_{i-1}} = \sqrt{\alpha_{\tau_{i-1}}} \hat{x}_0 + \sqrt{1 - \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \hat{\epsilon} + \sigma_{\tau_i} \cdot \epsilon_{\tau_i}$$
   where $\sigma_{\tau_i} = \eta \sqrt{\frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}}} \sqrt{1 - \frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}$.

Sprint 8 treats $\eta = 0$ and $\eta > 0$ as independent experimental axes. The research protocol establishes pure initial-noise diversity ($\eta = 0$) as the primary baseline before evaluating trajectory noise injection ($\eta \in \{0.25, 0.5, 1.0\}$) using identical, paired initial seed manifests.

---

## 3. Literature Review and Meteorological Grounding

### 3.1 Test-Time Scaling in Generative Diffusion Models
Recent literature in generative modeling highlights test-time compute allocation as a primary scaling frontier (Karras et al., EDM, NeurIPS 2022; Hoogeboom et al., 2023). While initial diffusion literature focused on minimizing ODE discretization error by increasing $S$, empirical work reveals that when diffusion models operate under heavy conditioning (e.g., NWP forecasts and antecedent history), low step counts ($S \in [4, 8]$) capture the vast majority of the conditional mean trajectory. Deeper trajectories ($S \ge 32$) offer diminishing returns or over-smooth high-frequency variability.

### 3.2 Diffusion Ensembles in Operational Numerical Weather Prediction
In operational meteorology, ensemble forecasting provides flow-dependent uncertainty estimates essential for risk assessment and extreme event early warning (Bauer et al., Nature 2015; Buizza, 2018). Machine learning downscaling and weather models (e.g., GenCast by Price et al., 2023; CorrDiff by Mardani et al., 2023; SEEDS by Rasp et al., 2024; DiffDA) demonstrate that diffusion models inherently function as probabilistic generative filters. Rather than relying on singular deterministic point predictions, drawing ensemble members directly samples the predictive posterior distribution $p(y \mid x_{\text{coarse}}, x_{\text{nwp}}, \text{terrain})$.

### 3.3 Scoring Rules and Uncertainty Evaluation
Following Gneiting and Raftery (JASA 2007) and Wilks (2019), probabilistic evaluation must use strictly proper scoring rules:
- **Continuous Ranked Probability Score (CRPS)**: Generalizes absolute error to probabilistic forecasts. Unbiased finite-sample estimators must be used to avoid penalizing small ensembles.
- **Brier Score and Brier Skill Score (BSS)**: Quantifies probabilistic skill for binary threshold events ($P > 15$ mm, $P > 30$ mm) and admits Murphy's three-component decomposition into Reliability, Resolution, and Uncertainty.
- **Spread-Skill Relationship**: An ideally calibrated ensemble satisfies $\text{Spread} \approx \text{RMSE}$. An under-dispersive ensemble exhibits $\text{Spread} < \text{RMSE}$, indicating overconfidence.
- **Energy Score**: A multivariate generalization of CRPS that evaluates joint spatial and inter-variable coherence across the 6-variable weather cube.

---

## 4. Formal Research Hypotheses

Sprint 8 evaluates 8 formal hypotheses (H0 through H7) with explicit falsification criteria:

### H0: Informative Initial-Noise Ensembles
- **Hypothesis**: At $\eta = 0$, generating $K > 1$ members via independent initial Gaussian noise seeds yields an informative predictive ensemble whose ensemble mean achieves equal or better CMVS than a single member ($K=1$), accompanied by meaningful ensemble spread.
- **Control**: DDIM-4, $K=1, \eta=0$.
- **Treatment**: DDIM-4, $K \in \{2, 4, 8, 16\}, \eta=0$.
- **Falsification Criterion**: Falsified if ensemble mean CMVS degrades by $> 2.0\%$ or if ensemble spread $\sigma \to 0$ across all variables.

### H1: Diminishing Marginal Returns of Ensemble Scaling
- **Hypothesis**: At fixed $S=4$ and $\eta=0$, expanding $K$ yields diminishing marginal reductions in CRPS and Brier score, with improvements saturating beyond $K = 8$ or $K = 16$.
- **Control**: DDIM-4 with $K \in \{1, 2, 4\}$.
- **Treatment**: DDIM-4 with $K \in \{8, 16, 32\}$.
- **Falsification Criterion**: Falsified if CRPS improves linearly with $K$ up to $K=32$ without slope attenuation.

### H2: Sample Broader Beats Denoise Deeper at Matched Compute
- **Hypothesis**: At matched total NFE ($B = 32$), a low-step ensemble ($K=8, S=4$) achieves superior CRPS, Brier score, and threshold reliability compared to a single deep deterministic trajectory ($K=1, S=32$), without sacrificing physical consistency.
- **Control**: $K=1, S=32$ (32 NFE).
- **Treatment**: $K=8, S=4$ (32 NFE).
- **Secondary Treatments**: $K=4, S=8$ (32 NFE), $K=2, S=16$ (32 NFE).
- **Falsification Criterion**: Falsified if $K=1, S=32$ achieves lower CRPS and higher BSS@30 than $K=8, S=4$.

### H3: Value of Trajectory Stochasticity ($\eta > 0$)
- **Hypothesis**: For a fixed $(K, S)$ configuration, injecting intermediate trajectory noise ($\eta \in \{0.25, 0.5, 1.0\}$) improves probabilistic calibration (reliability and spread-skill ratio) over pure initial-noise diversity ($\eta = 0$) without unacceptable deterioration in deterministic MAE or forecast sharpness.
- **Control**: DDIM-4, $K=4, \eta=0$.
- **Treatment**: DDIM-4, $K=4, \eta \in \{0.25, 0.5, 1.0\}$ using the identical initial seed manifest.
- **Falsification Criterion**: Falsified if $\eta > 0$ increases wet-MAE by $> 5.0\%$ or degrades Brier score relative to $\eta = 0$.

### H4: Decoupling of Probabilistic Skill from Deterministic CMVS
- **Hypothesis**: An ensemble configuration can produce substantial gains in CRPS, Brier score, and coverage without showing large numerical shifts in ensemble-mean CMVS.
- **Falsification Criterion**: Falsified if improvements in CRPS strictly correlate with and require proportional improvements in CMVS.

### H5: Precipitation Tail Weighting Transfers to Probabilistic Exceedance
- **Hypothesis**: The convective tail weighting ($w_p = 3.0$ on $> 15$ mm) introduced in Sprint 6 enables the ensemble to generate well-calibrated exceedance probabilities $P(P > 15)$ and $P(P > 30)$ that outperform unconditional climatology.
- **Control**: Climatological event-frequency baseline on 2022 validation data.
- **Treatment**: Ensemble-derived probabilities from DDIM-4 ($K \ge 4$).
- **Falsification Criterion**: Falsified if BSS@15 $\le 0$ or BSS@30 $\le 0$.

### H6: Lead-Time Uncertainty Responsiveness
- **Hypothesis**: The ensemble spread and CRPS exhibit physically meaningful growth across forecast horizons $D+0$ through $D+6$, reflecting accumulating NWP forecast uncertainty.
- **Metric**: Lead-specific spread $\sigma(d)$ and lead-specific CRPS across $d \in \{0, 1, \dots, 6\}$.
- **Falsification Criterion**: Falsified if ensemble spread at $D+6$ is equal to or lower than at $D+0$.

### H7: Member-Wise Physical Consistency Invariant
- **Hypothesis**: Applying non-linear physical clipping member-by-member before ensemble reduction guarantees zero physical violations in individual members and produces physically valid ensemble aggregates.
- **Falsification Criterion**: Falsified if any individual member or ensemble mean exhibits negative precipitation, $\text{RH} < 0\%$ or $> 100\%$, or $T_{\min} > T_{\max}$.

---

## 5. Mathematical Formulations and Estimators

### 5.1 Member-Wise Physical Inversion Invariant
Let $y_{\text{norm}}^{(k)} \in \mathbb{R}^{T_f \times C \times H_{\text{fine}} \times W_{\text{fine}}}$ be the standardized output of the diffusion reverse process for member $k \in \{1, \dots, K\}$.
All post-processing must strictly follow this order:

1. **Standardization Inversion**:
   $$\tilde{y}_v^{(k)} = y_{\text{norm}, v}^{(k)} \cdot \sigma_{\text{train}, v} + \mu_{\text{train}, v}$$

2. **Member-Wise Non-Linear Physical Bounds**:
   $$P^{(k)} = \max\left(0, \tilde{P}^{(k)}\right)$$
   $$\text{RH}^{(k)} = \text{clip}\left(\tilde{\text{RH}}^{(k)}, 0.0, 100.0\right)$$
   $$\text{Ensure } T_{\min}^{(k)} \le T_{\max}^{(k)} \text{ by setting } T_{\max}^{(k)} = \max\left(T_{\max}^{(k)}, T_{\min}^{(k)}\right)$$

3. **Ensemble Statistics Computation in Physical Space**:
   - Ensemble Mean: $\mu = \frac{1}{K}\sum_{k=1}^K y^{(k)}$
   - Ensemble Variance: $\sigma^2 = \frac{1}{K-1}\sum_{k=1}^K (y^{(k)} - \mu)^2$
   - Physical Quantiles: $q_\alpha = \text{Quantile}_\alpha\left(\{y^{(1)}, \dots, y^{(K)}\}\right)$ for $\alpha \in \{0.10, 0.25, 0.50, 0.75, 0.90\}$
   - Exceedance Probability: $P(P > \text{thresh}) = \frac{1}{K}\sum_{k=1}^K \mathbb{I}\left(P^{(k)} > \text{thresh}\right)$

### 5.2 Unbiased Finite-Ensemble CRPS Estimator
For a finite ensemble $\{y^{(1)}, \dots, y^{(K)}\}$ and scalar observation $y_{\text{true}}$, the empirical Fair-CRPS is calculated as:

$$\text{CRPS}_{\text{fair}}\left(\{y^{(k)}\}_{k=1}^K, y_{\text{true}}\right) = \frac{1}{K}\sum_{k=1}^K |y^{(k)} - y_{\text{true}}| - \frac{1}{2K(K-1)}\sum_{k=1}^K \sum_{j=1}^K |y^{(k)} - y^{(j)}|$$

The second term removes the positive finite-sample bias of the empirical CDF, ensuring fair comparisons across different ensemble sizes $K$.

### 5.3 Brier Score and Brier Skill Score
For binary threshold event $\mathcal{E} = \mathbb{I}(y_{\text{true}} > \tau)$ and predicted ensemble probability $p = \frac{1}{K}\sum_{k=1}^K \mathbb{I}(y^{(k)} > \tau)$:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - o_i)^2$$

where $o_i \in \{0, 1\}$ is the observed occurrence.
The Brier Skill Score relative to sample climatology $\bar{o} = \frac{1}{N}\sum_{i=1}^N o_i$ is:

$$\text{BS}_{\text{clim}} = \bar{o}(1 - \bar{o}), \quad \text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{clim}}}$$

A value of $\text{BSS} > 0$ indicates genuine predictive skill beyond climatological probability.

### 5.4 Spread-Skill Ratio (SSR)
The ensemble spread-skill relationship across all validation points $i \in \{1, \dots, N\}$ is:

$$\text{Ensemble Spread} = \sqrt{\frac{1}{N}\sum_{i=1}^N \sigma_i^2}, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mu_i - y_{\text{true}, i})^2}$$

$$\text{SSR} = \frac{\text{Ensemble Spread}}{\text{RMSE}}$$

An ideal ensemble yields $\text{SSR} = 1.0$. Values with $\text{SSR} < 1.0$ indicate under-dispersion (overconfidence).

### 5.5 Prediction Interval Coverage and Sharpness
For nominal coverage level $(1 - \alpha) \in \{0.50, 0.80, 0.90\}$ with quantile bounds $[q_{\alpha/2}, q_{1 - \alpha/2}]$:
- Empirical Coverage: $\text{Cov} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left(q_{\alpha/2, i} \le y_{\text{true}, i} \le q_{1 - \alpha/2, i}\right)$
- Interval Sharpness: $\text{Width} = \frac{1}{N}\sum_{i=1}^N \left(q_{1 - \alpha/2, i} - q_{\alpha/2, i}\right)$

### 5.6 Multivariate Energy Score
To assess inter-variable spatial coherence across all 6 variables jointly, the Energy Score is:

$$\text{ES}\left(\{\mathbf{y}^{(k)}\}_{k=1}^K, \mathbf{y}_{\text{true}}\right) = \frac{1}{K}\sum_{k=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}_{\text{true}}\|_2 - \frac{1}{2K(K-1)}\sum_{k=1}^K \sum_{j=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}^{(j)}\|_2$$

---

## 6. Staged Experimental Campaign

The experimental matrix is organized into 4 sequential phases:

```
+---------------------------------------------------------------------------------------------------+
| PHASE 0: REPRODUCIBILITY & HARDWARE AUDIT                                                          |
| Checkpoint Git LFS hash, parameter count (15,685,478), DDIM-4 regression on 2022 (|dCMVS| <= 1.0%) |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 1: DETERMINISTIC REFERENCE ANCHORS                                                          |
| K=1, eta=0 for S in {4, 8, 16} (Establishes point-forecast frontier)                              |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 2: STAGED ENSEMBLE COUNT & STOCHASTICITY SWEEPS                                             |
| Stage A: Fixed S=4, eta=0, sweep K in {2, 4, 8, 16} (Initial-noise diversity)                    |
| Stage B: Sweep trajectory noise eta in {0.25, 0.5, 1.0} paired on same initial seeds             |
| Stage C: Gate check: evaluate K=32 only if K=16 has not saturated                                |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 3: MATCHED COMPUTE BUDGET FRONTIER                                                          |
| Budget 8:  K=1,S=8  vs K=2,S=4                                                                    |
| Budget 16: K=1,S=16 vs K=2,S=8  vs K=4,S=4                                                        |
| Budget 32: K=1,S=32 vs K=2,S=16 vs K=4,S=8 vs K=8,S=4 (FLAGSHIP)                                  |
| Budget 64: K=1,S=64 vs K=2,S=32 vs K=4,S=16 vs K=8,S=8 vs K=16,S=4                              |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 4: VALIDATION SELECTION & QUARANTINED 2023 HOLDOUT EVALUATION                               |
| Evaluate single chosen champion configuration once on 2023 test set                               |
+---------------------------------------------------------------------------------------------------+
```

### 6.1 Phase 0: Reproducibility Gate
Before initiating any ensemble sampling, the script verifies:
1. Model weights file exists at `models/checkpoints/sprint6_candidate3_multitask_champion.pt`.
2. Git LFS object ID matches `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`.
3. Parameter count is exactly 15,685,478 with 0 missing and 0 unexpected keys.
4. Single-sample deterministic DDIM-4 run reproduces the reference:
   $$|\text{CMVS}_{\text{run}} - 0.5376| \le 0.0054 \quad (\le 1.0\% \text{ deviation})$$
   $$|\text{Wet-MAE}_{\text{run}} - 6.97| \le 0.15 \text{ mm}$$

If Phase 0 fails, execution terminates immediately with a hard error.

### 6.2 Phase 1: Deterministic Reference Anchors
Run deterministic DDIM ($K = 1, \eta = 0$) across $S \in \{4, 8, 16\}$ to establish baseline anchors for single-trajectory performance.

### 6.3 Phase 2: Staged Ensemble Count and Stochasticity Sweeps
- **Stage A (Initial-Noise Diversity)**:
  Evaluate DDIM-4 with $\eta = 0$ for $K \in \{2, 4, 8, 16\}$.
  Seed manifest is strictly nested:
  $$\text{Seed Manifest}(K) = \{s_1, \dots, s_K\}, \quad s_k = 20260900 + 1000 \cdot k$$
  This ensures that the $K=4$ run contains the exact same members as $K=2$ plus two additional members.
- **Stage B (Trajectory Stochasticity)**:
  Select the optimal $K$ from Stage A and evaluate $\eta \in \{0.25, 0.5, 1.0\}$ using the exact same seed manifest.
- **Stage C (Saturation Gate for $K=32$)**:
  Evaluate $K=32$ only if the relative improvement in fair-CRPS from $K=8$ to $K=16$ exceeds $2.0\%$.

### 6.4 Phase 3: The Matched-Compute Frontier Matrix
This represents the primary scientific deliverable of Sprint 8. For each budget $B \in \{8, 16, 32, 64\}$, all factorizations $(K, S)$ such that $K \times S = B$ are evaluated on the full 2022 validation season:

| Budget ($B$) | Member Count ($K$) | Denoising Steps ($S$) | Nominal NFE | Wall-Clock Latency (Sequential) | Wall-Clock Latency (Batched) |
|---|---|---|---|---|---|
| **8** | 1 | 8 | 8 | Profiling Target | Profiling Target |
| **8** | 2 | 4 | 8 | Profiling Target | Profiling Target |
| **16** | 1 | 16 | 16 | Profiling Target | Profiling Target |
| **16** | 2 | 8 | 16 | Profiling Target | Profiling Target |
| **16** | 4 | 4 | 16 | Profiling Target | Profiling Target |
| **32 (Flagship)** | 1 | 32 | 32 | Profiling Target | Profiling Target |
| **32 (Flagship)** | 2 | 16 | 32 | Profiling Target | Profiling Target |
| **32 (Flagship)** | 4 | 8 | 32 | Profiling Target | Profiling Target |
| **32 (Flagship)** | 8 | 4 | 32 | Profiling Target | Profiling Target |
| **64** | 1 | 64 | 64 | Profiling Target | Profiling Target |
| **64** | 2 | 32 | 64 | Profiling Target | Profiling Target |
| **64** | 4 | 16 | 64 | Profiling Target | Profiling Target |
| **64** | 8 | 8 | 64 | Profiling Target | Profiling Target |
| **64** | 16 | 4 | 64 | Profiling Target | Profiling Target |

Both nominal NFE and empirical wall-clock latency (sequential and batched) are reported to maintain strict compute fairness.

---

## 7. Statistical Rigor and Case-Level Bootstrap Protocol

### 7.1 Case-Level Paired Resampling
Because meteorological variables display strong spatial autocorrelation and multi-day temporal autocorrelation across the 7-lead sequence, individual grid cells must never be treated as independent statistical samples.
- **Resampling Unit**: The complete 7-day $\times$ 6-variable fine forecast cube for an entire forecast initialization date.
- **Bootstrap Replicates**: $B = 1000$ paired resamples with replacement.
- **Pairing Invariant**: For every bootstrap sample, differences between competing configurations are computed case-by-case on the identical sampled dates.
- **Confidence Intervals**: 95% empirical bootstrap percentile intervals $[q_{0.025}, q_{0.975}]$ reported for all primary metrics.

### 7.2 Quarantined Holdout Protocol
The 2023 holdout test set is quarantined. No exploratory parameter sweeps, threshold tuning, or hyperparameter selection are conducted on 2023 data. Exactly one validation-selected champion configuration is evaluated on 2023 test data at the end of the study.

---

## 8. Deployment Regimes and Operational Decision Framework

The final evaluation report will articulate recommendations across three distinct operational regimes:

1. **Ultra-Low-Latency Edge Deployment (Panchayat Advisory Unit)**:
   - Target Latency: $< 100$ ms per forecast cube.
   - Resource Profile: Single edge GPU or high-end CPU.
   - Candidate Configuration: $K = 1, S = 4, \eta = 0$.
   - Operational Focus: Rapid deterministic advisory alerts with minimum compute footprint.

2. **Balanced Operational Regional Deployment (District / Block Agro-Met Center)**:
   - Target Latency: $< 400$ ms per forecast cube.
   - Resource Profile: Standard workstation GPU (single Tesla T4 or RTX 4000).
   - Candidate Configuration: $K = 4, S = 4, \eta = 0.5$ (or $\eta = 0$).
   - Operational Focus: Balanced deterministic point accuracy and actionable probability intervals for convective alerts.

3. **High-Compute Uncertainty-Centric Regime (State / National Meteorological Center)**:
   - Target Latency: $< 1500$ ms per forecast cube.
   - Resource Profile: Multi-GPU cluster or high-throughput cloud batch server.
   - Candidate Configuration: $K = 8$ or $16, S = 4, \eta = 0.5$.
   - Operational Focus: High-resolution ensemble quantiles, full predictive PDF, calibrated tail exceedance probabilities ($P > 15, P > 30$ mm).

---

## 9. Compute Allocation on Kaggle Accelerators

Sprint 8 has an allocated compute budget of **6.0 hours (360 minutes) on 2x Tesla T4 accelerators**. The execution plan allocates this budget across the experimental phases:

| Phase | Description | Configurations | Estimated Runtime | Cumulative Time |
|---|---|---|---|---|
| **Phase 0** | Reproducibility gate, hash check, DDIM-4 validation | 1 | 8 min | 8 min |
| **Phase 1** | Deterministic reference baselines ($S \in \{4, 8, 16\}$) | 3 | 15 min | 23 min |
| **Phase 2 Stage A** | Ensemble count sweep ($K \in \{2, 4, 8, 16\}, S=4, \eta=0$) | 4 | 75 min | 98 min |
| **Phase 2 Stage B** | Trajectory noise sweep ($\eta \in \{0.25, 0.5, 1.0\}$ on $K=4$) | 3 | 55 min | 153 min |
| **Phase 3** | Matched compute frontier (Budgets 8, 16, 32, 64) | 12 | 140 min | 293 min |
| **Phase 4** | Quarantined 2023 holdout evaluation on champion | 1 | 20 min | 313 min |
| **Buffer** | Statistical bootstrap, plotting, serialization overhead | - | 47 min | **360 min (6.0 h)** |

---

## 10. Required Code and Documentation Deliverables

The implementation of Sprint 8 will produce the following structured files in the repository:

1. **Model and Sampler Logic**:
   - `src/models/ensemble.py`: Reusable ensemble wrapper supporting nested seeds, parallel batched sampling, trajectory noise $\eta$, and member retention.
2. **Verification and Test Suite**:
   - `tests/models/test_ensemble_reproducibility.py`: Unit tests validating exact seed repeatability, nested seed subsets, member-wise physical clipping, and finite-ensemble Fair-CRPS computation.
3. **Execution Scripts**:
   - `scripts/evaluate_sprint8_ensemble.py`: Full evaluation pipeline executing Phases 0 through 4 on Kaggle.
   - `scripts/benchmark_sprint8_compute_frontier.py`: Matched-compute latency and throughput benchmark.
4. **Documentation and Reports**:
   - `docs/plans/sprint_8_implementation_plan.md`: This comprehensive implementation specification.
   - `docs/sprint_8_model_training_audit.md`: Rigorous audit covering frozen checkpoint provenance, mathematical estimators, and test-time safeguards.
   - `reports/sprint8_compute_matched_table.md`: Comprehensive table of all matched-compute configurations.
   - `reports/sprint8_probabilistic_metrics.md`: CRPS, Brier score, reliability, coverage, and spread-skill results.
   - `reports/sprint8_per_lead_uncertainty.md`: Day-by-day uncertainty evolution ($D+0$ to $D+6$).
   - `reports/sprint8_precipitation_calibration.md`: Threshold calibration for $P > 15$ and $P > 30$ mm.
   - `reports/sprint8_champion_holdout_test.json`: Final 2023 holdout evaluation record.
   - `docs/walkthrough_sprint_8.md`: Complete summary walkthrough of findings for scientific stakeholders.
