# Sprint 8 Implementation Plan: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 27, 2026 (Revised with Review Amendments)  

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
- **Training Noise Schedule**: Linear beta schedule with `beta_start = 1e-4`, `beta_end = 0.035`, and $T = 100$ diffusion timesteps (Nichol & Dhariwal / Ho et al.)
- **Checkpoint Object ID**: Git LFS hash `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`
- **Hardware Budget**: 6.0 hours of 2x Tesla T4 Kaggle student-tier GPU accelerators

### 1.3 Scope and Non-Goals
Sprint 8 investigates the test-time scaling frontier between denoising steps and ensemble sample counts.
- **In-Scope**:
  - Initial-noise stochasticity ($\eta = 0, K > 1$) versus trajectory stochasticity ($\eta > 0$).
  - Matched-compute budget allocations ($B = K \times S \in \{8, 16, 32, 64\}$).
  - Probabilistic verification (Fair-CRPS for $K \ge 2$, deterministic CRPS = MAE for $K=1$, Brier score, Brier Skill Score against train and pooled climatology, reliability diagrams, coverage, sharpness, spread-skill ratio, rank histograms, energy score).
  - Explicit pairwise ensemble diversity diagnostics (pairwise RMSE, pairwise correlation, effective diversity ratio).
  - Member-wise physical constraints with repair-burden diagnostics (clipping rates, mass shifts, temperature ordering repairs).
  - Memory-safe chunked ensemble batching (`ensemble_member_chunk_size` in $\{1, 2, 4\}$).
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

Sprint 7 demonstrated that DDIM-4 ($S = 4$) reduces latency from 734.9 ms to 96.8 ms while outperforming DDIM-32 ($S = 32$) on deterministic error metrics. This computational surplus unlocks a fundamental question in generative weather forecasting:

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
- **Continuous Ranked Probability Score (CRPS)**: Generalizes absolute error to probabilistic forecasts. Unbiased finite-sample estimators (Fair-CRPS) must be used for $K \ge 2$ to avoid penalizing small ensembles. For $K=1$, CRPS degenerates to ordinary absolute error ($\text{CRPS}_{\text{det}} = \text{MAE}$).
- **Brier Score and Brier Skill Score (BSS)**: Quantifies probabilistic skill for binary threshold events ($P > 15$ mm, $P > 30$ mm). Skill scores must be evaluated against training-derived climatology as the primary no-skill baseline, with pooled validation climatology reported as secondary reference.
- **Spread-Skill Relationship**: An ideally calibrated ensemble satisfies $\text{Spread} \approx \text{RMSE}$. An under-dispersive ensemble exhibits $\text{Spread} < \text{RMSE}$, indicating overconfidence.
- **Ensemble Diversity Diagnostics**: Direct measurement of inter-member spread (pairwise RMSE, spatial correlation) to prevent degenerate or false ensembles.

---

## 4. Formal Research Hypotheses and System Invariants

Sprint 8 establishes 1 foundational system invariant and evaluates 7 formal research hypotheses (H0 through H6) with explicit falsification criteria:

### Foundational System Invariant: Physical Consistency Invariant (PCI)
Every reported ensemble member must strictly satisfy all physical constraints:
- Non-negative precipitation: $P \ge 0$ mm/day.
- Bounded relative humidity: $0 \le \text{RH} \le 100\%$.
- Thermodynamic temperature ordering: $T_{\min} \le T_{\max}$.

Rather than treating the elimination of physical violations as a tautological hypothesis, the evaluation suite mandates PCI as a hard system invariant and tracks the empirical **Physical Repair Burden**:
- Pre-repair violation rates for each variable.
- Post-repair violation rates (strictly $0.0\%$).
- Percentage of spatial cells modified by post-hoc physical adjustments.
- Mean and maximum adjustment magnitudes ($|\Delta P|$, $|\Delta \text{RH}|$, $|\Delta T_{\max}|$).

### H0: Informative Initial-Noise Ensembles
- **Hypothesis**: At $\eta = 0$, generating $K > 1$ members via independent initial Gaussian noise seeds yields an informative predictive ensemble whose ensemble mean achieves equal or better CMVS than a single member ($K=1$), accompanied by meaningful non-zero ensemble spread.
- **Control**: DDIM-4, $K=1, \eta=0$.
- **Treatment**: DDIM-4, $K \in \{2, 4, 8, 16\}, \eta=0$.
- **Falsification Criterion**: Falsified if ensemble mean CMVS degrades by $> 2.0\%$ or if mean pairwise member RMSE $\to 0$ across all variables.

### H1: Diminishing Marginal Returns of Ensemble Scaling
- **Hypothesis**: At fixed $S=4$ and $\eta=0$, expanding $K$ yields diminishing marginal reductions in Fair-CRPS and Brier score, with improvements saturating beyond $K = 8$ or $K = 16$.
- **Control**: DDIM-4 with $K \in \{2, 4\}$.
- **Treatment**: DDIM-4 with $K \in \{8, 16, 32\}$.
- **Falsification Criterion**: Falsified if Fair-CRPS improves linearly with $K$ up to $K=32$ without slope attenuation.

### H2: Sample Broader Beats Denoise Deeper at Matched Compute
- **Hypothesis**: At matched total NFE ($B = 32$), a low-step ensemble ($K=8, S=4$) achieves superior probabilistic skill (Fair-CRPS, Brier score, and threshold reliability) compared to a single deep deterministic trajectory ($K=1, S=32$), without sacrificing physical consistency.
- **Control**: $K=1, S=32$ (32 NFE, deterministic CRPS = MAE).
- **Treatment**: $K=8, S=4$ (32 NFE, Fair-CRPS).
- **Secondary Treatments**: $K=4, S=8$ (32 NFE), $K=2, S=16$ (32 NFE).
- **Falsification Criterion**: Falsified if $K=1, S=32$ achieves lower CRPS and higher BSS@30 than $K=8, S=4$.

### H3: Value of Trajectory Stochasticity ($\eta > 0$)
- **Hypothesis**: For a fixed $(K, S)$ configuration, injecting intermediate trajectory noise ($\eta \in \{0.25, 0.5, 1.0\}$) improves probabilistic calibration (reliability and spread-skill ratio) over pure initial-noise diversity ($\eta = 0$) without unacceptable deterioration in deterministic MAE or forecast sharpness.
- **Control**: DDIM-4, $K=4, \eta=0$.
- **Treatment**: DDIM-4, $K=4, \eta \in \{0.25, 0.5, 1.0\}$ using the identical initial seed manifest.
- **Falsification Criterion**: Falsified if $\eta > 0$ increases wet-MAE by $> 5.0\%$ or degrades Brier score relative to $\eta = 0$.

### H4: Decoupling of Probabilistic Skill from Deterministic CMVS
- **Hypothesis**: An ensemble configuration can produce substantial gains in Fair-CRPS, Brier score, and coverage without showing large numerical shifts in ensemble-mean CMVS.
- **Falsification Criterion**: Falsified if improvements in Fair-CRPS strictly correlate with and require proportional improvements in CMVS.

### H5: Precipitation Tail Weighting Transfers to Probabilistic Exceedance
- **Hypothesis**: The convective tail weighting ($w_p = 3.0$ on $> 15$ mm) introduced in Sprint 6 enables the ensemble to generate well-calibrated exceedance probabilities $P(P > 15)$ and $P(P > 30)$ that outperform training-derived climatology.
- **Control**: Historical training climatological event frequency baseline.
- **Treatment**: Ensemble-derived probabilities from DDIM-4 ($K \ge 4$).
- **Falsification Criterion**: Falsified if BSS@15 $\le 0$ or BSS@30 $\le 0$ relative to training climatology.

### H6: Lead-Time Uncertainty Responsiveness
- **Hypothesis**: The ensemble spread and CRPS exhibit physically meaningful growth across forecast horizons $D+0$ through $D+6$, reflecting accumulating NWP forecast uncertainty.
- **Metric**: Lead-specific spread $\sigma(d)$ and lead-specific CRPS across $d \in \{0, 1, \dots, 6\}$.
- **Falsification Criterion**: Falsified if ensemble spread at $D+6$ is equal to or lower than at $D+0$.

### H7: Physical Repair Burden and Intrinsic Model Realism
- **Hypothesis**: The unconstrained generative output of Candidate 3 exhibits high intrinsic physical realism, such that the required post-hoc physical repair burden is minimal (pre-repair violation rates $< 2.0\%$ across all variables), and physical repairs do not distort the underlying meteorological fields.
- **Metric**: Pre-repair violation frequency and clipping mass change $\Delta M_{\text{precip}}$.
- **Falsification Criterion**: Falsified if pre-repair $T_{\min} > T_{\max}$ violation rate exceeds $5.0\%$ or if precipitation clipping alters total domain rainfall mass by $> 3.0\%$.

---

## 5. Mathematical Formulations and Estimators

### 5.1 Member-Wise Physical Inversion and Repair Ordering
Let $y_{\text{norm}}^{(k)} \in \mathbb{R}^{T_f \times C \times H_{\text{fine}} \times W_{\text{fine}}}$ be the standardized output of the diffusion reverse process for member $k \in \{1, \dots, K\}$.
All post-processing must strictly follow this order:

1. **Standardization Inversion**:
   $$\tilde{y}_v^{(k)} = y_{\text{norm}, v}^{(k)} \cdot \sigma_{\text{train}, v} + \mu_{\text{train}, v}$$

2. **Member-Wise Non-Linear Physical Bounds and Repair Logging**:
   - Precipitation non-negativity:
     $$P^{(k)} = \max\left(0, \tilde{P}^{(k)}\right)$$
     Log clipped fraction and mass change: $\Delta M_P^{(k)} = \sum (\max(0, \tilde{P}^{(k)}) - \tilde{P}^{(k)})$.
   - Relative humidity bounds:
     $$\text{RH}^{(k)} = \text{clip}\left(\tilde{\text{RH}}^{(k)}, 0.0, 100.0\right)$$
   - Thermodynamic temperature ordering:
     $$\text{Violation}: \quad \mathcal{V}_T^{(k)} = \mathbb{I}\left(\tilde{T}_{\min}^{(k)} > \tilde{T}_{\max}^{(k)}\right)$$
     $$T_{\max}^{(k)} = \max\left(\tilde{T}_{\max}^{(k)}, \tilde{T}_{\min}^{(k)}\right), \quad T_{\min}^{(k)} = \tilde{T}_{\min}^{(k)}$$
     Log violation rate $\mathbb{E}[\mathcal{V}_T]$ and mean repair delta $|\Delta T_{\max}|$.

3. **Ensemble Statistics Computation in Physical Space**:
   - Ensemble Mean: $\mu = \frac{1}{K}\sum_{k=1}^K y^{(k)}$
   - Ensemble Variance: $\sigma^2 = \frac{1}{K-1}\sum_{k=1}^K (y^{(k)} - \mu)^2$ for $K \ge 2$
   - Physical Quantiles: $q_\alpha = \text{Quantile}_\alpha\left(\{y^{(1)}, \dots, y^{(K)}\}\right)$ for $\alpha \in \{0.10, 0.25, 0.50, 0.75, 0.90\}$
   - Exceedance Probability: $P(P > \text{thresh}) = \frac{1}{K}\sum_{k=1}^K \mathbb{I}\left(P^{(k)} > \text{thresh}\right)$

**Rationale on Member-Wise Inversion**:
Non-linear physical clipping is performed member-by-member before ensemble averaging to guarantee that every individual ensemble member represents a physically valid, realizable meteorological state and that exceedance event probabilities have proper semantics. By Jensen's Inequality, $\mathbb{E}[\max(0, X)] \ge \max(0, \mathbb{E}[X])$; member-wise clipping shifts the mean slightly relative to raw latents, and the magnitude of this shift is explicitly tracked as a diagnostic.

### 5.2 CRPS Evaluation Rules: Fair-CRPS for $K \ge 2$ vs Deterministic CRPS for $K=1$
For a finite ensemble $\{y^{(1)}, \dots, y^{(K)}\}$ and scalar observation $y_{\text{true}}$:

1. **For $K \ge 2$ (Unbiased Fair-CRPS)**:
   $$\text{CRPS}_{\text{fair}}\left(\{y^{(k)}\}_{k=1}^K, y_{\text{true}}\right) = \frac{1}{K}\sum_{k=1}^K |y^{(k)} - y_{\text{true}}| - \frac{1}{2K(K-1)}\sum_{k=1}^K \sum_{j=1}^K |y^{(k)} - y^{(j)}|$$
   The second term removes the positive finite-sample bias of the empirical CDF, ensuring fair comparisons across different ensemble sizes $K$.

2. **For $K = 1$ (Deterministic CRPS = MAE)**:
   For $K=1$, the Fair-CRPS denominator $K(K-1) = 0$ is mathematically undefined. A single member represents a degenerate deterministic point forecast. The continuous ranked probability score of a deterministic point forecast reduces identically to the absolute error:
   $$\text{CRPS}_{\text{det}}(y^{(1)}, y_{\text{true}}) = |y^{(1)} - y_{\text{true}}| = \text{MAE}$$
   **Strict Reporting Protocol**: Evaluation tables must never label a $K=1$ outcome as "Fair-CRPS". The metric must be explicitly labeled as $\text{CRPS}_{\text{det}}$ (or MAE).

### 5.3 Brier Score and Climatological Skill Decomposition
For binary threshold event $\mathcal{E} = \mathbb{I}(y_{\text{true}} > \tau)$ and predicted ensemble probability $p = \frac{1}{K}\sum_{k=1}^K \mathbb{I}(y^{(k)} > \tau)$:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - o_i)^2$$

where $o_i \in \{0, 1\}$ is the observed occurrence.

**Climatological Baselines for BSS**:
1. **Primary Reference: Training-Derived Climatology**:
   Computed from the 2014-2021 training split for each lead $d \in \{0, \dots, 6\}$:
   $$\bar{o}_{\text{train}}(d, \tau) = \frac{1}{N_{\text{train}}}\sum_{i \in \text{Train}} \mathbb{I}(y_{\text{train}, i}(d) > \tau)$$
   $$\text{BS}_{\text{clim, train}}(d) = \bar{o}_{\text{train}}(d, \tau) \cdot (1 - \bar{o}_{\text{train}}(d, \tau))$$
   $$\text{BSS}_{\text{train}}(d) = 1 - \frac{\text{BS}(d)}{\text{BS}_{\text{clim, train}}(d)}$$
2. **Secondary Reference: Pooled Validation Climatology**:
   Sample base rate over the 2022 validation split $\bar{o}_{\text{val}}$, reported as a secondary reference for transparency.

The Brier Score is partitioned into Murphy's canonical components:
$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

### 5.4 Pairwise Ensemble Diversity Diagnostics
To prevent degenerate or false ensembles (where members are nearly identical), every configuration reports:
1. **Mean Pairwise Member RMSE**:
   $$\text{Div}_{\text{RMSE}} = \frac{2}{K(K-1)} \sum_{k=1}^K \sum_{j=k+1}^K \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i^{(k)} - y_i^{(j)})^2}$$
2. **Mean Pairwise Spatial Correlation**:
   $$\bar{\rho}_{\text{pair}} = \frac{2}{K(K-1)} \sum_{k=1}^K \sum_{j=k+1}^K \text{Corr}\left(y^{(k)}, y^{(j)}\right)$$
3. **Ensemble Spatial Variance**:
   $$\bar{\sigma}_{\text{ens}}^2 = \frac{1}{N} \sum_{i=1}^N \left(\frac{1}{K-1}\sum_{k=1}^K (y_i^{(k)} - \mu_i)^2\right)$$
4. **Effective Diversity Ratio (EDR)**:
   $$\text{EDR}(K) = \frac{\text{Div}_{\text{RMSE}}(K)}{\text{Div}_{\text{RMSE}}(K=2)}$$

### 5.5 Spread-Skill Ratio (SSR)
The ensemble spread-skill relationship across all validation points $i \in \{1, \dots, N\}$ is:
$$\text{Ensemble Spread} = \sqrt{\frac{1}{N}\sum_{i=1}^N \sigma_i^2}, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mu_i - y_{\text{true}, i})^2}$$
$$\text{SSR} = \frac{\text{Ensemble Spread}}{\text{RMSE}}$$

### 5.6 Prediction Interval Coverage and Sharpness
For nominal coverage level $(1 - \alpha) \in \{0.50, 0.80, 0.90\}$ with quantile bounds $[q_{\alpha/2}, q_{1 - \alpha/2}]$:
- Empirical Coverage: $\text{Cov} = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left(q_{\alpha/2, i} \le y_{\text{true}, i} \le q_{1 - \alpha/2, i}\right)$
- Interval Sharpness: $\text{Width} = \frac{1}{N}\sum_{i=1}^N \left(q_{1 - \alpha/2, i} - q_{\alpha/2, i}\right)$

### 5.7 Multivariate Energy Score
To assess inter-variable spatial coherence across all 6 variables jointly:
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
| K=1, eta=0 for S in {4, 8, 16} (Establishes point-forecast frontier, CRPS_det = MAE)               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| PHASE 2: STAGED ENSEMBLE COUNT & STOCHASTICITY SWEEPS                                             |
| Stage A: Fixed S=4, eta=0, sweep K in {2, 4, 8, 16} (Initial-noise diversity + Pairwise Diversity)|
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
Run deterministic DDIM ($K = 1, \eta = 0$) across $S \in \{4, 8, 16\}$ to establish baseline anchors for single-trajectory performance. For these runs, report $\text{CRPS}_{\text{det}} = \text{MAE}$.

### 6.3 Phase 2: Staged Ensemble Count and Stochasticity Sweeps
- **Stage A (Initial-Noise Diversity)**:
  Evaluate DDIM-4 with $\eta = 0$ for $K \in \{2, 4, 8, 16\}$.
  Seed manifest is strictly nested:
  $$\text{Seed Manifest}(K) = \{s_1, \dots, s_K\}, \quad s_k = 20260900 + 1000 \cdot k$$
  This ensures that the $K=4$ run contains the exact same members as $K=2$ plus two additional members.
  Report Fair-CRPS, BSS@15, BSS@30, and Pairwise Diversity Diagnostics.
- **Stage B (Trajectory Stochasticity)**:
  Select the optimal $K$ from Stage A and evaluate $\eta \in \{0.25, 0.5, 1.0\}$ using the exact same seed manifest.
- **Stage C (Saturation Gate for $K=32$)**:
  Evaluate $K=32$ only if the relative improvement in Fair-CRPS from $K=8$ to $K=16$ exceeds $2.0\%$.

### 6.4 Phase 3: The Matched-Compute Frontier Matrix
This represents the primary scientific deliverable of Sprint 8. For each budget $B \in \{8, 16, 32, 64\}$, all factorizations $(K, S)$ such that $K \times S = B$ are evaluated on the full 2022 validation season:

| Budget ($B$) | Member Count ($K$) | Denoising Steps ($S$) | Nominal NFE | CRPS Metric Type | Wall-Clock Latency (Sequential) | Wall-Clock Latency (Chunked $C_{\text{ens}}=4$) |
|---|---|---|---|---|---|---|
| **8** | 1 | 8 | 8 | $\text{CRPS}_{\text{det}}$ (= MAE) | Profiling Target | Profiling Target |
| **8** | 2 | 4 | 8 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **16** | 1 | 16 | 16 | $\text{CRPS}_{\text{det}}$ (= MAE) | Profiling Target | Profiling Target |
| **16** | 2 | 8 | 16 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **16** | 4 | 4 | 16 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **32 (Flagship)** | 1 | 32 | 32 | $\text{CRPS}_{\text{det}}$ (= MAE) | Profiling Target | Profiling Target |
| **32 (Flagship)** | 2 | 16 | 32 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **32 (Flagship)** | 4 | 8 | 32 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **32 (Flagship)** | 8 | 4 | 32 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **64** | 1 | 64 | 64 | $\text{CRPS}_{\text{det}}$ (= MAE) | Profiling Target | Profiling Target |
| **64** | 2 | 32 | 64 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **64** | 4 | 16 | 64 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **64** | 8 | 8 | 64 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |
| **64** | 16 | 4 | 64 | $\text{CRPS}_{\text{fair}}$ | Profiling Target | Profiling Target |

Both nominal NFE and empirical wall-clock latency (sequential and chunked) are reported to maintain strict compute fairness.

---

## 7. Ensemble Batching, Chunking, and Memory Safeguards

### 7.1 Chunked Member Batching Architecture
To prevent CUDA out-of-memory errors on 16 GB Tesla T4 GPUs, the evaluation runner implements memory-safe ensemble chunking.
Directly expanding an evaluation batch by $K$ (e.g., $\text{batch}=8 \times K=16 \to 128$) exceeds VRAM limits.
Instead, the sampler processes members in bounded chunks:

$$\text{ensemble\_member\_chunk\_size} \in \{1, 2, 4\}$$

For a batch of evaluation cubes:
1. Divide the $K$ requested members into chunks of size $C_{\text{ens}} \le 4$.
2. For each chunk $c \in \{1, \dots, \lceil K / C_{\text{ens}} \rceil\}$, expand conditioning tensors by $C_{\text{ens}}$, execute the DDIM reverse loop, and immediately transfer completed physical outputs.
3. Concatenate outputs along the member dimension $[K, T_f, C, H_{\text{fine}}, W_{\text{fine}}]$.

### 7.2 Latency Profiling Specification
Every configuration benchmarks and reports two distinct latencies:
1. **Sequential Ensemble Latency**: Execution with $C_{\text{ens}} = 1$.
2. **Chunked Ensemble Latency**: Execution with $C_{\text{ens}} = 4$ (or $C_{\text{ens}} = 2$).

---

## 8. Statistical Rigor and Case-Level Bootstrap Protocol

### 8.1 Case-Level Paired Resampling
Because meteorological variables display strong spatial autocorrelation and multi-day temporal autocorrelation across the 7-lead sequence, individual grid cells must never be treated as independent statistical samples.
- **Resampling Unit**: The complete 7-day $\times$ 6-variable fine forecast cube for an entire forecast initialization date.
- **Bootstrap Replicates**: $B = 1000$ paired resamples with replacement.
- **Pairing Invariant**: For every bootstrap sample, differences between competing configurations are computed case-by-case on the identical sampled dates.
- **Confidence Intervals**: 95% empirical bootstrap percentile intervals $[q_{0.025}, q_{0.975}]$ reported for all primary metrics.

### 8.2 Quarantined Holdout Protocol
The 2023 holdout test set is quarantined. No exploratory parameter sweeps, threshold tuning, or hyperparameter selection are conducted on 2023 data. Exactly one validation-selected champion configuration is evaluated on 2023 test data at the end of the study.

---

## 9. Deployment Regimes and Operational Decision Framework

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

## 10. Compute Allocation on Kaggle Accelerators

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

## 11. Required Code and Documentation Deliverables

The implementation of Sprint 8 will produce the following structured files in the repository:

1. **Model and Sampler Logic**:
   - `src/models/ensemble.py`: Reusable ensemble wrapper supporting nested seeds, chunked batched sampling ($C_{\text{ens}} \le 4$), trajectory noise $\eta$, pairwise diversity diagnostics, and member retention.
2. **Verification and Test Suite**:
   - `tests/models/test_ensemble_reproducibility.py`: Unit tests validating exact seed repeatability, nested seed subsets, member-wise physical clipping, $K=1$ deterministic CRPS fallback, finite-ensemble Fair-CRPS computation ($K \ge 2$), and VRAM chunking.
3. **Execution Scripts**:
   - `scripts/evaluate_sprint8_ensemble.py`: Full evaluation pipeline executing Phases 0 through 4 on Kaggle.
   - `scripts/benchmark_sprint8_compute_frontier.py`: Matched-compute latency (sequential vs chunked) and throughput benchmark.
4. **Documentation and Reports**:
   - `docs/plans/sprint_8_implementation_plan.md`: This comprehensive implementation specification.
   - `docs/sprint_8_model_training_audit.md`: Rigorous audit covering frozen checkpoint provenance, mathematical estimators, and test-time safeguards.
   - `reports/sprint8_compute_matched_table.md`: Comprehensive table of all matched-compute configurations.
   - `reports/sprint8_probabilistic_metrics.md`: Fair-CRPS, Brier score, reliability, coverage, and spread-skill results.
   - `reports/sprint8_per_lead_uncertainty.md`: Day-by-day uncertainty evolution ($D+0$ to $D+6$).
   - `reports/sprint8_precipitation_calibration.md`: Threshold calibration for $P > 15$ and $P > 30$ mm.
   - `reports/sprint8_champion_holdout_test.json`: Final 2023 holdout evaluation record.
   - `docs/walkthrough_sprint_8.md`: Complete summary walkthrough of findings for scientific stakeholders.
