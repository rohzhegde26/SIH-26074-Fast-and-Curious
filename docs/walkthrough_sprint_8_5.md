# Sprint 8.5 Walkthrough: Predictive Uncertainty, Calibration and Output-Quality Diagnostic

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Date:** September 27, 2026  
**Auditor / Lead:** Senior Meteorological Systems Research Agent  
**Status:** Deep Research Complete (Awaiting User Review Before Implementation)  

---

## 1. Overview and Motivation

Sprint 8 established that inference compute can be traded between denoising depth ($S$) and ensemble breadth ($K$). Under a fixed budget of 32 neural function evaluations (NFEs):
- **Deeper ensembles** ($K=2, S=16, \eta=0.0$) achieve superior continuous probabilistic scores (Fair-CRPS = 0.5401, Multivariate Energy Score ≈ 0.2769).
- **Broader ensembles** ($K=8, S=4, \eta=0.5$) achieve superior deterministic point accuracy and convective storm recall (Wet-MAE = 6.51 mm, CSI@30 = 0.728).

However, Sprint 8 revealed a fundamental vulnerability:
- The champion configuration has a precipitation spread-skill ratio (SSR) of only **0.437**.
- Empirical coverage of nominal 90% prediction intervals is only **21.0%** on the 2022 validation set.
- Increasing ensemble size $K$ from 8 to 16 produces diminishing returns and fails to resolve the spread deficit.

Sprint 8.5 was initiated to perform a rigorous diagnostic research pass before committing to neural model scaling in Sprint 9. Its goal is to resolve the underlying question:
> **Is the uncertainty deficit primarily an artifact of post-hoc calibration scale, reverse diffusion sampler dynamics, or an intrinsic representation bottleneck in the Candidate 3 neural backbone?**

---

## 2. Phase 0 Audit: Metric Reconciliation

Before designing new experiments, a required audit resolved the numerical divergence between the condition-history CRPS values (~0.603) and the bootstrap point estimates (~0.550).

### 2.1 The Algebraic Root Cause
The validation dataset comprises 122 cubes. With a dataloader batch size of 4:
- 30 batches contain 4 cubes ($30 \times 4 = 120$ cubes).
- The 31st batch contains only the remaining 2 cubes.

In `evaluate_sprint8_ensemble.py`, `all_crps_list` recorded one value per batch and computed `np.mean(all_crps_list)`. This gave the 31st batch an unweighted $1/31 \approx 3.23\%$ weight instead of its legitimate sample proportion $2/122 \approx 1.64\%$. Because batch 31 happened to contain extreme late-monsoon storm events (mean CRPS = 3.8070 vs 0.4961 for batches 1-30), giving it double weight shifted the unweighted batch mean upward by $+0.0525$ points across every condition.

### 2.2 Proof of Exact Equivalence
When aggregated as the true sample-weighted mean across all 122 individual cubes:
- `B32_K2_S16`: **0.54009950** (matches bootstrap point estimate to 15 decimal places)
- `B32_K4_S8`: **0.54334022** (matches bootstrap point estimate to 15 decimal places)
- `B32_K8_S4`: **0.55235116** (matches bootstrap point estimate to 15 decimal places)
- `B32_K8_S4_ETA05`: **0.55038854** (matches bootstrap point estimate to 15 decimal places)

The relative rankings and paired differences are 100% identical and monotonic under both aggregations. The reconciliation has been formalized in `reports/sprint8_5_metric_reconciliation.md` and `reports/sprint8_5_metric_reconciliation.json`.

---

## 3. Deep Research Pillars and Proposed Diagnostics

### 3.1 Pillar 1: Spread Rescaling (EMOS Analogue)
In operational weather prediction, dynamical ensembles are post-processed via Ensemble Model Output Statistics (EMOS) (Gneiting et al., 2005; Scheuerer, 2014). For non-parametric diffusion outputs, we formulate multiplicative spread inflation around the ensemble mean:
$$x'_k = \bar{x} + \alpha (x_k - \bar{x}), \quad \alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$$
- **Mathematical Invariant:** The unconstrained ensemble mean $\bar{x} = \frac{1}{K}\sum x'_k$ is invariant to $\alpha$, preserving deterministic MAE prior to clipping.
- **Physical Enforcement:** Values are clipped to non-negative precipitation ($x''_k = \max(0, x'_k)$).
- **Lead-Dependent Generalization:** If global scaling succeeds, lead-dependent factors $\alpha_d$ ($d \in \{0, \dots, 6\}$) will be evaluated to accommodate growing forecast uncertainty.

### 3.2 Pillar 2: Threshold Probability Recalibration
Raw ensemble frequencies $\hat{p} = \frac{1}{K}\sum \mathbb{I}(x_k > \tau)$ at $K=8$ suffer from coarse $0.125$ quantile steps. Two post-processing recalibration methods will be tested:
1. **Isotonic Regression:** Non-parametric monotonic mapping fitted to minimize squared error on empirical calibration bins.
2. **Platt Logistic Scaling:** Parametric logistic mapping $\tilde{p} = \sigma(w_0 + w_1 \text{logit}(\hat{p}))$ providing tail regularization.

Parameters will be fitted on the internal 2022 calibration split (cubes 1-61) and evaluated out-of-sample on cubes 62-122 using Brier Score, Brier Skill Score (against 2015-2021 climatology), and reliability curves.

### 3.3 Pillar 3: Split-Conformal Non-Negative Intervals
To address the low nominal coverage (21.0% at 90%), split-conformal calibration will compute empirical non-conformity quantiles on the calibration split:
$$R_i = \frac{|y_i - \hat{\mu}_i|}{\hat{\sigma}_i + \epsilon}$$
Prediction intervals are constructed on unseen validation cases as:
$$C(x) = [\max(0, \hat{\mu}(x) - \hat{q}_{1-\gamma} \hat{\sigma}(x)), \; \hat{\mu}(x) + \hat{q}_{1-\gamma} \hat{\sigma}(x)]$$
Under the relevant exchangeability assumptions, this construction targets finite-sample marginal coverage while respecting the non-negative physical boundary ($P \ge 0$). Because the Sprint 8.5 split is chronological weather data, coverage must be verified empirically rather than assumed.

### 3.4 Pillar 4: Sampler vs Calibration Attribution
To cleanly isolate the source of the uncertainty deficit, we evaluate four orthogonal states:
1. **Baseline:** Uncalibrated $K=8, S=4, \eta=0.5$.
2. **Sampling Shift Only:** Modifying $\eta$ or step allocations without post-hoc calibration.
3. **Calibration Shift Only:** Applying spread rescaling and conformal adjustments to fixed Sprint 8 outputs.
4. **Combined:** Jointly optimized sampler plus post-hoc calibration.

### 3.5 Pillar 5: Physical Repair-Burden Attribution
Sprint 8 reported that 31% to 37% of precipitation predictions undergo non-negativity repair ($P = \max(0, P)$), shifting 7% to 8% of total rainfall mass. This is a candidate mechanism for under-dispersion, not yet a proven cause. We will conduct an attribution test on raw unclipped diffusion outputs to determine whether negative values indicate:
- A systematic negative bias in light rain areas, or
- Standard diffusion noise around the zero boundary.

### 3.6 Pillar 6: Spatial Sharpness and Texture Preservation
Ensemble averaging acts as a low-pass filter. To verify that gains in point accuracy and CSI@30 do not come from unphysical spatial blurring, we evaluate:
- **Laplacian Energy ($E_{\text{Lap}}$):** Mean squared second spatial derivatives.
- **2D Radial Power Spectral Density (RAPSD):** Retention of high-frequency convective kinetic energy ($k \ge 0.1 \text{ km}^{-1}$).

---

## 4. Internal Validation Split and Holdout Governance

To eliminate data leakage during post-hoc calibration:
- **Internal 2022 Split:** The 122 validation cubes are chronologically divided into:
  - **Calibration Block (Cubes 1 to 61):** Used to fit $\alpha$, isotonic maps, and conformal quantiles.
  - **Evaluation Block (Cubes 62 to 122):** Used to assess out-of-sample calibration performance.
- **2023 Holdout Quarantined:** The 2023 test set remains 100% untouched until a single final configuration is frozen.

---

## 5. Decision Gate for Sprint 9

| Finding | Diagnostic Conclusion | Decision for Sprint 9 |
|---|---|---|
| Post-hoc calibration resolves spread (SSR $\ge 0.85$, Cov@90 $\ge 75\%$) without harming Wet-MAE (within 1% relative shift) or CSI@30 (within 0.01 absolute). | Bottleneck was calibration scale. | Focus Sprint 9 model scaling entirely on advancing deterministic point accuracy and high-resolution storm features. |
| Post-hoc calibration fails or distorts physical fields, but sampler adjustments restore spread. | Bottleneck was sampler dynamics. | Introduce advanced diffusion samplers (predictor-corrector, Langevin) alongside capacity scaling. |
| Neither post-hoc calibration nor sampler tuning can resolve under-dispersion without degrading point skill. | Bottleneck is neural representation. | Increase model capacity and introduce explicit dispersion-promoting loss objectives (e.g. CRPS loss, ensemble distillation). |

---

## 6. Next Steps

With deep research and Phase 0 reconciliation complete, the implementation plan, model training audit, and metric reconciliation artifacts have been drafted. Following user review and approval, we will proceed to generate the calibration and diagnostic scripts (`src/models/calibration.py`, `scripts/evaluate_sprint8_5_calibration.py`, `tests/models/test_calibration.py`).


## 7. Diagnostic Interpretation Caveats

- Thermodynamic and wind variables show substantially better uncertainty behavior than precipitation, but they should not be described as perfectly calibrated solely from SSR and empirical coverage.
- For precipitation, use ensemble rank histograms; ordinary continuous PIT is inappropriate with zero-mass and finite K unless a randomized tie-handling procedure is explicitly implemented.
- Physical clipping is treated as a candidate mechanism for precipitation under-dispersion and is tested in Sprint 8.5 rather than assumed to be causal.
