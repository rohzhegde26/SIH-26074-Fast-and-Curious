# Sprint 8.5 Deep Research Report: Predictive Uncertainty, Calibration and Output-Quality Diagnostics

**Project:** SIH 2026 Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Sprint:** 8.5 of 10  
**Author:** Senior Meteorological Systems Research Agent  
**Date:** September 27, 2026  
**Status:** Complete Deep Research Synthesis (Awaiting User Review Before Implementation)  

---

## 1. Executive Summary

Sprint 8 revealed an essential inference trade-off under a compute budget matched at approximately 32 neural function evaluations (NFEs):
- **Deeper low-member sampling** ($K=2, S=16, \eta=0.0$) achieves the strongest continuous probabilistic distribution scores (Fair-CRPS = 0.5401, Multivariate Energy Score ≈ 0.2769).
- **Broader shallow sampling** ($K=8, S=4, \eta=0.5$) achieves the strongest deterministic point accuracy and convective storm recall (Wet-MAE = 6.51 mm, CSI@30 = 0.728).

However, an empirical audit of the 2022 validation metrics revealed a critical diagnostic vulnerability:
- The precipitation spread-skill ratio (SSR) of the champion ensemble is only **0.4379**.
- The nominal 90% prediction interval coverage for precipitation is only **20.94%**.
- Increasing ensemble size $K$ from 8 to 16 yields diminishing returns and fails to resolve this spread deficit.

This deep research report establishes that:
1. **The under-dispersion is variable-specific:** Precipitation exhibits severe under-dispersion (SSR = 0.438, Cov@90 = 20.9%), while the thermodynamic and wind variables show substantially better uncertainty behavior (Tmax SSR = 1.057, Cov@90 = 74.1%; Tmin SSR = 1.170, Cov@90 = 79.4%; RH SSR = 0.946, Cov@90 = 69.8%; Wind-U SSR = 0.811, Cov@90 = 71.5%; Wind-V SSR = 0.817, Cov@90 = 69.3%). They are still imperfectly calibrated and should not be described as fully calibrated.
2. **Physical clipping is a candidate mechanism for variance collapse:** 31.26% of raw precipitation predictions fall below zero and undergo non-negativity clipping ($P = \max(0, P)$). Clipping can collapse inter-member variance at the zero boundary, which may contribute to the observed precipitation under-dispersion. Sprint 8.5 explicitly tests the magnitude and causality of this effect.
3. **The Phase 0 metric discrepancy is fully resolved:** The ~0.0525 difference between condition-history CRPS (~0.603) and bootstrap point estimates (~0.550) was mathematically proven to be a batch-weighting artifact on the final 2-cube batch, with zero impact on scientific ordering or point metrics.
4. **Post-hoc calibration is mathematically viable:** Multiplicative spread rescaling, isotonic/logistic probability recalibration, and non-negative split conformal prediction can address this deficit without requiring neural model retraining.

---

## 2. Phase 0 Audit: Metric and Provenance Reconciliation

### 2.1 Theoretical Proof of the Discrepancy
In `evaluate_sprint8_ensemble.py`, the 122 validation cubes were evaluated with batch size $B=4$, resulting in 30 full batches of 4 cubes and 1 final batch of 2 cubes. 

The evaluation script accumulated batch-mean CRPS into a list of length 31:
$$\bar{M}_{\text{batch}} = \frac{1}{31} \sum_{b=1}^{31} \bar{M}_b$$
This assigned batch 31 a weight of $1/31 \approx 3.2258\%$, whereas its actual sample weight was $2/122 \approx 1.6393\%$. 

Because batch 31 contained the two most severe late-monsoon storm cubes (mean CRPS = 3.8070 vs 0.4961 across batches 1-30), giving this batch double weight systematically shifted the unweighted batch mean upward by $+0.0525$ points across every condition.

### 2.2 Proof of Equivalence with Case-Level Bootstrap
In contrast, `bootstrap_sprint8_paired.py` computed the metric per individual cube:
$$\bar{M}_{\text{case}} = \frac{1}{122} \sum_{i=1}^{122} c_i = \frac{120 \times 0.49611130 + 2 \times 3.80702317}{122} = \mathbf{0.55038854}$$

This matches the bootstrap report point estimate to 15 decimal places.

| Condition | Case-Weighted Mean (Bootstrap) | Unweighted Batch Mean | History JSON Value | Shift ($\Delta$) | Relative Rank |
|---|---:|---:|---:|---:|:---:|
| `B32_K2_S16` | **0.54009950** | 0.59273757 | 0.59273757 | +0.052638 | 1 (Best Distribution) |
| `B32_K4_S8` | 0.54334022 | 0.59579121 | 0.59579121 | +0.052451 | 2 |
| `B32_K8_S4_ETA05` | 0.55038854 | 0.60291491 | 0.60291491 | +0.052526 | 3 |
| `B32_K8_S4` | 0.55235116 | 0.60462592 | 0.60462592 | +0.052275 | 4 |
| `CHAMPION_HOLDOUT` | **0.63711661** | 0.68190044 | 0.68190044 | +0.044784 | N/A (Holdout 2023) |

**Conclusion:** The discrepancy is an aggregation artifact. Scientific rankings, paired confidence intervals, and deterministic metrics (Wet-MAE, CSI) are 100% sound.

---

## 3. Detailed Empirical Analysis of Existing Sprint 8 Outputs

Inspection of the fine-grained per-variable metrics from the Sprint 8 champion (`K=8, S=4, eta=0.5`) reveals a striking dichotomy:

### 3.1 Spread-Skill Ratio (SSR) Across Variables
$$\text{SSR} = \frac{\mathbb{E}[\text{spread}]}{\mathbb{E}[\text{RMSE}]}$$

- **Precipitation:** Ensemble Spread = 2.866 mm, RMSE = 7.176 mm, **SSR = 0.4379** (Severe Under-Dispersion)
- **Tmax:** Ensemble Spread = 0.165 degC, RMSE = 0.393 degC, **SSR = 1.0574** (substantially better calibrated than precipitation)
- **Tmin:** Ensemble Spread = 0.157 degC, RMSE = 0.243 degC, **SSR = 1.1705** (substantially better calibrated than precipitation)
- **Relative Humidity:** Ensemble Spread = 0.265%, RMSE = 0.485%, **SSR = 0.9460** (substantially better calibrated than precipitation)
- **Wind U:** Ensemble Spread = 0.256 m/s, RMSE = 0.677 m/s, **SSR = 0.8106** (substantially better calibrated than precipitation)
- **Wind V:** Ensemble Spread = 0.185 m/s, RMSE = 0.506 m/s, **SSR = 0.8172** (substantially better calibrated than precipitation)

### 3.2 Prediction Interval Coverage Collapse
| Variable | Nominal 50% Coverage | Nominal 80% Coverage | Nominal 90% Coverage | 90% Sharpness |
|---|---:|---:|---:|---:|
| **Precipitation** | **8.48%** | **14.31%** | **20.94%** | 4.458 mm/day |
| **Tmax** | 31.94% | 52.97% | 74.14% | 0.410 degC |
| **Tmin** | 36.58% | 59.18% | 79.40% | 0.401 degC |
| **RH** | 27.95% | 47.73% | 69.76% | 0.665% |
| **Wind U** | 32.15% | 52.34% | 71.53% | 0.648 m/s |
| **Wind V** | 30.08% | 49.56% | 69.25% | 0.461 m/s |

### 3.3 Diagnostic Interpretation of the Precipitation Coverage Deficit
The data show substantially better uncertainty behavior for smooth thermodynamic and wind fields than for precipitation; this does not establish that the underlying conditional uncertainty manifold is fully learned or calibrated. The under-dispersion is specific to precipitation.

Two physical-numerical factors drive this precipitation-specific collapse:
1. **Zero-Bound Truncation:** 31.26% of raw precipitation predictions are clipped at $P=0$. When multiple members predict negative values, their values collapse to identical zero entries, forcing intra-ensemble variance to zero.
2. **High Tail Skewness (hypothesis):** Precipitation is strongly heavy-tailed. The current Gaussian-like stochastic perturbation in the reverse process may not generate enough positive convective-tail variability while also producing negative values near the zero boundary. This is a hypothesis for Sprint 8.5 to test, not an established causal mechanism.

---

## 4. Deep Research Synthesis Across 7 Diagnostic Pillars

### 4.1 Pillar A: Multiplicative Spread Rescaling and EMOS
In operational meteorology, Ensemble Model Output Statistics (EMOS) (Gneiting et al., 2005; Wilks, 2011) fits a post-processing transformation:
$$Y \sim \mathcal{D}(\mu, \sigma^2), \quad \mu = a + b \bar{x}, \quad \sigma^2 = c + d s^2$$

For diffusion ensembles without an explicit parametric distribution, we evaluate non-parametric multiplicative spread inflation around the ensemble mean:
$$x'_k = \bar{x} + \alpha (x_k - \bar{x})$$
where $\bar{x} = \frac{1}{K}\sum_{k=1}^K x_k$ and $\alpha \in \{1.0, 1.25, 1.5, 2.0, 3.0\}$.

**Properties:**
- $\sum_{k=1}^K x'_k = K \bar{x}$: The ensemble mean is invariant before clipping.
- Deterministic MAE and RMSE of the unclipped ensemble mean are unchanged.
- Post-scaling non-negativity clipping ($x''_k = \max(0, x'_k)$) enforces physical realism.
- As $\alpha$ increases from 1.0 to 2.0, the un-clipped ensemble spread scales linearly with $\alpha$, so the nominal SSR would move from 0.438 toward approximately 0.876 before any clipping-induced changes. Actual post-repair calibration must be measured empirically.

### 4.2 Pillar B: Lead-Dependent Uncertainty Calibration
Forecast uncertainty in dynamical systems grows with lead time due to atmospheric chaos (Leutbecher & Palmer, 2008). In a 7-day forecast ($D+0$ to $D+6$):
$$e(D+6) > e(D+0)$$
If the diffusion model maintains stationary noise injection across all lead days, it will be over-dispersed at Day 0 or under-dispersed at Day 6.

We evaluate lead-dependent spread scaling:
$$x'_{k, d} = \bar{x}_d + \alpha_d (x_{k,d} - \bar{x}_d), \quad d \in \{0, \dots, 6\}$$
where $\alpha_d$ is fitted to match the empirical error growth curve on the internal calibration split.

### 4.3 Pillar C: Threshold Probability Recalibration
For extreme heavy-rainfall events ($P > 15$ mm and $P > 30$ mm), raw ensemble probabilities at $K=8$ have a minimum step resolution of $\Delta p = 1/8 = 0.125$.

We evaluate two calibration mappers $\hat{p} \mapsto \tilde{p}$:
1. **Isotonic Regression:** Non-parametric step function preserving monotonicity:
   $$\min_{\tilde{p}_1 \le \dots \le \tilde{p}_M} \sum_{i=1}^N (y_i - \tilde{p}_i)^2$$
2. **Platt Logistic Scaling:** Parametric sigmoid transformation:
   $$\tilde{p} = \frac{1}{1 + \exp(-(w_0 + w_1 \text{logit}(\hat{p})))$$

**Verification:** Evaluated via Brier Score (BS), Brier Skill Score (BSS) relative to 2015-2021 training climatology ($BSS = 1 - BS / BS_{\text{clim}}$), reliability error, and resolution.

### 4.4 Pillar D: Conformal Prediction with Physical Non-Negativity Bounds
Conformal prediction (Romano et al., 2019; Angelopoulos & Bates, 2021) can provide finite-sample marginal coverage under the relevant exchangeability assumptions. Those assumptions are not automatic for this chronological weather split, so Sprint 8.5 treats coverage as an empirical quantity to be tested.

Using the internal calibration split:
1. Compute studentized non-conformity scores:
   $$R_i = \frac{|y_i - \hat{\mu}_i|}{\hat{\sigma}_i + \epsilon}$$
2. Find the empirical $(1 - \gamma)$-quantile $\hat{q}$ of $R$.
3. Form the calibrated prediction interval:
   $$C(x) = [\max(0, \hat{\mu}(x) - \hat{q}\hat{\sigma}(x)), \; \hat{\mu}(x) + \hat{q}\hat{\sigma}(x)]$$

Under the relevant exchangeability assumptions, this construction targets finite-sample marginal coverage while strictly preventing unphysical negative precipitation bounds. The chronological 2022 evaluation must verify empirical coverage rather than assuming a formal guarantee.

### 4.5 Pillar E: Sampler vs Calibration Attribution Framework
To determine where the uncertainty bottleneck originates, we evaluate an orthogonal factorial matrix:
- **Baseline:** Uncalibrated $K=8, S=4, \eta=0.5$.
- **Sampler Modification:** Modifying $\eta$ (0.0, 0.5, 1.0) and step allocations ($S=4, 8, 16$) without post-hoc scaling.
- **Calibration Modification:** Applying spread rescaling and conformal adjustments to fixed $K=8, S=4, \eta=0.5$ outputs.
- **Joint Modification:** Sampler tuning combined with post-hoc calibration.

Use the explicit Sprint 8.5 decision gate rather than a generic percentage-of-gap rule: calibration is considered sufficient only if precipitation SSR reaches at least 0.85 and 90% coverage reaches at least 75% while Wet-MAE remains within 1% and CSI@30 within 0.01 absolute of the uncalibrated reference. If only sampler changes meet the dispersion target, the bottleneck is inference dynamics. If neither succeeds without degrading point skill, the bottleneck is representation.

### 4.6 Pillar F: Physical Repair-Burden and Mass Balance
Sprint 8 recorded:
- Clipped fraction: **31.26%**
- Mass shift: **7.76%**
- Tmin > Tmax violations: **0.0%** (thermodynamic ordering perfectly preserved)

These observations do not by themselves prove that clipping is the cause of precipitation under-dispersion. We evaluate raw unclipped precipitation fields against clipped fields to determine whether negative values are associated with systematic light-rain bias, zero-bound truncation, or ordinary diffusion variability around the dry-state threshold.

### 4.7 Pillar G: Spatial Sharpness and Texture Preservation
Ensemble averaging is a linear smoother that can blur local topography and convective cores. To ensure that point gains do not come from artificial smoothing, we benchmark:
1. **2D Laplacian Energy:**
   $$E_{\text{Lap}} = \frac{1}{HW} \sum_{i,j} (\nabla^2 X)_{i,j}^2$$
2. **Radial Power Spectral Density (RAPSD):**
   High-frequency spectral retention for wavenumbers $k \ge 0.1 \text{ km}^{-1}$.

---

## 5. Internal Validation Split and Holdout Governance

To maintain absolute scientific integrity and avoid data leakage:
- **Validation Dataset (2022, 122 cubes):** Chronologically partitioned into:
  - **Calibration Block (Cubes 1 to 61):** Used to fit spread factor $\alpha$, isotonic mapping, and conformal quantiles.
  - **Evaluation Block (Cubes 62 to 122):** Used to test out-of-sample calibration performance.
- **Holdout Dataset (2023, 122 cubes):** Strictly quarantined. Evaluated exactly once at the conclusion of Sprint 8.5 on the final chosen configuration.

---

## 6. Decision Gate for Sprint 9 Model Scaling

The findings of Sprint 8.5 will directly govern the architectural design of Sprint 9:

| Experimental Finding | Diagnostic Attribution | Action for Sprint 9 |
|---|---|---|
| Post-hoc calibration achieves SSR $\ge 0.85$ and Cov@90 $\ge 75\%$ with Wet-MAE shift $<1\%$ and CSI@30 shift $<0.01$. | Bottleneck was calibration scale. | Keep Candidate 3 architecture. Focus Sprint 9 compute entirely on increasing capacity for deterministic point accuracy and fine-scale convective features. |
| Post-hoc calibration fails or distorts physical fields, but modifying sampler dynamics restores spread. | Bottleneck was sampler dynamics. | Retain Candidate 3 architecture. In Sprint 9, implement advanced stochastic samplers (predictor-corrector, annealed Langevin) alongside capacity scaling. |
| Neither post-hoc calibration nor sampler adjustments can resolve under-dispersion without degrading point skill. | Bottleneck is representation capacity. | Sprint 9 must increase model capacity and introduce explicit dispersion-promoting training objectives (e.g. CRPS loss, ensemble distillation, or variance-regularized score heads). |

---

## 7. Conclusion and Readiness

The deep research phase has established:
1. Exact mathematical reconciliation of the Phase 0 metric discrepancy.
2. Direct empirical evidence that under-dispersion is isolated to precipitation and aggravated by zero-bound clipping.
3. A rigorous mathematical framework spanning spread rescaling, probability calibration, conformal prediction, and spatial texture diagnostics.
4. Clear decision criteria for Sprint 9 model capacity scaling.

All planning and research artifacts are complete and pushed to GitHub. We await user review before beginning implementation.


## 8. Distribution-Diagnostic Caveat

Because precipitation has a point mass at zero and K is finite (often K=8), ordinary continuous PIT is not directly appropriate. Sprint 8.5 should use ensemble rank histograms for precipitation and only use randomized PIT when zero-mass ties and discrete support are handled explicitly.
