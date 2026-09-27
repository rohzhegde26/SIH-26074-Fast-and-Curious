# Sprint 8 Walkthrough: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: Smart India Hackathon (SIH) 2026 - Problem Statement 26074  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Date**: September 27, 2026 (Revised with Review Amendments)  

---

## 1. Executive Orientation and Handoff

Sprint 8 transitions the research program from deterministic point forecasting to calibrated probabilistic weather downscaling.

Following the breakthrough results of Sprint 6 (Candidate 3 multi-task tail-weighted loss) and Sprint 7 (DDIM-4 reducing latency to 96.8 ms with superior validation metrics), Sprint 8 investigates how inference compute should be allocated between:
1. Deeper denoising steps ($S$) per member.
2. Broader stochastic ensemble sampling ($K$) across members.

### Core Scientific Identity
$$\text{Total NFE} = K \times S$$

The central question addressed is:
> At matched total inference compute ($B = K \times S$), is extra computation better invested in deeper trajectory denoising (increasing $S$) or in broader ensemble diversity (increasing $K$)?

---

## 2. Key Methodological Invariants and Review Amendments

1. **Frozen Weights**: Zero network retraining. All experiments evaluate the Sprint 6 Candidate 3 champion weights (`models/checkpoints/sprint6_candidate3_multitask_champion.pt`), locked at 15,685,478 parameters.
2. **Audited Noise Schedule**: Verified linear beta schedule (`beta_start = 1e-4`, `beta_end = 0.035`, $T = 100$), matching the codebase.
3. **Decoupled Stochasticity**: Initial-noise diversity ($\eta = 0$) is evaluated separately from intermediate trajectory noise injection ($\eta \in \{0.25, 0.5, 1.0\}$) using paired, nested seed manifests.
4. **Member-Wise Inversion Invariant & Repair Burden**: Predictions are converted to physical units and non-linear physical bounds ($P \ge 0$, $0 \le \text{RH} \le 100\%$, $T_{\min} \le T_{\max}$) applied member-by-member before ensemble reduction. Repair frequency and adjustment magnitudes are explicitly tracked as diagnostics.
5. **Fair Probabilistic Scoring**: Unbiased Fair-CRPS for $K \ge 2$; deterministic CRPS fallback ($\text{CRPS}_{\text{det}} = \text{MAE}$) for $K=1$. Brier Skill Score evaluated against training-derived climatology as primary reference.
6. **Ensemble Diversity Diagnostics**: Explicit tracking of mean pairwise member RMSE, spatial correlation, and effective diversity ratio to detect degenerate ensembles.
7. **Memory-Safe Chunked Execution**: Constrained member chunking ($C_{\text{ens}} \le 4$) to eliminate VRAM exhaustion risks on Tesla T4 GPUs.
8. **Statistical Rigor**: 7-lead cube-preserving paired bootstrap confidence intervals on 2022 validation; 2023 holdout evaluated exactly once.
9. **Compute Platform**: Scheduled across 6.0 hours of 2x Tesla T4 Kaggle student-tier GPU accelerators.

---

## 3. Matched-Compute Experimental Grid

| Budget ($B$) | Member Count ($K$) | Denoising Steps ($S$) | CRPS Metric Type | Primary Probabilistic Metrics |
|---|---|---|---|---|
| **8 NFE** | $K=1, S=8$ vs $K=2, S=4$ | 8 | $\text{CRPS}_{\text{det}}$ vs $\text{CRPS}_{\text{fair}}$ | Fair-CRPS, Wet-MAE, BSS@15, Pairwise RMSE |
| **16 NFE** | $K=1, S=16$ vs $K=2, S=8$ vs $K=4, S=4$ | 16 | $\text{CRPS}_{\text{det}}$ vs $\text{CRPS}_{\text{fair}}$ | Fair-CRPS, Wet-MAE, BSS@15, Pairwise Correlation |
| **32 NFE (Flagship)** | $K=1, S=32$ vs $K=2, S=16$ vs $K=4, S=8$ vs $K=8, S=4$ | 32 | $\text{CRPS}_{\text{det}}$ vs $\text{CRPS}_{\text{fair}}$ | Fair-CRPS, BSS@30, SSR, Reliability, Energy Score |
| **64 NFE** | $K=1, S=64$ vs $K=2, S=32$ vs $K=4, S=16$ vs $K=8, S=8$ vs $K=16, S=4$ | 64 | $\text{CRPS}_{\text{det}}$ vs $\text{CRPS}_{\text{fair}}$ | Fair-CRPS, BSS@30, Saturation Curve, Effective Diversity |

---

## 4. Key Documentation References

- **Implementation Plan**: [docs/plans/sprint_8_implementation_plan.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/docs/plans/sprint_8_implementation_plan.md)
- **Model Training and Evaluation Audit**: [docs/sprint_8_model_training_audit.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/docs/sprint_8_model_training_audit.md)
- **Matched Compute Table**: [reports/sprint8_compute_matched_table.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/reports/sprint8_compute_matched_table.md)
- **Probabilistic Metrics**: [reports/sprint8_probabilistic_metrics.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/reports/sprint8_probabilistic_metrics.md)
- **Quarantined Holdout Report**: [reports/sprint8_champion_holdout_test.json](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/reports/sprint8_champion_holdout_test.json)

---

## 5. Comprehensive Matched-Compute Pareto Frontier Results

The remote Kaggle evaluation campaign executed across the frozen 15.69M Candidate 3 champion weights over the 2022 validation season.

### 5.1 The Empirical Matched-Compute Matrix

| Budget | Configuration | $K$ | $S$ | $\eta$ | NFE | CRPS Metric Type | CRPS ↓ | Wet-MAE (mm) ↓ | CSI@30 ↑ | Latency (ms) | Speedup vs Ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **4 NFE** | `PHASE0_GATE_DDIM4` | 1 | 4 | 0.00 | 4 | CRPS_det | **0.8573** | 7.20 | 0.634 | 129.4 | 5.7x |
| **4 NFE** | `DET_S04` | 1 | 4 | 0.00 | 4 | CRPS_det | **0.8573** | 7.20 | 0.634 | 104.7 | 7.0x |
| **8 NFE** | `DET_S08` | 1 | 8 | 0.00 | 8 | CRPS_det | **0.8717** | 7.48 | 0.559 | 202.1 | 3.6x |
| **16 NFE** | `DET_S16` | 1 | 16 | 0.00 | 16 | CRPS_det | **0.8821** | 7.69 | 0.553 | 408.4 | 1.8x |
| **32 NFE** | `DET_S32` | 1 | 32 | 0.00 | 32 | CRPS_det | **0.8871** | 7.81 | 0.549 | 831.1 | 0.9x |
| **64 NFE** | `DET_S64` | 1 | 64 | 0.00 | 64 | CRPS_det | **0.8907** | 7.88 | 0.547 | 1653.9 | 0.4x |
| **8 NFE** | `ENS_K02_S04_ETA0` | 2 | 4 | 0.00 | 8 | Fair-CRPS | **0.6045** | 6.83 | 0.682 | 218.5 | 3.4x |
| **16 NFE** | `ENS_K04_S04_ETA0` | 4 | 4 | 0.00 | 16 | Fair-CRPS | **0.6043** | 6.63 | 0.721 | 428.1 | 1.7x |
| **32 NFE** | `ENS_K08_S04_ETA0` | 8 | 4 | 0.00 | 32 | Fair-CRPS | **0.6046** | 6.52 | 0.725 | 855.4 | 0.9x |
| **64 NFE** | `ENS_K16_S04_ETA0` | 16 | 4 | 0.00 | 64 | Fair-CRPS | **0.6046** | 6.47 | 0.730 | 1710.7 | 0.4x |
| **16 NFE** | `ENS_K04_S04_ETA025` | 4 | 4 | 0.25 | 16 | Fair-CRPS | **0.6038** | 6.62 | 0.723 | 429.7 | 1.7x |
| **16 NFE** | `ENS_K04_S04_ETA050` | 4 | 4 | 0.50 | 16 | Fair-CRPS | **0.6026** | 6.60 | 0.724 | 428.6 | 1.7x |
| **16 NFE** | `ENS_K04_S04_ETA100` | 4 | 4 | 1.00 | 16 | Fair-CRPS | **0.6027** | 6.53 | 0.729 | 429.1 | 1.7x |
| **8 NFE** | `B8_K1_S8` | 1 | 8 | 0.00 | 8 | CRPS_det | **0.8717** | 7.48 | 0.559 | 215.4 | 3.4x |
| **8 NFE** | `B8_K2_S4` | 2 | 4 | 0.00 | 8 | Fair-CRPS | **0.6045** | 6.83 | 0.682 | 218.3 | 3.4x |
| **16 NFE** | `B16_K1_S16` | 1 | 16 | 0.00 | 16 | CRPS_det | **0.8821** | 7.69 | 0.553 | 420.3 | 1.7x |
| **16 NFE** | `B16_K2_S8` | 2 | 8 | 0.00 | 16 | Fair-CRPS | **0.5959** | 6.99 | 0.675 | 424.0 | 1.7x |
| **16 NFE** | `B16_K4_S4` | 4 | 4 | 0.00 | 16 | Fair-CRPS | **0.6043** | 6.63 | 0.721 | 428.9 | 1.7x |
| **32 NFE** | `B32_K1_S32` | 1 | 32 | 0.00 | 32 | CRPS_det | **0.8871** | 7.81 | 0.549 | 831.6 | 0.9x |
| **32 NFE** | `B32_K2_S16` | 2 | 16 | 0.00 | 32 | Fair-CRPS | **0.5927** | 7.11 | 0.671 | 834.3 | 0.9x |
| **32 NFE** | `B32_K4_S8` | 4 | 8 | 0.00 | 32 | Fair-CRPS | **0.5958** | 6.72 | 0.717 | 839.5 | 0.9x |
| **32 NFE** | `B32_K8_S4` | 8 | 4 | 0.00 | 32 | Fair-CRPS | **0.6046** | 6.52 | 0.725 | 855.5 | 0.9x |
| **32 NFE** | `B32_K8_S4_ETA05` | 8 | 4 | 0.50 | 32 | Fair-CRPS | **0.6029** | **6.51** | **0.728** | **855.0** | **0.9x** |
| **64 NFE** | `B64_K1_S64` | 1 | 64 | 0.00 | 64 | CRPS_det | **0.8907** | 7.88 | 0.547 | 1650.3 | 0.4x |
| **64 NFE** | `B64_K2_S32` | 2 | 32 | 0.00 | 64 | Fair-CRPS | **0.5915** | 7.18 | 0.669 | 1653.4 | 0.4x |
| **64 NFE** | `B64_K4_S16` | 4 | 16 | 0.00 | 64 | Fair-CRPS | **0.5926** | 6.78 | 0.714 | 1660.2 | 0.4x |
| **64 NFE** | `B64_K8_S8` | 8 | 8 | 0.00 | 64 | Fair-CRPS | **0.5961** | 6.58 | 0.723 | 1675.5 | 0.4x |
| **64 NFE** | `B64_K16_S4` | 16 | 4 | 0.00 | 64 | Fair-CRPS | **0.6046** | 6.47 | 0.730 | 1713.7 | 0.4x |

---

## 6. The Flagship 32-NFE Comparison: Deep vs Broad

Under an identical compute budget of 32 NFE (approximately 830 to 909 ms of wall-clock inference time), we compare four distinct factorizations audited across both continuous distribution metrics and deterministic point verification:

| Metric | Deep Deterministic ($K=1, S=32$) | Shallow Ensemble ($K=2, S=16$) | Balanced ($K=4, S=8$) | Broad Ensemble ($K=8, S=4, \eta=0.5$) | Pareto Characterization |
|---|---|---|---|---|---|
| **CRPS Type** | $\text{CRPS}_{\text{det}}$ (= MAE) | $\text{CRPS}_{\text{fair}}$ | $\text{CRPS}_{\text{fair}}$ | $\text{CRPS}_{\text{fair}}$ | Finite-sample unbiased |
| **Composite CRPS** | 0.8871 | **0.5927** | 0.5958 | 0.6029 | Favors deeper steps ($K=2, S=16$) |
| **Precip CRPS (mm/day)** | 3.328 | **2.172** | 2.187 | 2.219 | Favors deeper steps ($K=2, S=16$) |
| **Wet-MAE (mm)** | 7.81 | 7.11 | 6.72 | **6.51** | Favors broader members ($K=8, S=4$) |
| **CSI@30 (Heavy Storm)** | 0.549 | 0.671 | 0.717 | **0.728** | Favors broader members ($K=8, S=4$) |
| **Brier@30** | 0.0290 | 0.0229 | 0.0199 | **0.0188** | Favors broader members ($K=8, S=4$) |
| **BSS@30 (True Train Clim)** | +0.471 | +0.582 | +0.636 | **+0.656** | Monotonic improvement with $K$ |
| **Precip Spread-Skill Ratio** | 0.000 | **0.519** | 0.509 | 0.438 | Underdispersive on precipitation |
| **2D Spatial Correlation** | 1.000 | 0.9356 | 0.9375 | **0.9487** | True spatial grid correlation |
| **Multivariate Energy Score** | 0.4120 | **0.2769** | 0.2782 | 0.2812 | Favors deeper steps ($K=2, S=16$) |
| **Wall Latency** | 831.6 ms | 834.3 ms | 839.5 ms | **909.0 ms** | Matched within 9% |

---

## 7. Scientific Findings from the Final Audit

1. **The Fundamental Tradeoff (Distribution Quality vs Point Recall)**:
   There is no single "universally optimal" configuration. Rather, inference compute allocation reveals a crisp Pareto frontier:
   - **Continuous Distribution Calibration**: Favors deeper trajectory denoising with small ensembles ($K=2, S=16, \eta=0$). This configuration achieves the lowest Composite CRPS (0.5927), the lowest Precipitation CRPS (2.172 mm/day), the best Multivariate Energy Score (0.2769), and the highest Spread-Skill Ratio (0.519).
   - **Deterministic Point Forecasts and Severe Storm Recall**: Favors broader ensemble averaging with shallow trajectories ($K=8, S=4, \eta=0.5$). This configuration achieves the lowest Wet-MAE (6.51 mm), highest CSI@30 (0.728), lowest Brier score (0.0188), and highest Brier Skill Score (+0.656).

2. **Resolution of Brier Skill Score Artifact**:
   Prior preliminary reports showed negative BSS@30 values due to an uncalibrated 2% placeholder reference rate. Using the authentic 2015-2021 training split empirical base rate (5.82% for $P > 30\text{ mm}$ and 11.03% for $P > 15\text{ mm}$ across 38.26 million grid points), all models demonstrate robust positive skill over climatology:
   - Deterministic 32-step DDIM: BSS@30 = +0.471.
   - Broad stochastic ensemble ($K=8, S=4, \eta=0.5$): BSS@30 = +0.656.

3. **Deterministic Diminishing Returns and Oversmoothing**:
   Single-trajectory deterministic sampling displays negative physical returns beyond 4 steps (Wet-MAE degrades from 7.20 mm at $S=4$ to 7.88 mm at $S=64$; CSI@30 drops from 0.634 to 0.547). This confirms that repeated reverse-diffusion filtering progressively strips high-frequency convective variance.

4. **Spread-Skill Dispersion Diagnostic**:
   Ensemble spread is well-calibrated for thermal and wind variables (SSR = 1.057 for Tmax, 1.170 for Tmin, 0.946 for RH, 0.811 for wind), but remains underdispersive for precipitation (SSR = 0.438 to 0.519). 90% prediction intervals cover 20.9% of observed precipitation events, underscoring that precipitation extremes require post-processing quantile recalibration.

5. **Spatial vs Global Correlation Distinctions**:
   Global flattened member correlation is 0.997, whereas true 2D spatial pattern correlation per meteorological grid slice is 0.936 to 0.949. This demonstrates that individual ensemble members generate meaningful local structural diversity across rainbands while remaining anchored to synoptic boundaries.

6. **Physical Conservation and Repair Burden**:
   Member-wise physical bounds applied prior to ensemble reduction eliminate thermodynamic violations: 0.00% diurnal temperature inversions ($T_{\min} > T_{\max}$), with an average precipitation mass shift of 7.76% to enforce non-negativity.

---

## 8. Phase 4 Confirmatory Quarantined Holdout Results (2023 Season)

Following the validation sweep, the chosen Flagship Champion (`CHAMPION_HOLDOUT_B32_K8_S4_ETA05`) was evaluated once on the quarantined 2023 holdout test set (122 samples):

* **Configuration**: $K=8$ members, $S=4$ DDIM steps, $\eta=0.50$, chunk size $C_{\text{ens}}=4$
* **Composite Fair-CRPS**: **0.6819**
* **Precipitation CRPS**: **2.0767 mm/day**
* **Holdout Wet-MAE**: **8.15 mm** (Candidate 3 deterministic baseline was 8.93 mm, yielding an 8.7% error reduction)
* **Holdout CSI@15**: **0.557**
* **Holdout CSI@30**: **0.484** (Validation CSI@30 of 0.728 drops to 0.484 on the drier 2023 drought season)
* **BSS@15 (Train Climatology)**: **+0.646** (Brier score 0.0347 vs climatology 0.0982)
* **BSS@30 (Train Climatology)**: **+0.682** (Brier score 0.0174 vs climatology 0.0548)
* **Mean Pairwise Member RMSE**: **1.523 mm/day**
* **2D Spatial Pattern Correlation**: **0.9463**
* **Multivariate Energy Score**: **0.3588**
* **Inference Latency**: **827.3 ms** per 7-day forecast cube (sequential member looping)
* **Physical Repair Burden**: 0.00% diurnal temperature violations, 0.00% RH violations.

### Key Generalization Insight
Validation recall (CSI@30 = 0.728) did NOT fully transfer to the 2023 holdout season (CSI@30 = 0.484). The 2023 season experienced severe regional monsoon deficiency in Mandya (mean precipitation dropped from 7.75 mm in 2022 to 4.53 mm in 2023, with $P > 30\text{ mm}$ frequency falling from 7.55% to 3.77%). While point recall dropped, probabilistic tail discrimination remained high (BSS@30 = +0.682).

---

## 9. Edge Deployment Guidance for Panchayat Advisory Systems

1. **Severe Weather Advisory Mode**:
   Deploy Broad Ensemble ($K=8, S=4, \eta=0.5$) where priority is detecting localized cloudbursts, severe flash flood triggers, and maximizing CSI@30 (0.728 val, 0.484 test).
2. **Probabilistic Risk & Crop Water Budgeting Mode**:
   Deploy Shallow Ensemble ($K=2, S=16, \eta=0$) where priority is well-calibrated rainfall volume distributions, minimizing CRPS (2.17 mm/day), and optimal multivariate energy score (0.2769).
3. **Low-Power Panchayat Edge Nodes**:
   Deploy Budget 8 ($K=2, S=4, \eta=0$), which delivers Fair-CRPS of 0.6045 and CSI@30 of 0.682 in just 218 ms (4.6 cubes/sec throughput).
