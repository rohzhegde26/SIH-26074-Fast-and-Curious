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
