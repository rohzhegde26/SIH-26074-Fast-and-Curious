# Sprint 8 Walkthrough: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: Smart India Hackathon (SIH) 2026 - Problem Statement 26074  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Date**: September 27, 2026  

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

## 2. Key Methodological Invariants

1. **Frozen Weights**: Zero network retraining. All experiments evaluate the Sprint 6 Candidate 3 champion weights (`models/checkpoints/sprint6_candidate3_multitask_champion.pt`), locked at 15,685,478 parameters.
2. **Decoupled Stochasticity**: Initial-noise diversity ($\eta = 0$) is evaluated separately from intermediate trajectory noise injection ($\eta \in \{0.25, 0.5, 1.0\}$) using paired, nested seed manifests.
3. **Member-Wise Inversion Invariant**: Predictions are converted to physical units and non-linear physical bounds ($P \ge 0$, $0 \le \text{RH} \le 100\%$, $T_{\min} \le T_{\max}$) applied member-by-member before ensemble reduction.
4. **Fair Probabilistic Scoring**: Unbiased Fair-CRPS and Brier Skill Score relative to sample climatology.
5. **Statistical Rigor**: 7-lead cube-preserving paired bootstrap confidence intervals on 2022 validation; 2023 holdout evaluated exactly once.
6. **Compute Platform**: Scheduled across 6.0 hours of 2x Tesla T4 Kaggle student-tier GPU accelerators.

---

## 3. Matched-Compute Experimental Grid

| Budget ($B$) | Member Count ($K$) | Denoising Steps ($S$) | Primary Evaluation Metric |
|---|---|---|---|
| **8 NFE** | $K=1, S=8$ vs $K=2, S=4$ | 8 | Fair-CRPS, Wet-MAE, BSS@15 |
| **16 NFE** | $K=1, S=16$ vs $K=2, S=8$ vs $K=4, S=4$ | 16 | Fair-CRPS, Wet-MAE, BSS@15 |
| **32 NFE (Flagship)** | $K=1, S=32$ vs $K=2, S=16$ vs $K=4, S=8$ vs $K=8, S=4$ | 32 | Fair-CRPS, BSS@30, SSR, Reliability |
| **64 NFE** | $K=1, S=64$ vs $K=2, S=32$ vs $K=4, S=16$ vs $K=8, S=8$ vs $K=16, S=4$ | 64 | Fair-CRPS, BSS@30, Saturation Curve |

---

## 4. Key Documentation References

- **Implementation Plan**: [docs/plans/sprint_8_implementation_plan.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/docs/plans/sprint_8_implementation_plan.md)
- **Model Training and Evaluation Audit**: [docs/sprint_8_model_training_audit.md](file:///c:/Users/rohit/.gemini/antigravity/playground/SIH/docs/sprint_8_model_training_audit.md)
