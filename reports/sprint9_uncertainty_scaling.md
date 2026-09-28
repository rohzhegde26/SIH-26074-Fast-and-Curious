# Sprint 9 Predictive Uncertainty and Calibration Scaling Report

## 1. Executive Summary
This report investigates the impact of neural capacity scaling on predictive uncertainty representation, ensemble spread calibration, and lead-time dispersion across the Sprint 9 ladder:
- Dense-S: 15.69M parameters (Candidate 3 control)
- Dense-M: 31.20M parameters (1.99x scale)
- Dense-L: 52.00M parameters (3.31x scale)
- MoE-4: 35.77M total parameters (15.69M active parameters)

## 2. Core Uncertainty Metrics
1. **Spread-Skill Ratio (SSR)**:
   $$\text{SSR} = \frac{\mathbb{E}[\sigma_{\text{ensemble}}]}{\text{RMSE}(\bar{x}_{\text{ensemble}}, y)}$$
   An ideal ensemble achieves $\text{SSR} = 1.0$. Values $< 1.0$ indicate under-dispersion.
2. **Empirical Prediction Interval Coverage (Cov@90)**:
   $$\text{Cov}_{90} = \frac{1}{N_{\text{pts}}} \sum_{i} \mathbb{I}(y_i \in [q_{0.05}, q_{0.95}])$$
3. **Continuous Ranked Probability Score (Fair-CRPS)**:
   $$\text{CRPS}_{\text{Fair}} = \mathbb{E}[|x - y|] - \frac{1}{2(K-1)} \sum_{k=1}^K \sum_{m=1}^K |x_k - x_m|$$
4. **Calibration Spread Multiplier Requirement ($\alpha^*$)**:
   The post-hoc rescaling factor $\alpha$ required to achieve calibrated interval coverage without distorting physical rainfall totals.

## 3. Probabilistic Uncertainty Frontier Across Capacity Tiers

| Model Tier | Base Ch | Status | Raw SSR | Raw Cov@90 | Selected $\alpha^*$ | Calibrated SSR | Calibrated Cov@90 | Precip Fair-CRPS |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 96 | **VALIDATED_BASELINE** | 0.284 | 0.178 | 3.0 | 0.582 | 0.284 | 1.842 |
| **Dense-M** | 136 | TARGET_PENDING_KAGGLE | Target > 0.350 | Target > 0.220 | Target <= 2.0 | Target > 0.700 | Target > 0.350 | Target < 1.650 |
| **Dense-L** | 176 | TARGET_PENDING_KAGGLE | Target > 0.450 | Target > 0.300 | Target <= 1.5 | Target > 0.800 | Target > 0.450 | Target < 1.500 |
| **MoE-4** | 96 | TARGET_PENDING_KAGGLE | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M |

## 4. Lead-Time Uncertainty Evolution (D+0 to D+6)
In Candidate 3 (Dense-S), ensemble spread suffers progressive decay over lead days, dropping from SSR 0.35 at D+0 down to 0.19 at D+6.
Larger capacity models are targeted to provide more resilient uncertainty retention across longer horizons:

| Lead Day | Dense-S (Validated) | Dense-M (Target) | Dense-L (Target) | MoE-4 (Target) |
| :--- | :---: | :---: | :---: | :---: |
| **D+0** | 0.348 | Target > 0.420 | Target > 0.520 | Match Dense-M |
| **D+1** | 0.322 | Target > 0.400 | Target > 0.500 | Match Dense-M |
| **D+2** | 0.301 | Target > 0.380 | Target > 0.480 | Match Dense-M |
| **D+3** | 0.278 | Target > 0.360 | Target > 0.460 | Match Dense-M |
| **D+4** | 0.245 | Target > 0.340 | Target > 0.440 | Match Dense-M |
| **D+5** | 0.218 | Target > 0.320 | Target > 0.420 | Match Dense-M |
| **D+6** | 0.192 | Target > 0.300 | Target > 0.400 | Match Dense-M |

## 5. Architectural Hypotheses
1. **Capacity Directly Reduces Calibration Burden**: Dense-S requires an aggressive post-hoc spread multiplier $\alpha^* = 3.0$ to compensate for severe under-dispersion. Dense-L is hypothesized to reduce this requirement to $\alpha^* \le 1.5$ while lifting calibrated SSR to > 0.800.
2. **Fair-CRPS Improvement**: Continuous Ranked Probability Score is hypothesized to improve by > 18% in Dense-L (< 1.500 mm vs 1.842 mm in Dense-S).
3. **Horizon Stability**: At D+6, Dense-L is targeted to retain an SSR of > 0.400, more than doubling Candidate 3's terminal spread-skill ratio (0.192).
4. **Validation Grounding**: Empirical results will be updated following execution of `sprint_9_phase1_dense_capacity_scaling.ipynb` on Kaggle.
