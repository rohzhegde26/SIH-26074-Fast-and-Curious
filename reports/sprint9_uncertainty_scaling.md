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

## 3. Empirical Calibration Scaling Results Across Capacity Tiers

| Model Tier | Base Ch | Raw SSR | Raw Cov@90 | Selected $\alpha^*$ | Calibrated SSR | Calibrated Cov@90 | Precip CRPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 96 | 0.284 | 0.178 | 3.0 | 0.582 | 0.284 | 1.842 |
| **Dense-M** | 136 | 0.395 | 0.245 | 2.0 | 0.714 | 0.388 | 1.625 |
| **Dense-L** | 176 | 0.512 | 0.320 | 1.5 | 0.825 | 0.495 | 1.480 |
| **MoE-4** | 96 | 0.382 | 0.238 | 2.0 | 0.698 | 0.375 | 1.648 |

## 4. Lead-Time Uncertainty Evolution (D+0 to D+6)
In Candidate 3 (Dense-S), ensemble spread suffers progressive decay over lead days, dropping from SSR 0.35 at D+0 down to 0.19 at D+6.
Larger capacity models provide more resilient uncertainty retention across longer horizons:

| Lead Day | Dense-S SSR | Dense-M SSR | Dense-L SSR | MoE-4 SSR |
| :--- | :---: | :---: | :---: | :---: |
| **D+0** | 0.348 | 0.445 | 0.562 | 0.432 |
| **D+1** | 0.322 | 0.428 | 0.548 | 0.418 |
| **D+2** | 0.301 | 0.412 | 0.531 | 0.401 |
| **D+3** | 0.278 | 0.395 | 0.512 | 0.384 |
| **D+4** | 0.245 | 0.378 | 0.495 | 0.366 |
| **D+5** | 0.218 | 0.356 | 0.478 | 0.345 |
| **D+6** | 0.192 | 0.334 | 0.458 | 0.328 |

## 5. Key Findings
1. **Capacity Directly Reduces Calibration Burden**: Dense-S requires an aggressive post-hoc spread multiplier $\alpha^* = 3.0$ to compensate for severe under-dispersion. Dense-L reduces this requirement to $\alpha^* = 1.5$ while lifting calibrated SSR to 0.825.
2. **Fair-CRPS Improvement**: Precipitation Fair-CRPS improves from 1.842 mm (Dense-S) to 1.480 mm (Dense-L), representing a 19.7% error reduction across continuous probabilistic forecasts.
3. **Horizon Stability**: At D+6, Dense-L retains an SSR of 0.458, more than doubling Candidate 3's terminal spread-skill ratio (0.192).
