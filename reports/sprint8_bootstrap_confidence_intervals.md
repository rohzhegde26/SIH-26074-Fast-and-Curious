# Sprint 8 Case-Level Paired Bootstrap Statistical Rigor Report

## 1. Statistical Rigor Protocol

In accordance with the Sprint 8 protocol, all primary verification metrics are evaluated via case-level paired bootstrap resampling:
- **Resampling Unit**: 7-day forecast cube (122 independent validation cubes, 122 holdout test cubes).
- **Replicates**: $B = 1,000$ paired resamples with replacement.
- **Pairing Invariant**: Competitor configurations are evaluated on the exact same resampled dates per replicate.
- **Interval Estimator**: 95% empirical percentile intervals $[q_{0.025}, q_{0.975}]$.
- **Hypothesis Testing**: Empirical two-sided p-values derived from paired difference bootstrap distributions ($H_0: \Delta = 0$).

## 2. Primary 32-NFE Metrics with 95% Bootstrap Confidence Intervals

| Configuration | Fair-CRPS ↓ | Precip CRPS (mm/day) ↓ | Wet-MAE (mm) ↓ | CSI@30 ↑ | Multivariate Energy Score ↓ |
|---|---|---|---|---|---|
| `B32_K2_S16` | 0.5401 [0.4498, 0.6444] | 2.137 [1.903, 2.371] | 7.13 [6.53, 7.76] | 0.708 [0.694, 0.722] | 0.2391 [0.1797, 0.3174] |
| `B32_K4_S8` | 0.5433 [0.4533, 0.6480] | 2.152 [1.919, 2.385] | 6.73 [6.14, 7.34] | 0.723 [0.709, 0.736] | 0.2405 [0.1813, 0.3184] |
| `B32_K8_S4` | 0.5524 [0.4623, 0.6560] | 2.191 [1.952, 2.428] | 6.53 [5.95, 7.13] | 0.730 [0.717, 0.743] | 0.2439 [0.1848, 0.3217] |
| `B32_K8_S4_ETA05` | 0.5504 [0.4602, 0.6546] | 2.184 [1.946, 2.420] | 6.52 [5.93, 7.11] | 0.732 [0.718, 0.745] | 0.2435 [0.1843, 0.3214] |
| `2023 Holdout (K=8, S=4, eta=0.5)` | 0.6371 [0.4344, 0.9065] | 2.081 [1.643, 2.638] | 8.00 [7.16, 8.87] | 0.581 [0.501, 0.645] | 0.3160 [0.2003, 0.4515] |

## 3. Paired Differences and Hypothesis Testing

Differences are computed case-by-case on identical bootstrap resamples ($\Delta = A - B$). A confidence interval strictly excluding zero indicates statistical significance at $\alpha = 0.05$.

| Comparison | Metric | Delta Point Estimate | 95% Bootstrap CI | p-value | Significant (p < 0.05) |
|---|---|---|---|---|---|
| Broad Stochastic (K=8, S=4, eta=0.5) minus Deep Low-Member (K=2, S=16) | **Wet-MAE (mm)** | -0.6114 | [-0.6424, -0.5800] | 0.0000 | Yes |
| Broad Stochastic (K=8, S=4, eta=0.5) minus Deep Low-Member (K=2, S=16) | **CSI@30** | +0.0235 | [+0.0221, +0.0251] | 0.0000 | Yes |
| Broad Stochastic (K=8, S=4, eta=0.5) minus Deep Low-Member (K=2, S=16) | **Fair-CRPS** | +0.0103 | [+0.0090, +0.0117] | 0.0000 | Yes |
| Broad Stochastic (K=8, S=4, eta=0.5) minus Deep Low-Member (K=2, S=16) | **Precip CRPS (mm/day)** | +0.0473 | [+0.0402, +0.0545] | 0.0000 | Yes |
| Broad Stochastic (K=8, S=4, eta=0.5) minus Deep Low-Member (K=2, S=16) | **Energy Score** | +0.0044 | [+0.0039, +0.0049] | 0.0000 | Yes |
| Broad Deterministic (K=8, S=4, eta=0) minus Deep Low-Member (K=2, S=16) | **Wet-MAE (mm)** | -0.5949 | [-0.6245, -0.5644] | 0.0000 | Yes |
| Broad Deterministic (K=8, S=4, eta=0) minus Deep Low-Member (K=2, S=16) | **CSI@30** | +0.0216 | [+0.0201, +0.0233] | 0.0000 | Yes |
| Broad Deterministic (K=8, S=4, eta=0) minus Deep Low-Member (K=2, S=16) | **Fair-CRPS** | +0.0123 | [+0.0109, +0.0137] | 0.0000 | Yes |
| Broad Deterministic (K=8, S=4, eta=0) minus Deep Low-Member (K=2, S=16) | **Precip CRPS (mm/day)** | +0.0540 | [+0.0466, +0.0612] | 0.0000 | Yes |
| Broad Deterministic (K=8, S=4, eta=0) minus Deep Low-Member (K=2, S=16) | **Energy Score** | +0.0048 | [+0.0043, +0.0053] | 0.0000 | Yes |
| Balanced (K=4, S=8) minus Deep Low-Member (K=2, S=16) | **Wet-MAE (mm)** | -0.3954 | [-0.4158, -0.3737] | 0.0000 | Yes |
| Balanced (K=4, S=8) minus Deep Low-Member (K=2, S=16) | **CSI@30** | +0.0145 | [+0.0132, +0.0159] | 0.0000 | Yes |
| Balanced (K=4, S=8) minus Deep Low-Member (K=2, S=16) | **Fair-CRPS** | +0.0032 | [+0.0026, +0.0039] | 0.0000 | Yes |
| Balanced (K=4, S=8) minus Deep Low-Member (K=2, S=16) | **Precip CRPS (mm/day)** | +0.0149 | [+0.0110, +0.0186] | 0.0000 | Yes |
| Balanced (K=4, S=8) minus Deep Low-Member (K=2, S=16) | **Energy Score** | +0.0013 | [+0.0010, +0.0017] | 0.0000 | Yes |
| Stochasticity Impact: eta=0.5 minus eta=0.0 on (K=8, S=4) | **Wet-MAE (mm)** | -0.0164 | [-0.0196, -0.0131] | 0.0000 | Yes |
| Stochasticity Impact: eta=0.5 minus eta=0.0 on (K=8, S=4) | **CSI@30** | +0.0019 | [+0.0015, +0.0023] | 0.0000 | Yes |
| Stochasticity Impact: eta=0.5 minus eta=0.0 on (K=8, S=4) | **Fair-CRPS** | -0.0020 | [-0.0025, -0.0014] | 0.0000 | Yes |
| Stochasticity Impact: eta=0.5 minus eta=0.0 on (K=8, S=4) | **Precip CRPS (mm/day)** | -0.0067 | [-0.0087, -0.0046] | 0.0000 | Yes |
| Stochasticity Impact: eta=0.5 minus eta=0.0 on (K=8, S=4) | **Energy Score** | -0.0004 | [-0.0005, -0.0002] | 0.0000 | Yes |

## 4. Key Scientific Inferences from the Bootstrap Evidence

1. **Statistical Significance of Point Accuracy Gain**: Broad ensemble averaging ($K=8, S=4, \eta=0.5$) reduces Wet-MAE relative to deep sampling ($K=2, S=16$). The paired bootstrap distribution confirms whether this point improvement is statistically significant under the finite 122-cube sample.
2. **Distribution Calibration vs Point Error Divergence**: Deeper sampling ($K=2, S=16$) improves continuous probabilistic metrics (lower Fair-CRPS, lower Precipitation CRPS, lower Energy Score). The paired confidence intervals confirm that the deep-vs-broad trade-off is genuine rather than sample noise.
3. **Impact of Stochastic Perturbations ($\eta=0.5$ vs $\eta=0.0$)**: The stochastic trajectory adds high-frequency variance that enhances heavy-storm recall (CSI@30) while preserving composite calibration.
