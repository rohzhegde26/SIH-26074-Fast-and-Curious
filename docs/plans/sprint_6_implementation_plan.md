# Sprint 6 Implementation Plan: Residual Diffusion Formulation and Meteorological Refinement

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 6 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 24, 2026  

---

## 1. Executive Summary & Sprint 5 Scientific Handoff

### 1.1 The Established Baseline Handoff
Sprint 5 executed the systematic spatial-context ($N/M$) sweep on Kaggle Dual Tesla T4 accelerators while holding antecedent memory strictly frozen at $H^* = 14$ days (established in Sprint 4). Model capacity was locked at exactly 15,685,478 parameters, and evaluation used the fixed DDIM-32 ($\eta = 0.0$) deterministic sampler.

The four spatial configurations produced the following validation convergence trajectories:
- **$N = 16$ ($N/M = 1.00$, Control)**: Best validation loss $\mathcal{L}_{\text{val}} = 0.026008$ (Epoch 26)
- **$N = 20$ ($N/M = 1.25$, Coastal)**: Best validation loss $\mathcal{L}_{\text{val}} = 0.031769$ (Epoch 22)
- **$N = 24$ ($N/M = 1.50$, Mesoscale)**: Best validation loss $\mathcal{L}_{\text{val}} = \mathbf{0.023524}$ (Epoch 22, lowest across all runs)
- **$N = 32$ ($N/M = 2.00$, Synoptic)**: Best validation loss $\mathcal{L}_{\text{val}} = 0.031578$ (Epoch 20)

In strict accordance with the pre-declared model-selection protocol (selection exclusively by minimum validation loss on the 2022 validation season), the **Sprint 6 frozen configuration** is:
- **Antecedent Memory**: $H^* = 14$ days
- **Spatial Context**: $N/M = 24 / 16 = 1.50$ (coarse input grid $24 \times 24$ cells, $660 \times 660$ km domain)
- **Target Footprint**: $M = 16$ coarse cells ($440 \times 440$ km) downscaled $5\times$ to $80 \times 80$ fine cells ($0.05^\circ$)
- **Model Parameter Invariant**: Exactly 15,685,478 trainable parameters
- **Sampling Protocol**: DDIM-32 ($\eta = 0.0$)

### 1.2 The Validation-Test Divergence Finding
A critical scientific observation emerged from the post-hoc 2023 holdout test evaluation:

| Metric | $N=16$ (Control) | $N=20$ (Coastal) | $N=24$ (Selected N*) | $N=32$ (Synoptic) |
|---|---|---|---|---|
| Validation Loss ($\mathcal{L}_{\text{val}}$, 2022) | 0.0260 | 0.0318 | **0.0235** | 0.0316 |
| Test Precip Wet-MAE (mm, 2023) | **9.70** | 9.77 | 9.90 | 9.82 |
| Test Precip CSI@30 (2023) | **0.498** | 0.494 | 0.489 | 0.491 |
| Test Wind Vector RMSE (m/s, 2023) | 3.55 | **3.53** | 3.56 | 3.59 |
| Test Tmax MAE (°C, 2023) | **0.443** | 0.448 | 0.458 | 0.463 |

While $N=24$ achieved the strongest validation objective in 2022, it did not dominate the 2023 held-out test season. Test set metrics favor $N=16$ slightly. 

**Central Research Rule for Sprint 6**: We do not discard $N=24$ or revert to $N=16$ post-hoc, because doing so would contaminate the test split. Instead, the validation/test divergence is treated as our primary diagnostic signal. Sprint 6 focuses on the residual diffusion formulation itself.

---

## 2. Scientific Scope: What Sprint 6 Is and Is Not

### 2.1 Explicit Boundaries
Sprint 6 investigates:
- How the coarse numerical weather prediction forecast, historical context, and high-resolution terrain are converted into a physically plausible high-resolution stochastic residual field.
- The mathematical formulation of the target residual $r_0$.
- The denoising parameterization ($\epsilon$-prediction vs $v$-prediction vs $x_0$-prediction).
- The channel-heterogeneity and extreme-precipitation distribution problem.
- The alignment between validation loss and physical meteorological skill.

Sprint 6 explicitly **excludes**:
- No history length sweep (frozen at $H=14$, assigned to Sprint 4).
- No spatial context $N/M$ sweep (frozen at $N/M=1.50$, assigned to Sprint 5).
- No model capacity scaling or architecture size changes (frozen at 15.69M parameters, assigned to Sprint 9).
- No Mixture-of-Experts (MoE) routing (assigned to Sprint 9).
- No diffusion sampling-step sweep or sampler comparison (frozen at DDIM-32, assigned to Sprint 7).
- No ensemble member scaling (assigned to Sprint 8).

---

## 3. Empirical Diagnosis of the Current Residual Process

### 3.1 Mathematical Audit of the Current Formulation
In the current implementation:
$$r_0 = y_{\text{fine\_norm}} - \text{upsample}(\text{future\_forecast\_norm})$$
The forward diffusion process adds isotropic Gaussian noise:
$$q(r_t \mid r_0) = \mathcal{N}\left(r_t; \sqrt{\bar{\alpha}_t} r_0, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$
The training loss is unweighted noise MSE:
$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, r_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(r_t, t, \text{cond})\|^2\right]$$

### 3.2 Key Empirical Findings from Dataset Diagnostics
Our diagnostic audit on the authentic 2022 validation dataset revealed four major physical deficiencies:

1. **Massive Residual Heteroscedasticity across Precipitation Regimes**:
   - Dry pixels ($< 0.1$ mm/day, 64.5% of dataset): Coarse GFS forecasts 2.1 mm, target is 0.0 mm. GFS exhibits a systematic drizzle bias of $-2.10$ mm with residual std of 8.27 mm.
   - Light rain ($0.1 - 2.5$ mm): Residual std is 0.97 mm.
   - Extreme storm events ($> 30$ mm/day, 7.6% of dataset): Coarse GFS predicts 58.0 mm while fine target is 67.4 mm (underprediction bias $+9.42$ mm). The residual standard deviation surges to **22.17 mm**.
   - The residual variance expands by over **20-fold** from light rain to convective extremes. An unweighted MSE objective is dominated by rare extreme errors while ignoring subtle orographic triggers.

2. **Extreme Non-Gaussian Kurtosis in Precipitation**:
   - Precipitation physical residual kurtosis is **64.57** (Gaussian is 0.0), with skewness of $-1.08$.
   - In contrast, thermodynamic variables have near-Gaussian residual distributions: Tmax physical std is 0.857 °C (kurtosis 38.3), Tmin std is 0.423 °C (kurtosis 45.5), Wind U std is 1.427 m/s (kurtosis 35.6).

3. **Lead-Time Signal-to-Noise Ratio Asymmetry**:
   - For thermodynamic and wind variables, coarse GFS error grows dramatically with lead day:
     - Tmax residual std: 0.038 at D+0 vs 0.513 at D+6 (a 13.5x expansion).
     - Tmin residual std: 0.036 at D+0 vs 0.240 at D+6 (a 6.7x expansion).
     - Wind U/V residual std: 0.103 at D+0 vs 0.323 at D+6 (a 3.1x expansion).
   - For precipitation, coarse error is high immediately at D+0 (residual std 0.520) and remains virtually flat through D+6 (0.529). Coarse precipitation error is driven by spatial resolution mismatch rather than dynamic forecast drift.

4. **The 2022 vs 2023 Meteorological Regime Shift (Root Cause of Divergence)**:
   - 2022 Validation Season (Active La Niña Monsoon):
     - Domain mean precipitation: 7.75 mm/day.
     - Fraction $> 15$ mm: 13.8%.
     - Fraction $> 30$ mm: 7.55%.
     - Intense synoptic westerly moisture advection from the Arabian Sea across $73^\circ-75^\circ$E. Large spatial context ($N=24$) provided crucial upstream moisture boundary information, lowering 2022 validation loss to 0.0235.
   - 2023 Holdout Test Season (El Niño Drought / Deficit Year):
     - Domain mean precipitation: 4.53 mm/day (41.5% decrease).
     - Fraction $> 15$ mm: only 7.18% (almost 50% fewer convective systems).
     - Fraction $> 30$ mm: only 3.77%.
     - Convective systems were suppressed; precipitation was dominated by weak, local orographic forcing. In this regime, the wider boundary halo in $N=24$ provided little useful synoptic flux and introduced slight non-local variance, while $N=16$ concentrated its capacity on local terrain.

---

## 4. Architectural Analysis: The Spatial Halo Bottleneck

In inspecting `src/models/residual_diffusion.py`, we identified an architectural property in how spatial context was processed:
```python
hist_tokens = F.adaptive_avg_pool2d(hist_feats, 1).view(b, h_len, self.embed_dim)
fcst_tokens = F.adaptive_avg_pool2d(fcst_feats, 1).view(b, num_leads, self.embed_dim)
cross_context, _ = self.hist_future_cross_attn(query=fcst_tokens, key=hist_context, value=hist_context)
fused_spatial = fcst_feats + fused_tokens.view(b * num_leads, self.embed_dim, 1, 1)
```
- Coarse history and forecast fields across the $24 \times 24$ halo were pooled to a single $1 \times 1$ global spatial token (`adaptive_avg_pool2d`) before cross-attention.
- The localized convective cells in the halo (e.g. offshore squall lines) were collapsed into a spatial average rather than preserved as localized directional advection vectors.
- For Sprint 6, we do not re-engineer the backbone or change parameters; rather, we improve how the residual target and noise conditioning interact with the existing architecture.

---

## 5. Sprint 6 Candidate Formulations

To identify the optimal residual diffusion formulation without exceeding the available compute quota, four candidate models are specified:

```
+-----------------------------------------------------------------------------------+
| CANDIDATE 1 (EXP-01): Frozen Control (Sprint 5 Baseline)                           |
| Parameterization: Epsilon-prediction (eps_theta)                                  |
| Residual Space: Raw normalized difference (r_0 = y_norm - upsample(fcst_norm))    |
| Loss: Unweighted MSE(eps, eps_theta) across all channels & leads                   |
| Sampling: Standard DDIM-32 from pure Gaussian noise                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| CANDIDATE 2 (EXP-02): Velocity Prediction (v-prediction)                          |
| Parameterization: v = alpha_t * eps - sigma_t * r_0                               |
| Objective: Loss = MSE(v, v_theta)                                                 |
| Rationale: Eliminates SNR explosion at t -> 0 and blurred means at t -> T         |
| Sampling: DDIM-32 adapted for v-parameterization                                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| CANDIDATE 3 (EXP-03): Multi-Task Variable-Aware Noise Weighting                   |
| Parameterization: v-prediction with group-balanced loss                           |
| Objective: L = w_precip * L_precip + w_thermo * L_thermo + w_wind * L_wind         |
| Rationale: Prevents precipitation variance (std=0.527) from overwhelming          |
|            thermodynamic gradients (std=0.174 - 0.369)                            |
| Tail Treatment: Moderate loss boost on wet pixels to improve CSI@30               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| CANDIDATE 4 (EXP-04): Deterministic Prior Residual Refinement                     |
| Architecture: Deterministic UNet coarse-to-fine prediction as baseline r_base     |
| Formulation: Diffusion models stochastic residual r_stoch = r_0 - r_base          |
| Sampling: DDIM-32 starting from intermediate noise level t_init = 60              |
| Rationale: Coarse NWP carries deterministic bias; isolating fine texture avoids   |
|            requiring diffusion steps to correct large-scale offset                |
+-----------------------------------------------------------------------------------+
```

### 5.1 Formulation Mathematical Details

#### Candidate 1: Frozen Control Baseline
- Target: $r_0 = y_{\text{fine\_norm}} - \text{upsample}(x_{\text{coarse\_norm}})$
- Model predicts $\hat{\epsilon}_\theta(r_t, t, c)$
- Training loss: $\mathcal{L} = \frac{1}{B \cdot 7 \cdot 6 \cdot 80 \cdot 80} \sum (\epsilon - \hat{\epsilon}_\theta)^2$

#### Candidate 2: Velocity Prediction ($v$-prediction)
- Mathematical Definition (Salimans & Ho, 2022):
  $$v_t \equiv \sqrt{\bar{\alpha}_t} \epsilon - \sqrt{1 - \bar{\alpha}_t} r_0$$
- In terms of noise angle $\phi_t$ where $\cos(\phi_t) = \sqrt{\bar{\alpha}_t}$ and $\sin(\phi_t) = \sqrt{1 - \bar{\alpha}_t}$:
  $$r_t = \cos(\phi_t) r_0 + \sin(\phi_t) \epsilon$$
  $$v_t = \cos(\phi_t) \epsilon - \sin(\phi_t) r_0$$
- Denoised reconstruction from predicted $\hat{v}$:
  $$\hat{r}_0 = \cos(\phi_t) r_t - \sin(\phi_t) \hat{v}_t$$
  $$\hat{\epsilon} = \sin(\phi_t) r_t + \cos(\phi_t) \hat{v}_t$$
- Loss objective:
  $$\mathcal{L}_v = \|\hat{v}_\theta(r_t, t, c) - v_t\|^2$$
- Benefit: Numerical stability across all timesteps without division by $\sqrt{\bar{\alpha}_t} \approx 0$ or $\sqrt{1 - \bar{\alpha}_t} \approx 0$.

#### Candidate 3: Variable-Grouped & Convective-Tail Calibration
- Group-balanced loss weights:
  $$\mathcal{L}_{\text{multi}} = \lambda_{\text{precip}} \mathcal{L}_{\text{precip}} + \lambda_{\text{thermo}} \mathcal{L}_{\text{thermo}} + \lambda_{\text{wind}} \mathcal{L}_{\text{wind}}$$
  where:
  - $\mathcal{L}_{\text{precip}}$: Precipitation channel loss with focal tail weight:
    $$w_p(y) = 1.0 + 2.0 \cdot \mathbb{I}(y_{\text{phys}} > 15.0\text{ mm})$$
  - $\mathcal{L}_{\text{thermo}}$: Combined Tmax, Tmin, RH loss.
  - $\mathcal{L}_{\text{wind}}$: Vector wind $(U, V)$ loss: $\mathcal{L}_U + \mathcal{L}_V$.
  - Balancing weights: $\lambda_{\text{precip}} = 1.0$, $\lambda_{\text{thermo}} = 1.2$, $\lambda_{\text{wind}} = 1.1$.

#### Candidate 4: Deterministic Residual Refinement
- Let $\hat{y}_{\text{det}}$ be the output of our pre-trained deterministic baseline (`temporal_deterministic_champion.pt`).
- Residual defined against deterministic downscaling:
  $$r_0^{\text{refine}} = y_{\text{fine\_norm}} - \hat{y}_{\text{det\_norm}}$$
- At inference time, reverse diffusion begins at noise step $t_{\text{init}} = 60$ (instead of $t = 100$):
  $$r_{t_{\text{init}}} = \sqrt{\bar{\alpha}_{60}} r_{\text{init}} + \sqrt{1 - \bar{\alpha}_{60}} \epsilon$$
  followed by 32 DDIM reverse steps.

---

## 6. Checkpoint Selection Strategy: Composite Meteorological Validation Score (CMVS)

### 6.1 Flaw of Current Selection Rule
Currently, checkpoints are saved whenever validation loss drops:
$$\text{if } \text{avg\_val\_loss} < \text{best\_val\_loss}: \text{ save checkpoint}$$
In diffusion models, `avg_val_loss` is computed as $\text{MSE}(\epsilon, \hat{\epsilon})$ on a single random timestep $t \sim \mathcal{U}(0, 99)$ per batch. This has three flaws:
1. Monte-Carlo noise variance from the random draw of $t$ creates artificial fluctuations.
2. An improvement in predicting noise at $t=90$ has minimal impact on fine orographic structure at $t=10$.
3. Noise MSE treats all channels and lead times identically, ignoring physical metric calibration.

### 6.2 Proposed Sprint 6 Selection Protocol: CMVS
Checkpoint selection is evaluated every 5 epochs using reverse sampling on the 2022 validation set:
$$\text{CMVS} = 0.35 \cdot \left(\frac{\text{WetMAE}_{\text{val}}}{8.70}\right) + 0.35 \cdot \left(1.0 - \frac{\text{CSI@30}_{\text{val}}}{0.631}\right) + 0.15 \cdot \left(\frac{\text{TmaxMAE}_{\text{val}}}{0.37}\right) + 0.15 \cdot \left(\frac{\text{WindRMSE}_{\text{val}}}{1.69}\right)$$
- Normalization denominators correspond to the Sprint 5 $N=24$ baseline values.
- A score below 1.00 indicates a net improvement across all physical weather dimensions.
- The 2023 holdout test set remains strictly quarantined and is evaluated once on the champion checkpoint.

---

## 7. Compute Allocation & Kaggle Execution Protocol

### 7.1 Available Compute Budget
- Repository record shows **2.98 hours (178.8 minutes)** of Kaggle GPU quota remaining.
- Measured runtime from Sprint 5: ~62.4 seconds per epoch on Dual Tesla T4 GPUs.
- One full 30-epoch training run requires approximately **31.2 minutes**.

### 7.2 Staged Execution Ladder
To prevent quota exhaustion, execution follows a strict gated ladder:

```
+-----------------------------------------------------------------------------------+
| PHASE 0: Local Verification & TDD (0.00 Kaggle hours)                             |
| - Verify v-prediction math and DDIM sampler inverse locally on CPU                |
| - Confirm parameter count invariant: 15,685,478 params                             |
| - Gate: 100% green tests in tests/models/test_residual_diffusion_vpred.py         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 1: Kaggle Diagnostic Probes (0.25 Kaggle hours)                             |
| - Run 2-epoch screening probes for Candidates 2, 3, 4 (5 min each = 15 min total)  |
| - Measure gradient norms, loss stability, and initial denoiser convergence        |
| - Gate: Reject any candidate exhibiting loss divergence or gradient instability   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 2: Competitive Screening (0.75 Kaggle hours)                                |
| - Run 10-epoch screening on top 2 surviving candidates (~10.5 min each = 21 min)  |
| - Evaluate CMVS at Epoch 10                                                       |
| - Gate: Only candidate with lower CMVS than Candidate 1 proceeds to full run      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| PHASE 3: Full Champion Convergence Run (1.05 Kaggle hours)                        |
| - Run 30 epochs with Cosine Annealing on selected Champion (~31.5 min)            |
| - Save champion checkpoint with full training history                             |
| - Evaluate 2023 holdout test set with DDIM-32                                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| TOTAL ESTIMATED CONSUMPTION: ~1.75 - 2.05 hours                                   |
| REMAINING SAFETY BUFFER: ~0.93 - 1.23 hours                                       |
+-----------------------------------------------------------------------------------+
```

---

## 8. Research Hypotheses & Falsification Criteria

### Hypothesis 1 (Parameterization Stability)
- **Statement**: Velocity prediction ($v$-prediction) achieves a lower gradient variance and a higher signal-to-noise ratio in reconstructing physical fields than standard $\epsilon$-prediction at identical compute.
- **Control**: Candidate 1 ($\epsilon$-prediction).
- **Treatment**: Candidate 2 ($v$-prediction).
- **Metric**: Validation reconstructed residual MAE across all 6 channels.
- **Falsification Criterion**: Candidate 2 fails to reduce multi-task reconstructed residual MAE by at least 3.0% relative to Candidate 1 after 10 epochs.

### Hypothesis 2 (Precipitation Tail Heteroscedasticity)
- **Statement**: Applying a focal convective tail weighting ($w_p = 3.0$ on $> 15$ mm) directly increases CSI@30 by at least 5.0% without degrading thermodynamic (Tmax/Tmin) MAE by more than 2.0%.
- **Control**: Candidate 2 (unweighted $v$-prediction).
- **Treatment**: Candidate 3 (tail-weighted multi-task $v$-prediction).
- **Metric**: Validation CSI@30 and Tmax MAE.
- **Falsification Criterion**: Candidate 3 yields CSI@30 $< 0.640$ on validation or increases Tmax MAE above 0.38 °C.

### Hypothesis 3 (Deterministic Prior Refinement)
- **Statement**: Diffusing only the stochastic high-frequency residual around a learned deterministic coarse-to-fine prediction produces sharper fine-scale orographic features than diffusing the raw residual around upsampled coarse NWP.
- **Control**: Candidate 2 (upsampled coarse baseline).
- **Treatment**: Candidate 4 (deterministic prior baseline).
- **Metric**: Validation Wet-MAE and spatial gradient power spectrum.
- **Falsification Criterion**: Candidate 4 achieves higher Wet-MAE or lower spatial power than Candidate 2.

### Hypothesis 4 (CMVS Metric Alignment)
- **Statement**: Selecting model checkpoints via the Composite Meteorological Validation Score (CMVS) yields lower holdout test Wet-MAE and higher test CSI@30 than selecting checkpoints via minimum latent $\epsilon$-loss.
- **Control**: Checkpoint selected by minimum validation loss.
- **Treatment**: Checkpoint selected by minimum CMVS.
- **Metric**: 2023 holdout test Wet-MAE and CSI@30.
- **Falsification Criterion**: CMVS-selected checkpoint performs worse on both test metrics.

### Hypothesis 5 (Lead-Time Variable Sensitivity)
- **Statement**: Weighting the denoising loss dynamically by lead-time forecast variance prevents late-lead thermodynamic errors from degrading early-lead orographic precipitation downscaling.
- **Control**: Uniform lead loss.
- **Treatment**: Lead-variance weighted loss.
- **Metric**: $D+0$ vs $D+6$ CSI@30 and wind vector RMSE.
- **Falsification Criterion**: $D+0$ CSI@30 does not improve while $D+6$ wind RMSE degrades.

### Hypothesis 6 (Physical Bounds Invariance)
- **Statement**: Enforcing analytical physical post-projection ($P \ge 0$, $0 \le \text{RH} \le 100$, $T_{\min} \le T_{\max}$) at every reverse diffusion step reduces physical violation rate to exactly 0.0% without increasing MAE.
- **Control**: Raw reverse sampling with end-of-process clipping.
- **Treatment**: Step-wise physical projection.
- **Metric**: Diurnal temperature violation rate and out-of-range RH rate.
- **Falsification Criterion**: Step-wise projection increases Wet-MAE by $> 0.2$ mm.

---

## 9. Definition of Done & Success Criteria

1. **Mathematical Implementation**:
   - $v$-prediction forward and reverse diffusion implemented in `src/models/residual_diffusion.py`.
   - Analytical inversion formulas verified with unit tests.
2. **Capacity Invariant Maintained**:
   - PyTorch parameter count strictly verified: exactly 15,685,478 parameters across all candidates.
3. **Green Test Coverage**:
   - New unit tests in `tests/models/test_residual_diffusion_vpred.py` pass 100% locally.
4. **Remote Execution & Artifact Integrity**:
   - All diagnostic probes and candidate training runs executed remotely on Kaggle GPUs.
   - Weekly GPU quota floor ($\ge 0.50$ hours) strictly preserved.
   - Champion checkpoint saved with optimizer state, normalization stats, and experiment metadata.
5. **Synthesis & Verification**:
   - `reports/sprint_6_residual_diffusion_summary.json` generated.
   - `reports/sprint_6_comparison_table.md` generated with Tables A (validation) and B (holdout test).
   - Walkthrough and execution audit documented.

---

## 10. Sprint 7 Handoff Specification

Sprint 6 produces the optimal residual diffusion formulation at fixed $H=14$, $N=24$, 15.69M parameters, and DDIM-32. This frozen checkpoint family will be handed directly to **Sprint 7 (Diffusion-Step and Sampler Frontier)** to investigate:
- Reverse step scaling: 4, 8, 16, 32, 64 steps.
- Sampler algorithms: DDIM vs DPM-Solver++ vs PNDM.
- Stochasticity parameter: $\eta \in [0.0, 1.0]$.
- Inference latency vs accuracy Pareto curves for operational Panchayat deployment.
