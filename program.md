# AutoResearch Master Playbook: SIH 26074 Weather Downscaling

## Active Compute Budget & Live Kaggle GPU Quota
> **LIVE STATUS (Updated 2026-09-10 20:32:39):**
> - **Total GPU Allowance:** 6.0 hours (21600 s)
> - **GPU Time Consumed:** 0.41 hours (1488 s)
> - **GPU TIME REMAINING:** **5.59 hours (335.2 minutes)**
> - **Quota Refreshes At:** 2026-09-12T00:00:00.000Z
>
> ⚠️ **HARD CONSTRAINT FOR AGENTS:**
> You have AT MOST **5.59 hours** of GPU time remaining for this cycle.
> Every training run depletes this quota. Budget your experiments carefully. If you run 5-minute runs,
> you have at most ~67 candidate iterations left before quota exhaustion!
## 1. Problem Statement & Objective
* **Target:** Downscale coarse IMD / NCUM Numerical Weather Prediction (NWP) precipitation forecasts from **Block level ($0.25^\circ \approx 27\text{ km}$, $16\times 16$ grid)** to **Panchayat level ($0.05^\circ \approx 5.5\text{ km}$, $80\times 80$ grid)**.
* **Pilot:** Mandya District, Karnataka (234–258 Gram Panchayats).
* **End Goal:** High-resolution precipitation, temperature, relative humidity, and wind fields for localized agro-meteorological advisories (Paddy, Ragi, Sugarcane).

---

## 2. Co-Evolutionary Agent Roles
* **Agent A (The Explorer / Researcher):**
  * Operates with full creative autonomy on `train.py`.
  * Free to redefine the architecture, loss functions, spatial representations, or downscaling formulation.
  * Must respect the compute budget and memory constraints.
* **Agent B (The Auditor / Red-Teamer):**
  * Operates independently to stress-test candidate models against physical realism, texture smoothing, and extreme weather failures.
  * Mines the 14-year national archive for challenging convective storm scenarios.
  * Formulates plain-language feedback and failure modes back to Agent A.

---

## 3. Core Conceptual Failure Archetypes to Explore & Probe
1. **Physical Conservation & Balance:**
   * Atmospheric water volume must balance across spatial scales ($0.000\%$ discrepancy). Can this be guaranteed by construction through differentiable projection or normalization?
2. **Spatial Texture & Sharpness vs. Blurry Averages:**
   * Standard loss functions often lead to blurry conditional means. Convective storms ($>30\text{ mm}$) must retain realistic high-frequency spatial texture.
3. **Topographical Coupling (Western Ghats & Rain Shadows):**
   * Terrain elevation, slope, aspect, and windward moisture convergence dictate orographic precipitation. The model must not decouple topography in favor of flat plains.
4. **Extreme Cloudburst Detection & Agricultural Decisions:**
   * Drainage and harvesting decisions require high recall on extreme convective events, not just low error on frequent dry/drizzle days.

---

## 4. Execution Sandbox
* **Remote Accelerator:** Kaggle Cloud GPU (Tesla T4, 16GB VRAM, AMP FP16 enabled).
* **Memory Ceiling:** Micro-batches must maintain peak memory $< 2.5\text{ GB}$ to guarantee safety.
* **Quota Tracking:** `scripts/kaggle/dispatch_kaggle.py` queries live quota after each run and updates this document.

- **Cycle 1 Rejected:** Baseline Model Characterization (Mass conservation breach (42.04%)).

- **Cycle 2 Rejected:** Differentiable Mass-Conserving Head (Blurry texture collapse (HF ratio=0.007)).

- **Cycle 1 Rejected:** Baseline Model Characterization (Mass conservation breach (59.53%)).

- **Cycle 1 Rejected:** Baseline Model Characterization (Mass conservation breach (32.40%)).

- **Cycle 4 Rejected:** High-Frequency Laplacian Sharpness Loss (Sub-optimal score).

- **Cycle 5 Rejected:** Terrain Spatial Cross-Attention (Sub-optimal score).

- **Cycle 6 Rejected:** Focal Convective Tail Weighting (Sub-optimal score).

- **Cycle 7 Rejected:** Two-Stage Hurdle Probability Gate (Sub-optimal score).

- **Cycle 8 Rejected:** High Learning Rate Exploration (Sub-optimal score).

- **Cycle 9 Rejected:** Over-Smoothed Regularization Probe (Mass conservation breach (58.00%)).

- **Cycle 10 Rejected:** Cosine Annealing with Warmup (Sub-optimal score).

- **Cycle 11 Rejected:** Extreme Gradient Clipping Test (Sub-optimal score).

- **Cycle 12 Rejected:** Pareto Unified Champion Architecture (Sub-optimal score).

- **Cycle 13 Rejected:** Directional Windward Orographic Lifting Scaling (Sub-optimal score).

- **Cycle 14 Rejected:** Aggressive Sharpness-Regularized Loss (Sub-optimal score).

- **Cycle 15 Rejected:** Final Multi-Objective Consolidated Model (Sub-optimal score).

- **Round 2 Cycle 2 Rejected:** Dynamic Orographic FiLM Modulation (Sub-optimal score).

- **Round 2 Cycle 3 Rejected:** Multi-Scale Atrous / Dilated ConvNeXt (Sub-optimal score).

- **Round 2 Cycle 4 Rejected:** 2D FFT Fourier Spectral Regularization (Sub-optimal score).

- **Round 2 Cycle 7 Rejected:** Orographic Soft Hurdle Masking (Sub-optimal score).

- **Round 2 Cycle 8 Rejected:** Multi-Octave Laplacian Pyramid Loss (Sub-optimal score).

- **Round 2 Cycle 9 Rejected:** Agent B Adversarial Stress Probe (Sub-optimal score).

- **Round 2 Cycle 10 Rejected:** Cosine Annealing with Warm Restarts (SGDR) (Sub-optimal score).

- **Round 2 Cycle 11 Rejected:** Thermodynamic Elevation Lapse Rate Prior (Sub-optimal score).

- **Round 2 Cycle 12 Rejected:** Round 2 Consolidated Super-Champion (Sub-optimal score).

- **Round 2 Cycle 13 Rejected:** Fine-Grained Basin Optimization Refinement (Sub-optimal score).

- **Round 2 Cycle 14 Rejected:** Balanced Multi-Objective Regularization (Sub-optimal score).

- **Round 2 Cycle 15 Rejected:** Final Calibrated Ensemble Checkpoint (Sub-optimal score).
- **Round 3 Cycle 2 Rejected:** Extreme Tail Pinball Loss (tau=0.92) (Sub-optimal score).

- **Round 3 Cycle 3 Rejected:** Storm-Core Spatial Focal Loss Mask (Sub-optimal score).

- **Round 3 Cycle 5 Rejected:** Dual-Stream Stratiform-Convective Head (Sub-optimal score).

- **Round 3 Cycle 6 Rejected:** Extreme Convective Hurdle Trigger (>30mm) (Sub-optimal score).

- **Round 3 Cycle 7 Rejected:** Topographic Gradient Skip Routing (Sub-optimal score).

- **Round 3 Cycle 8 Rejected:** Curvature + Extreme Hurdle Synthesis (Sub-optimal score).

- **Round 3 Cycle 9 Rejected:** Agent B Extreme Flood Stress Probe (Sub-optimal score).

- **Round 3 Cycle 10 Rejected:** Cyclic Cosine Annealing with Warm Restarts (SGDR) (Sub-optimal score).

- **Round 3 Cycle 11 Rejected:** High-Resolution Terrain Skip Refinement (Sub-optimal score).

- **Round 3 Cycle 12 Rejected:** Round 3 Consolidated Cloudburst Super-Champion (Sub-optimal score).

- **Round 3 Cycle 13 Rejected:** Fine-Grained Learning Rate Refinement (Sub-optimal score).

- **Round 3 Cycle 15 Rejected:** Top-K Pareto Ensemble Checkpoint (Sub-optimal score).

- **Round 4 Cycle 1 Rejected:** Round 3 Champion Calibrated Baseline (Sub-optimal score).

- **Round 4 Cycle 2 Rejected:** 2D Haar Wavelet Decomposition Head (Sub-optimal score).

- **Round 4 Cycle 3 Rejected:** Directional Squall-Line Wavelet Tuning (Sub-optimal score).

- **Round 4 Cycle 4 Rejected:** Multivariate Temperature-Moisture Coupling (Mass conservation breach (22.24%)).

- **Round 4 Cycle 5 Rejected:** Relative Humidity Saturation Prior (Mass conservation breach (21.22%)).

- **Round 4 Cycle 6 Rejected:** Calibrated Multi-Quantile P10/P50/P90 Head (Mass conservation breach (25.49%)).

- **Round 4 Cycle 7 Rejected:** Asymmetric Cloudburst Quantile Boosting (tau=0.95) (Mass conservation breach (100.00%)).

- **Round 4 Cycle 8 Rejected:** Froude Number Flow Regime Gating (Sub-optimal score).

- **Round 4 Cycle 9 Rejected:** Agent B Over-Smoothing Stress Probe (Sub-optimal score).

- **Round 4 Cycle 10 Rejected:** Cosine Annealing with Warm Restarts (SGDR) (Mass conservation breach (28.98%)).

- **Round 4 Cycle 11 Rejected:** Wavelet + Froude Flow Regime Synthesis (Sub-optimal score).

- **Round 4 Cycle 12 Rejected:** Multivariate Wind-Shear Flux Module (Mass conservation breach (15.22%)).

- **Round 4 Cycle 13 Rejected:** Monotonic Quantile Sorting Projection (Mass conservation breach (23.53%)).

- **Round 4 Cycle 14 Rejected:** Balanced Multi-Pillar Regularization (Sub-optimal score).

- **Round 4 Cycle 15 Rejected:** Multi-Scale Atrous Bottleneck Fusion (Sub-optimal score).

- **Round 4 Cycle 16 Rejected:** Orographic Blocking Barrier Dynamics (Sub-optimal score).

- **Round 4 Cycle 17 Rejected:** Agent B Gradient Noise Injection Probe (Sub-optimal score).

- **Round 4 Cycle 18 Rejected:** Stochastic Weight Averaging (SWA) Explorer (Sub-optimal score).

- **Round 4 Cycle 19 Rejected:** Sub-Band Attention Wavelet Gating (Sub-optimal score).

- **Round 4 Cycle 20 Rejected:** Focal Convective Upper Quantile Loss (Sub-optimal score).

- **Round 4 Cycle 21 Rejected:** Clausius-Clapeyron Moisture Limit (Sub-optimal score).

- **Round 4 Cycle 22 Rejected:** Multi-Scale Energy Conservation Check (Mass conservation breach (16.39%)).

- **Round 4 Cycle 23 Rejected:** Squeeze-and-Excitation Topographic Attention (Sub-optimal score).

- **Round 4 Cycle 24 Rejected:** Katabatic Valley Drainage Flow Prior (Sub-optimal score).

- **Round 4 Cycle 25 Rejected:** Lookahead Optimization Trajectory (Mass conservation breach (32.34%)).

- **Round 4 Cycle 26 Rejected:** Agent B Anti-Topographic Inversion Probe (Mass conservation breach (26.97%)).

- **Round 4 Cycle 27 Rejected:** Multi-Level Wavelet Decomposition (Sub-optimal score).

- **Round 4 Cycle 28 Rejected:** Extreme Upper Quantile Sharpening (Sub-optimal score).

- **Round 4 Cycle 29 Rejected:** Residual Dense Topographic Aggregation (Sub-optimal score).

- **Round 4 Cycle 30 Rejected:** Round 4 Consolidated Super-Champion (Sub-optimal score).

- **Round 4 Cycle 31 Rejected:** Conservative Basin Fine-Tuning (Sub-optimal score).

- **Round 4 Cycle 32 Rejected:** Pareto Multi-Objective Loss Calibration (Sub-optimal score).

- **Round 4 Cycle 34 Rejected:** Top-3 Checkpoint Weight Averaging (Sub-optimal score).

- **Round 4 Cycle 35 Rejected:** Calibrated Production Ensemble Checkpoint (Sub-optimal score).

- **Round 5 Cycle 1 Rejected:** Round 4 Champion Calibrated Baseline (Sub-optimal score).

- **Round 5 Cycle 2 Rejected:** Wavelet-Guided Convective Attention (W-GCA) (Sub-optimal score).

- **Round 5 Cycle 3 Rejected:** W-GCA High-Pass Directional Tuning (Sub-optimal score).

- **Round 5 Cycle 4 Rejected:** Gradient-Isolated Multivariate Co-Downscaling (Mass conservation breach (20.92%)).

- **Round 5 Cycle 5 Rejected:** Isolated Multivariate Saturation Deficit (Mass conservation breach (99.61%)).

- **Round 5 Cycle 6 Rejected:** Exact-Conserved Multi-Quantile P10/P50/P90 Head (Sub-optimal score).

- **Round 5 Cycle 7 Rejected:** Conserved Quantile Asymmetric Tail Sharpening (Mass conservation breach (34.73%)).

- **Round 5 Cycle 8 Rejected:** Cloudburst Vorticity & Streamline Dynamics (Mass conservation breach (16.26%)).

- **Round 5 Cycle 9 Rejected:** Agent B Extreme Inversion Stress Probe (Sub-optimal score).

- **Round 5 Cycle 10 Rejected:** Cyclic Cosine Annealing with Warm Restarts (SGDR) (Sub-optimal score).

- **Round 5 Cycle 11 Rejected:** W-GCA + Cloudburst Vorticity Synthesis (Sub-optimal score).

- **Round 5 Cycle 12 Rejected:** Isolated Multivariate Wind-Shear Flux (Mass conservation breach (22.19%)).

- **Round 5 Cycle 13 Rejected:** Strict Monotonic Quantile Projection (Mass conservation breach (25.27%)).

- **Round 5 Cycle 14 Rejected:** Harmonized Spectral-Quantile-Vorticity Loss (Sub-optimal score).

- **Round 5 Cycle 15 Rejected:** Multi-Scale Atrous Convective Fusion (Mass conservation breach (21.64%)).

- **Round 5 Cycle 16 Rejected:** Orographic Stagnation Barrier Dynamics (Sub-optimal score).

- **Round 5 Cycle 17 Rejected:** Agent B Gradient Noise Injection Probe (Sub-optimal score).

- **Round 5 Cycle 18 Rejected:** Stochastic Weight Averaging (SWA) Explorer (Mass conservation breach (34.39%)).

- **Round 5 Cycle 19 Rejected:** Sub-Band Wavelet Energy Weighting (Sub-optimal score).

- **Round 5 Cycle 20 Rejected:** Focal Convective Upper Quantile Loss (Mass conservation breach (27.26%)).

- **Round 5 Cycle 21 Rejected:** Isolated Clausius-Clapeyron Moisture Limit (Sub-optimal score).

- **Round 5 Cycle 22 Rejected:** Multi-Scale Energy Conservation at 5x (Mass conservation breach (17.08%)).

- **Round 5 Cycle 23 Rejected:** Squeeze-and-Excitation Topographic Attention (Mass conservation breach (25.70%)).

- **Round 5 Cycle 24 Rejected:** Katabatic Valley Drainage Flow Prior (Mass conservation breach (19.79%)).

- **Round 5 Cycle 25 Rejected:** Lookahead Optimization Trajectory (Sub-optimal score).

- **Round 5 Cycle 26 Rejected:** Agent B Anti-Topographic Inversion Probe (Mass conservation breach (58.26%)).

- **Round 5 Cycle 27 Rejected:** Multi-Level Wavelet Decomposition Level 2 (Sub-optimal score).

- **Round 5 Cycle 28 Rejected:** Conserved Extreme Upper Quantile Sharpening (Mass conservation breach (100.00%)).

- **Round 5 Cycle 29 Rejected:** Residual Dense Topographic Aggregation (Sub-optimal score).

- **Round 5 Cycle 30 Rejected:** W-GCA + Conserved Quantiles + Vorticity Super-Champion (Mass conservation breach (23.24%)).

- **Round 5 Cycle 32 Rejected:** Pareto Multi-Objective Loss Calibration (Sub-optimal score).

- **Round 5 Cycle 33 Rejected:** Single-Period Cosine Restart Basin Deepening (Sub-optimal score).

- **Round 5 Cycle 34 Rejected:** Top-3 Checkpoint Weight Averaging (Sub-optimal score).

- **Round 5 Cycle 35 Rejected:** Multi-Pillar Calibrated Production Checkpoint (Sub-optimal score).

- **Round 5 Cycle 36 Rejected:** Wavelet High-Frequency Energy Reinforcement (Sub-optimal score).

- **Round 5 Cycle 37 Rejected:** Vorticity-Enhanced Convective Updraft Prior (Sub-optimal score).

- **Round 5 Cycle 38 Rejected:** Gradient-Isolated Dewpoint Depression Coupling (Mass conservation breach (16.00%)).

- **Round 5 Cycle 39 Rejected:** Conserved P95 Extreme Cloudburst Gate (Mass conservation breach (27.59%)).

- **Round 5 Cycle 40 Rejected:** Dual-Band Directional Laplacian Wavelet Filter (Sub-optimal score).

- **Round 5 Cycle 41 Rejected:** Sub-Grid Terrain Roughness Index Module (Sub-optimal score).

- **Round 5 Cycle 42 Rejected:** Asymmetric Convective Core Spatial Booster (Sub-optimal score).

- **Round 5 Cycle 43 Rejected:** Multi-Octave Kolmogorov Power Spectrum Loss (Sub-optimal score).

- **Round 5 Cycle 44 Rejected:** Agent B Unconstrained Convective Flood Probe (Mass conservation breach (36.72%)).

- **Round 5 Cycle 45 Rejected:** Conserved Quantile Monotonic Projection Layer (Mass conservation breach (54.62%)).

- **Round 5 Cycle 46 Rejected:** Deep Loss Basin Trajectory Synchronization (Sub-optimal score).

- **Round 5 Cycle 47 Rejected:** Refined Orographic Updraft Scaling (Sub-optimal score).

- **Round 5 Cycle 48 Rejected:** Unified Round 5 Super-Champion Synthesis (Sub-optimal score).

- **Round 5 Cycle 49 Rejected:** Top-5 Checkpoint Polyak Averaging (Sub-optimal score).

- **Round 5 Cycle 50 Rejected:** Final Calibrated Production Master Ensemble (Sub-optimal score).
