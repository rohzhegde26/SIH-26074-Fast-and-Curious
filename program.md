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
