## Active Compute Budget & Live Kaggle GPU Quota
> **LIVE STATUS (Updated 2026-09-25 23:00:25):**
> - **Total GPU Allowance:** 6.0 hours (21600 s)
> - **GPU Time Consumed:** 0.47 hours (1685 s)
> - **GPU TIME REMAINING:** **5.53 hours (331.9 minutes)**
> - **Quota Refreshes At:** 2026-09-26T00:00:00.000Z
>
> ⚠️ **HARD CONSTRAINT FOR AGENTS:**
> You have AT MOST **5.53 hours** of GPU time remaining for this cycle.
> Every training run depletes this quota. Budget your experiments carefully. If you run 5-minute runs,
> you have at most ~66 candidate iterations left before quota exhaustion!
## Active Compute Budget & Live Kaggle Accelerator Quota
> **LIVE STATUS (Verified 2026-09-22):**
> - **Kaggle GPU Quota:** 6.0 hours (21,600 s) allowed | 0.0 s consumed | **6.0 hours remaining (100%)**
> - **Kaggle TPU Quota:** 20.0 hours (72,000 s) allowed | 0.0 s consumed | **20.0 hours remaining (100%)**
> - **Quota Refresh Date:** 2026-09-26T00:00:00.000Z
>
> ⚠️ **MANDATORY HARD CONSTRAINT FOR ALL AGENTS:**
> All model training workloads MUST run on Kaggle (GPU or TPU accelerators via Kaggle API / CLI dispatch).
> Do NOT execute heavy model training loops on the local CPU or machine.

## 1. Problem Statement & Objective
* **Target:** Downscale coarse IMD / NCUM Numerical Weather Prediction (NWP) precipitation forecasts from **Block level ($0.25^\circ \approx 27\text{ km}$, $16\times 16$ grid)** to **Panchayat level ($0.05^\circ \approx 5.5\text{ km}$, $80\times 80$ grid)**.
* **Pilot:** Mandya District, Karnataka (234 active Gram Panchayats in the current Mandya pilot dataset, filtered from 258 in the source cadastral listing).
* **End Goal:** High-resolution precipitation combined with physics-based thermodynamic refinement for localized agro-meteorological advisories (Paddy, Ragi, Sugarcane).

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
