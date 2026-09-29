# Sprint 10 Implementation Plan: Integrated Mature System & Final Multi-Dimensional Scaling Study

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 10 of 10 (Culminating Research and System Integration Sprint)  
**Author**: Lead Meteorological Systems Architecture Agent  
**Date**: September 29, 2026  

---

## 1. Executive Summary & Research Handoff

### 1.1 The Cumulative Evidence Chain (Sprints 1 through 9)
Sprint 10 represents the synthesis and culmination of the 10-sprint research roadmap. Sprints 1 through 9 systematically resolved individual architectural and physical variables in strict isolation:

```text
[Sprint 1-2: Data Validity]
  Authentic ERA5-Land + CHIRPS + GLO-30 DEM. Canonical 03:00-03:00 UTC day. 80x80 target, 16x16 input.
       |
       v
[Sprint 3-4: Temporal Context]
  Antecedent history sweep (H in {3, 5, 7, 10, 14}). H=14 selected for synoptic soil moisture memory.
       |
       v
[Sprint 5: Spatial Context]
  Surrounding context ratio sweep (N/M in {1.25, 1.5, 1.75, 2.0, 2.5}). N=24, M=16 (1.5x) selected.
       |
       v
[Sprint 6: Diffusion Formulation]
  Conditional residual diffusion with v-prediction, multi-task group loss, and focal tail weighting.
  Established Candidate 3 control baseline (15,685,478 parameters).
       |
       v
[Sprint 7: Test-Time Denoising]
  Sampler and step frontier (S in {4, 8, 12, 16, 24, 32}). DDIM-4 proved optimal for fast traversal.
       |
       v
[Sprint 8 / 8.5: Ensemble Compute & Calibration]
  Test-time compute scaling (B = K x S in {8, 16, 32, 64}). Established 6.51 mm Wet-MAE under physical repair.
  Identified conditional under-dispersion bottleneck requiring capacity scaling.
       |
       v
[Sprint 9: Capacity Scaling & Sparse Routing]
  Dense ladder (Dense-S 15.69M, Dense-M 31.20M, Dense-L 52.00M) vs Sparse MoE-4 (22.77M total / 15.69M active).
  Verified Outcome E partially supported: MoE-4 wins on efficiency & point MAE; Dense-L wins on extremes & texture.
       |
       v
[Sprint 10: Integrated Mature System & Final Scaling Study]
  Synthesis of optimal configurations into 4 operational deployment modes (FAST, BALANCED, ACCURATE, ENSEMBLE).
  Canonical physical-repair cross-evaluation, 2023 El Nino holdout unquarantining, and jury deployment package.
```

### 1.2 Core Scientific Objectives of Sprint 10
Sprint 10 directly addresses four critical operational and scientific goals:
1. **Unify the Operational Deployment Modes**: Map the empirical Pareto frontier into four discrete, production-ready operating configurations (`FAST`, `BALANCED`, `ACCURATE`, `ENSEMBLE`), tailored to specific Ministry of Earth Sciences (MoES) and IMD operational deployment tiers.
2. **Canonical Physical-Repair Cross-Evaluation**: Bridge the Sprint 9 exploratory notebook numbers by running the full 4-tier capacity ladder (Dense-S, Dense-M, Dense-L, MoE-4) through the canonical `sprint8_physical` evaluation pipeline (`invert_normalization`, member-wise physical bounds repair, standard $p_{\text{target}} > 2.5\text{ mm/day}$ wet threshold).
3. **Unquarantine the 2023 El Nino Holdout Year**: For the first time across the entire research program, evaluate the champion architectures against the strictly quarantined 2023 holdout test set (365 daily forecast cycles during an active El Nino phase) to measure out-of-distribution generalization.
4. **Multivariate Physical Consistency & Reliability Audit**: Audit the joint physical relationships across predicted variables (Precipitation vs RH, Orographic Precipitation vs Wind Divergence, $T_{\max} \ge T_{\min}$ thermal diurnal bounds) to certify meteorological plausibility.
5. **Final Jury and Production Deliverables**: Deliver locked deployment configurations, reproducible inference APIs, automated diagnostic visualizers, and the definitive master research monograph for the SIH hackathon jury.

---

## 2. Operating Modes & Deployment Architecture

To translate our scientific findings into practical meteorological service, Sprint 10 organizes the model capacity and test-time compute configurations into four distinct operational profiles:

```
+==================================================================================================+
|                                SPRINT 10 OPERATIONAL PROFILES                                    |
+==================================================================================================+
| Profile   | Target Model | NFE Budget | Sampler (K, S, eta) | Target Latency | Deployment Tier   |
+-----------+--------------+------------+---------------------+----------------+-------------------+
| FAST      | MoE-4        | 4 NFE      | K=1, S=4, eta=0.0   | < 250 ms/cube  | Panchayat Edge /  |
|           | (15.7M act)  |            |                     |                | Mobile Web API    |
+-----------+--------------+------------+---------------------+----------------+-------------------+
| BALANCED  | MoE-4        | 16 NFE     | K=2, S=8, eta=0.25  | ~ 700 ms/cube  | District Advisory |
|           | (15.7M act)  |            |                     |                | Workstations      |
+-----------+--------------+------------+---------------------+----------------+-------------------+
| ACCURATE  | Dense-L      | 32 NFE     | K=2, S=16, eta=0.50 | ~ 1.85 s/cube  | Severe Storm /    |
|           | (52.0M dens) |            |                     |                | Cloudburst Radar  |
+-----------+--------------+------------+---------------------+----------------+-------------------+
| ENSEMBLE  | Dense-L /    | 64 NFE     | K=8, S=8, eta=0.50  | ~ 3.50 s/cube  | State HPC / Flood |
|           | Dense-M      |            | (or K=16, S=4)      |                | Risk Management   |
+==================================================================================================+
```

### 2.1 Profile Specifications
1. **FAST (Edge / Real-Time Panchayat Tier)**:
   - **Target Environment**: Local edge devices, block-level servers, public web APIs (`/api/v1/forecast/fast`).
   - **Architecture**: MoE-4 denoiser (22.77M total, 15.69M active parameters).
   - **Inference Setup**: Single-member deterministic traversal ($K=1, S=4, \eta=0.0$).
   - **Primary Objective**: Lowest possible wall-clock latency ($< 250\text{ ms}$ on GPU, $< 2.5\text{ s}$ on 4-core CPU) while delivering state-of-the-art point Wet-MAE.
2. **BALANCED (District Operational Advisory Tier)**:
   - **Target Environment**: District Agromet Units (DAMUs) and regional advisory centers.
   - **Architecture**: MoE-4 denoiser.
   - **Inference Setup**: 2 stochastic members with 8 DDIM steps ($K=2, S=8, \eta=0.25$).
   - **Primary Objective**: Optimal tradeoff between point error, moderate convective recall (CSI@15), and baseline uncertainty bounds within a sub-second response window.
3. **ACCURATE (Severe Convective Storm & Cloudburst Tier)**:
   - **Target Environment**: State disaster management centers, Doppler weather radar integration nodes.
   - **Architecture**: Dense-L denoiser (51,997,958 parameters, 176 base channels).
   - **Inference Setup**: $K=2, S=16, \eta=0.50$ (or $K=4, S=8$).
   - **Primary Objective**: Maximum high-frequency spatial Laplacian sharpness ($R_{\text{Lap}} \ge 0.084$), crisp ridgeline gradient preservation, and peak extreme storm recall (CSI@30 $\ge 0.660$).
4. **ENSEMBLE (State HPC / Probabilistic Flood Risk Management Tier)**:
   - **Target Environment**: Central IMD / NCMRWF High-Performance Computing clusters.
   - **Architecture**: Dense-L or Dense-M.
   - **Inference Setup**: Multi-member stochastic ensemble ($K=8\text{ or }16, S=4\text{ to }8, \eta=0.50$).
   - **Primary Objective**: Full probabilistic distribution capture, lowest Fair-CRPS, calibrated spread-skill ratio (SSR), and reliable threshold exceedance curves for reservoir control and flood warning.

---

## 3. Core Scientific Hypotheses for Sprint 10

| Hypothesis | Description | Success Criterion |
| :--- | :--- | :--- |
| **H1: Canonical Physical Convergence** | Under full non-linear physical inversion and bounds repair, capacity gains from Sprint 9 persist. | Dense-L and MoE-4 achieve lower Wet-MAE and higher CSI@30 than Candidate 3's 6.51 mm baseline on the 2022 validation set. |
| **H2: Out-of-Distribution Generalization** | The mature architectures generalize to the quarantined 2023 El Nino holdout year without representation collapse. | Holdout 2023 Wet-MAE degradation is $< 15\%$ relative to 2022 validation, outperforming bilinear and coarse reanalysis baselines. |
| **H3: Multidimensional Operating Separation** | The 4 operational profiles form a non-dominated Pareto frontier across latency, point MAE, storm CSI, and Fair-CRPS. | No single profile dominates all others; each profile is strictly optimal for its designated compute/latency constraint. |
| **H4: Multivariate Thermodynamic Plausibility** | Spatiotemporal diffusion maintains coupled atmospheric balances across variables without post-hoc physical distortion. | Zero violations of $T_{\max} \ge T_{\min}$; negative precipitation mass shift under physical bounds repair is $< 1.0\%$; RH remains bounded in $[0, 100]\%$. |
| **H5: Calibrated Extreme Exceedance** | Combining capacity scaling with ensemble sampling ($K=8$) improves decision-relevant threshold probabilities. | Brier Skill Score (BSS) for precipitation $> 15\text{ mm}$ and $> 30\text{ mm}$ improves by $> 10\%$ over Candidate 3 baseline. |

---

## 4. Detailed Implementation Workstreams

Sprint 10 is structured into five cohesive execution workstreams:

```
[Workstream 1: Canonical Physical Evaluation Engine]
  Upgrade evaluate_sprint9_capacity.py to execute all 4 checkpoints under sprint8_physical protocol.
       |
       v
[Workstream 2: Operational Profile Configurations & Benchmarking Engine]
  Implement configs/final/{fast, balanced, accurate, ensemble}.yaml and scripts/benchmark_final_system.py.
       |
       v
[Workstream 3: Quarantined 2023 El Nino Holdout Generalization Study]
  Execute unbiased verification across all 365 daily samples of 2023. Record zero-leakage test metrics.
       |
       v
[Workstream 4: Multivariate Physical Consistency & Calibration Diagnostics]
  Audit inter-variable correlations (P-RH, P-Wind, Tmax-Tmin) and compute calibrated Brier Skill Scores.
       |
       v
[Workstream 5: Master Deliverables & SIH Jury Showcase Package]
  Author final research report, export executive summaries to Downloads, package FastAPI endpoints.
```

---

### Workstream 1: Canonical Physical Evaluation Engine

#### 1.1 Objective
Run all four model checkpoints (Dense-S, Dense-M, Dense-L, MoE-4) through the exact `sprint8_physical` pipeline to replace exploratory notebook numbers with canonical physical metrics directly comparable to Sprint 8's 6.51 mm benchmark.

#### 1.2 Execution Mechanics
- **Script**: `scripts/evaluate_final_canonical_benchmarks.py` (building on `scripts/evaluate_sprint9_capacity.py`).
- **Input Checkpoints**:
  - `models/checkpoints/sprint6_candidate3_multitask_champion.pt` (Dense-S, 15.69M)
  - `models/checkpoints/sprint9_dense_m_weights.pt` (Dense-M, 31.20M)
  - `models/checkpoints/sprint9_dense_l_weights.pt` (Dense-L, 52.00M)
  - `models/checkpoints/sprint9_moe4_weights.pt` (MoE-4, 22.77M total / 15.69M active)
- **Dataset**: `datasets/multitask_temporal_v2_h14.zarr` (2022 validation split, 122 cubes).
- **Physical Protocol**:
  - `invert_normalization(..., stats)` with proper exponential inversion for precipitation ($\max(0, \exp(y \cdot \sigma + \mu) - 1)$).
  - Member-wise physical bounds repair via `apply_member_wise_physical_bounds()`.
  - Meteorological wet threshold: $p_{\text{target}} > 2.5\text{ mm/day}$.
  - Finite-ensemble unbiased Fair-CRPS (Ferro et al., 2008):
    $$\text{CRPS}_{\text{Fair}} = \frac{1}{K}\sum_{k=1}^K |x_k - y| - \frac{1}{2K(K-1)}\sum_{k=1}^K\sum_{m=1}^K |x_k - x_m|$$
  - Router metrics accumulated across all batches to report true global dispatch distribution.

---

### Workstream 2: Operational Profile Configurations & Benchmarking Engine

#### 2.1 Objective
Formalize the 4 operational profiles into versioned configuration files and build an automated latency/throughput profiling suite.

#### 2.2 Configuration Architecture
Create four production configuration YAMLs under `configs/final/`:
1. `configs/final/profile_fast.yaml`:
   ```yaml
   profile_name: FAST
   model_tier: moe_4
   checkpoint: models/checkpoints/sprint9_moe4_weights.pt
   num_members: 1
   denoising_steps: 4
   eta: 0.0
   target_latency_ms: 250
   description: "Panchayat Edge / Web API real-time inference"
   ```
2. `configs/final/profile_balanced.yaml`:
   ```yaml
   profile_name: BALANCED
   model_tier: moe_4
   checkpoint: models/checkpoints/sprint9_moe4_weights.pt
   num_members: 2
   denoising_steps: 8
   eta: 0.25
   target_latency_ms: 750
   description: "District Agromet Advisory operational workstation"
   ```
3. `configs/final/profile_accurate.yaml`:
   ```yaml
   profile_name: ACCURATE
   model_tier: dense_l
   checkpoint: models/checkpoints/sprint9_dense_l_weights.pt
   num_members: 2
   denoising_steps: 16
   eta: 0.50
   target_latency_ms: 1850
   description: "Severe convective storm and cloudburst warning"
   ```
4. `configs/final/profile_ensemble.yaml`:
   ```yaml
   profile_name: ENSEMBLE
   model_tier: dense_l
   checkpoint: models/checkpoints/sprint9_dense_l_weights.pt
   num_members: 8
   denoising_steps: 8
   eta: 0.50
   target_latency_ms: 3500
   description: "State HPC flood risk management and calibrated probabilistic bounds"
   ```

#### 2.3 Automated Benchmarking Suite
Build `scripts/benchmark_final_system.py`:
- Measures cold-start load time, VRAM allocation (FP16 and FP32), batch inference throughput (cubes/sec), and wall-clock latency per lead-day.
- Benchmarks across both GPU (Dual Tesla T4 or single accelerator) and CPU (Intel/AMD multi-core) hardware targets.

---

### Workstream 3: Quarantined 2023 El Nino Holdout Generalization Study

#### 3.1 Objective
Execute the definitive generalization stress test against the quarantined 2023 holdout partition.

#### 3.2 Holdout Invariants & Context
- **Climatological Setting**: 2023 experienced significant climate anomalies across Peninsular India due to a developing moderate-to-strong El Nino and positive Indian Ocean Dipole (IOD), resulting in delayed monsoon onset and erratic intra-seasonal dry spells followed by intense late-season convective bursts.
- **Strict Quarantine Verification**: The 2023 partition (`split == "test"` in `sample_index_v2_h14.parquet`) was held completely untouched across Sprints 3 through 9 during model training, capacity tuning, and router optimization.
- **Evaluation Metrics**:
  - Wet-MAE (mm) across all 365 daily forecast horizons.
  - CSI@15 (moderate rain) and CSI@30 (heavy storm rain).
  - Fair-CRPS (probabilistic error).
  - Spatial Laplacian energy retention $R_{\text{Lap}}$.
  - Generalization Gap: $\Delta_{\text{gen}} = \text{Metric}_{2023} - \text{Metric}_{2022}$.

---

### Workstream 4: Multivariate Physical Consistency & Calibration Diagnostics

#### 4.1 Objective
Verify that joint multivariate downscaling adheres to physical laws and preserves meteorological relationships across all 6 prognostic variables ($P, T_{\max}, T_{\min}, \text{RH}, U, V$).

#### 4.2 Diagnostic Suite
Build `src/eval/multivariate_diagnostics.py`:
1. **Thermodynamic Invariant ($T_{\max} \ge T_{\min}$)**:
   - Measure empirical violation rate $\mathbb{P}(T_{\max} < T_{\min})$ before and after physical bounds repair.
   - Record maximum temperature inversion gap.
2. **Moisture-Precipitation Coupling**:
   - Compute bivariate spatial correlation between precipitation rate and relative humidity:
     $$r(P, \text{RH}) = \text{corr}(P, \text{RH})$$
   - Ground truth vs model comparison: models must replicate the physical reality that high rain rates occur in near-saturated air masses ($\text{RH} > 85\%$).
3. **Orographic Windward/Leeward Asymmetry**:
   - Compute wind divergence $\nabla \cdot \mathbf{u} = \frac{\partial U}{\partial x} + \frac{\partial V}{\partial y}$ along Western Ghats ridgelines.
   - Verify that positive moisture flux convergence ($-\nabla \cdot (q\mathbf{u}) > 0$) correlates with enhanced precipitation along windward slopes.
4. **Extreme Threshold Probabilistic Verification**:
   - Compute Brier Score (BS) and Brier Skill Score (BSS) against climatology for key agro-meteorological thresholds:
     - Light rain: $P \ge 2.5\text{ mm/day}$
     - Moderate rain: $P \ge 15.0\text{ mm/day}$
     - Heavy storm: $P \ge 30.0\text{ mm/day}$
     - Very heavy rainfall / Cloudburst: $P \ge 50.0\text{ mm/day}$

---

### Workstream 5: Master Deliverables & SIH Jury Showcase Package

#### 5.1 Objective
Synthesize all empirical findings into a comprehensive research report, update user-facing documentation, and package FastAPI serving endpoints.

#### 5.2 Deliverables List
1. **Executable Code & Tooling**:
   - `scripts/evaluate_final_canonical_benchmarks.py`: Full physical-repair evaluation runner.
   - `scripts/benchmark_final_system.py`: Multi-profile latency and memory profiler.
   - `src/eval/multivariate_diagnostics.py`: Joint physical consistency validator.
   - `src/api/forecast_service.py`: Multi-profile serving endpoint supporting `?profile={fast,balanced,accurate,ensemble}`.
2. **Configuration Profiles**:
   - `configs/final/profile_fast.yaml`
   - `configs/final/profile_balanced.yaml`
   - `configs/final/profile_accurate.yaml`
   - `configs/final/profile_ensemble.yaml`
3. **Reports & Research Documentation**:
   - `reports/sprint10_canonical_benchmarks.md`: Physical repair metrics for Dense-S, Dense-M, Dense-L, MoE-4 on 2022 validation.
   - `reports/sprint10_holdout_2023_generalization.md`: Out-of-distribution El Nino stress test report.
   - `reports/sprint10_multivariate_physical_consistency.md`: Thermodynamic and orographic consistency audit.
   - `reports/sprint10_final_system_scaling_report.md`: Consolidated master scaling monograph.
   - Synchronized export to `C:\Users\rohit\Downloads\sprint10_final_system_scaling_report.md`.
4. **Unit Tests**:
   - `tests/models/test_final_profiles.py`: Verification of profile instantiation, config integrity, and profile switching.
   - `tests/eval/test_physical_consistency.py`: Unit tests for thermodynamic bounds, mass shift diagnostics, and Brier skill score computation.

---

## 5. Execution Strategy & Compute Budget

In accordance with agent directives, all heavy validation passes and full-year holdout evaluations are executed on Kaggle cloud accelerators (or Kaggle notebooks dispatched via API) rather than running intensive multi-hour jobs on local CPU:

- **Hardware Target**: Kaggle Dual Tesla T4 GPU accelerators (16 GB VRAM each).
- **Execution Workflow**:
  1. Build and locally validate modular components (`pytest tests/ -q` on CPU for unit contracts).
  2. Assemble standalone evaluation notebook `notebooks/sprint_10_final_integrated_scaling.ipynb` configured with all 4 profiles, 2022 validation split, and 2023 holdout split.
  3. Execute remotely on Kaggle to evaluate all profiles and unquarantine the 2023 holdout.
  4. Download output metric JSONs and diagnostic logs into `reports/` and `data/cache/`.
  5. Commit and push versioned artifacts to GitHub.

---

## 6. Sprint 10 Completion Checklist & Exit Criteria

Sprint 10 will conclude when all of the following criteria are strictly met:

```text
[ ] Workstream 1: All 4 models evaluated under canonical sprint8_physical protocol on 2022 validation data.
[ ] Workstream 2: 4 operational profile YAMLs committed under configs/final/ and verified via pytest.
[ ] Workstream 3: Quarantined 2023 El Nino holdout year evaluated; generalization gap reported.
[ ] Workstream 4: Multivariate physical consistency verified (zero Tmax < Tmin violations, P mass shift < 1%).
[ ] Workstream 5: Comprehensive master scaling monograph reports/sprint10_final_system_scaling_report.md authored.
[ ] Deliverable: Final report copied to C:\Users\rohit\Downloads\sprint10_final_system_scaling_report.md.
[ ] Code Quality: All unit tests passing (tests/models/ and tests/eval/).
[ ] Style Rule: Strictly ZERO em dashes (\u2014 and \u2013) across all files, code, commits, and responses.
[ ] Repository: All code, configs, reports, and checkpoints committed and pushed to remote branch.
```

---

*This document establishes the definitive blueprint for Sprint 10, bringing the SIH 26074 Weather Downscaling research program to a rigorous, reproducible, and competition-winning conclusion.*
