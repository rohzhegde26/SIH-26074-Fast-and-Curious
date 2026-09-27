# Sprint 8.5 Model Training and Evaluation Audit

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Date:** September 27, 2026  
**Auditor:** Lead Meteorological Systems Research Agent  
**Operational Status:** Zero Training Backprop (Inference and Post-Processing Diagnostic Only)  

---

## 1. Model Training Status: Strictly Frozen

### 1.1 Model Checkpoint Invariants
In accordance with mandatory sprint directives, **zero model training iterations or backpropagation passes are permitted in Sprint 8.5**. The model weights remain 100% immutable and identical to the Sprint 6 multi-task champion checkpoint.

- **Checkpoint File:** `models/checkpoints/sprint6_candidate3_multitask_champion.pt`
- **Checkpoint SHA-256 Hash:**
  ```text
  f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92
  ```
- **Total Trainable Parameters:** 15,685,478
- **Architectural Specifications:**
  - Backbone: MultiTaskUNet5x with spatial conditioning and time-step embedding
  - Historical Temporal Context ($H$): 14 frames (03:00 UTC daily cycle)
  - Spatial Context Padding ($N$): 24 grid cells (coarse buffer)
  - Target Panchayat Output ($M$): 16 central grid cells (0.05 degree resolution)
  - Target Channels: 6 physical channels (Precipitation, Tmax, Tmin, RH, U-wind, V-wind)
  - Diffusion Formulation: $v$-prediction parameterization
  - Diffusion Noise Schedule: Linear schedule ($\beta_1 = 10^{-4}, \beta_T = 0.035$, $T = 100$ steps)
  - Training Loss Function: Multi-task group-weighted MSE with tail over-weighting for precipitation

---

## 2. Data Provenance and Evaluation Invariants

### 2.1 Partition Integrity and Temporal Leakage Prevention
All evaluation workflows strictly enforce physical temporal partitioning:
- **Historical Training Climatology (2015-2021):**
  - Grid points evaluated: 38,261,760
  - Extreme precipitation exceedance rates:
    - $P > 15$ mm/day: **11.0322%**
    - $P > 30$ mm/day: **5.8157%**
  - Used exclusively as the immutable climatological reference for Brier Skill Score (BSS) evaluations.
- **Validation Dataset (2022):**
  - 122 spatio-temporal forecast cubes (7 lead days each, 6 channels, 80x80 grid).
  - Used for uncertainty diagnostics and calibration parameter fitting.
  - **Internal Time-Respecting Partition:**
    - Early monsoon calibration block (cubes 1-61): Parameter estimation.
    - Late monsoon evaluation block (cubes 62-122): Out-of-sample validation.
- **Test Holdout Dataset (2023):**
  - 122 spatio-temporal forecast cubes.
  - Strictly quarantined during diagnostic and calibration exploration.
  - Evaluated exactly once at the final conclusion of Sprint 8.5 for the selected configuration.

### 2.2 Physical Atmospheric Invariants
- **Meteorological Day Alignment:** All coarse inputs, ground truth targets, and observational references adhere strictly to the **03:00-03:00 UTC** meteorological accumulation window (08:30-08:30 IST).
- **Physical Boundaries:**
  - Precipitation: $P \ge 0.0$ mm/day (strictly non-negative).
  - Relative Humidity: $0.0\% \le \text{RH} \le 100.0\%$.
  - Temperature Invariant: $T_{\min} \le T_{\max}$ enforced pairwise.
- **Atmospheric Driving Inputs:** Authentic GFS 0.25 degree forecasts only; no synthetic or reanalysis substitutes for future boundary conditions.

---

## 3. Pre-Experiment Metric Reconciliation Audit

Sprint 8.5 Phase 0 formally resolved the discrepancy between the headline condition-history Fair-CRPS and the case-level bootstrap point estimates.

### 3.1 Certified Audit Finding
- The dataloader evaluated 122 validation cubes with batch size $B=4$, yielding 30 full batches of 4 cubes and 1 final batch of 2 cubes.
- `evaluate_sprint8_ensemble.py` averaged batch-level metrics with `np.mean(all_crps_list)` (length 31), which weighted the final batch at $1/31 = 3.226\%$ rather than its proper sample proportion $2/122 = 1.639\%$.
- Because the final batch contained high-error monsoon storm events (mean CRPS = 3.8070 vs 0.4961 across batches 1-30), this unweighted batch average systematically shifted CRPS upwards by $+0.0525$ points across all conditions.
- When calculated as the true case-preserving sample mean across all 122 cubes, the values match the bootstrap point estimates exactly:
  - `B32_K2_S16`: 0.54009950
  - `B32_K4_S8`: 0.54334022
  - `B32_K8_S4`: 0.55235116
  - `B32_K8_S4_ETA05`: 0.55038854
- The relative rankings are 100% identical and monotonic under both aggregations.

### 3.2 Canonical Aggregation Invariant
All future metric reporting must strictly enforce:
$$\bar{M} = \frac{1}{N} \sum_{i=1}^N M_i$$
where each forecast cube $i$ carries an identical sample weight of $1/N$.

---

## 4. Evaluation Provenance Logging Schema

Every evaluation experiment executed in Sprint 8.5 must record its full provenance in the output metadata block conforming to the following JSON schema:

```json
{
  "checkpoint_sha256": "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92",
  "parameter_count": 15685478,
  "dataset_split": "validation_2022",
  "calibration_split": "internal_chronological_2022_b61",
  "holdout_split": "quarantined_2023",
  "sampler": "standard_DDIM",
  "ensemble_size_k": 8,
  "denoising_steps_s": 4,
  "eta": 0.5,
  "base_seed": 20260927,
  "calibration_method": "spread_rescaling",
  "calibration_parameters": {
    "alpha": 1.5,
    "lead_dependent": false
  },
  "physical_repair": "member_wise_non_negative_clip",
  "aggregation": "case_preserving_weighted_sample_mean"
}
```

---

## 5. Auditor Certification

I certify that:
1. No weights of Candidate 3 have been modified or subjected to further gradient updates.
2. The pre-experiment metric discrepancy has been verified and documented with mathematical certainty.
3. The 2023 holdout test set remains strictly quarantined.
4. All computational experiments in Sprint 8.5 adhere to the scientific constraints of Problem Statement 26074.
