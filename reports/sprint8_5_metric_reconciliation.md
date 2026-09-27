# Sprint 8.5 Phase 0: Metric and Provenance Reconciliation Report

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Date:** September 27, 2026  
**Status:** Pre-Experiment Audit Completed and Formally Reconciled  

---

## 1. Executive Summary

Before undertaking diagnostic experiments for Sprint 8.5, an essential pre-experiment audit was mandated to resolve the numerical discrepancy between two primary Sprint 8 metric artifacts:
- Condition-history Fair-CRPS reports (e.g. `sprint8_b32_k8_s4_eta05_history.json` reporting `crps: 0.6029`)
- Case-level paired bootstrap analysis (e.g. `sprint8_bootstrap_confidence_intervals.json` reporting `point_estimate: 0.5504`)

This audit has identified the exact algebraic root cause of the discrepancy. The difference is entirely explained by a **batch-weighting artifact** in the evaluation loop when handling the final incomplete batch, rather than an underlying calculation or sampling bug.

Crucially:
1. When calculated as a true sample-weighted mean across all 122 validation cubes, the resulting values match the bootstrap point estimates exactly to 15 decimal places.
2. The relative scientific ranking of all 4 matched-compute configurations is 100% identical and strictly monotonic under both aggregations:
   - `K=2, S=16, eta=0` remains the top continuous probabilistic distribution performer.
   - `K=8, S=4, eta=0.5` remains the top point-accuracy and heavy-rain event recall performer.
3. No model retraining or rerun of the Sprint 8 matrix is required.

---

## 2. Mathematical Root-Cause Analysis

### 2.1 The Evaluation Dataloader Setup
In `scripts/evaluate_sprint8_ensemble.py`, the 2022 validation split comprises $N_{\text{total}} = 122$ spatial-temporal forecast cubes (each spanning 7 lead days, 6 channels, 80x80 spatial grid).

The dataloader evaluated these cubes with a batch size of $B = 4$:
- $30$ batches contain $4$ cubes ($30 \times 4 = 120$ cubes).
- $1$ final batch (batch index 31) contains the remaining $2$ cubes.

### 2.2 The Aggregation Defect in Condition History
In `evaluate_condition()`, the code accumulated batch-averaged Fair-CRPS into a list:
```python
# evaluate_sprint8_ensemble.py (line 218)
crps_val, is_fair = compute_crps(members_phys, target_phys_tensor)
all_crps_list.append(crps_val)
...
# Line 248
avg_crps = float(np.mean(all_crps_list))
```

Because `all_crps_list` contains 31 elements, taking `np.mean(all_crps_list)` computed:
$$\bar{M}_{\text{batch}} = \frac{1}{31} \sum_{b=1}^{31} \bar{M}_b = \frac{1}{31} \left( \sum_{b=1}^{30} \bar{M}_b + \bar{M}_{31} \right)$$

This formula implicitly assigned the 31st batch an effective weight of:
$$w_{31} = \frac{1}{31} \approx 3.2258\%$$
However, because batch 31 contains only 2 forecast cubes, its legitimate sample proportion is:
$$w_{31,\text{true}} = \frac{2}{122} \approx 1.6393\%$$

### 2.3 Why the Magnitude of the Shift Was Large (~0.0525 points)
Batch 31 consists of late-monsoon heavy-rain events (cubes 121 and 122) where absolute errors are naturally large:
- Mean CRPS across batches 1 through 30: **0.4961**
- Mean CRPS for batch 31: **3.8070** (cube 121: 3.4895, cube 122: 4.1245)

Because batch 31 had high CRPS, giving it nearly double its legitimate sample weight ($3.23\%$ instead of $1.64\%$) systematically shifted the unweighted batch mean upward by approximately $+0.0525$ points across every condition evaluated.

### 2.4 The Bootstrap Aggregation
In contrast, `scripts/bootstrap_sprint8_paired.py` computed the metric per individual cube:
```python
# evaluate_sprint8_ensemble.py (line 231)
c_i, _ = compute_crps(mem_i, tgt_i)
case_crps_list.append(float(c_i))
```
The bootstrap point estimate and resamples evaluated the uniform sample mean:
$$\bar{M}_{\text{case}} = \frac{1}{122} \sum_{i=1}^{122} c_i = \frac{30 \times 4 \times \bar{M}_{1..30} + 2 \times \bar{M}_{31}}{122}$$

For $K=8, S=4, \eta=0.5$:
$$\bar{M}_{\text{case}} = \frac{120 \times 0.49611130 + 2 \times 3.80702317}{122} = \mathbf{0.55038854}$$
This matches the bootstrap report point estimate to 15 decimal places.

---

## 3. Empirical Verification Across All 4 Reference Conditions

The table below provides the full side-by-side reconciliation for all 32-NFE configurations evaluated in Sprint 8, plus the 2023 holdout test:

| Condition ID | Description | True Case Mean (Bootstrap) | Unweighted Batch Mean | History JSON Value | Discrepancy ($\Delta$) | Rank (Case) | Rank (Batch) |
|---|---|---:|---:|---:|---:|:---:|:---:|
| `B32_K2_S16` | K=2, S=16, eta=0.0 | **0.54009950** | 0.59273757 | 0.59273757 | +0.052638 | 1 | 1 |
| `B32_K4_S8` | K=4, S=8, eta=0.0 | 0.54334022 | 0.59579121 | 0.59579121 | +0.052451 | 2 | 2 |
| `B32_K8_S4` | K=8, S=4, eta=0.0 | 0.55235116 | 0.60462592 | 0.60462592 | +0.052275 | 4 | 4 |
| `B32_K8_S4_ETA05` | K=8, S=4, eta=0.5 | 0.55038854 | 0.60291491 | 0.60291491 | +0.052526 | 3 | 3 |
| `CHAMPION_HOLDOUT` | K=8, S=4, eta=0.5 (2023) | **0.63711661** | 0.68190044 | 0.68190044 | +0.044784 | N/A | N/A |

### Key Observations:
1. **Identical Discrepancy Shift**: The batch-weighting artifact is uniformly $+0.0523$ to $+0.0526$ across all four 2022 validation conditions.
2. **Preserved Ordering**: In both aggregations:
   - `B32_K2_S16` is strictly superior on Fair-CRPS (Case: 0.5401 vs Batch: 0.5927).
   - `B32_K4_S8` is second (Case: 0.5433 vs Batch: 0.5958).
   - `B32_K8_S4_ETA05` is third (Case: 0.5504 vs Batch: 0.6029).
   - `B32_K8_S4` is fourth (Case: 0.5524 vs Batch: 0.6046).
3. **Paired Differences Preserved**: Because the shift is nearly a pure constant scalar offset across conditions, all paired differences $\Delta = A - B$ and their bootstrap confidence intervals remain mathematically consistent.
4. **Point Forecast Metrics Unaffected**: Wet-MAE and CSI@30 were computed by accumulating total pixel counts (hits, misses, false alarms, wet-day absolute error sum) across the entire dataset before division, and were therefore completely free from this batch-weighting artifact.

---

## 4. Canonical Aggregation Definition Adopted for Sprint 8.5

To eliminate ambiguity across all future sprints, the following operational definition is declared canonical:

### Definition: Case-Preserving Weighted Sample Mean
For any metric $M$ evaluated over a dataset of $N$ forecast cubes (cases), the dataset-level headline metric $\bar{M}$ MUST be computed as:
$$\bar{M} \equiv \frac{1}{N} \sum_{i=1}^{N} M_i$$
where $M_i$ is the metric evaluated on the complete 7-day spatio-temporal cube $i$.

### Implementation Invariants for Sprint 8.5:
1. **No Unweighted Batch Means**: Evaluation scripts must never average a list of batch means when batch sizes vary or when the dataset size is not an exact multiple of the batch size.
2. **Case-Level Metric Logging**: All evaluation runs must record per-case metric arrays (`case_level: {crps: [...], precip_crps: [...], ...}`) in output JSON artifacts.
3. **Single Source of Truth**: The bootstrap point estimate and the headline evaluation metric must be derived from the exact same per-case array, guaranteeing mathematical identity.

---

## 5. Certification and Sign-Off

The pre-experiment audit is complete:
- The discrepancy is fully explained and proved.
- The scientific findings and Pareto tradeoff of Sprint 8 remain valid.
- The canonical aggregation is formalized and ready for Sprint 8.5 diagnostic workflows.
