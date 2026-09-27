# Sprint 8 Model Training and Evaluation Audit: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 27, 2026  

---

## 1. Audit Executive Summary

Sprint 8 addresses **Ensemble and Test-Time Scaling** under matched computational budgets. Unlike Sprints 1 through 6, which investigated model architectures, parameterizations, and loss formulations, Sprint 8 is strictly an **inference-only research sprint**.

This audit serves as the primary governance document for Sprint 8, providing rigorous verification across six core dimensions:
1. **Model Checkpoint Provenance and Weight Invariant**: Hard verification of the Sprint 6 Candidate 3 champion weights (`sprint6_candidate3_multitask_champion.pt`), confirming Git LFS hashes, parameter counts, and zero-modification enforcement.
2. **Phase 0 Reproducibility Gate**: Prescribing the strict tolerance bounds under which the deterministic DDIM-4 baseline must reproduce the Sprint 7 reference on the 2022 validation split before ensemble scaling begins.
3. **Mathematical Derivations of Probabilistic Estimators**: Rigorous formulation of finite-ensemble Fair-CRPS, Brier Skill Score, Spread-Skill Ratio, and Multivariate Energy Score, including proofs of finite-sample bias correction.
4. **Physical Transformation and Invariant Ordering**: Formal proof via Jensen's Inequality showing why member-wise non-linear physical clipping must strictly precede ensemble reduction.
5. **Kaggle Compute Budget Allocation**: Operational scheduling and memory profiling across **6.0 hours of 2x Tesla T4 GPU accelerators** to complete all matched-compute sweeps ($B \in \{8, 16, 32, 64\}$) within allocation.
6. **Data Leakage and Seed Determinism Audit**: Complete isolation of the 2023 quarantined holdout test set and specification of deterministic, nested pseudo-random number generator (PRNG) seed manifests.

---

## 2. Frozen Model Provenance and Checkpoint Verification

### 2.1 Checkpoint Provenance Invariant
The model evaluated in Sprint 8 is the Sprint 6 Candidate 3 champion, which achieved the best validation and test performance across all investigated architectures:

| Property | Required Specification | Verification Method |
|---|---|---|
| **Checkpoint Path** | `models/checkpoints/sprint6_candidate3_multitask_champion.pt` | File existence and path check |
| **Git LFS Object ID** | `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92` | SHA-256 hash match |
| **Trainable Parameters** | Exactly **15,685,478** | Model parameter inspection |
| **Training Epoch** | Epoch 30 (Final Convergence) | Checkpoint state dictionary |
| **Missing Keys** | **0** | PyTorch `load_state_dict` verification |
| **Unexpected Keys** | **0** | PyTorch `load_state_dict` verification |
| **Training Weights** | **Strictly Immutable (Read-Only)** | Zero backpropagation steps |

### 2.2 Hard Failure Policy on Provenance Violation
To prevent the recurrence of provenance ambiguities observed in early Sprint 7 runs, the evaluation runner enforces an unconditional hard failure policy:
- If the checkpoint file is missing, the script terminates immediately with exit code 1.
- If the SHA-256 hash diverges from `f3367f5fdd...`, the script terminates immediately.
- The evaluation harness will **never**:
  - Silently initialize random weights.
  - Fall back to an alternate checkpoint (e.g., Candidate 1 or Candidate 2).
  - Proceed with partial or missing keys.
  - Execute any optimizer steps (`optimizer.step()`, `loss.backward()`).

---

## 3. Architecture and Hyperparameter Integrity Audit

### 3.1 Neural Network Specification
The model architecture remains strictly identical to the frozen Candidate 3 design:
- **Base Architecture**: 3D Spatiotemporal U-Net with ResNet blocks and spatial self-attention.
- **Temporal Memory**: Antecedent history depth $H = 14$ days.
- **Forecast Horizon**: Output lead horizon $T_f = 7$ days ($D+0$ through $D+6$).
- **Spatial Grid**: Input coarse context $N = 24$ ($660 \times 660$ km), target crop $M = 16$, fine output $80 \times 80$ ($0.05^\circ$ resolution).
- **Target Channels**: 6 meteorological variables:
  1. Precipitation ($P$, mm/day)
  2. Maximum Temperature ($T_{\max}$, °C)
  3. Minimum Temperature ($T_{\min}$, °C)
  4. Relative Humidity ($\text{RH}$, %)
  5. Zonal Wind Component ($U$, m/s)
  6. Meridional Wind Component ($V$, m/s)

### 3.2 Diffusion Parameterization and Schedule
- **Objective Formulation**: Velocity prediction ($v$-prediction) where $v_t \equiv \alpha_t \epsilon - \sigma_t x_0$.
- **Training Noise Schedule**: Cosine variance schedule across $T = 100$ diffusion steps.
- **Inference Sampler**: Standard DDIM discretization with uniform spacing spanning $[T-1, 0]$ (starts at $t = 99$, terminates at $t = 0$).

---

## 4. Phase 0 Reproducibility Gate and Validation Baselines

### 4.1 Gate Criteria
Before any ensemble or matched-compute evaluations are executed on Kaggle, the evaluation script must run a deterministic single-member DDIM-4 regression on the full 2022 validation dataset.

The execution gate is satisfied if and only if:

$$|\text{CMVS}_{\text{run}} - 0.5376| \le 0.0054 \quad (\le 1.0\% \text{ relative error})$$
$$|\text{Wet-MAE}_{\text{run}} - 6.97| \le 0.15 \text{ mm}$$
$$|\text{CSI@30}_{\text{run}} - 0.704| \le 0.015$$

If any metric falls outside this tolerance envelope, Sprint 8 scaling halts.

### 4.2 Benchmark Baseline Reference Matrix
For reference, established baseline figures from Sprints 6 and 7 are summarized below:

| Model Configuration | Sampler & Steps | Val CMVS (2022) | Val Wet-MAE (mm) | Val CSI@30 | Latency (ms) |
|---|---|---|---|---|---|
| Sprint 5 Baseline (Candidate 1) | DDIM-32 Legacy | 0.6490 | 8.70 | 0.631 | 740 ms |
| Sprint 6 Candidate 2 ($v$-pred uniform) | DDIM-32 Legacy | 0.6493 | 8.65 | 0.635 | 738 ms |
| Sprint 6 Candidate 3 (Multi-Task Tail) | DDIM-32 Legacy | 0.5710 | 7.67 | 0.679 | 735 ms |
| Sprint 7 DDIM-32 Standard (Corrected) | DDIM-32 Standard | 0.5649 | 7.57 | 0.682 | 735 ms |
| **Sprint 7 Champion (DDIM-4 Standard)** | **DDIM-4 Standard** | **0.5376** | **6.97** | **0.704** | **96.8 ms** |

---

## 5. Mathematical Formulations of Probabilistic Estimators

### 5.1 Fair-CRPS (Finite-Ensemble Unbiased Estimator)
The Continuous Ranked Probability Score for a predictive cumulative distribution $F$ and realized observation $y$ is defined as:

$$\text{CRPS}(F, y) = \int_{-\infty}^\infty \left(F(z) - \mathbb{I}(z \ge y)\right)^2 dz$$

For an empirical ensemble of size $K$ with members $\{y^{(1)}, \dots, y^{(K)}\}$, the standard sample estimator is:

$$\text{CRPS}_{\text{emp}}(F_K, y) = \frac{1}{K}\sum_{k=1}^K |y^{(k)} - y| - \frac{1}{2 K^2}\sum_{k=1}^K \sum_{j=1}^K |y^{(k)} - y^{(j)}|$$

However, $\text{CRPS}_{\text{emp}}$ exhibits positive finite-sample bias: $\mathbb{E}[\text{CRPS}_{\text{emp}}(F_K, y)] > \text{CRPS}(F, y)$, which unfairly penalizes smaller ensembles when comparing $K=2$ versus $K=16$.

Following Ferro et al. (2008) and Gneiting & Raftery (2007), Sprint 8 implements the strictly unbiased Fair-CRPS estimator:

$$\text{CRPS}_{\text{fair}}(F_K, y) = \frac{1}{K}\sum_{k=1}^K |y^{(k)} - y| - \frac{1}{2 K (K - 1)}\sum_{k=1}^K \sum_{j=1}^K |y^{(k)} - y^{(j)}|$$

**Proof of Bias Correction**:
The expected value of the inter-member distance under independent draws from $F$ has expectation:
$$\mathbb{E}[|Y - Y'|] = \frac{1}{K(K-1)} \sum_{k=1}^K \sum_{j \ne k} \mathbb{E}[|y^{(k)} - y^{(j)}|]$$
The divisor $K(K-1)$ replaces $K^2$, eliminating the $k=j$ zero-distance self-terms and guaranteeing that $\mathbb{E}[\text{CRPS}_{\text{fair}}(F_K, y)] = \text{CRPS}(F, y)$ for any $K \ge 2$.

### 5.2 Brier Score and Climatological Skill Decomposition
For extreme precipitation threshold events ($P > \tau$ for $\tau \in \{15, 30\}$ mm/day), the event indicator is $o_i = \mathbb{I}(y_{\text{true}, i} > \tau)$. The ensemble forecast probability is:

$$p_i = \frac{1}{K}\sum_{k=1}^K \mathbb{I}\left(y_i^{(k)} > \tau\right)$$

The Brier Score across $N$ validation instances is:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - o_i)^2$$

To place raw Brier scores into proper meteorological perspective, Sprint 8 computes the Brier Skill Score (BSS) relative to the sample climatological base rate $\bar{o} = \frac{1}{N}\sum_{i=1}^N o_i$:

$$\text{BS}_{\text{clim}} = \bar{o}(1 - \bar{o})$$
$$\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{\text{clim}}}$$

Furthermore, the Brier Score is partitioned into Murphy's canonical components via binning predicted probabilities into $B_m$ bins ($m = 1, \dots, M$):

$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

where:
- $\text{Reliability} = \sum_{m=1}^M \frac{n_m}{N} (\bar{p}_m - \bar{o}_m)^2$ (measures conditional calibration; lower is better, 0 is perfect).
- $\text{Resolution} = \sum_{m=1}^M \frac{n_m}{N} (\bar{o}_m - \bar{o})^2$ (measures ability to distinguish event from non-event regimes; higher is better).
- $\text{Uncertainty} = \bar{o}(1 - \bar{o})$ (inherent climatological entropy of the event).

### 5.3 Spread-Skill Ratio and Fair Spread
For an ideally calibrated ensemble forecasting system, the ensemble spread matches the root-mean-square error of the ensemble mean.
For finite ensemble size $K$, the sample variance per instance $i$ is:

$$s_i^2 = \frac{1}{K - 1}\sum_{k=1}^K \left(y_i^{(k)} - \mu_i\right)^2, \quad \mu_i = \frac{1}{K}\sum_{k=1}^K y_i^{(k)}$$

The root-mean-square ensemble spread is:

$$\text{Spread} = \sqrt{\frac{1}{N}\sum_{i=1}^N s_i^2}$$

The root-mean-square error of the ensemble mean is:

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left(\mu_i - y_{\text{true}, i}\right)^2}$$

The Spread-Skill Ratio (SSR) is:

$$\text{SSR} = \frac{\text{Spread}}{\text{RMSE}}$$

An ensemble is under-dispersive (overconfident) when $\text{SSR} < 1.0$ and over-dispersive when $\text{SSR} > 1.0$.

### 5.4 Multivariate Energy Score
To assess inter-variable correlation structures (e.g., wind vector $(U, V)$ consistency, precipitation-humidity coupling), Sprint 8 evaluates the Energy Score:

$$\text{ES}(F_K, \mathbf{y}) = \frac{1}{K}\sum_{k=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}\|_2 - \frac{1}{2 K (K - 1)}\sum_{k=1}^K \sum_{j=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}^{(j)}\|_2$$

where $\mathbf{y} \in \mathbb{R}^6$ represents the normalized multi-variable state vector at a given spatial grid cell.

---

## 6. Test-Time Physical Safeguards and Transformation Invariants

### 6.1 Member-Wise Transformation Ordering Invariant
A critical source of potential bias in probabilistic weather downscaling is the ordering of non-linear physical clipping operations relative to ensemble aggregation.

**Mathematical Rule**: All denormalization and physical constraint enforcements must be executed on individual ensemble members **prior** to computing ensemble summary statistics (mean, variance, quantiles, exceedance probabilities).

```
CORRECT (Member-Wise Invariant):
Raw Latents y_norm^(k) 
  --> Unstandardize to Physical Space y_phys^(k) 
  --> Non-Linear Physical Bounds [P >= 0, 0 <= RH <= 100, Tmin <= Tmax]
  --> Compute Ensemble Mean, Quantiles, Spread, P(P > thresh)

INCORRECT (Ensemble-First Violation):
Raw Latents y_norm^(k) 
  --> Average in Latent/Normalized Space 
  --> Unstandardize 
  --> Apply Non-Linear Bounds (DESTROYS PROBABILISTIC CALIBRATION)
```

### 6.2 Mathematical Proof of the Non-Linear Clipping Invariant
Let $X \in \mathbb{R}$ represent an unconstrained precipitation prediction before non-negativity clipping, and let $g(x) = \max(0, x)$ be the ReLU physical projection.
The function $g(x)$ is convex.

By **Jensen's Inequality**, for any random variable $X$:

$$\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$$

Equality holds if and only if $X \ge 0$ almost everywhere.
In regions with dry weather or light rainfall where the diffusion latents fluctuate around zero:
1. If clipping is applied **after** averaging:
   $$\mu_{\text{wrong}} = \max\left(0, \frac{1}{K}\sum_{k=1}^K X^{(k)}\right)$$
   Negative member perturbations cancel positive member perturbations, systematically underestimating total precipitation volume and completely extinguishing convective tail signals.
2. If clipping is applied **member-by-member**:
   $$\mu_{\text{correct}} = \frac{1}{K}\sum_{k=1}^K \max\left(0, X^{(k)}\right)$$
   Every positive excursion contributes physical rainfall volume, maintaining an unbiased expected precipitation rate.
3. For threshold probability estimation:
   $$P(P > 15) = \frac{1}{K}\sum_{k=1}^K \mathbb{I}\left(\max(0, X^{(k)}) > 15\right)$$
   Averaging first would replace this with a binary step function $\mathbb{I}(\mu_{\text{wrong}} > 15)$, destroying the entire continuous probability distribution.

### 6.3 Thermodynamic Ordering Safeguard
For temperature downscaling, physical consistency demands that $T_{\min} \le T_{\max}$ at every spatial location and forecast lead:
$$\text{Violation Indicator}: \quad \mathcal{V}_T = \mathbb{I}\left(T_{\min}^{(k)} > T_{\max}^{(k)}\right)$$
The evaluation audit tracks $\mathcal{V}_T$ across all members. If $\mathcal{V}_T = 1$, the member is repaired via:
$$T_{\max, \text{repaired}}^{(k)} = \max\left(T_{\max}^{(k)}, T_{\min}^{(k)}\right)$$
The occurrence rate of such violations is logged as a primary physical diagnostic metric.

---

## 7. Compute Budget and Accelerator Resource Allocation Audit

### 7.1 Hardware Platform Specification
All Sprint 8 benchmark runs execute on Kaggle accelerators under authenticated API credentials (`rohitajitbharadwaj`):
- **Accelerators**: 2x Tesla T4 (16 GB GDDR6 per GPU, 32 GB total VRAM).
- **Weekly Allowance**: 6.0 hours (360 minutes).
- **Target Utilization**: Mixed-precision FP16 with PyTorch `torch.cuda.amp.autocast(dtype=torch.float16)`.
- **Peak VRAM Cap**: Maximum batch size capped at 12 GB per GPU to eliminate CUDA out-of-memory risks.

### 7.2 Detailed Execution Time Budget
The 360-minute GPU quota is allocated as follows:

| Campaign Phase | Runs | Steps / Member | Sample Count ($K$) | Target GPU | Time / Run | Total Time | Cumulative |
|---|---|---|---|---|---|---|---|
| **Phase 0: Regression Gate** | 1 | 4 | 1 | GPU 0 | 8 min | 8 min | 8 min |
| **Phase 1: Deterministic Anchors** | 3 | {4, 8, 16} | 1 | GPU 0 | 5 min | 15 min | 23 min |
| **Phase 2 Stage A: $K$ Scaling** | 4 | 4 | {2, 4, 8, 16} | GPUs 0, 1 | 18 min avg | 75 min | 98 min |
| **Phase 2 Stage B: $\eta$ Sweeps** | 3 | 4 | 4 ($\eta \in \{0.25, 0.5, 1.0\}$) | GPUs 0, 1 | 18 min | 55 min | 153 min |
| **Phase 3: Matched Budget 8** | 2 | (8,1), (4,2) | {1, 2} | GPU 0 | 8 min | 16 min | 169 min |
| **Phase 3: Matched Budget 16** | 3 | (16,1), (8,2), (4,4) | {1, 2, 4} | GPUs 0, 1 | 10 min | 30 min | 199 min |
| **Phase 3: Matched Budget 32** | 4 | (32,1), (16,2), (8,4), (4,8) | {1, 2, 4, 8} | GPUs 0, 1 | 14 min | 56 min | 255 min |
| **Phase 3: Matched Budget 64** | 5 | (64,1), (32,2), (16,4), (8,8), (4,16) | {1, 2, 4, 8, 16} | GPUs 0, 1 | 16 min | 80 min | 335 min |
| **Phase 4: Holdout Champion Run** | 1 | Champion $(K^*, S^*, \eta^*)$ | $K^*$ | GPUs 0, 1 | 15 min | 15 min | 350 min |
| **Reserve Buffer** | - | Bootstrap, plotting, disk I/O | - | Host CPU | 10 min | 10 min | **360 min** |

---

## 8. Seed Manifest and Random Number Generator Determinism Audit

### 8.1 Seed Manifest Architecture
To guarantee paired, reproducible comparisons across varying ensemble sizes $K$ and trajectory noise parameters $\eta$, Sprint 8 establishes an explicit nested seed architecture:

$$\text{Seed}(k, b) = \text{base\_seed} + 1000 \cdot k + b$$

where:
- $\text{base\_seed} = 20260927$ (sprint launch date).
- $k \in \{1, \dots, K\}$ is the ensemble member index.
- $b$ is the validation batch index.

### 8.2 Nested Property Invariant
For any two ensemble sizes $K_1 < K_2$:
$$\text{Seeds}(K_1) \subset \text{Seeds}(K_2)$$
For example:
- $K=2$: seeds $[s_1, s_2]$
- $K=4$: seeds $[s_1, s_2, s_3, s_4]$
- $K=8$: seeds $[s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]$

This nested property guarantees that any difference observed when moving from $K=4$ to $K=8$ is strictly attributable to the addition of members $5 \dots 8$, rather than a confounding resampling of members $1 \dots 4$.

### 8.3 Paired $\eta$ Invariant
When comparing $\eta = 0$ against $\eta \in \{0.25, 0.5, 1.0\}$, the initial latent noise $z_T^{(k)}$ is drawn using the identical seed $s_k$. Only the subsequent intermediate trajectory noise updates differ.

---

## 9. Data Leakage, Holdout Isolation, and Provenance Metadata Audit

### 9.1 Quarantined Holdout Protocol
The 2023 holdout test set represents unseen operational evaluation data.
- **Strict Quarantine**: No tuning, selection of $K$, optimization of $\eta$, or calibrator fitting is performed on 2023 data.
- **Single Shot Evaluation**: Once all 2022 validation evaluations are complete and the champion ensemble configuration is selected, exactly one evaluation run is executed on the 2023 test set.

### 9.2 Provenance Metadata Schema
Every evaluation artifact produced in Sprint 8 must persist a complete provenance header conforming to the following JSON schema:

```json
{
  "provenance": {
    "sprint": 8,
    "checkpoint_path": "models/checkpoints/sprint6_candidate3_multitask_champion.pt",
    "checkpoint_sha256": "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92",
    "parameter_count": 15685478,
    "model_architecture": "MultiTaskUNet_ResidualDiffusion",
    "sampler": "Standard_DDIM",
    "total_nfe": 32,
    "ensemble_size_K": 8,
    "denoising_steps_S": 4,
    "eta": 0.5,
    "base_seed": 20260927,
    "dataset_split": "validation_2022",
    "batch_size": 4,
    "precision": "torch.float16",
    "accelerator": "2x Tesla T4",
    "execution_timestamp": "2026-09-27T12:00:00Z"
  }
}
```

---

## 10. Audit Conclusion and Execution Readiness

1. **Model Weights**: Frozen Candidate 3 weights are verified and protected by hard failure gates.
2. **Mathematical Foundations**: Fair-CRPS, Brier Skill Score, and Spread-Skill metrics are properly formulated to eliminate finite-sample bias.
3. **Physical Transformation**: Member-by-member transformation ordering is formally proven and enforced.
4. **Compute Allocation**: 6.0 hours of 2x Tesla T4 GPU compute is realistically allocated across all matched-budget permutations.
5. **Readiness**: The specifications are complete, scientifically grounded, and approved for Phase 0 execution.
