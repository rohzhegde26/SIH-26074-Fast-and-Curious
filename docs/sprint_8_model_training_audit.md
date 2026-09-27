# Sprint 8 Model Training and Evaluation Audit: Ensemble and Test-Time Scaling Under Matched Compute Budgets

**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  
**Branch**: `feat/spatiotemporal-diffusion-downscaler`  
**Sprint**: 8 of 10 (Research Roadmap)  
**Author**: Antigravity Research Agent  
**Date**: September 27, 2026 (Revised with Review Amendments)  

---

## 1. Audit Executive Summary

Sprint 8 addresses **Ensemble and Test-Time Scaling** under matched computational budgets. Unlike Sprints 1 through 6, which investigated model architectures, parameterizations, and loss formulations, Sprint 8 is strictly an **inference-only research sprint**.

This audit serves as the primary governance document for Sprint 8, providing rigorous verification across eight core dimensions:
1. **Model Checkpoint Provenance and Weight Invariant**: Hard verification of the Sprint 6 Candidate 3 champion weights (`sprint6_candidate3_multitask_champion.pt`), confirming Git LFS hashes, parameter counts, and zero-modification enforcement.
2. **Noise Schedule Documentation Rectification**: Correction of the documented variance schedule to strictly match the codebase: linear beta schedule (`beta_start=1e-4`, `beta_end=0.035`, $T=100$).
3. **Phase 0 Reproducibility Gate**: Prescribing the strict tolerance bounds under which the deterministic DDIM-4 baseline must reproduce the Sprint 7 reference on the 2022 validation split before ensemble scaling begins.
4. **Mathematical Derivations and Domains of Probabilistic Estimators**: Rigorous formulation of finite-ensemble Fair-CRPS for $K \ge 2$, deterministic CRPS fallback ($\text{CRPS}_{\text{det}} = \text{MAE}$) for $K=1$, Brier Skill Score against training climatology, and Spread-Skill Ratio.
5. **Physical Transformation and Invariant Ordering**: Formal proof via Jensen's Inequality showing why member-wise non-linear physical clipping must strictly precede ensemble reduction, alongside diagnostic tracking of clipping rates, mass shifts, and temperature ordering repairs.
6. **Pairwise Ensemble Diversity Diagnostics**: Direct mathematical metrics to quantify inter-member spread and prevent degenerate or false ensembles.
7. **Memory-Safe Chunked Ensemble Execution**: Defining chunked batching (`ensemble_member_chunk_size` in $\{1, 2, 4\}$) to prevent VRAM exhaustion on Tesla T4 GPUs.
8. **Kaggle Compute Budget Allocation**: Operational scheduling and memory profiling across **6.0 hours of 2x Tesla T4 GPU accelerators** to complete all matched-compute sweeps ($B \in \{8, 16, 32, 64\}$) within allocation.

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

### 3.2 Diffusion Parameterization and Schedule Audit
- **Objective Formulation**: Velocity prediction ($v$-prediction) where $v_t \equiv \alpha_t \epsilon - \sigma_t x_0$.
- **Training Noise Schedule**: **Linear beta schedule**, `beta_start = 1e-4`, `beta_end = 0.035`, and $T = 100$ diffusion timesteps.
  *(Audit Correction: Previous drafts incorrectly referred to a cosine variance schedule. The underlying codebase `src/models/residual_diffusion.py` lines 331-346 defines `betas = torch.linspace(beta_start, beta_end, timesteps)`. This audit formally rectifies the documentation to match the true model weights.)*
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

### 5.1 Fair-CRPS for $K \ge 2$ and Deterministic CRPS Fallback for $K = 1$
The Continuous Ranked Probability Score for a predictive cumulative distribution $F$ and realized observation $y$ is:

$$\text{CRPS}(F, y) = \int_{-\infty}^\infty \left(F(z) - \mathbb{I}(z \ge y)\right)^2 dz$$

#### Case 1: Finite Ensemble with $K \ge 2$ (Fair-CRPS)
For an empirical ensemble of size $K \ge 2$ with members $\{y^{(1)}, \dots, y^{(K)}\}$, the standard empirical estimator has positive finite-sample bias. Sprint 8 implements the strictly unbiased Fair-CRPS estimator (Ferro et al., 2008; Gneiting & Raftery, 2007):

$$\text{CRPS}_{\text{fair}}(F_K, y) = \frac{1}{K}\sum_{k=1}^K |y^{(k)} - y| - \frac{1}{2 K (K - 1)}\sum_{k=1}^K \sum_{j=1}^K |y^{(k)} - y^{(j)}|$$

The denominator $K(K-1)$ eliminates the self-distance terms ($k=j$), guaranteeing that $\mathbb{E}[\text{CRPS}_{\text{fair}}(F_K, y)] = \text{CRPS}(F, y)$.

#### Case 2: Deterministic Reference with $K = 1$ (Deterministic CRPS = MAE)
When $K = 1$, the term $K(K-1) = 0$ in the denominator makes Fair-CRPS mathematically undefined. A single forecast represents a degenerate point distribution $F(z) = \mathbb{I}(z \ge y^{(1)})$. The integral reduces to:

$$\text{CRPS}_{\text{det}}(y^{(1)}, y) = \int_{-\infty}^\infty \left(\mathbb{I}(z \ge y^{(1)}) - \mathbb{I}(z \ge y)\right)^2 dz = |y^{(1)} - y| = \text{MAE}$$

**Mandatory Reporting Policy**:
In all matched-compute comparison tables (such as the flagship 32-NFE comparison: $K=1, S=32$ vs $K=2, S=16$ vs $K=4, S=8$ vs $K=8, S=4$), the metric for $K=1$ must be explicitly designated as $\text{CRPS}_{\text{det}}$ (= MAE). It must **never** be labeled as Fair-CRPS.

### 5.2 Brier Score and Climatological Skill Baselines
For extreme precipitation events ($P > \tau$ for $\tau \in \{15, 30\}$ mm/day), the observed event indicator is $o_i = \mathbb{I}(y_{\text{true}, i} > \tau)$ and predicted ensemble probability is $p_i = \frac{1}{K}\sum_{k=1}^K \mathbb{I}(y_i^{(k)} > \tau)$.

The Brier Score across $N$ validation instances is:

$$\text{BS} = \frac{1}{N}\sum_{i=1}^N (p_i - o_i)^2$$

To ensure rigorous meteorological attribution, Sprint 8 defines two explicit climatological baselines:
1. **Primary Reference: Training-Derived Climatology**:
   Calculated from the 2014-2021 training set for each lead $d \in \{0, \dots, 6\}$:
   $$\bar{o}_{\text{train}}(d, \tau) = \frac{1}{N_{\text{train}}}\sum_{i \in \text{Train}} \mathbb{I}(y_{\text{train}, i}(d) > \tau)$$
   $$\text{BS}_{\text{clim, train}}(d) = \bar{o}_{\text{train}}(d, \tau) \cdot (1 - \bar{o}_{\text{train}}(d, \tau))$$
   $$\text{BSS}_{\text{train}}(d) = 1 - \frac{\text{BS}(d)}{\text{BS}_{\text{clim, train}}(d)}$$
2. **Secondary Reference: Pooled Validation Climatology**:
   Calculated across the 2022 validation split $\bar{o}_{\text{val}}$, reported as a secondary reference.

Murphy's canonical decomposition is computed across $M$ probability bins:
$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

### 5.3 Direct Pairwise Ensemble Diversity Diagnostics
To prevent the deceptive failure mode of an ensemble generating identical members, the audit mandates logging four diversity diagnostics for every configuration:

1. **Mean Pairwise Member RMSE**:
   $$\text{Div}_{\text{RMSE}} = \frac{2}{K(K-1)} \sum_{k=1}^K \sum_{j=k+1}^K \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i^{(k)} - y_i^{(j)})^2}$$
2. **Mean Pairwise Spatial Correlation**:
   $$\bar{\rho}_{\text{pair}} = \frac{2}{K(K-1)} \sum_{k=1}^K \sum_{j=k+1}^K \text{Corr}\left(y^{(k)}, y^{(j)}\right)$$
3. **Ensemble Spatial Variance**:
   $$\bar{\sigma}_{\text{ens}}^2 = \frac{1}{N} \sum_{i=1}^N \left(\frac{1}{K-1}\sum_{k=1}^K (y_i^{(k)} - \mu_i)^2\right)$$
4. **Effective Diversity Ratio (EDR)**:
   $$\text{EDR}(K) = \frac{\text{Div}_{\text{RMSE}}(K)}{\text{Div}_{\text{RMSE}}(K=2)}$$

### 5.4 Spread-Skill Ratio and Fair Spread
For finite ensemble size $K$, the sample variance per instance $i$ is:
$$s_i^2 = \frac{1}{K - 1}\sum_{k=1}^K \left(y_i^{(k)} - \mu_i\right)^2, \quad \mu_i = \frac{1}{K}\sum_{k=1}^K y_i^{(k)}$$
$$\text{Ensemble Spread} = \sqrt{\frac{1}{N}\sum_{i=1}^N s_i^2}, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N \left(\mu_i - y_{\text{true}, i}\right)^2}$$
$$\text{SSR} = \frac{\text{Ensemble Spread}}{\text{RMSE}}$$

### 5.5 Multivariate Energy Score
To assess inter-variable correlation structures (e.g., wind vector $(U, V)$ consistency, precipitation-humidity coupling):
$$\text{ES}(F_K, \mathbf{y}) = \frac{1}{K}\sum_{k=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}\|_2 - \frac{1}{2 K (K - 1)}\sum_{k=1}^K \sum_{j=1}^K \|\mathbf{y}^{(k)} - \mathbf{y}^{(j)}\|_2$$

---

## 6. Test-Time Physical Safeguards, Invariants, and Diagnostics

### 6.1 Member-Wise Transformation Ordering Invariant
All denormalization and physical constraint enforcements must be executed on individual ensemble members **prior** to computing ensemble summary statistics:

```
CORRECT (Member-Wise Invariant):
Raw Latents y_norm^(k) 
  --> Unstandardize to Physical Space y_phys^(k) 
  --> Member-Wise Non-Linear Physical Bounds [P >= 0, 0 <= RH <= 100, Tmin <= Tmax]
  --> Compute Ensemble Mean, Quantiles, Spread, P(P > thresh), Fair-CRPS

INCORRECT (Ensemble-First Violation):
Raw Latents y_norm^(k) 
  --> Average in Latent/Normalized Space 
  --> Unstandardize 
  --> Apply Non-Linear Bounds (CORRUPTS PROBABILISTIC DISTRIBUTIONS)
```

### 6.2 Mathematical Analysis of Non-Linear Clipping
Let $X \in \mathbb{R}$ represent an unconstrained precipitation prediction before non-negativity clipping, and let $g(x) = \max(0, x)$ be the ReLU physical projection. The function $g(x)$ is convex.

By **Jensen's Inequality**:
$$\mathbb{E}[g(X)] \ge g(\mathbb{E}[X])$$

Because $g(x)$ is non-linear, $\mathbb{E}[\max(0, X)] \neq \mathbb{E}[X]$ in regions where $X$ takes negative values. Member-wise clipping does **not** preserve an unbiased rainfall expectation; rather, it guarantees that every individual ensemble member represents a physically realizable state ($P \ge 0$) and that threshold exceedance probabilities $P(P > \tau) = \frac{1}{K}\sum \mathbb{I}(P^{(k)} > \tau)$ follow correct probabilistic semantics.

### 6.3 Mandatory Physical Repair Burden Tracking
To turn potential distribution shifts into transparent scientific metrics, the audit mandates tracking:
1. **Pre-Repair Violation Rates**:
   - Fraction of precipitation grid cells with $\tilde{P} < 0$.
   - Fraction of RH grid cells outside $[0, 100\%]$.
   - Fraction of temperature cells with $\tilde{T}_{\min} > \tilde{T}_{\max}$.
2. **Precipitation Mass Shift**:
   $$\Delta M_{\text{precip}} = \frac{\sum \max(0, \tilde{P}) - \sum \tilde{P}}{\sum \max(0, \tilde{P})} \times 100\%$$
3. **Ensemble Mean Shift**:
   $$|\Delta \mu_P| = \frac{1}{N}\sum_{i=1}^N \left| \frac{1}{K}\sum_{k=1}^K \max(0, \tilde{P}_i^{(k)}) - \max\left(0, \frac{1}{K}\sum_{k=1}^K \tilde{P}_i^{(k)}\right) \right|$$
4. **Thermodynamic Ordering Modification**:
   Logging the empirical frequency of $T_{\max} = \max(T_{\max}, T_{\min})$ and mean absolute adjustment magnitude $|\Delta T_{\max}|$, as this is a distribution-altering operation.

---

## 7. Compute Budget, Memory Chunking, and Resource Allocation

### 7.1 Memory-Safe Chunked Ensemble Execution
To eliminate CUDA out-of-memory risks on 16 GB Tesla T4 GPUs:
- Parameter `ensemble_member_chunk_size` is constrained to $C_{\text{ens}} \in \{1, 2, 4\}$.
- For evaluation cubes, members are generated in chunks of size $C_{\text{ens}}$. Intermediate latents are cleared from VRAM immediately upon physical conversion.
- Peak VRAM is capped at $< 12$ GB per GPU.

### 7.2 Detailed Execution Time Budget (6.0 Hours Total)

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
| **Reserve Buffer** | - | Bootstrap, plotting, disk I/O | - | Host CPU | 10 min | 10 min | **360 min (6.0 h)** |

---

## 8. Seed Manifest and RNG Determinism Audit

### 8.1 Seed Manifest Architecture
$$\text{Seed}(k, b) = \text{base\_seed} + 1000 \cdot k + b$$
where $\text{base\_seed} = 20260927$, $k \in \{1, \dots, K\}$, and $b$ is the validation batch index.

### 8.2 Nested and Paired Invariants
- **Nested Property**: $\text{Seeds}(K_1) \subset \text{Seeds}(K_2)$ for $K_1 < K_2$.
- **Paired $\eta$ Invariant**: Identical initial latent noise seeds $s_k$ used across all $\eta$ values.

---

## 9. Data Leakage, Holdout Isolation, and Provenance Metadata Audit

### 9.1 Quarantined Holdout Protocol
The 2023 holdout test set is quarantined. No exploratory parameter sweeps, threshold tuning, or hyperparameter selection are conducted on 2023 data. Exactly one evaluation run is executed on 2023 data after all validation evaluations are finalized.

### 9.2 Provenance Metadata Schema
Every evaluation artifact produced in Sprint 8 must persist a complete provenance header:

```json
{
  "provenance": {
    "sprint": 8,
    "checkpoint_path": "models/checkpoints/sprint6_candidate3_multitask_champion.pt",
    "checkpoint_sha256": "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92",
    "parameter_count": 15685478,
    "model_architecture": "MultiTaskUNet_ResidualDiffusion",
    "noise_schedule": "linear_beta_1e-4_to_0.035_T100",
    "sampler": "Standard_DDIM",
    "total_nfe": 32,
    "ensemble_size_K": 8,
    "denoising_steps_S": 4,
    "eta": 0.5,
    "base_seed": 20260927,
    "ensemble_member_chunk_size": 4,
    "dataset_split": "validation_2022",
    "precision": "torch.float16",
    "accelerator": "2x Tesla T4",
    "execution_timestamp": "2026-09-27T12:00:00Z"
  }
}
```

---

## 10. Audit Conclusion and Execution Readiness

1. **Noise Schedule Rectified**: Documented as linear beta schedule (`beta_start=1e-4`, `beta_end=0.035`, $T=100$), exactly matching code.
2. **Estimator Domains Clarified**: Fair-CRPS strictly defined for $K \ge 2$; $K=1$ fallback formally designated as $\text{CRPS}_{\text{det}} = \text{MAE}$.
3. **Physical Transformation Validated**: Member-wise clipping Jensen analysis corrected; physical repair burden tracking established.
4. **Diversity Diagnostics Integrated**: Pairwise RMSE and correlation metrics added to detect degenerate ensembles.
5. **Memory Safety Established**: Chunked batching ($C_{\text{ens}} \le 4$) prevents OOM risks.
6. **Execution Readiness**: All review amendments are fully integrated; documentation is ready for stakeholder sign-off prior to implementation.
