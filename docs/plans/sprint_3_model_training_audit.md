# Sprint 3: Model Training Audit & Spatiotemporal Architecture

**Project**: SIH Problem Statement 26074 (Multivariate Spatiotemporal Weather Downscaler)  
**Target Domain**: Mandya District, Karnataka, Peninsular India (11.0°N to 15.0°N, 74.0°E to 78.0°E)  
**Grid Transformation**: Coarse 0.25° (16x16) -> Fine 0.05° (80x80) [5x Spatial Downscaling]  
**Temporal Extent**: 7 Forecast Lead Days (D through D+6) conditioned on 3 Antecedent Days (D-3 through D-1)  

---

## 1. Sprint 2 Data-Freeze Sanity Check

Before initiating model development, the Sprint 2 data foundation was audited across all storage layers:

### A. Zarr Store Verification (`datasets/multitask_temporal_v1.zarr`)
- **Total Samples**: Exactly 1,098 samples across 9 monsoon seasons (2015 to 2023, 122 days/season).
- **History Array**: Shape `(1098, 3, 6, 16, 16)`, dtype `float32`, chunk size `(1, 3, 6, 16, 16)`.
- **Forecast Array**: Shape `(1098, 7, 6, 16, 16)`, dtype `float32`, chunk size `(1, 7, 6, 16, 16)`.
- **Target Array**: Shape `(1098, 7, 6, 80, 80)`, dtype `float32`, chunk size `(1, 7, 6, 80, 80)`.
- **Terrain Array**: Shape `(5, 80, 80)`, dtype `float32`, chunk size `(5, 80, 80)`.
- **Disk Footprint**: 976.6 MB across 3,330 chunk files.
- **Quarantine Check**: 2014 pre-operational archive strictly quarantined (0 samples from 2014 present in forecast contract).

### B. Catalog & Provenance Integrity (`data/sample_index.parquet`)
- **Sample Count**: Exactly 1,098 rows.
- **Split Distribution**: Train = 854 (2015-2021), Val = 122 (2022), Test = 122 (2023).
- **QA Status**: 1,098 / 1,098 samples marked `PASSED` (100%).
- **GFS Provenance Breakdown**:
  - `ncar_rda_ds084_1` via `ncar_thredds_subset`: 732 samples (monsoons 2015-2020).
  - `aws_open_data` via `aws_http_range`: 366 samples (monsoons 2021-2023).
  - Synthetic simulation fallback: 0 samples (zero fallback policy strictly enforced).
  - Every sample records verified remote source GRIB URLs/keys in `gfs_source_files`.

### C. Invertible Normalization (`data/normalization_stats.yaml`)
- Parameters fitted strictly on the `train` partition (854 samples, 5,978 daily lead observations).
- Channel 0 (Precipitation): `log1p_zscore` (mean = 0.7670, std = 1.2793, physical range [0.0, 500.0] mm).
- Channel 1 (Tmax): `zscore` (mean = 31.21 °C, std = 2.33 °C, physical range [23.78, 38.87] °C).
- Channel 2 (Tmin): `zscore` (mean = 22.57 °C, std = 2.43 °C, physical range [14.81, 30.38] °C).
- Channel 3 (RH): `zscore` (mean = 70.60 %, std = 3.82 %, physical range [58.68, 84.60] %).
- Channel 4 (Wind U): `zscore` (mean = +11.78 m/s, std = 6.75 m/s, physical range [-19.47, 45.43] m/s).
- Channel 5 (Wind V): `zscore` (mean = +1.07 m/s, std = 4.61 m/s, physical range [-23.73, 40.25] m/s).
- Invertible round-trip error certified < 1e-5.

---

## 2. Model Training Audit: Architectural Analysis

### A. The Core Scientific Challenge
The goal is 5x spatial downscaling (0.25° to 0.05°) over a 7-day forecast horizon. This requires solving two coupled problems:
1. **Dynamic Synoptic Propagation**: Translating coarse NWP trajectories across lead days $D \dots D+6$ into local weather evolution, informed by recent antecedent state ($D-3 \dots D-1$).
2. **Topographic & Microclimatic Disaggregation**: Injecting high-resolution orographic details (slope, elevation, windward lift) into coarse atmospheric fields while preserving physical mass and energy conservation.

### B. Why Phased Progression: Baseline -> Residual Diffusion
Direct end-to-end training of an unguided diffusion model from pure noise across 7 lead days $\times$ 6 variables ($7 \times 6 \times 80 \times 80 = 268,800$ continuous dimensions) is computationally heavy and inefficient:
- The network would waste thousands of iterations learning coarse synoptic geography and large-scale temperature gradients that the coarse GFS forecast already provides.
- Benchmarking the value of diffusion requires an uncompromised, high-quality deterministic baseline to quantify the exact marginal gain of stochastic generative sampling.

Therefore, our architecture implements a two-phase hierarchy:
1. **Phase 1: Deterministic Spatiotemporal Baseline (`TemporalMultiTaskUNet5x`)**:
   - Computes deterministic multivariate prediction $\hat{\mathbf{y}}_{\text{det}} \in \mathbb{R}^{B \times 7 \times 6 \times 80 \times 80}$.
   - Establishes canonical benchmark metrics: All-Day MAE, Wet-Day MAE, RMSE, CSI@15, CSI@30, Pearson correlation, and mass conservation error.
2. **Phase 2: Conditional Residual Diffusion (`SpatiotemporalDiffusionDownscaler`)**:
   - Formulates diffusion on the fine-scale spatial residual:
     $$\mathbf{r} = \mathbf{y}_{\text{fine}} - \mathbf{y}_{\text{coarse\_interp}}$$
   - Reverse diffusion starts from Gaussian noise $\mathbf{r}_T \sim \mathcal{N}(0, \mathbf{I})$ and denoises to recover $\mathbf{r}_0$, conditioned on:
     $$\mathbf{c} = \{\mathbf{x}_{\text{history}}, \mathbf{x}_{\text{forecast\_coarse}}, \mathbf{x}_{\text{terrain}}\}$$
   - Output prediction:
     $$\hat{\mathbf{y}} = \mathbf{y}_{\text{coarse\_interp}} + \mathbf{r}_\theta$$
   - Advantages:
     * Fast convergence: The coarse synoptic structure is preserved by construction.
     * Generative focus: 100% of diffusion capacity models fine convective cells, orographic wind channeling, and sharp microclimatic fronts.
     * Compact test-time sampling: 8 to 16 DDIM steps produce calibrated ensemble members.

---

## 3. Detailed Tensor Dimensions & Conditioning Pathways

| Tensor Role | Tensor Name | Input Shape | Normalized Space | Physical Space |
| :--- | :--- | :---: | :---: | :---: |
| **Antecedent History** | `history` | `[B, 3, 6, 16, 16]` | log1p-zscore (P), zscore (T, RH, UV) | mm, °C, %, m/s |
| **Coarse NWP Forecast** | `future_forecast` | `[B, 7, 6, 16, 16]` | log1p-zscore (P), zscore (T, RH, UV) | mm, °C, %, m/s |
| **Coarse Interp Base** | `forecast_interp` | `[B, 7, 6, 80, 80]` | Bilinear 5x upsampling of coarse NWP | mm, °C, %, m/s |
| **High-Res Topography** | `terrain` | `[B, 5, 80, 80]` | Min-max scaled: elev, slope, aspect, lift | [0, 1] |
| **Target Supervision** | `target` | `[B, 7, 6, 80, 80]` | Normalized ground truth (CHIRPS + ERA5) | mm, °C, %, m/s |
| **Diffusion Timestep** | `t` | `[B]` | Uniform integer $t \in [1, T]$ | Scaled noise level $\sigma_t$ |

### Conditioning Flow:
1. **Temporal Context Encoding**:
   - `history` `[B, 3, 6, 16, 16]` passes through a 1D temporal convolution + spatial projection to yield temporal context tokens $\mathbf{h}_{\text{hist}} \in \mathbb{R}^{B \times d_{\text{ctx}}}$.
2. **Spatial Alignment**:
   - Coarse forecast `[B, 7, 6, 16, 16]` is upsampled 5x via bilinear interpolation to match target resolution `[B, 7, 6, 80, 80]`.
3. **Multi-Scale Topographic Fusion**:
   - Terrain channels `[B, 5, 80, 80]` are concatenated with the upsampled atmospheric fields and projected into latent feature space via ConvNeXt blocks.
4. **Lead-Time Cross-Attention**:
   - At the bottleneck resolution ($10 \times 10$), multi-head cross-attention attends across the 7 forecast lead days, enabling dynamic synoptic information flow between consecutive lead steps.

---

## 4. Physics Constraints & Loss Formulation

### A. Mathematical Loss Definition
The composite loss balances score matching / regression with exact physical conservation laws:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{mass}} \mathcal{L}_{\text{mass}} + \lambda_{\text{thermo}} \mathcal{L}_{\text{thermo}} + \lambda_{\text{lapse}} \mathcal{L}_{\text{lapse}}$$

Where:
1. **Multi-Task Task Loss ($\mathcal{L}_{\text{task}}$)**:
   - For Deterministic Baseline: Balanced via Kendall homoscedastic uncertainty log-variances:
     $$\mathcal{L}_{\text{task}} = \sum_{k=1}^5 \frac{1}{2} \exp(-s_k) \mathcal{L}_k + \frac{1}{2} s_k$$
     * Precipitation ($\mathcal{L}_{\text{rain}}$): Log-Cosh + Asymmetric Quantile Pinball ($\tau = 0.90$):
       $$\mathcal{L}_{\text{pinball}}(y, \hat{y}) = \max(\tau (y - \hat{y}), (\tau - 1)(y - \hat{y}))$$
     * Temperatures ($\mathcal{L}_{T_{\max}}, \mathcal{L}_{T_{\min}}$): Huber loss with $\beta = 1.0$.
     * Relative Humidity ($\mathcal{L}_{\text{rh}}$): L1 loss.
     * Wind Vector ($\mathcal{L}_{\text{wind}}$): Log-Cosh on vector magnitude $\|(u - \hat{u}, v - \hat{v})\|$.
   - For Diffusion: Mean Squared Error on residual noise:
     $$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{r}_0, \epsilon} [\|\epsilon - \epsilon_\theta(\mathbf{r}_t, t, \mathbf{c})\|^2]$$

2. **Area-Weighted Mass Conservation ($\mathcal{L}_{\text{mass}}$)**:
   - Evaluated strictly in physical precipitation space (mm):
     $$\hat{P}_{\text{phys}} = \text{expm1}(\hat{P}_{\text{norm}} \cdot \sigma_P + \mu_P)$$
   - Coarsening operator $\mathcal{C}$ aggregates $5 \times 5$ fine cells with cosine of latitude weighting:
     $$\mathcal{C}[\hat{P}_{\text{phys}}] = \frac{\text{avg\_pool2d}(\hat{P}_{\text{phys}} \cdot \cos(\text{lat}), k=5, s=5)}{\text{avg\_pool2d}(\cos(\text{lat}), k=5, s=5)}$$
   - Penalty:
     $$\mathcal{L}_{\text{mass}} = \frac{1}{7} \sum_{l=0}^6 \|\mathcal{C}[\hat{P}_{\text{phys}, l}] - P_{\text{coarse}, l}\|_2^2$$

3. **Diurnal Temperature Spread Guarantee ($\mathcal{L}_{\text{thermo}}$)**:
   - Guaranteed structurally in the model head:
     $$T_{\max} = T_{\min} + \text{Softplus}(w_{\Delta T})$$
   - Supplemented by soft penalty in physical space:
     $$\mathcal{L}_{\text{thermo}} = \frac{1}{7} \sum_{l=0}^6 \text{mean}(\text{ReLU}(T_{\min, l} - T_{\max, l})^2)$$

4. **Orographic Lapse Rate Bounds ($\mathcal{L}_{\text{lapse}}$)**:
   - Constrains $-\partial T/\partial z$ on steep terrain ($|\nabla z| > 20\,\text{m/cell}$) to meteorological bounds $[0.004, 0.0098] \,^\circ\text{C/m}$.

---

## 5. Kaggle Remote Training Infrastructure & Budget

### A. Infrastructure Audit
- **Authentication**: Kaggle API authenticated via Personal Access Token at `~/.kaggle/access_token`.
- **Username**: `rohitajitbharadwaj`.
- **Live Quota**:
  - GPU: Exactly 6.0 hours (21,600 seconds) remaining, refreshes 2026-09-26.
  - TPU: 20.0 hours / week available.
- **Hardware Targets**:
  - Primary: Nvidia Dual Tesla T4 GPUs (32 GB total VRAM).
  - Secondary: TPU VM v3-8 (128 GB TPU HBM).

### B. Training Run Budgeting
| Phase | Experiment Name | Epochs | Batch Size | Estimated Time | GPU Budget Cost | Remaining Allowance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1** | `temporal-baseline-t4-v1` | 25 | 8 | ~16 mins | 0.27 hrs | 5.73 hrs |
| **Phase 1b** | `temporal-baseline-tuning` | 25 | 8 | ~16 mins | 0.27 hrs | 5.46 hrs |
| **Phase 2** | `spatiotemporal-diffusion-v1`| 35 | 8 | ~22 mins | 0.37 hrs | 5.09 hrs |
| **Phase 2b** | `diffusion-ddim-sampler` | 35 | 8 | ~22 mins | 0.37 hrs | 4.72 hrs |

### C. Dispatch Workflow via `scripts/kaggle/`
1. `prepare_kernel_bundle()`: Copies models, losses, and training script into `.kaggle_staging`.
2. `kernel-metadata.json`: Sets `"enable_gpu": true`, `"machine_shape": "NvidiaTeslaT4"`, and attaches dataset.
3. `kaggle kernels push`: Initiates remote execution.
4. Remote Polling: Tracks stdout and GPU status every 20 seconds.
5. Checkpoint & Report Retrieval: Downloads `models/checkpoints/` and `reports/` upon completion.
