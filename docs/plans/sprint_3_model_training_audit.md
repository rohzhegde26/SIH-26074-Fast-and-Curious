# Sprint 3 Model Training Audit: Architecture, Data, and Research Readiness

**Project:** SIH 26074 Weather Downscaling  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`

This audit evaluates the Sprint 3 plan against:
1. the frozen Sprint 1-2 data pipeline
2. the previously agreed temporal Transformer + spatial ConvNeXt/U-Net diffusion architecture
3. the Deep Research review supplied for Sprint 3

---

## 1. Executive Verdict

The revised Sprint 3 plan is ready to implement, with explicit gates.

The key scientific conclusion is:

**Sprint 2 is sufficient for the deterministic baseline and the core conditional residual diffusion model, but it is not sufficient for every longer-term architecture experiment discussed before the sprints.**

Supported now:
- 7-day forecasting/downscaling
- 6-channel joint output
- U/V wind prediction
- three-day historical context
- history-to-future cross-attention
- temporal Transformer across seven future leads
- terrain conditioning
- conditional residual diffusion
- diffusion-step scaling
- forecast-initialized refinement
- dense model-size scaling
- selective MoE
- ensemble-size versus denoising-step studies

Not supported by the frozen artifact:
- H>3 history studies
- true N/M surrounding-area context
- prior forecast trajectories inside history
- vertical lapse-rate supervision

Those must be explicit future dataset extensions, not inferred from current tensors.

---

## 2. Alignment With the Agreed Architecture

| Architecture objective | Frozen Sprint 2 support | Sprint 3 implementation | Assessment |
|---|---|---|---|
| Historical context | 3 days of six-variable history | H=1/2/3 adapter | Supported |
| Historical forecast context | Not present | Optional future interface only | Requires data extension |
| Future forecast conditioning | 7-day authentic GFS | Dedicated future encoder | Supported |
| 7-day temporal modeling | 7 explicit lead days | Lead embeddings + temporal attention | Supported |
| History-to-future interaction | History and future tensors available | Explicit cross-attention | Supported |
| Spatial downscaling | 16x16 to 80x80 | ConvNeXt/U-Net | Supported |
| Terrain conditioning | 5x80x80 static channels | Multi-scale terrain fusion | Supported |
| Joint multivariate model | 6 channels | One joint output tensor | Supported |
| Wind representation | U/V | Separate joint U/V outputs | Supported |
| Residual diffusion | Fine reference + coarse GFS | Model-space residual diffusion | Supported |
| Forecast-conditioned refinement | Future forecast available | Separate inference mode | Supported |
| Diffusion-step scaling | Sampler configurable | 4/8/16/32 | Supported |
| Model-size scaling | No new data required | small/base/large | Supported |
| Sparse MoE | No new data required | temporal/bottleneck only | Supported |
| Ensemble vs denoising compute | Multiple diffusion seeds possible | fixed-budget comparison | Supported |
| Larger N/M context | Not in frozen Zarr | future data revision | Not yet supported |
| H>3 | Not in frozen Zarr | future data revision | Not yet supported |

The supplied Deep Research report describes the current plan as broadly aligned but recommends clarifications around temporal handling, joint six-channel output, baselines, transforms, splits, leakage, metrics, normalization, and resource feasibility. Those recommendations are incorporated here. fileciteturn165file0L5-L19

---

## 3. Sprint 1-2 Data Integrity Audit

### 3.1 Split integrity

The trainer must use the explicit split field from `data/sample_index.parquet`.

Required:
- train = 2015-2021, 854 samples
- validation = 2022, 122
- test = 2023, 122

Do not slice the first 854 rows by position. Use the stored `split` labels.

The test partition is never used for:
- early stopping
- hyperparameter selection
- architecture selection
- normalization fitting

### 3.2 Leakage

The model input must satisfy:
- history = D-3, D-2, D-1 only
- future GFS = D...D+6
- no target-derived tensor reaches the conditioning path
- no test sample reaches training

A model-level unit test should assert the temporal boundaries from sample metadata.

### 3.3 Forecast provenance

The trainer must consume the frozen Zarr artifact.

It must not:
- call the GFS downloader during training
- rebuild forecast fields
- substitute ERA5, CHIRPS, or ERA5-Land when GFS is missing
- use station interpolation

The existing Sprint 2 Gate 7 provenance contract remains authoritative.

### 3.4 Manifest integrity

At training start, record:
- dataset version
- exact `builder_git_commit` from the frozen `data/dataset_manifest.yaml`
- exact preprocessing `config_sha256`
- normalization file identifier/hash

The check is against the frozen dataset artifact, not against the current Sprint 3 source commit.

---

## 4. Normalization Audit

The Deep Research recommendation to use the exact Sprint 2 train-fit normalization is accepted.

Required:
- load `data/normalization_stats.yaml`
- do not fit statistics in the trainer
- do not refit on validation/test
- use the same transforms for history, future forecast, and target channels where the Sprint 2 contract specifies them

One-time audit:
1. scan the complete training partition or a documented full subset
2. verify transform type per channel
3. verify the stored mean/std against the actual train representation within a declared floating-point tolerance

Do not assert that each minibatch mean/std exactly matches the global training statistics. That would be an incorrect test because minibatch statistics naturally vary.

---

## 5. Architecture Audit

### 5.1 Expected model I/O

Expected:
- history: [B,H,6,16,16]
- future forecast: [B,7,6,16,16]
- terrain: [B,5,80,80]
- output: [B,7,6,80,80]

For H=3 this exactly matches the frozen Sprint 2 contract.

### 5.2 Temporal structure

The seven future leads must remain an explicit temporal dimension.

Required checks:
- lead-day embedding exists
- future lead representations retain ordering
- temporal self-attention receives the seven leads
- history-to-future cross-attention exists
- gradients reach the temporal modules

Do not implement the model as a flat 42-channel image and then describe it as a temporal Transformer. A 2D spatial backbone may be shared across leads, but temporal attention must remain an explicit operation.

### 5.3 Spatial structure

Required:
- 80x80 target resolution
- multi-scale feature hierarchy
- ConvNeXt/U-Net style skip connections
- terrain fusion
- fine spatial output

### 5.4 Joint six-channel prediction

All six channels are part of one joint output:
- precipitation
- Tmax
- Tmin
- RH
- U
- V

Wind speed/direction are derived diagnostics only.

---

## 6. Loss Audit

### Deterministic model

The loss must contain all six target channels.

The exact implementation should be traceable from:
- configured per-channel loss
- task weights
- regularizer weights

### Diffusion model

The declared Sprint 3 v1 objective is:

`MSE(epsilon, epsilon_theta(r_t,t,c))`

where `r_t` is formed from the model-space residual.

Do not combine DDPM and EDM parameterizations in one run.

### Physical-space checks

The following are evaluated after inverse transform:
- precipitation non-negativity
- Tmax/Tmin ordering
- RH range
- wind physical values
- conservation error

The audit must not claim that z-score-space activations are physical constraints.

The Deep Research report specifically calls out positivity, normalized loss, and de-normalized evaluation as required checks. fileciteturn165file0L11-L25

---

## 7. Baseline Audit

Before diffusion:

### Baseline 0
Non-learned coarse-to-fine interpolation.

### Baseline 1
Deterministic learned U-Net/ConvNeXt refinement.

### Baseline 2
Single-step deterministic residual refinement matching the diffusion conditioning path.

These baselines answer whether the stochastic diffusion machinery adds value over simple resolution conversion and deterministic learned correction. The Deep Research review explicitly recommends simple upsampling and a single-step UNet-style comparator. fileciteturn165file0L7-L10

---

## 8. Metrics Audit

Two metric spaces are mandatory.

### Training/optimization space
- normalized-space loss
- exact Sprint 2 transforms

### Scientific reporting space
- inverse-transformed physical units
- de-logged precipitation
- interpretable temperature/RH/wind errors

Core metrics:
- precipitation MAE/RMSE, wet-day MAE, bias
- CSI@15, CSI@30
- spatial event metric such as FSS
- temperature MAE/RMSE/bias
- RH MAE/RMSE/bias
- wind U/V and vector RMSE
- conservation error
- CRPS and Brier score for stochastic outputs

Report per lead D...D+6 and aggregate.

The report's recommendation to include de-normalized metrics and qualitative held-out comparisons is incorporated. fileciteturn165file0L11-L28

---

## 9. Qualitative Verification Audit

For validation and final test reporting, save side-by-side:
- coarse/interpolation field
- reference target
- deterministic prediction
- diffusion mean/median
- multiple ensemble members
- residual field

Also save:
- precipitation histograms or quantiles
- extreme-event case plots

Numerical metrics alone are not sufficient to evaluate downscaling texture and spatial event structure.

---

## 10. Reproducibility Audit

Required:
- fixed Python/NumPy/PyTorch seeds
- deterministic settings where practical
- configuration snapshot
- model architecture configuration
- dataset manifest identifiers
- preprocessing config hash
- normalization identifier
- environment/PyTorch version
- GPU metadata
- wall-clock time
- peak GPU memory

The Deep Research report explicitly recommends fixed seeds and explicit environment/GPU checks. fileciteturn165file0L41-L45

---

## 11. Kaggle Resource Audit

The previous static GPU budget table is treated as a planning estimate, not a guaranteed timing model.

Before each run:
1. query the live quota
2. record remaining seconds
3. run a one-epoch timing probe for a new architecture
4. select the number of epochs based on measured runtime

Initial pilot:
- 5-10 epochs with early stopping for baseline
- 5-10 epochs with early stopping for diffusion

Longer runs are allowed only when the measured time fits the remaining quota.

The report's recommendation to keep training modest under the approximately six-hour GPU budget is incorporated, while avoiding an unsupported assumption about exact minutes per epoch. fileciteturn165file0L23-L27

### Hardware check

The trainer must detect:
- number of GPUs
- GPU model
- available VRAM

Never assume two T4 devices or silently run on CPU.

### Checkpoint preservation

Save to:
- `/kaggle/working/models/`
- `/kaggle/working/reports/`

Before job termination verify:
- checkpoint exists
- metrics exist
- config exists
- sample visualizations exist where requested

The Deep Research review specifically recommends preserving checkpoints and outputs before the session ends. fileciteturn165file0L43-L47

---

## 12. Diffusion Research Audit

### Diffusion horizon

Sprint 3 v1:
- T=100 training diffusion steps
- 4/8/16/32-step DDIM inference comparison

This is a feasibility-oriented starting point. The research question is how much inference compute improves the forecast, not whether a particular step count is assumed optimal.

### Initialization study

Compare:
- standard Gaussian residual initialization
- forecast-conditioned intermediate-noise refinement

Use the same trained model family and matched inference budgets.

### History length

Current:
- H=1/2/3

Do not report H>3 until the historical dataset is expanded.

### N/M context

Do not report a true surrounding-area context study from the current 16x16 crop.

This is an important reconciliation with the Deep Research report: its description of a domain with up to 2.5x spatial context is not evidence that the frozen Zarr currently contains that context. The repository's current Sprint 2 tensors must remain the source of truth.

### Model-size scaling

Small/base/large after core model stability.

### MoE

Only after dense scaling is stable.

Start at temporal/deep bottleneck blocks.

### Ensemble versus denoising depth

At fixed inference compute:
- more denoising steps
- more ensemble members

Primary probabilistic metrics:
- CRPS
- Brier score
- coverage
- spread-skill
- extreme event probability quality

---

## 13. Research Gates

### Gate A: data/model contract

Pass:
- split integrity
- no leakage
- provenance intact
- normalization intact
- all shape tests pass

### Gate B: deterministic baseline

Pass:
- reproducible validation metrics
- qualitative maps
- interpolation comparison

### Gate C: diffusion

Pass:
- reproducible checkpoint
- valid epsilon loss
- physical metrics
- stochastic sample diversity

### Gate D: denoising depth

Pass:
- 4/8/16/32 comparison
- skill/runtime/calibration recorded

### Gate E: initialization

Pass:
- Gaussian vs forecast-initialized refinement comparison

### Gate F: context/history

Pass only with appropriate data:
- H>3 requires additional history
- N/M requires wider coarse context

### Gate G: scaling

Pass:
- dense capacity results
- MoE results
- fixed-budget uncertainty results

---

## 14. Sprint 3 Exit Criteria

Sprint 3 can move to the next research sprint when:

1. deterministic baselines are reproduced
2. conditional residual diffusion trains successfully
3. data contracts remain intact
4. normalized training losses and de-normalized scientific metrics are logged
5. per-lead metrics exist
6. qualitative held-out comparisons exist
7. 4/8/16/32 denoising comparison exists
8. forecast-initialized refinement is tested or explicitly deferred
9. current-data limits for H>3 and N/M are documented
10. no synthetic/reanalysis forecast fallback appears
11. Sprint 2 regression tests remain green

---

## 15. Final Go/No-Go

**GO for Sprint 3 implementation, with gates.**

The correct interpretation is:
- the frozen Sprint 2 artifact is strong enough for the core deterministic and diffusion model
- the temporal Transformer/history-to-future cross-attention design should be explicit in code, not merely described
- normalized losses and physical-unit reporting must coexist
- the current data do not justify H>3, N/M, or prior-forecast-history claims
- diffusion, model-size, sparse-MoE, and ensemble allocation studies should be isolated and compute-budgeted

This keeps the Sprint 3 model scientifically continuous with Sprints 1-2 and with the longer-term architecture research objective.
