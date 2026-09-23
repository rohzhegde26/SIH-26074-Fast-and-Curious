# Sprint 3 Implementation Plan: Multivariate Spatiotemporal Downscaler and Diffusion Training

This plan translates the previously agreed model architecture into an executable Sprint 3 implementation while preserving the frozen Sprint 1-2 data contracts.

The intended research architecture is:

**historical context encoder + future NWP encoder + history-to-future cross-attention + temporal Transformer + spatial ConvNeXt/U-Net hybrid + conditional residual diffusion**

The design also preserves the longer-term research axes agreed before the sprint sequence:
- history-length scaling
- surrounding-area context ratio N/M
- diffusion denoising-step scaling
- model-capacity scaling
- selective MoE sparsity
- inference-budget allocation between more denoising steps and more ensemble members

Sprint 3 must distinguish between axes supported by the frozen dataset and axes that require a deliberate dataset revision.

---

## 1. Frozen Sprint 2 Data Contract

The Sprint 2 artifact is treated as immutable for Sprint 3 training.

- Zarr: `datasets/multitask_temporal_v1.zarr`
- Samples: 1,098
- Train: 854 samples from 2015-2021
- Validation: 122 samples from 2022
- Test: 122 samples from 2023
- History: `[B, 3, 6, 16, 16]`
- Future GFS forecast: `[B, 7, 6, 16, 16]`
- Fine reference target: `[B, 7, 6, 80, 80]`
- Terrain: `[B, 5, 80, 80]`
- Weather channels: precipitation, Tmax, Tmin, RH, U, V
- Future forecast source: authentic NOAA GFS
- Forecast fallback: none
- Normalization: exact train-fitted `data/normalization_stats.yaml`

The fine target is a high-resolution reference/supervision product assembled from CHIRPS and ERA5/ERA5-Land. It is not dense ground-truth telemetry. Station data remain independent point validation only.

### 1.1 What Sprint 2 supports

The frozen dataset supports:
1. three antecedent days of historical meteorological context
2. seven future daily GFS forecast leads
3. high-resolution terrain conditioning
4. joint six-variable supervision at 80x80

The frozen dataset does **not** contain:
- archived prior forecast trajectories inside the history tensor
- a larger surrounding coarse context crop for N/M experiments
- vertical atmospheric temperature profiles

Therefore:
- Sprint 3 v1 uses authentic historical weather as the history stream and authentic GFS as future conditioning.
- The model interface should remain extensible to an optional historical-forecast substream, but Sprint 3 must not synthesize or reconstruct one from targets.
- H>3 and true N/M context experiments are future data-extension tracks.

---

## 2. Sprint 3 Research Execution Order

### Stage A: deterministic reference

1. Validate the frozen data contract in the trainer.
2. Implement non-learned baselines.
3. Implement the deterministic learned refinement baseline.
4. Freeze the evaluation and metric pipeline.

### Stage B: core diffusion

5. Implement joint 7-lead, 6-variable conditional residual diffusion.
6. Train one reproducible diffusion checkpoint.
7. Compare deterministic and diffusion performance on validation, then evaluate selected models on the untouched test partition.

### Stage C: first architecture research axes

8. History length: H=1, H=2, H=3 using the current artifact.
9. Sampling depth: 4, 8, 16, 32 denoising steps.
10. Gaussian residual initialization versus forecast-conditioned intermediate-noise refinement.
11. Warm-start or backbone transfer from the deterministic model as an optional compute optimization.

### Stage D: future research axes

12. N/M surrounding context after an authentic larger-context dataset revision.
13. Dense model-size scaling.
14. Selective MoE sparsity in the temporal/bottleneck path.
15. Fixed-compute comparison of denoising depth versus ensemble count.

No later scaling study should start before the deterministic baseline and core diffusion model are reproducible.

---

## 3. Model Architecture

### 3.1 Input contract

Canonical Sprint 3 input:

- `history`: `[B, H, 6, 16, 16]`, H in {1,2,3}; H=3 is the canonical configuration.
- `future_forecast`: `[B, 7, 6, 16, 16]`
- `terrain`: `[B, 5, 80, 80]`
- `target`: `[B, 7, 6, 80, 80]`

Optional future architecture input:
- `history_forecast_aux`: archived prior forecast trajectories, only when a future dataset revision provides authentic data. It is absent in Sprint 3 v1.

### 3.2 Temporal representation

The seven forecast days remain an explicit **time dimension**. They are not flattened into an unrelated set of image channels.

For each future lead:
1. encode the 16x16 GFS field
2. add a learned or sinusoidal lead-day embedding for D...D+6
3. project the lead into the spatial feature hierarchy

Historical days are encoded as a temporal sequence.

### 3.3 History encoder and history-to-future cross-attention

Historical context:
- spatial projection at 16x16
- temporal self-attention over H historical days

Future conditioning:
- one representation per lead day
- explicit lead embedding

Cross-attention:
- future lead queries attend to historical keys/values
- this is the required history-to-future information pathway

If a future dataset revision adds prior forecast history, its encoder should enter the same historical context fusion module rather than bypassing the temporal attention mechanism.

### 3.4 Spatial backbone

Use a shared spatial ConvNeXt/U-Net hierarchy:
- 80x80
- 40x40
- 20x20
- 10x10

Terrain is fused at high resolution and available at selected downsampled stages.

Skip connections preserve fine spatial structure.

### 3.5 Temporal coupling across the seven future leads

At the bottleneck, the seven lead representations are processed jointly using temporal self-attention.

The denoiser therefore models:
- spatial structure within each lead
- temporal interaction across D...D+6
- historical influence on every future lead

This is the intended temporal Transformer + spatial ConvNeXt/U-Net hybrid.

---

## 4. Deterministic Baselines

All baselines use the same frozen train/validation/test partition.

### Baseline 0A: channel-aware coarse-to-fine interpolation

- precipitation: conservative/area-preserving 5x5 disaggregation
- Tmax, Tmin, RH, U, V: bilinear interpolation

### Baseline 0B: all-bilinear interpolation

Bilinearly upsample all six GFS channels. This provides a simple common reference even where it is not physically ideal.

### Baseline 0C: persistence sanity check

Repeat the latest available historical day over the seven future leads. This is a weak lower-bound style reference, not a production forecast.

### Baseline 1: deterministic learned refinement

A single deterministic ConvNeXt/U-Net predicts the fine-grid correction from:
- history
- future GFS
- terrain

Output:
`[B,7,6,80,80]`

### Baseline 2: single-step refinement U-Net

A deterministic one-pass correction model using the same conditioning and residual formulation as diffusion, but without iterative noise denoising.

The diffusion model must demonstrate value beyond Baseline 0 and Baseline 1/2.

---

## 5. Conditional Residual Diffusion

### 5.1 Residual definition

Because Sprint 2 uses channel-specific train-fitted transforms, the core diffusion target is defined in **model space**:

`r_0 = y_fine_norm - upsample(future_forecast_norm)`

where:
- precipitation uses log1p-zscore
- other variables use z-score
- the same saved Sprint 2 statistics are used for both streams

Physical-unit reconstruction occurs only after inverse normalization.

### 5.2 Joint seven-day diffusion

A single training sample contains the full cube:

`r_0: [B,7,6,80,80]`

The diffusion process therefore operates on the complete seven-lead target while preserving the lead dimension internally.

A practical implementation is:
- shared 2D spatial blocks applied per lead
- temporal attention blocks operating over the lead dimension
- lead embeddings
- history-to-future cross-attention
- terrain conditioning
- timestep embedding injected into denoiser blocks

This provides a true joint spatiotemporal model without requiring full 3D convolution over the entire cube.

### 5.3 Diffusion formulation for Sprint 3 v1

Use one explicit formulation, not a mixture of incompatible formulations:

- DDPM-style Gaussian forward process
- epsilon prediction
- training diffusion horizon T=100 as the initial configuration
- sinusoidal timestep embedding
- AdaGN or equivalent timestep-conditioned modulation

EDM is a separate later experiment if research results justify it.

### 5.4 Sampling-depth research

At a fixed trained checkpoint compare:
- 4 DDIM steps
- 8 DDIM steps
- 16 DDIM steps
- 32 DDIM steps

Measure:
- deterministic skill of the sample mean
- probabilistic calibration
- extreme-event performance
- wall time
- peak VRAM

The optimal number of steps is an empirical result.

### 5.5 Forecast-conditioned refinement initialization

Support two inference modes:

1. standard residual initialization from Gaussian noise
2. refinement initialization from a deterministic residual estimate plus controlled noise at an intermediate noise level

Keep this as an explicitly labeled research experiment. Do not silently replace the primary diffusion objective.

---

## 6. Loss and Physical Constraints

### 6.1 General rule

Training losses operate in normalized model space unless a term is explicitly defined in physical space.

Use the exact saved train statistics from `data/normalization_stats.yaml`.

Never refit normalization on validation or test.

### 6.2 Deterministic loss

Use a multi-task objective covering all six outputs:
- precipitation: log-cosh plus optional high-quantile pinball component
- Tmax/Tmin: Huber
- RH: L1
- U/V: joint vector-aware loss plus component losses

Homoscedastic task weighting may be used, but it must be logged so task weights are reproducible.

### 6.3 Diffusion loss

For Sprint 3 v1:

`L_diff = E[||epsilon - epsilon_theta(r_t,t,c)||^2]`

The diffusion objective is defined on the residual noise target.

### 6.4 Precipitation positivity

Do not impose physical non-negativity by clamping a z-scored precipitation output to zero.

The correct chain is:
- model log1p-zscore precipitation
- invert using the saved statistics
- apply the existing physical non-negative guard `max(0, expm1(...))`
- report negative precipitation count/rate before final clipping if any numerical issue occurs

### 6.5 Relative humidity

Do not apply a physical sigmoid directly to the normalized z-score field.

Sprint 3 v1 predicts normalized RH and reports:
- physical MAE/RMSE
- bias
- out-of-range fraction after inverse transformation

A bounded physical-space head is an optional ablation.

### 6.6 Tmax/Tmin relation

Do not impose `Tmax >= Tmin` directly in normalized coordinates because the two channels have different saved means/stds.

Sprint 3 v1:
- predict normalized Tmax and Tmin
- compute a physical-space ordering penalty after inverse transformation
- report the violation rate

A coupled physical-space temperature head can be a later ablation.

### 6.7 Wind

Predict U and V jointly.

Report:
- U MAE/RMSE
- V MAE/RMSE
- vector RMSE
- derived wind-speed error

Wind speed/direction are reporting quantities, not replacements for U/V targets.

### 6.8 Mass conservation

Mass conservation is a **soft regularizer**, not a claim of exact physical truth.

Compute:
1. predicted precipitation in physical mm
2. area-weighted 80x80 to 16x16 aggregation
3. error against the corresponding GFS coarse precipitation
4. a separate error against aggregated reference precipitation

The regularization coefficient is tuned on validation. The model must not be forced to reproduce systematic GFS precipitation bias.

### 6.9 Lapse-rate constraint

Do not use a vertical 4.0-9.8 K/km lapse-rate loss in Sprint 3 v1.

The frozen target contains surface 2D Tmax/Tmin, not a vertical atmospheric profile.

Use terrain-conditioned spatial diagnostics if desired. A true vertical lapse-rate experiment requires additional vertical data.

---

## 7. Metrics and Evaluation

All model selection uses validation only. The test set is opened once for final comparison.

### Precipitation
- MAE
- RMSE
- wet-day MAE
- bias
- CSI@15 mm
- CSI@30 mm
- FSS or another clearly defined spatial event metric
- physical conservation error
- CRPS for stochastic forecasts
- Brier score for precipitation thresholds

### Temperature
- Tmax MAE/RMSE
- Tmin MAE/RMSE
- bias
- Tmax<Tmin violation rate

### RH
- MAE/RMSE
- bias
- out-of-range rate

### Wind
- U/V MAE/RMSE
- vector RMSE
- wind-speed error

### Per-lead reporting

Every metric is reported separately for:
D, D+1, D+2, D+3, D+4, D+5, D+6

Also report seven-day aggregate values.

### Qualitative validation

For selected held-out dates, produce:
- reference target maps
- interpolation baseline maps
- deterministic model maps
- diffusion mean/median
- several independent diffusion members
- residual maps
- precipitation histograms/quantiles
- at least one high-precipitation event example

All qualitative figures must use validation during development and the untouched test set only for final reporting.

---

## 8. Research Matrix

### 8.1 History length

Current frozen-data experiment:
- H=1
- H=2
- H=3

Future extension:
- H>3 from authentic historical archives

Do not claim H>3 results from the three-day frozen tensor.

### 8.2 N/M spatial context

The current frozen Zarr contains only the target coarse 16x16 crop and corresponding 80x80 target. It does not support a true surrounding-area N/M experiment.

Future data extension:
- build an authentic coarse context crop larger than the target crop
- keep the target M fixed
- vary N/M across a predeclared grid

Candidate ratios such as 1.0, 1.5, 2.0, 2.5 are hypotheses, not conclusions.

No N/M result may be claimed until the wider-context artifact passes the same Sprint 2 provenance and leakage gates.

### 8.3 Model-size scaling

Create at least:
- small
- base
- large

Hold data and objective fixed.

Report:
- parameter count
- active FLOPs
- peak VRAM
- wall time
- validation/test skill

### 8.4 Sparse MoE

Do not make every layer MoE.

First experiment at:
- temporal bottleneck, or
- deepest shared feature block

Compare:
- dense baseline
- sparse top-1 routing
- sparse top-2 routing

Report total parameters and active parameters/FLOPs.

### 8.5 Denoising steps versus ensemble members

At a fixed inference compute budget compare, for example:
- 32 steps x 1 member
- 16 steps x 2 members
- 8 steps x 4 members

Measure:
- CRPS
- Brier score
- interval coverage
- spread-skill relationship
- extreme-event probability quality

Do not assume more denoising or more ensemble members is universally better.

---

## 9. Training Budget and Kaggle Execution

The Deep Research report's suggested 5-10 epoch initial training range is adopted as a **pilot budget**, not as a guaranteed wall-clock claim.

Before every remote run:
1. query live Kaggle quota
2. record available time
3. run a one-epoch timing probe when a new architecture is introduced
4. project feasible epochs from measured time
5. stop before the quota margin is exhausted

### Initial experiment ladder

1. CPU local shape/loss smoke test
2. one-epoch Kaggle baseline timing run
3. 5-10 epoch deterministic baseline pilot with early stopping
4. one-epoch diffusion timing run
5. 5-10 epoch diffusion pilot with early stopping
6. reserve remaining quota for only the most informative research axes

A longer 15-20 epoch run is permitted only when measured epoch time and remaining quota justify it.

### Hardware

Do not assume dual-GPU availability.

Log:
- `torch.cuda.device_count()`
- GPU names
- VRAM
- AMP mode
- wall time
- peak memory

Use T4-compatible FP16 AMP and gradient scaling as required.

TPU is not on the critical path unless a verified TPU implementation exists.

### Checkpoints and artifacts

Save during the remote run to:
- `/kaggle/working/models/`
- `/kaggle/working/reports/`

Checkpoint must include:
- model state
- optimizer state
- scheduler state
- epoch
- random seed
- experiment configuration
- dataset manifest identifier
- preprocessing config hash
- normalization statistics identifier

Do not rely on notebook memory for artifact preservation.

---

## 10. Implementation Files

### Models
- `src/models/temporal_multitask_baseline.py`
- `src/models/residual_diffusion.py`

### Losses
- `src/losses/spatiotemporal_multitask_loss.py`

### Training
- `scripts/train_temporal_downscaler.py`

### Kaggle dispatch
- `scripts/kaggle/dispatch_temporal_baseline.py`

### Tests
- `tests/models/test_spatiotemporal_baseline.py`
- additional diffusion and data-contract tests described below

No trainer should directly rebuild raw GFS or reference data.

---

## 11. Mandatory Tests Before Spending Serious GPU Quota

### Data contract
1. Exact split counts and year membership from `sample_index.parquet`
2. DataLoader selects by split field, never by positional assumptions
3. No sample with `qa_status=EXCLUDED`
4. History ends at D-1
5. Future forecast spans D...D+6
6. Forecast source remains authentic GFS
7. Exact `normalization_stats.yaml` is loaded
8. Manifest and preprocessing config identifiers are recorded

### Normalization audit
Run a one-time training-set audit utility that:
- loads the complete train partition or a documented full scan
- verifies the saved transform type for each channel
- verifies the stored mean/std against the actual normalized train representation within a declared numerical tolerance

Do not require an individual minibatch to equal the global train mean/std. Minibatch statistics are expected to vary.

### Architecture
1. H=1, H=2, H=3 execute
2. history-to-future cross-attention exists and receives gradients
3. lead embedding exists and changes the future representation
4. temporal attention operates across seven leads
5. spatial output is [B,7,6,80,80]
6. terrain branch affects the forward path
7. U and V remain separate

### Diffusion
1. residual target calculation is deterministic and tested
2. DDPM noise schedule bounds are correct
3. epsilon target shape is correct
4. timestep sampling is correct
5. 4/8/16/32-step sampler configurations execute
6. fixed seed reproduces the same sample
7. different seeds produce distinct members
8. inverse normalization produces finite physical fields

### Loss
1. all six channels contribute
2. diffusion loss is epsilon MSE in the declared parameterization
3. physical regularizers operate after inverse normalization where required
4. conservation error is finite
5. no unsupported vertical lapse-rate term is active

---

## 12. Definition of Done

Sprint 3 is complete when:

1. deterministic interpolation baselines exist
2. deterministic learned refinement is reproducible
3. conditional residual diffusion trains end-to-end
4. train/val/test contracts remain leak-free
5. one validated diffusion checkpoint exists
6. normalized training loss and de-normalized physical metrics are both reported
7. metrics are reported per variable and per lead
8. qualitative held-out comparisons exist
9. 4/8/16/32-step sampling results exist
10. forecast-initialized refinement is evaluated or explicitly deferred
11. H>3 and N/M limitations are documented
12. no synthetic/reanalysis forecast fallback returns
13. Sprint 2 regression tests remain green

---

## 13. No-Go Conditions

Stop Sprint 3 and return to data engineering if:
- synthetic or reanalysis-derived forecast data appear in the model input
- split membership changes without a new frozen dataset version
- normalization is refit on validation/test
- history crosses the D-1 cutoff
- a research axis silently uses unavailable data
- station data are interpolated into training targets
- the training script rebuilds a different dataset than the frozen Sprint 2 artifact

This keeps Sprint 3 scientifically continuous with Sprints 1-2 while preserving the original longer-term research program.
