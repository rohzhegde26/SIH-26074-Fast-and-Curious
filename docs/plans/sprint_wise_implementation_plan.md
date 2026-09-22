# Sprint-Wise Implementation Plan
## Multivariate Spatiotemporal Diffusion Weather Downscaling

This document converts the master implementation/research plan into **10 sequential sprints**.

The sprint structure is deliberately progressive: each sprint should produce a usable artifact and a checkpoint before the next major architectural variable is introduced.

---

# Sprint 1 — Data Audit & Source Finalization

## Objective

Establish the complete historical-data stack and verify that every required input can actually be obtained for the intended India/Peninsular-India period.

## Scope

Finalize:

- ERA5-Land historical weather source.
- CHIRPS v2.0 p05 precipitation.
- CHIRPS p25 where required for coarse precipitation.
- ERA5 0.25° where required for coarse meteorological inputs.
- Optional freely accessible NOAA GSOD / ISD / GHCN station data.
- Historical forecast archive investigation.

## Key decisions

- Exact 2014–2023 usable period.
- Canonical target domain.
- Context-domain extraction strategy.
- Canonical 03:00–03:00 UTC daily aggregation.
- Exact variables and units.
- Data provenance classes.

## Deliverables

```text
data/source_manifest.yaml
data/source_coverage_report.md
data/data_availability_report.md
data/domain_config.yaml
```

## Exit criteria

- Every required historical source has been tested.
- No source is being silently substituted with synthetic/proxy data.
- Missing historical-forecast archive requirements are explicitly documented.
- Data-license/access assumptions are documented.

---

# Sprint 2 — Dataset Builder & Data QA

## Objective

Build the reproducible model-ready dataset.

## Scope

Implement:

```text
raw download/cache
      ↓
temporal alignment
      ↓
spatial extraction
      ↓
unit conversion
      ↓
quality checks
      ↓
history tensor assembly
      ↓
target tensor assembly
```

Implement hard validation for:

- HTTP/request failures.
- Missing timestamps.
- Duplicate timestamps.
- Missing variables.
- Wrong coordinates.
- Wrong spatial resolution.
- Invalid physical values.
- Missing required samples.

## Dataset structure

Initial conceptual structure:

```text
history_actual
future_forecast
terrain
target
```

Support `history_forecasts` only once a genuinely archived forecast source is available.

## Deliverables

```text
data/build_dataset.py
data/validate_dataset.py
data/sample_index.parquet
data/preprocessing_config.yaml
datasets/multitask_temporal_v1/
```

## Exit criteria

- Dataset can be rebuilt deterministically.
- Same raw inputs produce the same processed sample.
- Chronological train/validation/test splits are generated.
- Dataset contains no synthetic scientific values.

---

# Sprint 3 — 7-Day Deterministic Multivariate Baseline

## Objective

Establish the first true baseline for the new research direction.

## Architecture

```text
historical actual/reanalysis
          +
future coarse forecast
          +
terrain
          ↓
temporal encoder
          +
spatial model
          ↓
7 × 6-channel deterministic output
```

Output:

```text
7 days ×
[P, Tmax, Tmin, RH, U, V]
```

## Important constraints

- Joint multivariate prediction.
- No diffusion yet.
- No MoE yet.
- Keep model deliberately small.
- Maintain complete experiment logging.

## Deliverables

```text
models/temporal_multitask_baseline.py
configs/baseline_temporal.yaml
checkpoints/baseline_temporal.pt
reports/baseline_metrics.md
```

## Exit criteria

- End-to-end training works.
- End-to-end inference works.
- Seven-day predictions are produced.
- Per-variable and per-lead-time metrics are reported.

---

# Sprint 4 — History-Length Research

## Objective

Determine how much historical context is useful before building the full diffusion system.

## Controlled sweep

```text
H = 3 days
H = 5 days
H = 7 days
H = 10 days
H = 14 days
```

Keep all other architecture/data choices fixed.

## Evaluate

### Deterministic

- MAE
- RMSE
- bias
- spatial correlation
- precipitation metrics

### Temporal

- lead-time degradation
- day-to-day consistency

### Compute

- training time
- inference latency
- memory

## Deliverables

```text
experiments/history_length/
results/history_length_comparison.csv
reports/history_length_research.md
```

## Exit criteria

Select a **provisional history length** for downstream experiments.

The decision should be based on validation performance plus compute cost, not intuition alone.

---

# Sprint 5 — Spatial Context Research

## Objective

Determine how much area surrounding the target should be visible to the model.

## Research variable

Let:

- M = target area
- N = context area

Test:

```text
N/M = 1.25
N/M = 1.50
N/M = 1.75
N/M = 2.00
N/M = 2.50
```

## Architecture

Use encoded context tokens rather than full attention over every context pixel.

```text
N × N context
      ↓
spatial encoder
      ↓
context latent tokens
      ↓
cross-attention
      ↓
target representation
```

## Evaluate

- Overall error.
- Precipitation structure.
- Spatial correlation.
- Boundary consistency.
- Extreme rainfall localization.
- Memory/latency.

## Deliverables

```text
experiments/spatial_context/
results/spatial_context_comparison.csv
reports/spatial_context_research.md
```

## Exit criteria

Select a provisional **context ratio** for the diffusion experiments.

---

# Sprint 6 — Conditional Residual Diffusion Prototype

## Objective

Introduce diffusion without simultaneously scaling everything else.

## Architecture

```text
history
+
future forecast
+
spatial context
+
terrain
       ↓
deterministic backbone
       ↓
fine-resolution baseline
       +
conditional residual diffusion
       ↓
final 7 × 6 output
```

Important:

> The reverse diffusion process still starts from noise. The forecast-derived baseline is the conditioning/reference state; diffusion generates a conditional fine-scale residual.

## Initial experiment

Start with:

- small model;
- fixed history length from Sprint 4;
- fixed context ratio from Sprint 5;
- one sampler;
- modest diffusion-step count.

## Compare

```text
deterministic baseline
vs
deterministic + residual diffusion
```

## Deliverables

```text
models/residual_diffusion.py
configs/diffusion_small.yaml
checkpoints/diffusion_small.pt
reports/deterministic_vs_diffusion.md
```

## Exit criteria

- Stable diffusion training.
- Valid 7 × 6 joint generation.
- Residual formulation verified.
- Probabilistic evaluation pipeline operational.

---

# Sprint 7 — Diffusion-Step & Sampler Research

## Objective

Determine the practical test-time denoising range.

## Primary sweep

```text
4 steps
8 steps
12 steps
16 steps
24 steps
32 steps
```

## Secondary variable

Test multiple compatible samplers where practical, e.g.:

```text
DDIM
Euler
Heun
DPM-style solver
```

## Evaluate

### Forecast quality

- MAE
- RMSE
- precipitation-event metrics
- extreme rainfall metrics
- spatial metrics

### Probabilistic quality

- CRPS
- calibration
- coverage
- ensemble spread
- exceedance probabilities

### Compute

- latency
- peak VRAM
- throughput

## Deliverables

```text
experiments/diffusion_steps/
results/diffusion_steps.csv
reports/diffusion_compute_frontier.md
```

## Exit criteria

Identify:

- minimum practical step count;
- point of diminishing returns;
- preferred sampler;
- quality/latency frontier.

---

# Sprint 8 — Ensemble & Test-Time Compute Scaling

## Objective

Investigate the third scaling axis and directly answer:

> Does spending compute on better denoising or on more ensemble members improve decision-relevant uncertainty more?

## Ensemble sweep

```text
1
4
16
32
64
```

## Matched-compute comparisons

Examples:

```text
32 steps × 1 sample
16 steps × 2 samples
8 steps × 4 samples
8 steps × 8 samples
4 steps × 16 samples
```

Use measured runtime rather than assuming identical costs per step/sample.

## Evaluate

Focus particularly on:

- CRPS.
- Calibration.
- Coverage.
- Ensemble spread.
- Extreme-event probabilities.
- Decision-relevant threshold probabilities.

Also report deterministic mean-field accuracy.

## Deliverables

```text
experiments/test_time_scaling/
results/test_time_scaling.csv
reports/denoising_vs_ensemble_scaling.md
```

## Exit criteria

Characterize whether compute is better spent on:

```text
deeper denoising
vs
more samples
```

and whether the answer changes with:

- lead time;
- precipitation intensity;
- variable.

---

# Sprint 9 — Model Scaling & MoE Sparsity Research

## Objective

Establish the second major scaling axis: model capacity.

## Phase 1 — Dense scaling

Train:

```text
Small
Medium
Large
```

Keep the data and diffusion configuration fixed.

Measure:

- total parameters;
- training compute;
- inference compute;
- memory;
- quality;
- probabilistic metrics.

## Phase 2 — MoE

Introduce MoE into selected Transformer FFN blocks.

Research variables:

### Expert count

```text
4
8
16
```

### Top-k

```text
1
2
4
```

### MoE block coverage

```text
25%
50%
75%
100%
```

## Key question

How much total parameter capacity can be added without proportionally increasing active computation?

## Deliverables

```text
models/moe_diffusion_transformer.py
experiments/moe_scaling/
results/moe_scaling.csv
reports/model_scaling_and_moe.md
```

## Exit criteria

Identify a provisional efficient configuration based on:

```text
quality
+
active parameters
+
latency
+
VRAM
```

not simply total parameter count.

---

# Sprint 10 — Integrated System & Final Scaling Study

## Objective

Combine the strongest findings into the mature architecture.

## Final architecture target

```text
history actual
       +
historical forecasts [when available]
       +
7-day future forecast
       +
N×N spatial context
       +
terrain
       ↓
history encoder
       +
future encoder
       +
history↔future cross-attention
       ↓
spatiotemporal DiT
       +
selected MoE configuration
       ↓
deterministic fine-resolution baseline
       +
conditional residual diffusion
       ↓
7 × 6 probabilistic output
```

## Final research axes

Run carefully selected combinations of:

```text
history length
×
context ratio
×
model size
×
MoE sparsity
×
diffusion steps
×
ensemble size
```

Do **not** run the full Cartesian product.

Select a small number of representative operating points.

## Suggested operating modes

### FAST

```text
small/moderate model
few diffusion steps
1 sample
```

### BALANCED

```text
medium/large model
moderate steps
few samples
```

### ACCURATE

```text
large model
higher diffusion steps
1–4 samples
```

### ENSEMBLE

```text
selected model
moderate steps
many samples
```

## Final evaluation

Report:

### Accuracy

- MAE
- RMSE
- bias
- precipitation-event skill
- extreme-event skill
- spatial structure

### Probabilistic quality

- CRPS
- calibration
- coverage
- ensemble spread
- threshold exceedance probabilities

### Multivariate consistency

- precipitation ↔ humidity
- precipitation ↔ temperature
- wind ↔ precipitation
- Tmax ↔ Tmin
- other physically meaningful relationships

### Scaling

- parameter count
- active parameters
- diffusion steps
- ensemble samples
- latency
- VRAM
- quality

## Deliverables

```text
models/final/
configs/final/
results/final_scaling_matrix.csv
reports/final_system_report.md
checkpoints/final/
```

## Exit criteria

A final report should identify the empirical tradeoffs among:

```text
model capacity
test-time denoising compute
ensemble sampling compute
accuracy
uncertainty quality
```

---

# Sprint Dependency Graph

```text
Sprint 1
   |
   v
Sprint 2
   |
   v
Sprint 3
   |
   +------------+
   |            |
   v            v
Sprint 4     Sprint 5
   |            |
   +------┬-----+
          v
       Sprint 6
          |
          v
       Sprint 7
          |
          v
       Sprint 8
          |
          v
       Sprint 9
          |
          v
      Sprint 10
```

Sprints 4 and 5 can proceed partly in parallel after Sprint 3, but their provisional choices should be fixed before Sprint 6 becomes the main diffusion experiment.

---

# Sprint Completion Checklist

Every sprint should finish with:

```text
[ ] Code committed
[ ] Config committed
[ ] Dataset version recorded
[ ] Experiment ID recorded
[ ] Metrics saved
[ ] Checkpoint saved where applicable
[ ] Compute statistics recorded
[ ] Validation/test protocol documented
[ ] Research conclusion written
[ ] Decision for next sprint documented
```

---

# Sprint Philosophy

The central rule is:

> **Change one major research variable at a time until its effect is understood.**

Do not jump directly from the current baseline to:

```text
7-day
+
large spatial context
+
diffusion
+
large DiT
+
MoE
+
many ensemble samples
```

because that would make improvements or failures impossible to attribute.

The sprint structure is intended to produce a chain of evidence:

```text
Data validity
    ↓
Temporal benefit
    ↓
Spatial-context benefit
    ↓
Diffusion benefit
    ↓
Denoising scaling
    ↓
Ensemble scaling
    ↓
Model scaling
    ↓
MoE efficiency
    ↓
Integrated scaling frontier
```

The final result should therefore be a **research-backed architecture**, not merely the largest model that happens to train.
