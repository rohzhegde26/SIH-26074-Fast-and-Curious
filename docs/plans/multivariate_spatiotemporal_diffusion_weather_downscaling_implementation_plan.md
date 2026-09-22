# Multivariate Spatiotemporal Diffusion Transformer Weather Downscaling
## Implementation + Research Plan

> **Status:** Research/implementation master plan  
> **Goal:** Build a 7-day, multivariate, probabilistic weather downscaler that maps coarse forecast information to fine-resolution fields while supporting two primary scaling axes—**model capacity** and **test-time diffusion compute**—plus a third uncertainty/sampling axis through **ensemble size**.

---

## 1. Core research thesis

The project should evolve from the current deterministic single-day downscaler into a:

> **Multivariate Spatiotemporal Diffusion Transformer with Hierarchical Spatial Context and Sparse Mixture-of-Experts (MoE)**

The target system should jointly generate a **7-day fine-resolution trajectory** for multiple meteorological variables, using:

- recent **actual weather + historical forecasts** as history context;
- only **future forecasts** for the upcoming 7 days;
- a spatial context region larger than the target downscaling region;
- terrain/geographic information;
- a deterministic coarse-to-fine baseline;
- a conditional diffusion model for fine-scale residual generation;
- joint generation of all target variables;
- optional sparse MoE layers for model-capacity scaling.

The model should be developed progressively rather than all at once.

---

# 2. Data pipeline phase — real, freely accessible historical context

This phase is a **prerequisite to the temporal-model experiments**.

The deep-research review found an important constraint:

> There is no identified freely accessible Indian station/AWS dataset that provides complete 2014–2023 coverage for all five required variables across the target region.

Therefore, the core history stream should use the best freely accessible **real meteorological products**, while explicitly preserving their provenance:

- **ERA5-Land via Open-Meteo** for temperature, humidity, and wind.
- **CHIRPS v2.0 p05** for fine-resolution precipitation.
- **CHIRPS p25 and/or ERA5 0.25°** for appropriate coarse historical fields.
- **NOAA GSOD/ISD or GHCN-Daily** only as optional station-observation supplements where usable Indian stations exist; they must not be assumed to provide complete spatial coverage.
- **NASA POWER** may be used as an optional independent sanity-check source, but it is also a derived/reanalysis product and should not be presented as ground-truth observations.

### Important provenance distinction

The history stream must distinguish:

```text
Direct station observations
        ≠
Satellite/gauge-derived products
        ≠
Reanalysis
```

In particular:

- ERA5-Land is **reanalysis**, not station telemetry.
- CHIRPS is a **gauge-adjusted satellite precipitation product**, not raw station telemetry.
- NOAA GSOD/ISD/GHCN are **actual station observations** where a station record exists.

The absence of freely accessible dense Indian station data means the project should **not claim that the history context is direct observational telemetry**.

No RTI process or credential-gated Indian station-data acquisition is part of this plan.

---

## 2.1 Canonical data period and geography

Use the same historical window as the downscaling dataset:

```text
Historical period: 2014–2023
Region: India / primary Peninsular-India study domain
Canonical core tile: 11–15°N, 74–78°E
```

The pipeline must support extracting a larger spatial context around the target once the N/M research matrix is run.

The historical source datasets must cover the complete set of dates needed by the selected training/validation/test splits.

---

## 2.2 History-variable source map

| Target variable | Primary history source | Observation class | Initial role |
|---|---|---|---|
| Precipitation | CHIRPS v2.0 p05 | Satellite + gauge-derived | Fine historical precipitation context |
| Tmax | ERA5-Land | Reanalysis | Historical temperature context |
| Tmin | ERA5-Land | Reanalysis | Historical temperature context |
| RH | ERA5-Land | Reanalysis / derived field | Historical humidity context |
| Wind U | ERA5-Land | Reanalysis | Historical wind context |
| Wind V | ERA5-Land | Reanalysis | Historical wind context |
| Station observations where available | NOAA GSOD/ISD/GHCN | Direct observations | Optional independent observation/sanity check |

The six-channel representation remains:

```text
[P, Tmax, Tmin, RH, U, V]
```

For consistency with the model, all sources must be converted into the same canonical units, day definition, coordinate convention, and spatial indexing scheme before being assembled into history tensors.

---

## 2.3 Do not silently mix products into a fake "observation" tensor

The pipeline should retain per-channel provenance metadata.

Example:

```yaml
history:
  precipitation:
    source: CHIRPS_v2_p05
    provenance_class: satellite_gauge_product

  tmax:
    source: ERA5-Land
    provenance_class: reanalysis

  tmin:
    source: ERA5-Land
    provenance_class: reanalysis

  rh:
    source: ERA5-Land
    provenance_class: reanalysis

  wind_u:
    source: ERA5-Land
    provenance_class: reanalysis

  wind_v:
    source: ERA5-Land
    provenance_class: reanalysis
```

This prevents later experiments from accidentally describing the history stream as "ground observations."

---

# 3. Data acquisition pipeline

The implementation should proceed as a distinct engineering phase before model research.

```text
Source verification
       |
       v
Domain/date requests
       |
       v
Raw-data cache
       |
       v
Temporal alignment
       |
       v
Spatial extraction/regridding
       |
       v
Unit + physical-range validation
       |
       v
Missing-data validation
       |
       v
History tensor assembly
       |
       v
Provenance manifest
       |
       v
Model-ready dataset
```

## 3.1 Step A — ERA5-Land history ingestion

Use the Open-Meteo Historical API for the freely accessible ERA5-Land history stream.

Retrieve, as required:

```text
temperature_2m
dew_point_2m / relative_humidity_2m
wind_u / wind_v or an equivalent representation
```

Prefer hourly retrieval when constructing the canonical meteorological day.

The ingestion code must:

1. request the exact date range;
2. request the exact target/context coordinates;
3. verify HTTP status;
4. verify returned coordinates;
5. verify timestamps;
6. verify the requested variables exist;
7. verify all expected hourly records are present;
8. aggregate to the canonical daily window in code.

### Canonical day

Use:

```text
03:00 UTC → next-day 03:00 UTC
```

which corresponds to:

```text
08:30 IST → next-day 08:30 IST
```

for the project's canonical daily aggregation.

Do not rely on an API's precomputed daily aggregate if doing so would obscure the exact 24-hour window.

---

## 3.2 Step B — CHIRPS precipitation ingestion

Use CHIRPS v2.0 p05 daily COGs from the UCSB/Climate Hazards Center archive.

For each date:

```text
date
  |
  v
CHIRPS p05 COG
  |
  v
extract target/context window
  |
  v
daily precipitation field
```

The pipeline should:

- construct the file path from the exact calendar date;
- fail hard if the expected tile/file is unavailable;
- validate CRS/grid metadata;
- extract the exact geographic window;
- preserve the raw precipitation units;
- record source URL/path and date in provenance metadata.

CHIRPS p25 may be used for a coarse historical precipitation input where scientifically appropriate, but it must be labeled as the same product family rather than an independent observation.

---

## 3.3 Step C — Coarse historical fields

Where the architecture requires a coarse historical state, use an independently defined coarse source.

Initial candidate:

```text
ERA5 0.25°
```

for coarse thermodynamic/wind fields, and:

```text
CHIRPS p25
```

for coarse precipitation where that design is retained.

The pipeline must not create "coarse observations" by simply pooling the fine target data unless the experiment explicitly defines a same-product regridding baseline.

---

## 3.4 Step D — Optional direct station observations

No RTI or restricted Indian-agency acquisition is part of the implementation.

However, freely downloadable NOAA station archives should be checked for stations within/near the target region:

```text
GSOD
ISD
GHCN-Daily
```

If usable stations exist:

```text
station observation
       |
       v
quality-control
       |
       v
independent validation/sanity-check layer
```

These observations should **not automatically be spatially interpolated into dense "observed" grids** for the model.

A station-to-grid conversion would itself introduce a modeling/interpolation assumption and should therefore be treated as a separate experiment.

---

# 4. Data-quality gates

The data pipeline must hard-fail rather than silently substitute a fallback.

### Request integrity

```text
HTTP status = success
coordinates = expected
dates = expected
variables = expected
```

### Temporal integrity

For every sample:

```text
expected 24 hourly values
no duplicate timestamps
no missing timestamps
correct 03:00–03:00 UTC grouping
```

### Spatial integrity

Verify:

```text
target bounds
context bounds
grid spacing
orientation
latitude order
longitude order
```

### Physical sanity checks

At minimum:

```text
precipitation >= 0
RH within physical range
Tmax >= Tmin
finite wind components
finite temperatures
```

Do not silently clip impossible values; flag them and investigate the source.

### Missing data

The default policy should be:

```text
required source missing
        ↓
sample invalid
        ↓
fail / exclude explicitly
```

Never generate replacement weather values using random numbers, climatological placeholders, or handcrafted proxies.

---

# 5. Temporal alignment with forecast data

The history stream and future forecast stream need explicit initialization-time alignment.

For a forecast initialized at \(t_0\):

```text
HISTORY
actual/reanalysis history:
t-H ... t0

HISTORICAL FORECASTS
only forecasts that would have been available by t0

FUTURE FORECAST
t+1 ... t+7
```

Every future forecast input must retain:

```text
forecast initialization time
valid time
lead time
source/model
```

### Important research/data gap

The reviewed freely accessible datasets solve the **historical weather-state** problem, but they do not automatically provide a complete 2014–2023 archive of historical forecasts matching the future-forecast branch.

Therefore:

> Do not fabricate the historical-forecast stream by relabeling ERA5/ERA5-Land analysis as a forecast.

Before the temporal architecture experiment that explicitly uses `actual + historical forecasts`, verify and acquire a genuinely archived forecast source with the required period, variables, geography, and initialization/lead-time metadata.

Until then, the model should support a controlled:

```text
actual/reanalysis history
+
future forecast
```

configuration as the clean baseline.

---

# 6. History tensor construction

Each training example should eventually contain something conceptually equivalent to:

```text
history_actual
    [H, C, context_H, context_W]

history_forecasts
    [H', C_f, context_H, context_W]    # only when a real archive is available

future_forecast
    [7, C_f, context_H, context_W]

terrain
    [C_terrain, target_H, target_W]

target
    [7, 6, target_H, target_W]
```

The model can later decide whether temporal compression, patchification, or latent encoding changes these physical dimensions.

---

# 7. Data split policy

The split must be chronological.

Do not randomly split individual daily samples across train/validation/test, because adjacent weather days are strongly correlated.

Example structure:

```text
2014–2020 → training
2021–2022 → validation
2023       → held-out test
```

The exact split may be adjusted, but it must remain **time-contiguous and leakage-safe**.

Any history window crossing a split boundary must be handled carefully so that future information from validation/test periods cannot leak into training samples.

---

# 8. Reproducible data manifest

Every source and derived tensor should have a manifest containing:

```yaml
dataset_version:
region:
target_region:
context_ratio:
date_start:
date_end:

sources:
  - provider:
    product:
    version:
    url:
    retrieval_time:
    provenance_class:
    spatial_resolution:
    temporal_resolution:

processing:
  daily_window_utc: "03:00–03:00"
  units:
  grid_convention:
  interpolation:
  aggregation:
```

The resulting model-ready dataset should therefore be reproducible from the manifest plus the raw/open source data.

---

# 9. Data-pipeline validation experiments

Before beginning architecture research, run these experiments/checks.

| Experiment | Purpose | Pass condition |
|---|---|---|
| ERA5-Land coverage audit | Verify complete historical history | All requested dates/hours available |
| CHIRPS coverage audit | Verify complete precipitation history | All requested daily fields available |
| Cross-source calendar audit | Verify exact sample matching | One unambiguous record per sample/date |
| Spatial-grid audit | Verify target/context geometry | Exact expected bounds/resolution |
| Unit audit | Verify units/conversions | Expected canonical units |
| Missingness audit | Quantify gaps | No unexplained required-source gaps |
| Physical-range audit | Detect corrupted values | No unexplained invalid values |
| Station-overlap audit | Check NOAA station usefulness | Report actual station count/coverage |
| Forecast-archive audit | Verify historical forecast availability | Real initialization/lead-time archive exists before use |
| Reproducibility test | Rebuild same sample from raw sources | Identical processed output within defined tolerance |

---

# 10. Data pipeline deliverable

The data phase is complete only when it produces:

```text
raw/
  era5_land/
  chirps_p05/
  chirps_p25/
  era5_coarse/
  optional_noaa/

processed/
  history/
  future_forecast/
  terrain/
  targets/

metadata/
  source_manifest.yaml
  sample_index.parquet
  preprocessing_config.yaml

datasets/
  multitask_temporal_v1/
```

The dataset builder should be deterministic and should have **no synthetic fallback path**.

---
The proposed architecture does **not** mean that inference should literally take the forecast and run a diffusion process directly on that forecast instead of starting from noise.

The preferred design is:

```text
Coarse forecast + history + spatial context + terrain
                       |
                       v
          Deterministic spatiotemporal backbone
                       |
                       v
             Fine-resolution baseline
                       |
                       v
          -----------------------------
          |                           |
          |      conditional          |
          |   residual diffusion      |
          |                           |
          -----------------------------
                       |
              reverse diffusion
              starts from noise
                       |
                       v
            fine-scale residual
                       |
                       v
       baseline + generated residual
                       |
                       v
              final 0.05° field
```

Mathematically:

\[
Y_{\mathrm{fine}} = B(X) + R
\]

where:

- \(X\) = history, future forecast, spatial context, terrain, etc.
- \(B(X)\) = deterministic fine-resolution baseline
- \(R\) = diffusion-generated fine-scale residual
- \(Y_{\mathrm{fine}}\) = final prediction

### Why this is preferred

The deterministic component can learn the large-scale, relatively predictable structure, while diffusion concentrates its capacity on the unresolved fine-scale structure and uncertainty.

At inference:

- the **reverse diffusion process still starts from noise**;
- the denoiser is **strongly conditioned on the forecast-derived baseline and all other context**;
- the model learns a conditional distribution of plausible residuals.

A separate experiment may later investigate **warm-start / non-noise initialization**, but this should not be the default architecture.

---

# 12. Target prediction representation

## 3.1 Joint output tensor

The first target representation should be:

\[
Y \in \mathbb{R}^{T \times C \times H \times W}
\]

with:

- \(T = 7\) forecast days
- \(C = 6\) primary meteorological channels
- \(H,W\) = fine-resolution spatial dimensions

### Proposed channels

| Channel | Variable | Representation |
|---|---|---|
| 1 | Precipitation | transformed precipitation, e.g. `log1p(P)` |
| 2 | Tmax | normalized temperature |
| 3 | Tmin | normalized temperature |
| 4 | Relative humidity | normalized RH |
| 5 | Wind U | zonal wind |
| 6 | Wind V | meridional wind |

Wind should initially be represented as **U/V components**, not speed + direction, because direction is circular and has a discontinuity at 0/360°.

## 3.2 Joint generation

The model should generate all variables jointly:

\[
p(Y_{1:7} \mid H,\ F_{1:7},\ C,\ T_{\mathrm{terrain}})
\]

where:

- \(H\) = historical actual + archived forecast context
- \(F_{1:7}\) = future coarse forecasts
- \(C\) = surrounding spatial context
- \(T_{\mathrm{terrain}}\) = terrain/geographic features

The model should therefore learn:

- cross-variable dependencies;
- cross-day dependencies;
- spatial dependencies;
- relationships between large-scale forecast structure and fine-scale residual structure.

---

# 13. Variable/channel design research

Although the initial design should use six channels, channel representation itself should be treated as a research area.

### Baseline

```text
[P, Tmax, Tmin, RH, U, V]
```

### Candidate extensions

Potential future experiments could include:

```text
[P, Tmax, Tmin, RH, U, V, pressure]
[P, Tmax, Tmin, RH, U, V, geopotential]
[P, Tmax, Tmin, RH, U, V, moisture variables]
```

The rule should be:

> Add a channel only if it is available consistently in training, validation, and deployment data and has a defensible physical role.

Each channel should also receive a learned **variable embedding**, so the Transformer knows the semantic identity of each channel rather than treating channels as anonymous numbers.

---

# 14. Temporal architecture

## 5.1 History stream

History should contain two distinct information sources:

```text
Actual observations / analysis
        +
Historical forecast information
```

These should remain provenance-distinguishable.

Proposed history length:

\[
H \in \{3,5,7,10\}\ \text{days initially}
\]

A larger candidate of 14 days can be investigated if justified by data/compute.

## 5.2 Future stream

The future stream contains:

```text
Forecast D+1
Forecast D+2
...
Forecast D+7
```

Only information that would genuinely be available at forecast initialization may enter this stream.

## 5.3 Information-flow constraint

At forecast issue time \(t_0\):

```text
PAST
actual/available observations:
t-7 ... t-1, t0

HISTORICAL FORECASTS:
archived forecasts available before / around t0

FUTURE:
forecast:
t+1 ... t+7
```

Actual future observations must never be used as future conditioning variables.

This cutoff/provenance rule is a first-class data invariant.

---

# 15. Temporal Transformer design

The preferred design is not one giant attention operation over all spatiotemporal pixels.

Instead:

```text
History Encoder
       |
       v
History latent tokens
       |
       |\
       | \
       |  \  cross-attention
       |   \
       |    v
       |  Future Forecast Encoder
       |         |
       +---------+
             |
             v
    joint temporal representation
```

The intended meaning of "bidirectional context" is:

- history can inform the representation used for the future;
- future forecast information can condition the interpretation of historical state/context;
- however, **future actual observations are never allowed into the system**.

Cross-attention is preferred over simply concatenating everything and allowing unrestricted attention.

---

# 16. Spatial-context architecture

## 7.1 Target vs context

Let:

- \(m \times m\) = target region
- \(n \times n\) = larger context region
- \(n > m\)

```text
+---------------------------------------+
|                                       |
|        larger N × N context           |
|                                       |
|      +-----------------------+        |
|      |                       |        |
|      |      M × M target     |        |
|      |                       |        |
|      +-----------------------+        |
|                                       |
+---------------------------------------+
```

The model should exploit surrounding regions for:

- upstream weather systems;
- moisture transport;
- pressure/temperature gradients;
- spatial continuity;
- neighboring precipitation structures;
- orographic effects beyond the target boundary.

## 7.2 Do not use full attention over every context pixel

Recommended design:

```text
N × N coarse context
        |
        v
spatial encoder / patch encoder
        |
        v
context latent tokens
        |
        v
cross-attention into target representation
```

This allows the receptive field to grow without quadratic attention over the entire fine-resolution area.

---

# 17. Terrain and static context

Terrain should be injected as a spatial conditioning stream.

Initial terrain representation:

```text
DEM
slope
sin(aspect)
cos(aspect)
orographic / terrain-derived feature
```

The same feature definitions, ordering, normalization, and preprocessing must be used in:

- training;
- validation;
- inference;
- benchmarking.

Static geographic embeddings may also be investigated later.

---

# 18. Diffusion design

## 9.1 Recommended default

Use a **conditional residual diffusion decoder**.

```text
Baseline B(X)
      +
conditioning representation Z
      +
noisy residual R_t
      |
      v
Diffusion denoiser
      |
      v
predicted noise / velocity / residual objective
      |
      v
reverse diffusion
      |
      v
R
      |
      v
B + R
```

The initial prediction target should be the residual between a deterministic baseline and the fine-resolution target.

## 9.2 Diffusion formulation to investigate

The research implementation should support interchangeable prediction objectives, at minimum:

- epsilon prediction;
- v-prediction.

Sampler implementation should also be modular so that multiple solvers can be tested without rewriting the model.

Potential candidates:

- DDIM;
- Euler;
- Heun;
- DPM-style solvers if supported by the implementation stack.

The exact sampler is an experimental variable, not a permanent architecture decision.

---

# 19. Test-time compute scaling

This should be treated as a major research axis.

## Axis A: diffusion steps

Example sweep:

```text
4
8
12
16
24
32
```

Measure quality and compute jointly.

Important:

> More diffusion steps must not be assumed to produce better outputs.

The experiment is intended to determine the practical quality/compute frontier.

## Axis B: ensemble/sample count

Example sweep:

```text
1
4
16
32
64
```

Generate multiple conditional realizations and evaluate:

- uncertainty calibration;
- CRPS;
- ensemble spread;
- coverage;
- extreme-event probabilities;
- downstream decision metrics.

## Axis C: denoising compute vs ensemble compute

This is a dedicated research question:

> **Does spending compute on better denoising or on more ensemble members improve decision-relevant uncertainty more?**

For a roughly fixed inference budget, compare configurations such as:

| Configuration | Steps | Samples |
|---|---:|---:|
| A | 32 | 1 |
| B | 16 | 2 |
| C | 8 | 4 |
| D | 8 | 8 |
| E | 4 | 16 |

The exact budget should be normalized by measured wall-clock time and/or GPU compute rather than assuming steps and samples have identical cost.

This gives a meaningful test-time scaling curve.

---

# 20. Model-size scaling

Model capacity is a second major scaling axis.

Initial progression:

```text
Small dense
Medium dense
Large dense
Large sparse-MoE
Larger sparse-MoE
```

The goal is to investigate:

\[
\text{model capacity}
\rightarrow
\text{forecast skill / calibration}
\]

while tracking:

- total parameters;
- active parameters per token;
- training FLOPs;
- peak VRAM;
- inference latency;
- throughput.

---

# 21. MoE research

MoE should initially be used in **selected Transformer FFN blocks**, not every layer.

Preferred starting structure:

```text
Attention
   |
Dense / normalization
   |
MoE FFN
   |
Attention
   |
Dense / normalization
   |
MoE FFN
```

## 12.1 Sparsity-ratio research

Sparsity itself should be a research axis.

Define:

\[
\text{sparsity} =
1 -
\frac{\text{active experts}}
{\text{total experts}}
\]

Equivalent implementation choices can be expressed through:

- number of experts;
- top-k routing;
- fraction of blocks using MoE;
- active expert fraction.

### Example matrix

| Total experts | Top-k | Approx. active fraction |
|---:|---:|---:|
| 4 | 1 | 25% |
| 4 | 2 | 50% |
| 8 | 1 | 12.5% |
| 8 | 2 | 25% |
| 8 | 4 | 50% |
| 16 | 2 | 12.5% |
| 16 | 4 | 25% |

The final comparison should also investigate **MoE block density**:

```text
25% of Transformer blocks are MoE
50%
75%
100%
```

This directly tests the hypothesis:

> sparse expert capacity can increase representational capacity more efficiently than making every block dense.

---

# 22. Primary research matrix

This matrix is the central experiment space.

| Research axis | Candidate values | Primary question |
|---|---|---|
| History length | 3, 5, 7, 10, 14 days | How much past context is useful? |
| Spatial context N/M | 1.25×, 1.5×, 1.75×, 2×, 2.5× | How much surrounding area is useful? |
| Forecast horizon | 1–7 days | How does quality degrade with lead time? |
| Diffusion steps | 4, 8, 12, 16, 24, 32 | What is the useful denoising compute range? |
| Ensemble size | 1, 4, 16, 32, 64 | How much sampling improves uncertainty? |
| Denoising vs sampling budget | matched-compute configurations | Better denoising or more members? |
| Model size | S, M, L, XL | How does scaling affect skill? |
| MoE experts | 4, 8, 16 | How much sparse capacity helps? |
| MoE top-k | 1, 2, 4 | What sparsity is optimal? |
| MoE block coverage | 25%, 50%, 75%, 100% | Should every block be sparse? |
| Diffusion objective | epsilon, v | Which training target works best? |
| Sampler | DDIM, Euler, Heun, etc. | Which sampler gives the best compute/quality tradeoff? |
| Output representation | 6 channels vs expanded channel variants | Which variable representation works best? |

This full matrix should **not** be run as a Cartesian product.

Use staged experimentation and only expand dimensions that show promise.

---

# 23. Research progression

The project should follow this progression.

## Stage 0 — Data and experiment infrastructure

Before adding architectural complexity:

- strict train/validation/test temporal separation;
- exact forecast-issue-time provenance;
- no future-observation leakage;
- reproducible preprocessing;
- deterministic baseline;
- evaluation pipeline;
- experiment configuration files;
- checkpoint/version tracking;
- GPU memory and runtime measurement.

Every experiment should record:

```text
git commit
dataset version
date range
history length
context ratio
model configuration
parameter count
active parameter count
diffusion steps
sampler
ensemble size
training compute
inference compute
metrics
```

---

## Stage 1 — Small deterministic temporal model

Start with:

```text
history
+
future forecast
+
terrain
+
small spatial context
        |
        v
temporal Transformer + spatial CNN/UNet
        |
        v
7-day × 6-channel deterministic prediction
```

Purpose:

- establish the 7-day multivariate baseline;
- test whether temporal context helps before diffusion;
- make sure the data pipeline is correct.

---

## Stage 2 — Small diffusion model

Keep the architecture small.

Add:

```text
deterministic baseline
+
conditional residual diffusion
```

Purpose:

- verify that the residual distribution is learnable;
- measure deterministic vs probabilistic improvement;
- establish initial diffusion-step scaling.

---

## Stage 3 — Joint multivariate diffusion

Generate:

```text
7 days ×
[P, Tmax, Tmin, RH, U, V]
```

jointly.

Compare against:

```text
independent variable heads/models
```

Measure both individual skill and cross-variable consistency.

---

## Stage 4 — Spatial context

Introduce:

```text
N × N context
      >
M × M target
```

and run the first controlled N/M experiment.

Do not simultaneously increase model size.

---

## Stage 5 — History-length research

Run the initial temporal sweep:

```text
3
5
7
10
14 days
```

Select a provisional best range based on validation results, then carry that choice into later architecture experiments.

---

## Stage 6 — Diffusion-step research

After fixing a reasonable architecture:

```text
4
8
12
16
24
32 steps
```

Measure:

- forecast metrics;
- uncertainty metrics;
- latency;
- GPU memory;
- cost.

Determine the practical test-time compute frontier.

---

## Stage 7 — Ensemble compute research

At selected diffusion-step settings:

```text
1
4
16
32
64 samples
```

Measure whether additional ensemble members improve uncertainty estimates more than additional denoising steps.

Then perform matched-compute comparisons.

---

## Stage 8 — Dense model scaling

Scale:

```text
Small
→ Medium
→ Large
```

Keep diffusion and data setup fixed.

This isolates model-capacity scaling.

---

## Stage 9 — MoE scaling

Add sparse MoE only after dense scaling is understood.

Investigate:

1. expert count;
2. top-k;
3. expert sparsity;
4. MoE block coverage;
5. routing stability;
6. active parameters vs total parameters.

---

## Stage 10 — Combined scaling

Only after individual axes are understood should combinations be tested:

```text
large model
+
MoE
+
higher diffusion steps
+
larger ensemble
```

The goal is to determine whether the scaling axes are complementary or whether one saturates another.

---

# 24. Dedicated near-term research matrix

The immediate research priority is to determine:

1. optimal history length;
2. optimal N/M spatial context;
3. practical diffusion-step range.

These should be investigated **before aggressively scaling the model**.

## Experiment A — History length

Keep everything else fixed.

| Experiment | History |
|---|---:|
| H3 | 3 days |
| H5 | 5 days |
| H7 | 7 days |
| H10 | 10 days |
| H14 | 14 days |

### Metrics

- MAE
- RMSE
- precipitation skill
- extreme precipitation metrics
- spatial correlation
- temporal consistency
- multivariate consistency
- inference/training cost

### Decision

Identify:

- minimum history providing most of the available benefit;
- point of diminishing returns;
- whether different variables require different temporal memory.

---

## Experiment B — Spatial context N/M

Keep target size fixed.

Example:

| Experiment | N/M |
|---|---:|
| C1 | 1.25× |
| C2 | 1.50× |
| C3 | 1.75× |
| C4 | 2.00× |
| C5 | 2.50× |

### Metrics

In addition to normal forecast metrics:

- boundary consistency;
- spatial power/spectral statistics;
- precipitation structure;
- spatial correlation;
- extreme-event localization;
- compute/memory.

### Decision

Find the smallest context that captures most of the benefit.

This is preferable to blindly maximizing context.

---

## Experiment C — Diffusion-step range

Hold model and conditioning constant.

| Experiment | Steps |
|---|---:|
| D4 | 4 |
| D8 | 8 |
| D12 | 12 |
| D16 | 16 |
| D24 | 24 |
| D32 | 32 |

### Metrics

Separate:

**Forecast quality**

- MAE
- RMSE
- precipitation metrics
- extremes
- spatial metrics

**Probabilistic quality**

- CRPS
- calibration
- coverage
- ensemble spread
- reliability of extreme-event probabilities

**Compute**

- wall-clock latency;
- GPU utilization;
- peak VRAM;
- samples/sec;
- energy/compute proxy where feasible.

### Decision

Identify:

- lowest useful step count;
- point of diminishing returns;
- whether additional steps improve deterministic quality, uncertainty quality, or both.

---

# 25. Dedicated test-time compute research matrix

After the diffusion-step sweep, compare **compute allocation strategies**.

## Matched-compute experiment

Example:

| Strategy | Steps | Ensemble members | Intended test |
|---|---:|---:|---|
| A | 32 | 1 | deep denoising |
| B | 16 | 2 | balanced |
| C | 8 | 4 | more sampling |
| D | 8 | 8 | sampling-heavy |
| E | 4 | 16 | highly sample-heavy |

Actual combinations should be selected using measured runtime.

### Core question

> For a fixed inference budget, is compute better spent on deeper denoising of each sample or on more independently sampled forecasts?

This should be evaluated primarily using **decision-relevant probabilistic metrics**, not just mean RMSE.

---

# 26. Evaluation framework

The project should not rely only on average deterministic error.

## Deterministic accuracy

- MAE
- RMSE
- bias
- spatial correlation

## Precipitation-specific

- thresholded event metrics;
- heavy-rain recall/precision;
- extreme quantile error;
- spatial localization;
- precipitation structure.

## Probabilistic

- CRPS;
- ensemble spread;
- coverage;
- reliability/calibration;
- rank histograms where appropriate;
- probability of threshold exceedance.

## Multivariate consistency

Measure correlations and physically meaningful relationships between:

```text
P ↔ RH
P ↔ temperature
U/V ↔ precipitation
Tmax ↔ Tmin
wind ↔ humidity / moisture transport
```

The exact evaluation should use physically defensible relationships rather than rewarding correlation alone.

## Temporal consistency

Measure:

- day-to-day continuity;
- lead-time degradation;
- persistence;
- transitions;
- temporal extremes.

## Computational scaling

Every major experiment should report:

```text
parameter count
active parameter count
training FLOPs / proxy
inference latency
GPU memory
ensemble generation cost
diffusion steps
ensemble members
```

---

# 27. Required baselines

The research program should retain strong baselines so that every new component can justify itself.

### Baseline 0

Current deterministic single-day model.

### Baseline 1

Small deterministic multivariate 7-day model.

### Baseline 2

Temporal Transformer + spatial model, no diffusion.

### Baseline 3

Temporal model + deterministic residual head.

### Baseline 4

Temporal model + conditional residual diffusion.

### Baseline 5

Joint 7-day × 6-variable diffusion.

### Baseline 6

Joint diffusion + spatial context.

### Baseline 7

Joint diffusion + sparse MoE.

This creates a clean attribution chain.

---

# 28. Ablation philosophy

Every major research claim should have a controlled ablation.

Examples:

```text
Does history help?
    full history vs no history

Does future conditioning help?
    forecast vs no forecast

Does spatial context help?
    M×M vs N×N

Does joint generation help?
    independent vs joint channels

Does diffusion help?
    deterministic vs diffusion

Does more diffusion compute help?
    4 vs 8 vs 16 vs 32

Does more ensemble sampling help?
    1 vs 4 vs 16 vs 32

Does MoE help?
    dense vs sparse

Does increasing sparsity help?
    multiple top-k / expert configurations
```

---

# 29. Data and leakage rules

These are hard constraints.

## Forecast-time availability

Every input must be reproducibly available at the model's forecast initialization time.

## Future observations

Never use actual future observations as conditioning information.

## Historical forecasts

When used, retain:

- forecast initialization time;
- lead time;
- variable;
- spatial grid;
- model/source identity.

## Fine-resolution target provenance

Record the exact target source and date/time aggregation rule.

## Context window

Context must be derived from the same valid forecast-time information as the target conditioning region.

---

# 30. Compute/deployment constraint

The early implementation should remain compatible with the current student-tier target environment:

```text
2 × NVIDIA T4
```

Therefore:

- avoid full 3-D global attention;
- use patch/token representations;
- use spatial downsampling where possible;
- use gradient checkpointing;
- use mixed precision where numerically safe;
- profile memory before increasing context/model size;
- keep inference configuration modular.

The architecture should be designed so that a larger GPU/system can scale up naturally rather than forcing the base model to require it.

---

# 31. Suggested experiment configuration structure

Each run should be expressible through a configuration like:

```yaml
model:
  history_days: 7
  forecast_days: 7
  target_channels:
    - precipitation
    - tmax
    - tmin
    - rh
    - wind_u
    - wind_v

spatial:
  target_ratio: 1.0
  context_ratio: 1.75

diffusion:
  enabled: true
  objective: v
  sampler: euler
  steps: 12
  residual: true

ensemble:
  samples: 4

moe:
  enabled: false
  experts: 8
  top_k: 2
  block_fraction: 0.5

compute:
  mixed_precision: true
  gradient_checkpointing: true
```

This is important because the research matrix should be implemented through configuration rather than code edits.

---

# 32. Experiment naming

Use machine-readable IDs.

Examples:

```text
H07_C175_D12_E04_M_S
H05_C150_D08_E16_M_S
H10_C200_D16_E04_L_DENSE
H07_C175_D12_E16_XL_MOE8_K2
```

Suggested semantics:

```text
H07      = 7-day history
C175     = N/M = 1.75
D12      = 12 diffusion steps
E04      = 4 ensemble members
S/M/L    = model size
MOE8     = 8 experts
K2       = top-2 routing
```

A metadata file should record the full human-readable configuration.

---

# 33. Final target architecture

The mature architecture should look approximately like:

```text
                         PAST
             actual + historical forecasts
                         |
                         v
                  History Encoder
                         |
                         v
                   Temporal Latents
                         |
                         |\
                         | \
                         |  \
                         |   \
                         |    \
                         v     v
                 Future Forecast Encoder
                         |
                         v
                 History ↔ Future
                   Cross-Attention
                         |
              +----------+----------+
              |                     |
              v                     v
       Large N×N context       Target region
        spatial encoder       representation
              |                     |
              +----------+----------+
                         |
                     Terrain
                         |
                         v
               Spatiotemporal DiT
                         |
                  sparse MoE FFNs
                         |
                         v
               deterministic baseline
                         |
                         v
             Conditional residual DiT
                         |
                 reverse diffusion
                         |
              +----------+----------+
              |                     |
        sample 1 ... sample N       |
              |                     |
              +----------+----------+
                         |
                         v
                 7 × 6 × H × W
                         |
                         v
              probabilistic 7-day
              fine-resolution weather
```

---

# 34. Research questions the project should ultimately answer

### Temporal

1. How many days of historical context are actually useful?
2. Is actual weather alone sufficient, or do archived forecasts add useful information?
3. Does additional history improve all variables equally?

### Spatial

4. How large should the surrounding context be?
5. Does increasing N/M primarily help precipitation and wind?
6. Is there a physical/context-scale point of diminishing returns?

### Diffusion

7. How many denoising steps are practically useful?
8. Which sampler gives the best quality/latency tradeoff?
9. Does residual diffusion outperform deterministic residual prediction?

### Multivariate generation

10. Does joint generation improve cross-variable physical consistency?
11. Which variables benefit most from joint conditioning?
12. Which additional channels, if any, improve the representation?

### Uncertainty / test-time scaling

13. Is deeper denoising better than generating more ensemble members?
14. Does the optimal compute allocation depend on lead time?
15. Does the optimal compute allocation depend on the variable or event type?

### Model scaling

16. How does dense model size scale?
17. Does MoE provide a better parameter-quality scaling curve?
18. What expert count is useful?
19. What top-k / sparsity ratio is useful?
20. How many Transformer blocks should use MoE?

### Combined scaling

21. Are model scaling and diffusion-step scaling complementary?
22. Does model scaling reduce the number of diffusion steps required?
23. Does a larger model produce better-calibrated ensembles?
24. Where does each scaling axis saturate?

---

# 35. Definition of success

The project should not define success as:

> "the largest model with the lowest RMSE."

Instead, success means identifying a useful frontier across:

\[
\boxed{
\text{accuracy}
+
\text{physical consistency}
+
\text{uncertainty quality}
+
\text{model capacity}
+
\text{test-time compute}
}
\]

The strongest final result would show something like:

```text
                 QUALITY
                    ^
                    |
          ● ● ● ●
       ●
    ●
  ●
  +--------------------------> COMPUTE
```

with multiple selectable operating points:

```text
FAST       → few steps, few samples
BALANCED   → moderate steps, moderate samples
ACCURATE   → more steps
ENSEMBLE   → more samples
RESEARCH   → large MoE + higher compute
```

That makes the architecture useful not only as a single model, but as a **scalable weather-downscaling system whose inference quality can be traded against available compute**.
