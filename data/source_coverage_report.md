# Sprint 1 Deliverable: Source Coverage Report

**Project**: SIH-26074 Multivariate Spatiotemporal Diffusion Weather Downscaler  
**Audit Executed**: 2026-09-22T14:50:57.176651  
**Status**: All 5 Decision Gates Formally Evaluated  

---

## 1. Executive Summary & Source Decision Matrix

| Stream | Source | Temporal Range | Variables | Native Res | Provenance Class | Role in Pipeline | Audit Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **History Context** | ECMWF ERA5-Land | 2014–2023 | Tmax, Tmin, RH, U, V | ~0.1° | `reanalysis` | Past atmospheric context | Verified (HTTP 200) |
| **History Context** | UCSB CHIRPS p05 | 2014–2023 | Precipitation | 0.05° | `satellite_gauge_product` | Past precipitation context | Verified (HTTP 200) |
| **Future Forecast** | NOAA GFS 0.25° | 2015–2023 | P, Tmax, Tmin, RH, U, V | 0.25° | `numerical_weather_prediction` | Coarse 7-day forecast conditioning | Verified (AWS S3) |
| **Supervision Target**| UCSB CHIRPS p05 | 2014–2023 | Precipitation | 0.05° | `satellite_gauge_product` | Fine precipitation supervision | Verified (HTTP 200) |
| **Supervision Target**| ECMWF ERA5-Land | 2014–2023 | Tmax, Tmin, RH, U, V | 0.1° → 0.05° | `reanalysis` | Fine thermodynamic supervision | Verified (HTTP 200) |
| **Station Check** | NOAA GSOD / GHCN | 2014–2023 | Subset (T, P, DewPt) | Point AWS | `direct_observation` | Independent point validation | Verified (NCEI) |
| **Geophysical Prior**| Copernicus GLO-30| Static | Elev, Slope, Aspect, Curv, Lift | 30 m → 0.05° | `terrain_dsm` | Static topographical input | Verified (Local NC) |

---

## 2. Sprint 1 Binding Exit Decision Gates

### Gate 1: Daily Temporal Aggregation Convention (GO / NO-GO)
- **Option A (Source-Compatible Calendar Day, 00:00–24:00 UTC)**:
  - *Scientific Consequence*: Matches native daily CHIRPS v2.0 p05 aggregation. Zero temporal interpolation or slicing error on precipitation supervision targets. ERA5-Land and GFS hourly series are cleanly re-aggregated to 00:00–24:00 UTC.
  - *Status*: **RECOMMENDED BASELINE FOR MODEL TRAINING**.
- **Option B (Dual Temporal Definition)**:
  - *Scientific Consequence*: Model trained on 00:00–24:00 UTC calendar day; operational packaging creates 03:00–03:00 UTC advisories with an explicit 3-hour lag disclaimer.
  - *Status*: **ACCEPTABLE OPERATIONAL ADAPTER**.
- **Option C (Hourly Precipitation Reanalysis Source)**:
  - *Scientific Consequence*: Replace CHIRPS with ERA5-Land `total_precipitation` hourly. Allows arbitrary 03:00–03:00 UTC slicing, but degrades spatial accuracy from CHIRPS 0.05° satellite-gauge product down to 0.1° model reanalysis precipitation.
  - *Status*: **NOT RECOMMENDED**.

### Gate 2: 2015–2023 Forecast Archive Coverage Boundary
- **Empirical Finding**: The NOAA GFS 0.25° public archive on AWS Open Data (`s3://noaa-gfs-bdp-pds`) began on **January 15, 2015**. Year 2014 does not exist in this archive.
- **Binding Decision**:
  - **2015–2023 (9 years)**: The canonical dataset for all **forecast-conditioned spatiotemporal diffusion experiments**.
  - **2014 (1 year)**: Reserved strictly for history-only downscaling pretraining or unsupervised feature representation. Strictly excluded from forecast-conditioned benchmarks to prevent synthetic forecast fabrication.

### Gate 3: Exact 6-Variable GFS Forecast Derivation
- **Precipitation ($P$)**: `APCP` surface 6-hour buckets de-accumulated and summed over forecast day ($kg/m^2 \equiv mm$).
- **Max Temperature ($T_{\\max}$)**: `TMAX` 2m (or maximum across 3-hourly `TMP` 2m values) over forecast day minus 273.15 (°C).
- **Min Temperature ($T_{\\min}$)**: `TMIN` 2m (or minimum across 3-hourly `TMP` 2m values) over forecast day minus 273.15 (°C).
- **Relative Humidity ($RH$)**: `RH` 2m (or derived via August-Roche-Magnus from 2m temperature and dew point) clipped strictly to $[0, 100]\%$.
- **Wind ($U, V$)**: `UGRD` and `VGRD` at 10m above ground (3-hourly sequence and daily vector mean).

### Gate 4: Scalable Spatial Context Geometry ($M$ to $2.5M$)
- **Core Target Domain ($M$)**: 11.0°N–15.0°N, 74.0°E–78.0°E ($4^\circ \\times 4^\circ$, 80×80 fine at 0.05°, 16×16 coarse at 0.25°).
- **Maximum Context Domain ($2.5M$)**: 8.0°N–18.0°N, 71.0°E–81.0°E ($10^\circ \\times 10^\circ$, 40×40 coarse at 0.25°).
- **Binding Decision**: The data ingestion architecture will extract the full $10^\circ \\times 10^\circ$ maximum context domain, enabling Sprint 5 to empirically benchmark $1.0\\times, 1.25\\times, 1.5\\times, 1.75\\times, 2.0\\times, 2.5\\times$ context ratios without pipeline re-engineering.

### Gate 5: Provenance Classification & 00Z Anti-Leakage Boundary
- **Anti-Leakage Rule**: For a 00Z forecast initialized on Day $D$, history context must terminate strictly at $t \\le 00:00\\text{ UTC}$ of Day $D$ (completed Day $D-1$). Day $D$ completed observations cannot be included in history inputs.
- **Supervision Target Policy**: Zero claims of "ground truth". The fine target stack is formally documented as a **fine-resolution reference target** (`reanalysis` + `satellite_gauge_product`).
- **Station Layer Policy**: NOAA GSOD stations serve strictly as a sparse **independent point-validation layer**, evaluating only the subset of variables reported.
