# Panchayat-scale weather downscaling (v3)

**Smart India Hackathon 2026 · Problem statement SIH26074:** *Downscaling of Weather Forecast from Block level to Panchayat level*
**Team FAST&CURIOUS** · Theme: Agriculture, FoodTech & Rural Development · Category: Software

The v3 system turns the coarse **NOAA GFS 0.25° (~25 km)** forecast into a **0.05° (~5 km)** panchayat-scale forecast:
* **Coverage:** 7 daily lead times and 6 variables (rain, Tmax, Tmin, RH, U and V wind) over Mandya and its surroundings (11–15°N, 74–78°E).
* **Model:** a spatiotemporal transformer, plus a cross-fitted residual-diffusion ensemble for uncertainty.
* **Operating modes:** four, from a 0.06 s deterministic forecast to a 16-member ensemble.
* **Web app:** serves 7-day forecasts and crop advisories for all 234 gram panchayats of Mandya.

---

## Run the web demo

```bash
git clone https://github.com/rohzhegde26/SIH-26074-Fast-and-Curious.git
cd SIH-26074-Fast-and-Curious
pip install fastapi uvicorn pydantic numpy
python -m uvicorn demo.server:app --port 8000
```

Open <http://localhost:8000>. The demo replays the **2023 monsoon** (Jun–Sep), the held-out test season that no model, calibration or selection step ever saw.

For every init date it shows:
* the v3 forecast in each of the four modes, next to the raw GFS input and the observed rain;
* 5–95 % ranges and threshold probabilities;
* proof plots of the real fields;
* the panchayat-level skill table.

See [`demo/README.md`](demo/README.md).

## Use the model

```bash
pip install -r requirements.txt
python scripts/download_weights.py          # 4 weight files (~416 MB) from release v3.0, SHA-256 verified
```

```python
from sihv3.predict import FinalDownscaler
m = FinalDownscaler("models/final")
out = m.predict(history, forecast, mode="BALANCED")   # history [B,3,6,40,40], forecast [B,7,6,40,40], normalised
out["mean"], out["prob"], out["rain_pm"]             # 0.05° fields, threshold probabilities, probability-matched rain
```

| Mode | What it runs | Test CSS | Latency (T4, batch 1) |
|---|---|---|---|
| FAST | 3-seed averaged deterministic backbone + rain quantile mapping | 0.266 | ~0.17 s |
| BALANCED | + residual diffusion, 24 DPM-Solver++ steps × 8 members | 0.283 | ~4 s |
| ACCURATE | 32 steps × 8 members | 0.284 | 5.3 s |
| ENSEMBLE | 24 steps × 16 members | **0.289** | 8.3 s |

All modes use < 1 GB VRAM.

* **CSS:** the composite skill score vs. bilinear-interpolated GFS, on the 2023 test season (higher is better).
* **Mode settings:** `configs/final/modes.yaml`.

## Results on the held-out 2023 season

Panchayat level: 122 forecasts × 234 panchayats × 7 days. Observed rain is CHIRPS.

| Forecast | Rain MAE (mm) | Wet-day MAE | Rain bias ratio | Tmax MAE (°C) | 5–95 % coverage |
|---|---|---|---|---|---|
| Raw GFS (interpolated) | 3.02 | 7.28 | 0.87 | 1.50 | – |
| v3 FAST | 2.99 | 6.39 | 1.02 | 1.15 | – |
| v3 BALANCED | 2.79 | 5.92 | 0.88 | 1.14 | 83 % |
| v3 ENSEMBLE | **2.75** | **5.83** | 0.88 | **1.14** | **88 %** |

* **Cross-fitted residual diffusion:** each training season's residual comes from a model that never saw that season. This cuts rain CRPS by 19 % compared with the standard in-sample recipe, and lifts the ensemble spread ratio from 0.4 to 1.1.
* **Rain tail cap:** stops unphysical extreme members (up to 1,100 mm/day before). No real 2023 day exceeds the cap.
* **Honest limits:**
  * With 8 training seasons the system is data-limited: bigger models did not help.
  * The 3-seed expectation for the diffusion pipeline is 0.263 ± 0.018 CSS; the shipped seed sits at the good end of that range.
  * Panchayat-level threshold probabilities do not yet beat climatology.

Full reports:
* [`reports/final_system_report.md`](reports/final_system_report.md): the shipped system (Sprints 10 and 11);
* [`docs/final_report.md`](docs/final_report.md): every sprint question, re-run on real data;
* [`docs/results_s11.md`](docs/results_s11.md): the Sprint 11 history tests and post-training improvements.

## How it was built

| Stage | Code |
|---|---|
| Real GFS 0.25° archive (AWS 2021+, NCAR RDA 2015–2020), ERA5 / ERA5-Land history, Copernicus GLO-30 terrain | `data_pipeline/` |
| Dataset: 2015–2023 monsoons, 7 leads × 6 variables, 40×40 coarse context → 80×80 fine target | `data_pipeline/build_dataset_v3.py` |
| Spatiotemporal transformer (MoE, 3 seeds), cross-fitted out-of-fold predictions, residual diffusion, spread calibration, rain QM and tail cap | `sihv3/` |
| Training and evaluation on Kaggle T4 GPUs (multi-account job launcher) | `kaggle/` |
| Final bundle export and reports | `scripts/export_final.py`, `scripts/report_final.py` |

**Splits:** trained on 2015–2022 and tested on 2023. Every setting was chosen on 2022 or out-of-fold data, never on 2023.

**Why v3 exists:** the audit [`docs/audit_previous_sprints.md`](docs/audit_previous_sprints.md) found that the earlier pipeline's "GFS" input was coarsened observations plus a little noise, not a real forecast. v3 rebuilds the dataset from genuine GFS forecasts and re-runs every sprint question, so all numbers here are what a real operational forecast would get.

## Repository layout

| Path | Contents |
|---|---|
| `sihv3/` | Model, data loader, training, sampling, calibration, prediction API |
| `models/final/` | Calibration, normalisation and static inputs (weights: `scripts/download_weights.py`) |
| `configs/final/` | Operating-mode settings and provenance |
| `demo/` | FastAPI server + PWA frontend, precomputed 2023 forecasts |
| `data_pipeline/` | GFS / ERA5 fetchers and dataset builder |
| `kaggle/` | GPU job launcher, scheduler and dashboard |
| `docs/`, `reports/`, `results/` | Sprint reports, decision log, per-run results |

## Data sources

All data is public:
* **NOAA GFS 0.25°:** AWS Open Data and NCAR RDA ds084.1.
* **ECMWF ERA5:** NCAR RDA ds633.0.
* **ERA5-Land:** Copernicus CDS.
* **CHIRPS v2.0:** UCSB Climate Hazards Center.
* **Copernicus GLO-30 DEM.**

## License

Code: [MIT](LICENSE). Data remains under its providers' terms. Leaflet is BSD-2-Clause.

Earlier prototypes (Dense-L, U-Net) are preserved in this repository's git history and on its other branches.
