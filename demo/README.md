# SIH-26074 v3 web demo — Mandya panchayat agro-weather advisory

The Mandya PWA from the original project (`frontend/`, copied unchanged in layout) served on top of the **v3 model**
(`models/final`): spatiotemporal transformer + cross-fitted residual diffusion, 0.25° GFS → 0.05°, 7 days × 6
variables, four operating modes.

```bash
pip install fastapi uvicorn
python -m uvicorn demo.server:app --port 8000      # from the repo root, then open http://localhost:8000
```

## What you see

* **Mode selector** — FAST / BALANCED / ACCURATE / ENSEMBLE (`configs/final/modes.yaml`). FAST is deterministic
  (no range); the diffusion modes give a 5–95 % range and threshold probabilities (`prob_ge_15mm`, …).
* **Date selector** — every init date of the **2023 monsoon (Jun–Sep)**, the held-out test season that no model,
  calibration or selection decision used. The default is the date whose observed week was wettest. This is a replay:
  each forecast is shown next to what was observed.
* **Map layers** — v3 rain, Tmax, humidity, wind, risk, uncertainty, **raw GFS 0.25°** (the model input at each
  panchayat) and **observed** rain (CHIRPS).
* **Proof plots** — real fields over Mandya for the selected day: raw GFS 0.25° blocks, the v3 FAST forecast at 0.05°,
  and the observed rain.
* **Skill table / model card** (`/api/v1/model-card`) — panchayat-level error of each mode vs observed, next to
  raw GFS, computed by the server from the 2023 data.
* **Live inference** (`POST /api/v1/infer`) — runs FAST on this machine from the real GFS + ERA5 inputs of the chosen
  date (needs the v3 dataset: set `SIH_DATA`), and reports how closely it matches the GPU-precomputed forecast.

## Skill on the 2023 held-out season (panchayat level, 122 forecasts × 234 GPs × 7 days)

Computed by the server (`/api/v1/model-card`) from `data/serving/precomputed/`:

| forecast | rain MAE (mm) | wet-day MAE (obs > 2.5 mm) | rain bias ratio | Tmax MAE (°C) | 5–95 % range coverage |
|---|---|---|---|---|---|
| raw GFS (interpolated) | 3.02 | 7.28 | 0.87 | 1.50 | – |
| v3 FAST | 2.99 | 6.39 | 1.02 | 1.15 | – (deterministic) |
| v3 BALANCED | 2.79 | 5.92 | 0.88 | 1.14 | 83 % |
| v3 ACCURATE | 2.85 | 5.83 | 0.93 | 1.14 | 84 % |
| v3 ENSEMBLE | **2.75** | **5.83** | 0.88 | **1.14** | **88 %** |

Known weakness: the panchayat-level threshold probabilities (share of members whose panchayat-mean rain exceeds
15 / 30 mm) do not beat climatology (Brier ≥ 15 mm 0.028 vs 0.027, with climatology taken from 2023 itself). With
8–16 members they come in coarse steps; a pixel-level exceedance probability averaged over the panchayat is the
obvious next step.

## Data flow

| step | code | output |
|---|---|---|
| panchayat ↔ 0.05° cell area weights (234 GPs, 1–14 cells each) | `backend/gp_weights.py` | `data/serving/gp_cells.json` |
| all four modes × 122 dates on a GPU, aggregated per panchayat | `sihv3/precompute.py` (Kaggle) | `data/serving/precomputed/gp_<MODE>.npz`, `gp_obs.npz` |
| proof-plot fields (raw GFS, v3 FAST, observed) | `backend/precompute_fields.py` (CPU) | `data/serving/precomputed/fields_FAST.npz` |
| records in the PWA's schema, real 7-day items, advisories | `backend/forecasts.py`, `backend/advisory/` | `/api/forecasts` |

## Honest notes

* **Changed from the original demo** (whose claims belonged to its earlier U-Net model): days 2–7 and their
  temperature/humidity were extrapolated from day 1 there; here every day and variable comes from the model.
  The "IMD block" layer painted the model's own district mean; here it is the real raw GFS forecast. The proof plots
  were drawn from a formula; here they are real fields. Hard-coded scores (CQR coverage, mass-conservation figures)
  were replaced by numbers the server computes from the 2023 data. The validation store starts empty
  (no seeded submissions).
* v3 is **not mass-conserving**: it corrects GFS rain, which is ~15–18 % too dry over this land.
* Crop stages follow a simple kharif calendar (ragi: sowing Jun, vegetative Jul–Aug, flowering Sep; paddy: sowing
  Jun–Jul, vegetative Aug, flowering Sep; sugarcane: grand growth). The advisory rules themselves are the original
  project's (`backend/advisory/`).
* Panchayat values are area-weighted means of 0.05° cells (~30 km²), so neighbouring small panchayats can share a cell.
* "SIH 2026 · PS 26074" prototype: not an official service of any ministry.
