# Audit of Sprints 1-9 (old repo, branch `feat/spatiotemporal-diffusion-downscaler`)

Audited 2026-09-29 .. 2026-10-01. The old repo was left untouched; everything new lives here.

## Data problems that invalidate the Sprint 3-9 numbers

| # | Problem | Evidence | Fix in v3 |
|---|---|---|---|
| 1 | `future_forecast` "GFS" was coarsened **observations** + small noise with fabricated GFS provenance (deleted script `scripts/stage_gfs_forecast_cache.py`, see `git show 089174e^:...`). | Precip anomaly corr vs truth 0.97 at day 7; forecasts for the same valid day from consecutive inits nearly identical. | Real GFS 0.25 deg 00Z, 1,096/1,098 monsoon inits 2015-2023 (AWS 2021-23, NCAR RDA ds084.1 2015-20). Two genuine archive holes (2015-09-06, 2016-09-26). Real skill: precip ACC 0.27 (D+0) -> 0.16 (D+6). (An early check suggested GFS was 1.4-2.2x too wet; that compared against CHIRPS coarse cells that included 0-filled sea pixels. Over land GFS is ~15-18 % too **dry** (bias ratio 0.82 val, 0.86 test).) |
| 2 | Tmax/Tmin/RH targets labelled ERA5-Land were not ERA5 at all. | Point corr with live ERA5/ERA5-Land -0.15..+0.22; 3-5 C too warm; no elevation dependence; zero corr with GFS at any lag. | ERA5-Land hourly from Copernicus CDS, 00-24 UTC daily stats. Validated: anomaly corr 0.84-0.99 with ERA5, Tmin-elevation corr -0.90, Bangalore Tmax 27.1 C (ECMWF archive 26.7 C). |
| 3 | Wind in km/h stored as m/s. | ERA5 file 15.2 m/s at Bangalore vs 4.1 m/s from ECMWF. | ERA5/ERA5-Land u10/v10 in m/s; mean speed 2.8-4.9 m/s. |
| 4 | Stores were 16x16 only; Sprint 5 "context" N=20/24/32 was **edge padding** (`np.pad(mode="edge")`). | `src/data/temporal_dataset.py:127`. | 40x40 coarse cells stored (2.5x context); centre crop gives real surroundings. |
| 5 | 21.6 % Arabian-Sea pixels (CHIRPS no-data) filled with 0 mm and scored. | 5,020 land px per UCSB vs 6,400 stored. | NaN + land mask; losses and metrics land-only. |
| 6 | Precip normalisation linear although documented as log1p; loss had hard-coded log1p constants; diffusion tail threshold hard-coded to the linear scale. | `normalization_stats_v2.yaml` (no transform key); `spatiotemporal_multitask_loss.py:62`; `residual_diffusion.py:417`. | log1p z-score from train split; constants read from the stats file. |
| 7 | Mass-conservation loss tied fine precip to the coarse input. | Real GFS is biased (0.82-0.86x over land), so this would teach the model GFS's bias. | Removed. |
| 8 | ERA5-Land 03-03 UTC day vs CHIRPS/GFS 00-24 UTC. | Ingestion provenance. | All sources 00-24 UTC. |

ERA5 0.25 deg history comes from the NCAR ds633.0 mirror, verified **bit-identical** to CDS ERA5 for June 2015
(T/Td/u/v max diff 2e-4; precip corr 1.00000 at 0 h offset, 0.68 at +-1 h).

## What each sprint was supposed to deliver vs. what it delivered

| Sprint | Plan deliverable for Sprint 10 | Status before this work |
|---|---|---|
| 1-2 | Real, documented, leakage-free dataset | Forecast and thermo targets not real (see above). |
| 3 | Deterministic 7x6 baseline, per-variable/lead metrics | Trained on fake inputs. |
| 4 | Provisional history length (H = 3/5/7/10/14) from validation + compute | Run with diffusion (not deterministic), 1 seed, near-tied losses, H=5/H=14 test rows identical. Oracle future makes history nearly useless, so the result cannot transfer. |
| 5 | Provisional context ratio via context tokens + cross-attention | Context was edge padding; 1.75/2.5 never run; N=16 had best val loss but N=24 chosen. |
| 6 | Deterministic vs deterministic+residual diffusion | Never compared (compared diffusion variants). |
| 7 | Min steps, diminishing returns, sampler, frontier, probabilistic metrics, VRAM | Quality decreased with steps; no CRPS/coverage; Euler/Heun, 12/24 steps missing. |
| 8 | Denoising vs ensemble compute, by lead/intensity/variable | Mostly done (on fake data); ensemble 4-5x under-dispersed; conformal bug (25.8 % coverage at 90 % nominal). |
| 9 | Dense S/M/L + MoE (experts 4/8/16, top-k 1/2/4, coverage 25-100 %) in transformer FFNs | Dense ladder done; MoE 1 of 36 configs; U-Net, not transformer; Wet-MAE 62 mm vs 6.5 mm protocol mismatch. |

Conclusion: every sprint question has to be re-answered on the real data; this repo does that.
