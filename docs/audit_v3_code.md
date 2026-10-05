# Audit of the v3 pipeline (2026-10-01, 22:00-22:20 IST)

Triggered after two errors were found in my own work (CSS bias term, tie rule). Every component that feeds a
decision was re-checked against an independent computation.

| Component | Check | Result |
|---|---|---|
| Dataset `multitask_temporal_v3_realgfs.zarr` | 3 random samples (test 2023, train 2020, train 2021) recomputed from raw GFS npz, CHIRPS, ERA5-Land hourly | forecast, precip (7 leads), Tmax/Tmin/RH/U/V: max abs diff **0.0**; sea pixels NaN; history finite |
| Normaliser | fwd -> inv round trip (numpy and torch) | max err 4e-5 |
| Fair CRPS | vs brute-force double sum, K = 2/5/9 | 7e-15 |
| FSS box filter | vs brute-force window mean | 3e-15 |
| CSI / wet MAE / bias | hand-built hit/miss/false-alarm field | exact (0.3333 / 10.0 / 1.0) |
| Land mask | corrupt masked pixels | metrics unchanged |
| Spread-skill / coverage | synthetic calibrated 32-member ensemble | SSR 1.000; cov90 0.85 = finite-K expectation 0.9(K-1)/(K+1) -> report now shows the ideal per K |
| DDIM, DPM-Solver++(2M), stochastic DDIM | oracle denoiser must return x0 exactly | max err 5e-7 for 2-32 steps |
| Stochastic DDIM variance (eta>0) | vs DDPM posterior variance | **BUG FIXED**: extra `s_next^2/s^2` factor; now matches exactly. Only Sprint 7 eta=1 arm affected; it had not run |
| CSS bias component | ratio skill exploded when GFS nearly unbiased (D+5/6) | **BUG FIXED**: bounded `exp(-lb) - exp(-lb_ref)`; all CSS recomputed from stored aggregates |
| Tie rule | 2 x pooled sd tied configs 0.036 apart | **BUG FIXED**: 1 SE of the difference of seed means |
| Summary table | header missing the +precipQM column | **FIXED** |
| GFS-QM vs model reference | baseline script interpolated GFS in mm, trainer in log space | CSS 0.1868 vs 0.1886: negligible; matched value (0.1886 val / 0.1805 test) used from now on |
| Reporting coverage | S6 table lacked its deterministic rows; confirmation runs had no table | **FIXED** |

Runs launched before a fix carry the old code: their *stored* CSS uses the old bias term (the summarizer
recomputes CSS from the stored per-metric aggregates, so tables are correct), and their per-epoch checkpoint
selection used the old CSS (best epoch was the last epoch in every Phase A run, so selection was unaffected).
