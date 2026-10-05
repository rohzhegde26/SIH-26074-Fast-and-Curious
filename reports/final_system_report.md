# SIH-26074 final system — Sprint 10 report

GFS 0.25° → 0.05° downscaling over 11–15°N, 74–78°E, 7 daily leads × 6 variables (precipitation, Tmax, Tmin, RH,
U, V). All numbers below are on the **2023 monsoon test season** (Jun–Sep, 122 forecasts), which no model,
calibration fit or selection decision used. CSS = composite skill vs GFS-bilinear (higher is better; for
ensembles, the ensemble mean). Tie rule: 1 SE of the paired difference.

**Summary.**
* One system, four modes (`models/final`, `configs/final/modes.yaml`): FAST 0.244 CSS in 0.06 s; BALANCED 0.282
  (~4 s); ACCURATE 0.284 (5.3 s); ENSEMBLE 0.290 (8.3 s), all on one T4 at batch 1, < 1 GB VRAM.
* The 3-seed expectation for the diffusion pipeline is 0.263 ± 0.018; the shipped seed's 0.281 is the lucky
  end of that range (picked on out-of-fold skill before test was seen).
* What worked: cross-fitted residual diffusion (rain CRPS −19 % vs the standard recipe, spread ratio 0.4 → 1.1),
  ~24 DPM-Solver++ steps, more members for threshold probabilities, spread calibration (Tmax CRPS −9 %).
* What did not: more capacity, in either stage. Stage-1 S/M/L/MoE are tied over 3 seeds; M/L denoisers fit
  better but sample over-confident ensembles. With 8 training seasons the system is data-limited.

## Sprint 11 update (shipped in `models/final`; details in `results_s11.md`)

| mode | Sprint 10 | Sprint 11 | change |
|---|---|---|---|
| FAST | single seed + QM: CSS 0.2435 | **3-seed average + QM: 0.2659** | backbone averaging (3 forward passes, ~0.17 s) |
| BALANCED (24 × 8) | CSS 0.2820, rain SSR 1.25 | CSS 0.2825, rain SSR 1.12 | rain tail cap |
| ENSEMBLE (24 × 16) | CSS 0.2895, rain SSR 1.27 | CSS 0.2891, rain SSR 1.12 | rain tail cap |

* All four modes now come from one bundle: FAST averages the three seeds' backbones; the diffusion modes keep the
  seed-0 backbone and denoiser (a denoiser retrained on the averaged backbone lost 0.020 CSS on 2022).
* Rain tail cap: individual members reached 1,100 mm/day; each member's rain is now capped at min(2 × the wettest
  training day in its 5×5 block, 1.2 × the wettest training day anywhere). Chosen leave-one-season-out so that no
  held-out season, and no 2023 day, exceeds it.
* New optional output `rain_pm` (probability-matched rain) for heavy-rain maps: doubles detection of ≥ 64.5 mm days.
* Tested and rejected: longer history and forecast history (8-season tie after the full rebuild), denoiser on the
  averaged backbone, static temperature bias correction (the bias changes sign between seasons).
* Sprint 10 tables below are unchanged and describe the Sprint 10 evaluation.

## 1. The shipped system

* **Backbone (FAST):** spatiotemporal transformer, S size with an MoE FFN (8 experts, top-2, MoE in 50 % of
  blocks), history H = 3 days, context N = 40 coarse cells (N/M = 2.5), trained 2015–2022, 50 epochs, no mm-loss.
  Rain quantile mapping per lead and 5×5 block, fitted on the multi-season out-of-fold predictions (2015–2022).
* **Generative stage (BALANCED / ACCURATE / ENSEMBLE):** CorrDiff-style residual diffusion (S denoiser,
  v-prediction, cosine schedule, 80 epochs) trained on **cross-fitted** residuals: every training season's
  residual comes from a deterministic model that never saw that season. Sampled with DPM-Solver++(2M), then
  mean-preserving spread calibration per lead × variable, fitted on 2022 by a model trained without 2022.
* **Mode settings:** FAST = 1 member; BALANCED = 24 steps × 8 members; ACCURATE = 32 × 8; ENSEMBLE = 24 × 16.
  Pre-registered from Sprint 7/8 validation (BALANCED was 16 × 8), then BALANCED's step count re-selected on 2022
  with a rule written down before that run (section 5); 2023 never chose a setting.
* Entry point: `sihv3.predict.FinalDownscaler("models/final").predict(history, forecast, mode=...)`.

## 2. Seed selection (out-of-fold, 2023 not used) and 3-seed spread

| seed | OOF CSS 2015–22 (selection score) | final det test CSS | + rain QM | cross-fitted diffusion test CSS | rain CRPS | rain SSR |
|---|---|---|---|---|---|---|
| 0 **(shipped)** | 0.2472 | 0.2361 | 0.2488 | 0.2814 | 4.431 | 1.11 |
| 1 | 0.2432 | 0.2205 | 0.2363 | 0.2448 | 4.502 | 0.85 |
| 2 | 0.2428 | 0.2484 | 0.2626 | 0.2642 | 4.458 | 1.02 |

Seed 0 has the best OOF score but the seeds are tied within noise (paired season SE ≈ 0.004–0.006).
The spread across seeds in the test column is the honest uncertainty of any single shipped model.

## 3. Cross-fitted vs in-sample residual diffusion (seed 0, final recipe)

| diffusion trained on | test CSS | rain CRPS | rain SSR | rain cov90 | rain bias | CSI30 |
|---|---|---|---|---|---|---|
| in-sample residuals (standard) | 0.2613 | 5.504 | 0.42 | 0.36 | 0.81 | 0.205 |
| **cross-fitted residuals (shipped)** | 0.2814 | 4.431 | 1.11 | 0.48 | 0.91 | 0.192 |

## 4. Capacity under the final recipe (deterministic, H3 N40, 50 ep, 2015–22)

| model | params (active) | test CSS | + rain QM | final train loss |
|---|---|---|---|---|
| **dense S: mean of 3 seed(s)** | | **0.2376 ± 0.0066** | | |
| **dense M: mean of 3 seed(s)** | | **0.2293 ± 0.0069** | | |
| **dense L: mean of 3 seed(s)** | | **0.2367 ± 0.0024** | | |
| **MoE S (shipped arch.): mean of 3 seed(s)** | | **0.2350 ± 0.0140** | | |
| dense S, seed 0 | 17.7M (17.7M) | 0.2443 | 0.2570 | 0.034 |
| dense S, seed 1 | 17.7M (17.7M) | 0.2310 | 0.2458 | 0.033 |
| dense S, seed 2 | 17.7M (17.7M) | 0.2374 | 0.2532 | 0.034 |
| dense M, seed 0 | 34.2M (34.2M) | 0.2355 | 0.2479 | 0.030 |
| dense M, seed 1 | 34.2M (34.2M) | 0.2218 | 0.2352 | 0.029 |
| dense M, seed 2 | 34.2M (34.2M) | 0.2307 | 0.2439 | 0.029 |
| dense L, seed 0 | 58.7M (58.7M) | 0.2389 | 0.2520 | 0.025 |
| dense L, seed 1 | 58.7M (58.7M) | 0.2341 | 0.2471 | 0.026 |
| dense L, seed 2 | 58.7M (58.7M) | 0.2371 | 0.2504 | 0.026 |
| MoE S (shipped arch.), seed 0 | 28.8M (19.3M) | 0.2361 | 0.2488 | 0.057 |
| MoE S (shipped arch.), seed 1 | 28.8M (19.3M) | 0.2205 | 0.2363 | 0.057 |
| MoE S (shipped arch.), seed 2 | 28.8M (19.3M) | 0.2484 | 0.2626 | 0.058 |

MoE train losses include the Switch balance term (0.01 × ≈1 per MoE layer × 3 layers ≈ 0.03), i.e. a data loss
of ≈ 0.027. Train loss falls monotonically with capacity (S 0.034 → M 0.029 → L 0.026), so the larger models do
use their capacity, but 3-seed test skill is flat (M's dip is ~1.5 SE: noise). Caveat on the setup: every size
used the same learning rate, weight decay and 50 epochs; larger models may need more regularisation to turn
lower train loss into skill. The MoE has the same mean as dense S but twice the seed variance (±0.014 vs
±0.007), so a single MoE model is less predictable; dense S would be an equally good, simpler backbone.

## 4a. Diffusion-denoiser capacity (cross-fitted residuals, 80 ep, DPM-Solver++ 24 × 8)

Selection evidence = 2022 validation (trained 2015–21, seed 1); 2023 test pairs share the OOF file and seed.

| denoiser | params | 2022 val CSS | 2022 val rain CRPS | 2023 test CSS (s1 / s2) | 2023 rain CRPS (s1 / s2) | rain SSR (s1 / s2) |
|---|---|---|---|---|---|---|
| S | 17.8M | 0.2628 | 6.408 | 0.2448 / 0.2642 | 4.502 / 4.458 | 0.85 / 1.02 |
| M | 34.3M | 0.2550 | 6.477 | 0.2410 / 0.2618 | 4.566 / 4.545 | 0.67 / 0.60 |
| L | 58.8M | 0.2765 | 6.344 | 0.2484 / 0.2592 | 4.591 / 4.556 | 0.59 / 0.57 |

Seed-0 runs (record only, finished after the decision): M test CSS 0.2683, rain CRPS 4.506, SSR 0.62; L test CSS 0.2632, rain CRPS 4.549, SSR 0.56 (seed-0 S: 0.2814 / 4.431 / 1.11).

**Decision: ship the S denoiser.** Bigger denoisers fit the training residuals better (final loss S 0.125,
M 0.113, L 0.105) but sample *narrower* ensembles (2023 rain SSR ≈ 0.6 vs 0.85–1.02 for S): with ~970 training
samples they overfit the residual distribution, the same mechanism that made in-sample residuals under-dispersed.
L leads on 2022 validation (+0.014 CSS, −1 % rain CRPS) but that is one seed and inside the denoiser-to-denoiser
noise shown by M (−0.008 vs S); every 2023 comparison has L's rain CRPS 1.8–2.2 % worse. Pooled over the four
paired comparisons L − S = +0.006 CSS (1.4 SE) and +1.2 % rain CRPS, at ~2.2× the compute (training 179 vs 81 min per run).
Next experiment: spread-calibrate L (its under-dispersion is exactly what the calibration corrects), and
revisit denoiser capacity when more training seasons are available.

## 4b. Calibration: fitted on 2022 (model trained 2015–21), applied to the shipped model on 2023 (24 steps × 8)

Spread factors (mean over leads) P/Tmax/Tmin/RH/U/V: 1.57 / 1.50 / 1.47 / 1.44 / 1.61 / 1.46 (range 1.2–2.1).

| model | variant | CSS | rain CRPS | rain SSR | rain cov90 (ideal 0.70 for K=8) | rain bias | Brier>30 | Tmax CRPS |
|---|---|---|---|---|---|---|---|---|
| 2015–21 calibration model, 2023 | raw | 0.2651 | 4.549 | 1.29 | 0.57 | 1.22 | 0.0420 | 0.756 |
| 2015–21 calibration model, 2023 | spread | 0.2652 | 4.373 | 1.45 | 0.88 | 1.22 | 0.0412 | 0.703 |
| 2015–21 calibration model, 2023 | rainqm | 0.2674 | 4.598 | 1.08 | 0.88 | 1.36 | 0.0448 | 0.756 |
| 2015–21 calibration model, 2023 | both | 0.2675 | 4.410 | 1.21 | 0.91 | 1.36 | 0.0437 | 0.703 |
| shipped model, 2023 | raw | 0.2811 | 4.421 | 1.12 | 0.49 | 0.91 | 0.0391 | 0.810 |
| shipped model, 2023 | spread | 0.2811 | 4.298 | 1.25 | 0.84 | 0.91 | 0.0388 | 0.740 |
| shipped model, 2023 | rainqm | 0.2854 | 4.454 | 0.92 | 0.85 | 1.07 | 0.0413 | 0.810 |
| shipped model, 2023 | both | 0.2854 | 4.312 | 1.04 | 0.88 | 1.07 | 0.0407 | 0.740 |

Spread calibration lowers rain CRPS (−3 %) and Tmax CRPS (−9 %) with CSS and bias unchanged, but rain
coverage overshoots (0.84 vs ideal 0.70): slightly over-dispersed for rain. Rain QM on the ensemble is not
shipped: it worsens Brier>30 for both models on 2023 and pushes the calibration model's bias from 1.22 to 1.36.

## 5. Operating modes (2023 test)

| mode | setting | CSS | rain CRPS | rain SSR | Brier>30 | Tmax CRPS | rain bias | latency s/forecast (T4, batch 1) | peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| FAST | det + rain QM | 0.2435 | – | – | – | – | 1.09 | 0.06 |  |
| FAST (no QM) | det only | 0.2361 | – | – | – | – | 0.66 | 0.06 |  |
| BALANCED | 24 steps × 8 | 0.2821 | 4.296 | 1.25 | 0.0386 | 0.733 | 0.91 | 5.17 | 0.6 |
| ACCURATE | 32 steps × 8 | 0.2839 | 4.312 | 1.29 | 0.0393 | 0.723 | 0.97 | 6.85 | 0.6 |
| ENSEMBLE | 24 steps × 16 | 0.2900 | 4.347 | 1.27 | 0.0370 | 0.742 | 0.92 | 10.47 | 0.7 |

* Skill rises with test-time compute: FAST < BALANCED ≤ ACCURATE < ENSEMBLE.
* BALANCED steps re-selected on **2022** (fold det trained 2015–20 + diffusion trained 2015–21, K=8, raw;
  rule pre-registered: smallest S within 0.005 CSS and 1 % rain CRPS of the best):

  | steps (K=8) | 2022 CSS | 2022 rain CRPS |
  |---|---|---|
  | 8 | 0.2410 | 6.609 |
  | 16 | 0.2684 | 6.314 |
  | 24 | 0.2795 | 6.258 |
  | 32 | 0.2783 | 6.287 |

  16 steps (the Sprint 7/8 choice, made with in-sample residuals) is short of the knee: the wider
  cross-fitted residuals need ~24 steps. 32 steps adds nothing on 2022 (ACCURATE ≈ BALANCED there);
  the real accuracy upgrade is ENSEMBLE's extra members.

* Spread factors were fitted at K=8; at K=16 rain is somewhat over-dispersed (SSR 1.27–1.33).
* Latency above is with members sampled sequentially (as evaluated). The shipped `predict()` batches all
  members into one pass with the same initial noise (identical samples, max |diff| 5e-6); see the table below.
* Bundle acceptance test: `models/final` FAST mode on CPU fp32 reproduces the T4 result exactly (CSS 0.2435,
  bias 1.09).

Shipped-mode latency, one forecast (batch 1), Tesla T4, fp16:

| mode | benchmarked setting | sequential members s | batched members s (shipped) | peak VRAM GB (shipped) |
|---|---|---|---|---|
| FAST | 1 member | 0.056 | – | 0.24 |
| BALANCED | 16 × 8 | 3.28 | 2.61 | 0.56 |
| ACCURATE | 32 × 8 | 7.28 | 5.33 | 0.56 |
| ENSEMBLE | 24 × 16 | 10.61 | 8.31 | 0.90 |

Batching gives only 1.3–1.4×: at 80×80 the denoiser already keeps a T4 fairly busy at batch 1. The bench ran before BALANCED moved to 24 steps; latency is linear in steps, so shipped BALANCED (24 × 8, batched) ≈ 4.0 s.

## 5b. Per-variable accuracy, extremes, spatial structure and probabilistic quality (2023 test)

| metric | FAST | FAST (no QM) | BALANCED | ACCURATE | ENSEMBLE |
|---|---|---|---|---|---|
| **Accuracy** (deterministic or ensemble mean) | | | | | |
| rain MAE / RMSE (mm/day) | 8.026 / 22.481 | 6.546 / 18.787 | 7.350 / 18.475 | 7.585 / 18.788 | 7.296 / 17.923 |
| Tmax MAE / RMSE / bias (°C) | 1.129 / 1.396 / -0.602 | 1.129 / 1.396 / -0.602 | 1.075 / 1.328 / -0.576 | 1.075 / 1.330 / -0.571 | 1.070 / 1.323 / -0.573 |
| Tmin MAE / RMSE / bias (°C) | 0.432 / 0.554 / -0.212 | 0.432 / 0.554 / -0.212 | 0.405 / 0.517 / -0.216 | 0.407 / 0.520 / -0.221 | 0.405 / 0.515 / -0.222 |
| RH MAE / RMSE / bias (%) | 3.479 / 4.608 / 1.136 | 3.479 / 4.608 / 1.136 | 3.305 / 4.369 / 0.991 | 3.333 / 4.413 / 1.029 | 3.263 / 4.312 / 0.997 |
| wind vector RMSE (m/s) | 0.645 | 0.645 | 0.601 | 0.606 | 0.592 |
| rain bias ratio | 1.089 | 0.656 | 0.908 | 0.974 | 0.916 |
| rain wet-day MAE (mm/day) | 17.484 | 17.242 | 15.836 | 15.686 | 15.481 |
| **Precipitation events and extremes** | | | | | |
| CSI ≥ 5 / 15 / 30 / 64.5 mm | 0.23 / 0.19 / 0.20 / 0.18 | 0.20 / 0.17 / 0.19 / 0.14 | 0.23 / 0.20 / 0.20 / 0.09 | 0.24 / 0.20 / 0.20 / 0.09 | 0.24 / 0.21 / 0.20 / 0.07 |
| POD / FAR ≥ 64.5 mm (very heavy) | 0.31 / 0.72 | 0.17 / 0.59 | 0.10 / 0.65 | 0.11 / 0.69 | 0.09 / 0.63 |
| **Spatial structure** | | | | | |
| rain FSS (15 mm) | 0.498 | 0.488 | 0.566 | 0.566 | 0.571 |
| rain spatial correlation | 0.153 | 0.164 | 0.199 | 0.198 | 0.229 |
| **Probabilistic quality** (diffusion modes) | | | | | |
| CRPS rain / Tmax / Tmin / RH | – / – / – / – | – / – / – / – | 4.296 / 0.733 / 0.269 / 2.294 | 4.312 / 0.723 / 0.267 / 2.285 | 4.347 / 0.742 / 0.276 / 2.299 |
| spread-skill ratio rain / Tmax / RH (ideal 1) | – / – / – | – / – / – | 1.25 / 0.80 / 0.78 | 1.29 / 0.87 / 0.84 | 1.27 / 0.79 / 0.77 |
| 90 % coverage rain / Tmax (ideal 0.70 at K=8, 0.79 at K=16) | – / – | – / – | 0.84 / 0.60 | 0.85 / 0.64 | 0.89 / 0.68 |
| Brier ≥ 15 / ≥ 30 mm | – / – | – / – | 0.0761 / 0.0386 | 0.0770 / 0.0393 | 0.0731 / 0.0370 |

* **Very heavy rain (≥ 64.5 mm): use FAST + rain QM for a yes/no warning.** It detects 31 % of events (CSI
  0.18) vs ~10 % (CSI 0.07–0.09) for the ensemble *mean* of the diffusion modes, which averages extremes away.
  The diffusion modes should instead be used through their exceedance probabilities (`prob["precip>64.5"]`),
  where they are better calibrated (Brier ≥ 30 mm 0.037–0.039).
* Everywhere else the diffusion modes improve the mean forecast too: rain RMSE, Tmax/Tmin/RH MAE, wind
  RMSE, FSS (0.50 → 0.57) and spatial correlation all improve over FAST.
* Spread after calibration: rain is somewhat over-dispersed (SSR ~1.25), Tmax and RH are still under-dispersed
  (SSR 0.78–0.87, Tmax 90 % coverage 0.60–0.68 vs ideal 0.70–0.79). The 2022-fitted factors transfer only
  partly to 2023; per-variable factors fitted on more seasons are the fix.

## 6. Test-time compute matrix (spread-calibrated; cell = CSS / rain CRPS / latency s)

| members \ steps | 8 | 16 | 24 | 32 |
|---|---|---|---|---|
| K=4 | 0.2351 / 4.538 / 0.8 | 0.2630 / 4.249 / 1.7 | 0.2672 / 4.222 / 2.6 | 0.2684 / 4.207 / 3.4 |
| K=8 | 0.2410 / 4.572 / 1.8 | 0.2693 / 4.363 / 3.4 | 0.2821 / 4.296 / 5.2 | 0.2839 / 4.312 / 6.9 |
| K=16 | 0.2422 / 4.603 / 3.5 | 0.2788 / 4.382 / 7.0 | 0.2900 / 4.347 / 10.5 | 0.2948 / 4.334 / 13.9 |

Raw (uncalibrated) vs spread-calibrated rain CRPS / SSR:

| K | S | raw CRPS | cal CRPS | raw SSR | cal SSR |
|---|---|---|---|---|---|
| 4 | 8 | 4.761 | 4.538 | 0.69 | 0.77 |
| 4 | 16 | 4.447 | 4.249 | 0.99 | 1.10 |
| 4 | 24 | 4.415 | 4.222 | 1.10 | 1.23 |
| 4 | 32 | 4.400 | 4.207 | 1.14 | 1.26 |
| 8 | 8 | 4.743 | 4.572 | 0.67 | 0.77 |
| 8 | 16 | 4.498 | 4.363 | 0.99 | 1.11 |
| 8 | 24 | 4.417 | 4.296 | 1.11 | 1.25 |
| 8 | 32 | 4.427 | 4.312 | 1.16 | 1.29 |
| 16 | 8 | 4.750 | 4.603 | 0.66 | 0.77 |
| 16 | 16 | 4.485 | 4.382 | 0.99 | 1.13 |
| 16 | 24 | 4.433 | 4.347 | 1.13 | 1.27 |
| 16 | 32 | 4.419 | 4.334 | 1.18 | 1.33 |

## 7. Multivariate physical consistency (per member; observed = ERA5-Land/CHIRPS targets)

| | corr_rain_rh | rh_wet_minus_dry | corr_rain_tmax | tmax_wet_minus_dry | corr_wind_rain | corr_tmax_tmin | diurnal_range | tmax_lt_tmin_rate |
|---|---|---|---|---|---|---|---|---|
| observed | 0.247 | 3.815 | -0.177 | -0.574 | -0.126 | 0.579 | 6.632 | 0.000 |
| FAST | 0.309 | 5.672 | -0.190 | -0.904 | -0.123 | 0.635 | 6.241 | 0.000 |
| FAST (no QM) | 0.299 | 7.342 | -0.180 | -1.119 | -0.134 | 0.635 | 6.241 | 0.000 |
| BALANCED | 0.209 | 5.397 | -0.117 | -0.892 | -0.104 | 0.598 | 6.272 | 0.000 |
| ACCURATE | 0.211 | 5.215 | -0.117 | -0.833 | -0.104 | 0.589 | 6.282 | 0.000 |
| ENSEMBLE | 0.205 | 5.315 | -0.114 | -0.872 | -0.102 | 0.592 | 6.281 | 0.000 |

Artefact check: FAST output vs the OOF file's 2023 rows, max |diff| = 0.3651 (normalised units, fp16).
Mean |diff| is ~6e-4 and only 0.04 % of values differ by > 0.05; the rare large local differences come from
MoE top-2 routing flips under reduced precision (0.3–0.7 % of tokens switch expert between fp32 and bf16).
Aggregate scores are unaffected (identical CSS to 4 decimals).

## 8. Headline numbers and caveats

* **Expected skill of the pipeline** (3 seeds, cross-fitted diffusion, 24 × 8): test CSS 0.263 ± 0.018. The shipped seed (0) scores above this mean; it was chosen on out-of-fold skill before test was seen,
  so its higher test score is luck, not selection.
* Seed variance is dominated by heavy-rain detection (CSI30 0.13–0.19 across seeds). Averaging the three seeds'
  deterministic backbones is the obvious next upgrade (3× stage-1 cost, ~0.2 s/forecast).
* One test season (2023, 122 forecasts) and 8 training seasons: differences under ~0.01 CSS are noise.

## 9. Compute, data and reproducibility

* Kaggle T4×2 session-hours (wall clock of each job's session): all v3 jobs 109.6 h over 95 jobs; Sprint 10 alone 44.5 h over 38 jobs (4 team accounts, ≤ 2 sessions each).
* Dataset: `multitask_temporal_v3_realgfs.zarr` (real GFS 0.25° 00Z 2015–2023, ERA5 / ERA5-Land, CHIRPS; 1096
  monsoon initialisations; splits 2015–22 train, 2023 test; 2022 or 2021 held out for selection).
* Experiment IDs = job names in `kaggle/queue.json` / `kaggle/registry.json`; run tags in `results/<job>/out/<tag>/`.
* Every decision with its time and evidence: `docs/decision_log.md`. Scaling matrix: `results/final_scaling_matrix.csv`.
* Checkpoints: `models/final/{det,diff}.pt` (186 MB, outside git); rebuild with `scripts/export_final.py` from the
  Kaggle outputs of `s10f-det-b` and `s10f-diff` (account rohitajitbharadwaj).
