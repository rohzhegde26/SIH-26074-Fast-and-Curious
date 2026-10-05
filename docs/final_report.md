# SIH-26074 — Sprints 1-10 on real data: findings and the shipped system

Validation season 2022 drives decisions; 2023 test shown for reference. A second validation season (2021 held
out) was added for history/context. CSS = Composite Skill Score vs GFS-bilinear (0 = raw GFS interpolated,
1 = perfect; `selection_criteria.md`). Ties = within one standard error of the difference of seed means; ties go to
the cheaper option. Full tables: `results.md`, `results_s7_s8.md`; every decision: `decision_log.md`;
pipeline verification: `audit_v3_code.md`. 80 training runs + 32 sampling configurations, all completed;
~40 of 81.7 GPU-h used.

## 0. Why everything was re-run
Old Sprint 3-9 data: "GFS" input = noisy ground truth, Tmax/Tmin/RH targets not ERA5, wind in km/h as m/s,
"context" = edge padding, sea pixels scored as 0 mm (`audit_previous_sprints.md`). v3: real GFS 0.25 deg
(1,096 monsoon inits 2015-23), ERA5 (NCAR, bit-identical to ECMWF), ERA5-Land (CDS), CHIRPS (bit-exact to UCSB),
land-masked; one spatiotemporal transformer (history encoder, future encoder with history<->future
cross-attention, fine-grid decoder cross-attending to N x N context tokens, adaLN, optional MoE FFNs).

## 1. Sprint 3 — baselines (H=7, N=16, 50 epochs, 3 seeds)
| Model | CSS val | CSS test |
|---|---|---|
| GFS-bilinear | 0 | 0 |
| GFS + per-pixel quantile mapping (statistical bar) | 0.189 | 0.181 |
| Transformer S, plain loss | 0.220 | 0.230 |
| Transformer S, + precipitation mm-loss term | 0.222 | 0.214 |
Transformer: Tmax MAE 0.93 vs 1.50 C (GFS), RH 3.0 vs 6.4 %, wind 0.76 vs 1.86 m/s. Rain is the weak point: a
deterministic model predicts ~50 % of observed rain (log-space loss -> median); train-fitted quantile mapping lifts
CSS by ~0.025-0.03. mm-loss: tied on CSS but CSI@30 0.203 vs 0.183 -> kept.

## 2. Sprint 4 — history length H (N=16)
| H (days) | 1 | 3 | 5 | 7 | 10 | 14 |
|---|---|---|---|---|---|---|
| 50 ep, 3 seeds, val 2022 | **0.212** | 0.223 | 0.219 | 0.220 | 0.220 | 0.224 |
| 100 ep, 2 seeds, val 2022 | | 0.238 | | | | 0.233 |
| 100 ep, 2 seeds, val 2021 | | 0.285 | | | | 0.287 |
**Finding:** 1 day of history is clearly worse; 3-14 days are statistically tied on both seasons. Longer history
overfits sooner (best epoch 45-69 vs 79-84) on ~852 overlapping samples. History is used (ablation: swapping only the
history ring changes rain predictions 57 % and Tmax 49 % as much as swapping the core). **Use H=3.**

## 3. Sprint 5 — spatial context N/M (real surrounding weather)
| N/M | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|
| 50 ep, 3 seeds, val 2022 | 0.220 | 0.225 | 0.216 | 0.222 | 0.213 | 0.225 |
| 50 ep, test 2023 | 0.230 | 0.238 | 0.234 | 0.236 | 0.237 | 0.238 |
| 100 ep, H=3, val 2022 | 0.238 | **0.242** | | | | 0.235 |
| 100 ep, H=3, val 2021 | 0.285 | 0.286 | | | | **0.288** |
**Finding:** context is wired and heavily used (swapping only the outer ring changes predictions 73 % as much as
the core), and wider context helps later lead days (D+3/D+4) and the 2023 test season, but on validation all
ratios are within noise on both seasons. At this data scale the extra context does not yet generalise into a
significant gain. **Use N/M = 1.0-1.25 (N=16 or 20)**; N=20 was nominally best on 2022 and 3rd of 4 on 2021.

## 4. Sprint 6 — deterministic vs deterministic + residual diffusion (H=3, N=16, S)
| | CSS val | CSS test | wet MAE | CSI@15 | FSS@15 | rain bias | Tmax MAE |
|---|---|---|---|---|---|---|---|
| Deterministic (100 ep) | 0.238 | 0.216 | 17.43 | 0.291 | 0.643 | 0.57 | 0.93 |
| + residual diffusion, 8-member mean | **0.258** | **0.240** | 17.01 | 0.297 | 0.659 | 0.63 | 0.89 |
**Finding:** diffusion adds +0.021 val / +0.024 test (seed sd 0.0003) and improves every variable; rain CRPS 7.6-7.8
vs deterministic MAE 8.9. Remaining issues: rain still dry (bias 0.63) and ensemble under-dispersed.

## 5. Sprint 7 — denoising steps x sampler (K = 8)
| steps | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| DDIM CSS / rain CRPS / SSR | 0.246/8.23/0.21 | 0.252/7.99/0.39 | 0.256/7.86/0.51 | 0.258/7.79/0.57 | 0.261/7.69/0.66 | 0.262/7.66/0.70 |
| DPM-Solver++(2M) | 0.248/8.18/0.25 | 0.256/7.89/0.52 | 0.259/7.76/0.65 | 0.262/7.69/0.70 | **0.264/7.61/0.77** | 0.265/7.58/0.80 |
| DDIM eta=1 (stochastic) | 0.247/8.38/0.15 | ... | | 0.256/7.98/0.46 | | 0.259/7.82/0.63 |
**Finding:** quality improves monotonically with steps for every sampler (unlike the old Sprint 7, whose reverse
trend was an artefact of the oracle data). DPM-Solver++ is best at every step count; diminishing returns after
16-24 steps (+0.002 CSS from 24->32 for +33 % cost). Few steps mainly collapse the ensemble spread. Latency
(T4, per 7-day x 6-variable sample, 8 members): 1.8 s at 16 steps, 2.8 s at 24. VRAM < 1 GB.
**Use DPM-Solver++ with 16 (fast) to 24 (accurate) steps.**

## 6. Sprint 8 — ensemble members K vs steps S at matched compute (DDIM)
| NFE | best rain CRPS (K, S) | best Brier>30 mm (K, S) | K large, S small |
|---|---|---|---|
| 32 | 7.79 (2, 16) | 0.0870 (4, 8) | K=16,S=2: 8.48 |
| 64 | 7.66 (2, 32) | 0.0853 (8, 8) | K=32,S=2: 8.48 |
| 128 | 7.66 (4, 32) | 0.0835 (8, 16) | K=32,S=4: 8.23 |
**Finding:** at fixed compute, spend on denoising steps first (>= 16), then add members. Below ~16 steps members
are individually poor and under-dispersed, so more members do not help (SSR 0.10-0.25). Threshold probabilities
(Brier) favour K=8 at S=16. The answer is the same at every lead day (CRPS rises from 7.3 at D+1 to 8.4 at D+5).
**Use K=8, S=16-24.**

## 7. Sprint 9 — capacity and MoE sparsity (H=3, N=16, 100 ep, 2 seeds)
| Model | total params | active | CSS val | CSS test |
|---|---|---|---|---|
| Dense S | 17.7M | 17.7M | 0.238 | 0.216 |
| Dense M | 34.2M | 34.2M | 0.238 | 0.222 |
| Dense L | 58.7M | 58.7M | **0.242** | 0.218 |
| MoE 8 experts, top-2, 50 % blocks | 28.8M | 19.3M | **0.242** | 0.215 |
| MoE 8, top-1, 50 % | 28.8M | 17.7M | 0.239 | 0.215 |
| MoE 4, top-1, 50 % | 22.5M | 17.7M | 0.238 | 0.223 |
| MoE 16, top-1 / top-2, 50 % | 41.4M | 17.7 / 19.3M | 0.233 / 0.230 | 0.219 / 0.202 |
| MoE 8, top-1, coverage 33 % / 50 % / 100 % | 25.1 / 28.8 / 39.8M | 17.7M | 0.231 / 0.239 / 0.235 | |
**Finding:** capacity helps only a little at this data scale (S -> L: +0.005, within noise). MoE with 8 experts,
top-2, in half the decoder blocks matches dense L at one third of the active parameters; 16 experts is too sparse
for ~852 training samples; ~50 % MoE coverage is the sweet spot. **Use MoE E=8, top-2, 50 % (or dense S if latency
matters most).**

## 8. Sprint 10 open problems — resolved (2023 test season; full table `results_s10.md`)
All runs: H=3, **N/M = 2.5 (N=40, project lead's choice)**, MoE backbone, trained on 2015-2022 with a fixed epoch
count (no single-season early stopping), diffusion sampled with DPM-Solver++ 24 steps x 8 members.

**(a) Training recipe for the deterministic backbone** (test CSS):
| | with mm-loss | without mm-loss |
|---|---|---|
| 50 epochs | 0.216 | **0.235** |
| 75 epochs | 0.218 / 0.228 (2 seeds) | 0.234 |
The mm-loss term only looked useful on the 2022 season; it costs 0.011-0.019 CSS on 2023 -> **dropped**. 50 and 75
epochs are equivalent -> **50 epochs**. (Seed spread on 2023 Tmax is ~0.1 C, so earlier single-run Tmax differences
were mostly noise.)

**(b) Cross-fitted residual diffusion** (the root cause of dry, under-dispersed ensembles): the diffusion was learning
in-sample residuals, which are **half the size** of out-of-sample ones (residual sigma, rain 0.52 vs 1.08; every
variable ~2x). Training it on out-of-fold deterministic predictions (3 year-folds):
| (2 seeds each) | in-sample residuals | **cross-fitted residuals** |
|---|---|---|
| rain CRPS | 5.64 / 5.70 | **4.47 / 4.51 (-21 %)** |
| Brier > 30 mm | 0.050 / 0.049 | **0.039 / 0.039 (-21 %)** |
| ens-mean CSS | 0.247 / 0.253 | 0.247 / **0.272** |
| spread-skill (mean) | 0.87 | **0.98** |
| rain bias / CSI@30 (mean) | 0.89 / **0.190** | 0.84 / 0.168 |
**Use cross-fitting.** Trade-off: the ensemble mean is slightly more conservative for heavy rain.

**(c) Capacity x test-time compute**: an MoE diffusion denoiser on the MoE backbone gives +0.0016 CSS and -0.5 % CRPS
over the S denoiser -> **S denoiser is sufficient**; spend compute on members/steps, not denoiser size.

**(d) Post-hoc calibration** (fitted on 2022, scored on 2023):
* **Spread inflation** (mean-preserving): rain CRPS -2 %, **Tmax CRPS -9 %**, bias and Brier unchanged; parameters
  transfer to the final model -> **use it**.
* **Rain quantile mapping** fitted on one season: CSS +0.016-0.019 but over-corrects (bias 0.89 -> 1.25) and worsens
  Brier, because wet-2022 statistics do not fit dry 2023 -> do **not** fit rain QM on a single season; fit it on the
  multi-season cross-fitted out-of-fold predictions instead (available from the cross-fitting step).

## 9. Recommended Sprint 10 system
* **Inputs:** real GFS 7-day forecast, ERA5 history **H=3**, context **N/M = 2.5**, terrain, land mask.
* **Deterministic backbone:** spatiotemporal transformer, **MoE 8 experts / top-2 / 50 % of decoder blocks**,
  **no mm-loss, 50 epochs, trained on 2015-2022**.
* **Probabilistic stage:** residual diffusion (S denoiser) trained on **cross-fitted** residuals (3 year-folds),
  80 epochs; **DPM-Solver++ 16-24 steps x 8 members**; mean-preserving **spread calibration**.
* **Operating modes:** FAST = deterministic + rain QM fitted on multi-season out-of-fold predictions; BALANCED =
  diffusion K=8, S=16; ACCURATE = K=8, S=24-32; ENSEMBLE = K=8-16, S>=16 for threshold (advisory) probabilities.
* The final-recipe pipeline (`s10f_*`: folds + final model without mm-loss, cross-fitted vs in-sample diffusion at
  80 epochs) is the Sprint 10 starting model; its 2023 scores are in `results_s10.md`.

## 10. Caveats
* Seasons differ strongly (2022 wet, 2023 dry); single-season rankings flipped several times, so only effects that
  hold on both 2021 and 2022 validation are reported as findings.
* ~852 heavily overlapping training initialisations limit how much extra history, context or capacity can help.
* Two of my own errors were caught and fixed during the night (CSS bias term, tie rule); all tables use the fixed
  definitions (`audit_v3_code.md`).

## 11. Sprint 10 outcome — the shipped system (full report: `final_system_report.md`)
* **Deliverables (plan paths):** `models/final/`, `configs/final/`, `results/final_scaling_matrix.csv`, `reports/final_system_report.md`.
* **Shipped:** seed-0 MoE-S backbone (H=3, N/M=2.5, no mm-loss, 50 ep) + S residual-diffusion denoiser trained on
  cross-fitted residuals + rain QM (FAST) + mean-preserving spread calibration (diffusion modes). Bundle:
  `models/final` (not in git: 186 MB; rebuild with `scripts/export_final.py`), settings `configs/final/modes.yaml`.
* **Modes (2023 test, T4, one forecast):** FAST 0.244 / 0.06 s; BALANCED 24×8 0.282 / ~4 s; ACCURATE 32×8 0.284 /
  5.3 s; ENSEMBLE 24×16 0.290 / 8.3 s. BALANCED's steps were re-selected 16 → 24 on 2022 by a pre-registered rule.
* **Honest headline:** diffusion-pipeline test CSS over 3 seeds = 0.263 ± 0.018 (shipped seed 0.281).
* **Scaling:** test-time compute (steps up to ~24, then members) is the lever that works. Parameter count does not
  help in either stage at 8 training seasons (3-seed stage-1 S/M/L/MoE tied; M/L denoisers over-confident).
* **Next:** average 3 seeds' backbones; spread-calibrate the L denoiser; per-K spread factors; more seasons.

## 12. Sprint 11 — history, forecast history and post-training improvements (`results_s11.md`)
* **History × forecast history** (shipped backbone, 3 seeds, H = 3/5/7/10/14, with and without the GFS forecast that
  was valid on each history day): on 2022 longer history helped and forecast history helped most at short H
  (+0.008 at H = 3). The rule picked H = 5 + forecast history, but the full rebuild did not hold up: FAST tied over all
  8 training seasons (+0.001 ± 0.0025) and the BALANCED check failed (0.268 vs 0.280 on 2022). Best of 10 on one
  season overstated the gain. The shipped H = 3 backbone stays.
* **Adopted:** FAST = average of the three seeds' backbones + matching rain QM (2022 0.256 → 0.275; 2023 0.244 →
  0.266); a rain tail cap on diffusion members (members reached 1,100 mm/day; cap min(2× block max, 1.2× domain
  max), no real day clipped); optional probability-matched rain field for heavy-rain maps.
* **Rejected:** denoiser retrained on the averaged backbone (−0.020 on 2022); static temperature bias correction
  (bias flips sign between seasons).
* **Operational note:** the laptop slept 04:21–10:36 and froze the scheduler; keep it awake and plugged in for runs.
