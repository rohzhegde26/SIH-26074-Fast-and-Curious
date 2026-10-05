# Improved sprint plan (v3) — overnight 2026-10-01/02

Follows the original sprint-wise plan (spatiotemporal context; scaling by test-time compute, parameters and
sparse MoE) with these improvements:

1. **Real data only** (see `audit_previous_sprints.md`): real GFS inputs, ERA5 history, CHIRPS + ERA5-Land
   targets, land-masked scoring.
2. **One architecture for every sprint**: a spatiotemporal transformer (`sihv3/model.py`) with
   history encoder, future encoder with history<->future cross-attention, and a fine-grid decoder that
   cross-attends to N x N *context tokens* (Sprint 5's intended design, never built before).
3. **Regression + diffusion (CorrDiff-style)**: diffusion learns the residual of the *deterministic
   transformer*, not of interpolated GFS, so Sprint 6 is a clean "deterministic vs deterministic + diffusion"
   comparison and the diffusion only has to model what the regression cannot.
4. **Robust selection** (`selection_criteria.md`): composite skill vs GFS-bilinear over all 6 variables,
   2 seeds per config, ties go to the cheaper option.
5. **Throughput**: every Kaggle T4x2 session runs two experiments in parallel (one per GPU) at no extra quota.

## Experiment matrix

S = 17.7M-param transformer unless noted; all deterministic runs predict y - upsample(GFS).

| Phase | Sprint | Runs | Decision |
|---|---|---|---|
| A | 3 | det S, H=7, N=16, seeds 0/1 (shared with S4) + GFS-bilinear reference | baseline skill on real data |
| A | 4 | det S, N=16, H in {1,3,7,14} x seeds {0,1} | provisional H* |
| A | 5 | det S, H=7, N in {24,32,40} x seeds {0,1} (N=16 shared) — N/M = 1.0/1.5/2.0/2.5 | provisional N* |
| B | 6 | det S at (H*, N*) seed 0/1, then residual diffusion S x seeds {0,1} on the better det seed | does diffusion add value (pCSS, CRPS, spread) |
| C | 9 | at (H*, N*): det M (34M), det L (59M); MoE on S: experts {4,8,16} x top-k {1,2} with 50 % of decoder blocks MoE; coverage {25 %, 100 %} for (8, top-1) | capacity/sparsity frontier on active params, latency, VRAM |
| D | 7 | eval-only on S6 diffusion: samplers {DDIM, DDIM eta=1, DPM-Solver++2M} x steps {4,8,12,16,24,32}, K=8 | min steps, knee, preferred sampler |
| D | 8 | eval-only: (K, S) at matched 32/64/128 NFE, K up to 32; by lead, rain intensity, variable | denoise vs ensemble trade-off |

If time allows: diffusion on top of the best Sprint 9 backbone (capacity x test-time compute interaction).
