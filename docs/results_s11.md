# Sprint 11 — history, forecast history and post-training improvements

Selection season 2022 (models trained 2015–2021); 2023 test reported only. Recipe = the shipped deterministic
backbone (MoE S, E8 top-2 50 %, N/M 2.5, no mm-loss, 50 epochs). Means ± seed SD over seeds 0–2.

## 1. History length H × forecast history

`obs` = observed ERA5 history only (as shipped); `+fc` = observed history plus the GFS forecast that was valid
on each history day (from the run issued that day), i.e. GFS's recent errors are visible to the model.

| H | input | 2022 val CSS | 2023 test CSS | 2022 Tmax MAE | 2022 rain CSI30 | paired Δ(+fc − obs) 2022 |
|---|---|---|---|---|---|---|
| 3 | obs | 0.2106 ± 0.0052 (n=3) | 0.2251 ± 0.0082 (n=3) | 0.914 | 0.183 |  |
| 3 | obs + fc | 0.2190 ± 0.0044 (n=3) | 0.2228 ± 0.0034 (n=3) | 0.907 | 0.187 | +0.0084 (SE 0.0012, n=3) |
| 5 | obs | 0.2205 ± 0.0064 (n=3) | 0.2303 ± 0.0012 (n=3) | 0.922 | 0.200 |  |
| 5 | obs + fc | 0.2252 ± 0.0018 (n=3) | 0.2309 ± 0.0047 (n=3) | 0.922 | 0.197 | +0.0047 (SE 0.0047, n=3) |
| 7 | obs | 0.2101 ± 0.0113 (n=3) | 0.2385 ± 0.0028 (n=3) | 0.921 | 0.185 |  |
| 7 | obs + fc | 0.2209 ± 0.0046 (n=3) | 0.2294 ± 0.0048 (n=3) | 0.900 | 0.172 | +0.0108 (SE 0.0073, n=3) |
| 10 | obs | 0.2173 ± 0.0101 (n=3) | 0.2366 ± 0.0090 (n=3) | 0.908 | 0.178 |  |
| 10 | obs + fc | 0.2244 ± 0.0018 (n=3) | 0.2255 ± 0.0147 (n=3) | 0.904 | 0.183 | +0.0071 (SE 0.0068, n=3) |
| 14 | obs | 0.2301 ± 0.0037 (n=3) | 0.2332 ± 0.0058 (n=3) | 0.900 | 0.185 |  |
| 14 | obs + fc | 0.2322 ± 0.0138 (n=3) | 0.2277 ± 0.0058 (n=3) | 0.889 | 0.181 | +0.0021 (SE 0.0059, n=3) |

## 2. Post-training improvements on the shipped model (s11-post, s11-post2)


**post22** (season 2022)

| variant | CSS | rain CRPS | rain SSR | Brier>30 | CSI30 | CSI64.5 | POD64.5 | rain bias |
|---|---|---|---|---|---|---|---|---|
| FAST single seed raw | 0.2097 | – | – | – | 0.213 | 0.098 | 0.12 | 0.52 |
| FAST single seed + rain QM | 0.2558 | – | – | – | 0.284 | 0.180 | 0.28 | 0.87 |
| FAST 3-seed average raw | 0.1827 | – | – | – | 0.136 | 0.037 | 0.04 | 0.39 |
| FAST 3-seed average + rain QM | 0.2749 | – | – | – | 0.280 | 0.183 | 0.27 | 0.80 |
| diff single seed K=8 raw | ens mean | 0.2811 | 6.238 | 1.21 | 0.0692 | 0.244 | 0.082 | 0.10 | 0.89 |
| diff single seed K=8 raw | PM mean | 0.2346 | 6.238 | 1.21 | 0.0692 | 0.236 | 0.118 | 0.19 | 0.89 |
| diff single seed K=8 spread-cal | ens mean | 0.2813 | 6.167 | 1.40 | 0.0691 | 0.244 | 0.082 | 0.10 | 0.89 |
| diff single seed K=8 spread-cal | PM mean | 0.2190 | 6.167 | 1.40 | 0.0691 | 0.234 | 0.122 | 0.20 | 0.89 |
| diff single seed K=16 raw | ens mean | 0.2958 | 6.235 | 1.22 | 0.0661 | 0.256 | 0.071 | 0.09 | 0.89 |
| diff single seed K=16 raw | PM mean | 0.2434 | 6.235 | 1.22 | 0.0661 | 0.243 | 0.123 | 0.19 | 0.89 |
| diff single seed K=16 spread-cal | ens mean | 0.2960 | 6.247 | 1.42 | 0.0664 | 0.256 | 0.071 | 0.09 | 0.89 |
| diff single seed K=16 spread-cal | PM mean | 0.2260 | 6.247 | 1.42 | 0.0664 | 0.240 | 0.127 | 0.20 | 0.89 |
| diff 3-seed average K=8 raw | ens mean | 0.2755 | 6.299 | 1.15 | 0.0704 | 0.217 | 0.064 | 0.08 | 0.84 |
| diff 3-seed average K=8 raw | PM mean | 0.2345 | 6.299 | 1.15 | 0.0704 | 0.221 | 0.107 | 0.16 | 0.84 |
| diff 3-seed average K=8 spread-cal | ens mean | 0.2756 | 6.233 | 1.32 | 0.0703 | 0.217 | 0.064 | 0.08 | 0.84 |
| diff 3-seed average K=8 spread-cal | PM mean | 0.2210 | 6.233 | 1.32 | 0.0703 | 0.221 | 0.111 | 0.17 | 0.84 |
| diff 3-seed average K=16 raw | ens mean | 0.2897 | 6.298 | 1.16 | 0.0673 | 0.225 | 0.050 | 0.06 | 0.84 |
| diff 3-seed average K=16 raw | PM mean | 0.2438 | 6.298 | 1.16 | 0.0673 | 0.229 | 0.111 | 0.17 | 0.84 |
| diff 3-seed average K=16 spread-cal | ens mean | 0.2898 | 6.312 | 1.34 | 0.0676 | 0.225 | 0.050 | 0.06 | 0.84 |
| diff 3-seed average K=16 spread-cal | PM mean | 0.2285 | 6.312 | 1.34 | 0.0676 | 0.228 | 0.116 | 0.18 | 0.84 |

**post23** (season 2023)

| variant | CSS | rain CRPS | rain SSR | Brier>30 | CSI30 | CSI64.5 | POD64.5 | rain bias |
|---|---|---|---|---|---|---|---|---|
| FAST single seed raw | 0.2361 | – | – | – | 0.193 | 0.139 | 0.17 | 0.66 |
| FAST single seed + rain QM | 0.2435 | – | – | – | 0.205 | 0.175 | 0.31 | 1.09 |
| FAST 3-seed average raw | 0.2232 | – | – | – | 0.154 | 0.090 | 0.10 | 0.49 |
| FAST 3-seed average + rain QM | 0.2659 | – | – | – | 0.223 | 0.178 | 0.30 | 1.01 |
| diff single seed K=8 raw | ens mean | 0.2820 | 4.445 | 1.11 | 0.0392 | 0.194 | 0.088 | 0.11 | 0.92 |
| diff single seed K=8 raw | PM mean | 0.2455 | 4.445 | 1.11 | 0.0392 | 0.176 | 0.114 | 0.18 | 0.92 |
| diff single seed K=8 spread-cal | ens mean | 0.2820 | 4.317 | 1.25 | 0.0388 | 0.194 | 0.088 | 0.11 | 0.92 |
| diff single seed K=8 spread-cal | PM mean | 0.2360 | 4.317 | 1.25 | 0.0388 | 0.169 | 0.113 | 0.18 | 0.92 |
| diff single seed K=16 raw | ens mean | 0.2896 | 4.442 | 1.12 | 0.0372 | 0.204 | 0.081 | 0.09 | 0.92 |
| diff single seed K=16 raw | PM mean | 0.2499 | 4.442 | 1.12 | 0.0372 | 0.180 | 0.117 | 0.18 | 0.92 |
| diff single seed K=16 spread-cal | ens mean | 0.2895 | 4.354 | 1.27 | 0.0371 | 0.204 | 0.081 | 0.09 | 0.92 |
| diff single seed K=16 spread-cal | PM mean | 0.2392 | 4.354 | 1.27 | 0.0371 | 0.173 | 0.115 | 0.18 | 0.92 |
| diff 3-seed average K=8 raw | ens mean | 0.2796 | 4.394 | 1.09 | 0.0386 | 0.188 | 0.077 | 0.09 | 0.88 |
| diff 3-seed average K=8 raw | PM mean | 0.2455 | 4.394 | 1.09 | 0.0386 | 0.173 | 0.109 | 0.16 | 0.88 |
| diff 3-seed average K=8 spread-cal | ens mean | 0.2796 | 4.279 | 1.22 | 0.0384 | 0.188 | 0.077 | 0.09 | 0.88 |
| diff 3-seed average K=8 spread-cal | PM mean | 0.2367 | 4.279 | 1.22 | 0.0384 | 0.167 | 0.108 | 0.17 | 0.88 |
| diff 3-seed average K=16 raw | ens mean | 0.2870 | 4.393 | 1.10 | 0.0368 | 0.196 | 0.068 | 0.08 | 0.88 |
| diff 3-seed average K=16 raw | PM mean | 0.2502 | 4.393 | 1.10 | 0.0368 | 0.178 | 0.112 | 0.17 | 0.88 |
| diff 3-seed average K=16 spread-cal | ens mean | 0.2870 | 4.316 | 1.24 | 0.0368 | 0.196 | 0.068 | 0.08 | 0.88 |
| diff 3-seed average K=16 spread-cal | PM mean | 0.2404 | 4.316 | 1.24 | 0.0368 | 0.172 | 0.111 | 0.17 | 0.88 |

**post22b** (season 2022)

| variant | CSS | rain CRPS | rain SSR | Brier>30 | CSI30 | CSI64.5 | POD64.5 | rain bias |
|---|---|---|---|---|---|---|---|---|
| FAST single seed raw | 0.2097 | – | – | – | 0.213 | 0.098 | 0.12 | 0.52 |
| FAST single seed + rain QM | 0.2558 | – | – | – | 0.284 | 0.180 | 0.28 | 0.87 |
| FAST 3-seed average raw | 0.1827 | – | – | – | 0.136 | 0.037 | 0.04 | 0.39 |
| FAST 3-seed average + rain QM | 0.2749 | – | – | – | 0.280 | 0.183 | 0.27 | 0.80 |
| diff single seed K=8 raw | ens mean | 0.2811 | 6.238 | 1.21 | 0.0692 | 0.244 | 0.082 | 0.10 | 0.89 |
| diff single seed K=8 raw | PM mean | 0.2346 | 6.238 | 1.21 | 0.0692 | 0.236 | 0.118 | 0.19 | 0.89 |
| diff single seed K=8 spread-cal | ens mean | 0.2813 | 6.167 | 1.40 | 0.0691 | 0.244 | 0.082 | 0.10 | 0.89 |
| diff single seed K=8 spread-cal | PM mean | 0.2190 | 6.167 | 1.40 | 0.0691 | 0.234 | 0.122 | 0.20 | 0.89 |
| diff single seed K=8 spread-cal + tail cap | ens mean | 0.2822 | 6.166 | 1.21 | 0.0691 | 0.244 | 0.072 | 0.09 | 0.86 |
| diff single seed K=8 spread-cal + tail cap | PM mean | 0.2293 | 6.166 | 1.21 | 0.0691 | 0.235 | 0.123 | 0.20 | 0.86 |
| diff single seed K=16 raw | ens mean | 0.2958 | 6.235 | 1.22 | 0.0661 | 0.256 | 0.071 | 0.09 | 0.89 |
| diff single seed K=16 raw | PM mean | 0.2434 | 6.235 | 1.22 | 0.0661 | 0.243 | 0.123 | 0.19 | 0.89 |
| diff single seed K=16 spread-cal | ens mean | 0.2960 | 6.247 | 1.42 | 0.0664 | 0.256 | 0.071 | 0.09 | 0.89 |
| diff single seed K=16 spread-cal | PM mean | 0.2260 | 6.247 | 1.42 | 0.0664 | 0.240 | 0.127 | 0.20 | 0.89 |
| diff single seed K=16 spread-cal + tail cap | ens mean | 0.2954 | 6.246 | 1.20 | 0.0664 | 0.255 | 0.056 | 0.06 | 0.86 |
| diff single seed K=16 spread-cal + tail cap | PM mean | 0.2367 | 6.246 | 1.20 | 0.0664 | 0.241 | 0.129 | 0.21 | 0.86 |

Member rain tails on land (mm/day): single seed: p99 117, p99.9 301, p99.99 804, max 1096, above training block max 0.142 %

**post23b** (season 2023)

| variant | CSS | rain CRPS | rain SSR | Brier>30 | CSI30 | CSI64.5 | POD64.5 | rain bias |
|---|---|---|---|---|---|---|---|---|
| FAST single seed raw | 0.2361 | – | – | – | 0.193 | 0.139 | 0.17 | 0.66 |
| FAST single seed + rain QM | 0.2435 | – | – | – | 0.205 | 0.175 | 0.31 | 1.09 |
| FAST 3-seed average raw | 0.2232 | – | – | – | 0.154 | 0.090 | 0.10 | 0.49 |
| FAST 3-seed average + rain QM | 0.2659 | – | – | – | 0.223 | 0.178 | 0.30 | 1.01 |
| diff single seed K=8 raw | ens mean | 0.2820 | 4.445 | 1.11 | 0.0392 | 0.194 | 0.088 | 0.11 | 0.92 |
| diff single seed K=8 raw | PM mean | 0.2455 | 4.445 | 1.11 | 0.0392 | 0.176 | 0.114 | 0.18 | 0.92 |
| diff single seed K=8 spread-cal | ens mean | 0.2820 | 4.317 | 1.25 | 0.0388 | 0.194 | 0.088 | 0.11 | 0.92 |
| diff single seed K=8 spread-cal | PM mean | 0.2360 | 4.317 | 1.25 | 0.0388 | 0.169 | 0.113 | 0.18 | 0.92 |
| diff single seed K=8 spread-cal + tail cap | ens mean | 0.2820 | 4.317 | 1.04 | 0.0388 | 0.194 | 0.083 | 0.10 | 0.90 |
| diff single seed K=8 spread-cal + tail cap | PM mean | 0.2449 | 4.317 | 1.04 | 0.0388 | 0.170 | 0.115 | 0.18 | 0.90 |
| diff single seed K=16 raw | ens mean | 0.2896 | 4.442 | 1.12 | 0.0372 | 0.204 | 0.081 | 0.09 | 0.92 |
| diff single seed K=16 raw | PM mean | 0.2499 | 4.442 | 1.12 | 0.0372 | 0.180 | 0.117 | 0.18 | 0.92 |
| diff single seed K=16 spread-cal | ens mean | 0.2895 | 4.354 | 1.27 | 0.0371 | 0.204 | 0.081 | 0.09 | 0.92 |
| diff single seed K=16 spread-cal | PM mean | 0.2392 | 4.354 | 1.27 | 0.0371 | 0.173 | 0.115 | 0.18 | 0.92 |
| diff single seed K=16 spread-cal + tail cap | ens mean | 0.2879 | 4.353 | 1.03 | 0.0371 | 0.203 | 0.067 | 0.07 | 0.89 |
| diff single seed K=16 spread-cal + tail cap | PM mean | 0.2486 | 4.353 | 1.03 | 0.0371 | 0.175 | 0.119 | 0.19 | 0.89 |

Member rain tails on land (mm/day): single seed: p99 82, p99.9 236, p99.99 778, max 1096, above training block max 0.055 %

## 3. Denoiser trained on the 3-seed averaged backbone (s11-davg)

| denoiser trained on | 2022 val CSS | 2022 rain CRPS | 2023 CSS | 2023 rain CRPS |
|---|---|---|---|---|
| single-seed residuals (shipped) | 0.2803 | 6.238 | 0.2814 | 4.431 |
| 3-seed averaged residuals | 0.2600 | 6.396 | 0.2774 | 4.398 |

## 4. Full-pipeline rebuild of the sweep winner (H = 5 + forecast history, prefix s12)

| seed | OOF CSS 2015–22 (H5+fc) | OOF CSS 2015–22 (shipped H3 obs) | final det 2023 (H5+fc) | final det 2023 (H3 obs) |
|---|---|---|---|---|
| 0 | 0.2364 | 0.2472 | 0.2407 | 0.2361 |
| 1 | 0.2482 | 0.2432 | 0.2293 | 0.2205 |
| 2 | 0.2422 | 0.2428 | 0.2367 | 0.2484 |

FAST (3-seed average + rain QM fitted on the other seasons), every training season scored out of fold:

| season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | mean Δ ± SE |
|---|---|---|---|---|---|---|---|---|---|
| shipped H3 obs | 0.3146 | 0.2959 | 0.2800 | 0.2248 | 0.2825 | 0.2688 | 0.2649 | 0.2749 | |
| H5 + fc | 0.3132 | 0.2927 | 0.2782 | 0.2384 | 0.2758 | 0.2637 | 0.2713 | 0.2808 | +0.0010 ± 0.0025 |

BALANCED proxy on 2022 (denoiser trained 2015–21, 24 × 8, raw ensemble):

| system | seed | 2022 CSS | 2022 rain CRPS | 2023 CSS (2015–22 denoiser) | 2023 rain CRPS |
|---|---|---|---|---|---|
| shipped H3 obs | 0 | 0.2803 | 6.238 | 0.2814 | 4.431 |
| shipped H3 obs | 1 | 0.2628 | 6.408 | 0.2448 | 4.502 |
| H5 + fc | 0 | 0.2726 | 6.327 | 0.2774 | 4.430 |
| H5 + fc | 1 | 0.2680 | 6.313 | 0.2497 | 4.523 |
| H5 + fc | 2 | 0.2595 | 6.424 | 0.2735 | 4.481 |

**Decision:** the replacement rule (beat the shipped system on 2022 in both FAST and BALANCED, rain CRPS
≤ +1 %) fails on BALANCED for the selected seed (seed 1). Seed for seed the H5 + fc pipeline is level with the
shipped one, matching the 8-season FAST tie: the history effects seen in the sweep are real but small, and the
sweep's 2022 margin was inflated by choosing the best of 10 configurations on one season.

## 5. Sprint 11 outcome

| change | evidence (2022 selection) | status |
|---|---|---|
| FAST = 3-seed averaged backbone + matching rain QM | CSS 0.2558 → 0.2749 (2023: 0.2435 → 0.2659) | **adopted** |
| Rain tail cap min(2× block max, 1.2× domain max) | CSS +0.0003–0.0009, CRPS unchanged; removes 800–1,100 mm members; 0 real 2023 pixel-days clipped | **adopted** |
| Probability-matched rain field | POD ≥ 64.5 mm 0.10 → 0.20 but CSS −0.05 as the mean | **optional output** (`rain_pm`) |
| Denoiser retrained on the averaged backbone | CSS 0.2803 → 0.2600, CRPS +2.5 % | rejected |
| H = 5 + forecast history (full pipeline) | FAST 8-season tie; BALANCED 0.2680 vs 0.2803 | rejected |
| Static Tmax / RH bias correction | season-to-season bias changes sign (Tmax −0.33 … +0.27 °C) | not applicable |
