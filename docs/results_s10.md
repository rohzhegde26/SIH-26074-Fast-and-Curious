# Sprint 10 open problems — results on the 2023 test season

All models: H=3, N=40 (N/M 2.5), mm-loss, MoE backbone (E8 top-2, 50 %), trained 2015-2022 for fixed epochs;
diffusion sampled with DPM-Solver++ 24 steps x 8 members. CSS vs GFS-bilinear (ensemble mean for diffusion).

| model | CSS | CSS +precipQM | precip_crps | precip_ssr | precip_cov90 | precip_bias_ratio | brier30 | precip_csi30 | precip_wet_mae | tmax_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deterministic MoE (seed 0) | 0.2183 | 0.2323 | nan | nan | nan | 0.689 | nan | 0.185 | 17.274 | 1.217 | 3.602 | 0.695 |
| deterministic MoE (seed 1) | 0.2278 | 0.2409 | nan | nan | nan | 0.687 | nan | 0.184 | 17.276 | 1.116 | 3.399 | 0.670 |
| diffusion, in-sample residuals (seed 0) | 0.2466 |  | 5.640 | 1.004 | 0.361 | 0.921 | 0.050 | 0.191 | 16.546 | 1.189 | 3.518 | 0.657 |
| diffusion, in-sample residuals (seed 1) | 0.2531 |  | 5.696 | 0.743 | 0.364 | 0.858 | 0.049 | 0.189 | 16.561 | 1.094 | 3.305 | 0.635 |
| diffusion, CROSS-FITTED residuals (seed 0) | 0.2466 |  | 4.474 | 0.865 | 0.487 | 0.773 | 0.039 | 0.155 | 16.463 | 1.136 | 3.356 | 0.621 |
| diffusion, CROSS-FITTED residuals (seed 1) | 0.2722 |  | 4.506 | 1.085 | 0.524 | 0.903 | 0.039 | 0.180 | 16.023 | 1.074 | 3.175 | 0.611 |
| diffusion, cross-fitted, MoE denoiser (seed 0) | 0.2482 |  | 4.449 | 0.811 | 0.438 | 0.786 | 0.039 | 0.157 | 16.274 | 1.156 | 3.409 | 0.613 |
| diffusion, cross-fitted, trained 2015-21 (calibration model) | 0.2447 |  | 4.499 | 1.020 | 0.516 | 0.912 | 0.041 | 0.151 | 16.282 | 1.145 | 3.384 | 0.663 |

## Calibration (cal_MoE): fitted on [2022], scored on [2023]

| ensemble | variant | CSS (ens mean) | precip_crps | precip_ssr | precip_cov90 | precip_bias_ratio | brier30 | tmax_crps |
|---|---|---|---|---|---|---|---|---|
| 2015-21 model | raw | 0.2414 | 4.536 | 0.994 | 0.512 | 0.891 | 0.041 | 0.813 |
| 2015-21 model | spread | 0.2415 | 4.435 | 1.088 | 0.852 | 0.891 | 0.041 | 0.744 |
| 2015-21 model | rainqm | 0.2571 | 4.528 | 1.060 | 0.889 | 1.251 | 0.044 | 0.813 |
| 2015-21 model | both | 0.2571 | 4.417 | 1.153 | 0.907 | 1.251 | 0.044 | 0.744 |
| final model (params transferred) | raw | 0.2479 | 4.440 | 0.819 | 0.440 | 0.789 | 0.039 | 0.885 |
| final model (params transferred) | spread | 0.2480 | 4.350 | 0.891 | 0.817 | 0.789 | 0.039 | 0.799 |
| final model (params transferred) | rainqm | 0.2636 | 4.528 | 0.935 | 0.867 | 1.193 | 0.043 | 0.885 |
| final model (params transferred) | both | 0.2636 | 4.415 | 1.009 | 0.886 | 1.193 | 0.043 | 0.799 |

## Calibration (cal_S): fitted on [2022], scored on [2023]

| ensemble | variant | CSS (ens mean) | precip_crps | precip_ssr | precip_cov90 | precip_bias_ratio | brier30 | tmax_crps |
|---|---|---|---|---|---|---|---|---|
| 2015-21 model | raw | 0.2414 | 4.536 | 0.994 | 0.512 | 0.891 | 0.041 | 0.813 |
| 2015-21 model | spread | 0.2415 | 4.435 | 1.088 | 0.852 | 0.891 | 0.041 | 0.744 |
| 2015-21 model | rainqm | 0.2571 | 4.528 | 1.060 | 0.889 | 1.251 | 0.044 | 0.813 |
| 2015-21 model | both | 0.2571 | 4.417 | 1.153 | 0.907 | 1.251 | 0.044 | 0.744 |
| final model (params transferred) | raw | 0.2449 | 4.477 | 0.860 | 0.490 | 0.769 | 0.039 | 0.861 |
| final model (params transferred) | spread | 0.2449 | 4.385 | 0.945 | 0.832 | 0.769 | 0.039 | 0.779 |
| final model (params transferred) | rainqm | 0.2637 | 4.521 | 0.939 | 0.867 | 1.152 | 0.043 | 0.861 |
| final model (params transferred) | both | 0.2637 | 4.410 | 1.025 | 0.887 | 1.152 | 0.043 | 0.779 |
