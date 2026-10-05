# Results (validation 2022; test 2023 shown for reference only)


## Sprint 3 — deterministic baseline + loss variant (H=7, N=16)

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **lin_h7_n16** | 3 | 0.2224 ± 0.0070 | nan | 0.2137 | 17.780 | 0.276 | 0.203 | 0.610 | 0.527 | 0.930 | 0.420 | 3.042 | 0.781 |
| s4_h7_n16 | 3 | 0.2203 ± 0.0012 | nan | 0.2304 | 17.785 | 0.258 | 0.183 | 0.588 | 0.502 | 0.934 | 0.402 | 2.965 | 0.761 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0050; configs within it of the best: lin_h7_n16, s4_h7_n16. **Selected: lin_h7_n16** (cheapest within the floor).


## Sprint 4 — history length (N=16)

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s4_h14_n16 | 3 | 0.2238 ± 0.0098 | nan | 0.2310 | 17.912 | 0.261 | 0.182 | 0.598 | 0.499 | 0.927 | 0.389 | 2.957 | 0.754 |
| **s4_h3_n16** | 3 | 0.2230 ± 0.0057 | nan | 0.2261 | 17.871 | 0.259 | 0.186 | 0.590 | 0.507 | 0.926 | 0.391 | 2.935 | 0.763 |
| s4_h7_n16 | 3 | 0.2203 ± 0.0012 | nan | 0.2304 | 17.785 | 0.258 | 0.183 | 0.588 | 0.502 | 0.934 | 0.402 | 2.965 | 0.761 |
| s4_h10_n16 | 3 | 0.2196 ± 0.0044 | 0.2452 | 0.2313 | 17.888 | 0.257 | 0.182 | 0.585 | 0.508 | 0.934 | 0.392 | 2.969 | 0.766 |
| s4_h5_n16 | 3 | 0.2193 ± 0.0119 | 0.2451 | 0.2335 | 17.875 | 0.260 | 0.184 | 0.587 | 0.507 | 0.936 | 0.395 | 2.985 | 0.770 |
| s4_h1_n16 | 3 | 0.2121 ± 0.0152 | nan | 0.2082 | 18.130 | 0.248 | 0.176 | 0.569 | 0.473 | 0.942 | 0.389 | 2.965 | 0.757 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0076; configs within it of the best: s4_h14_n16, s4_h3_n16, s4_h7_n16, s4_h10_n16, s4_h5_n16. **Selected: s4_h3_n16** (cheapest within the floor).


## Sprint 5 — spatial context (H=7; N=16 row shared with Sprint 4)

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s5_h7_n40 | 3 | 0.2254 ± 0.0139 | nan | 0.2383 | 17.768 | 0.260 | 0.187 | 0.592 | 0.524 | 0.896 | 0.395 | 2.982 | 0.757 |
| s5_h7_n20 | 3 | 0.2250 ± 0.0181 | 0.2518 | 0.2380 | 17.663 | 0.265 | 0.191 | 0.597 | 0.513 | 0.923 | 0.402 | 2.969 | 0.757 |
| s5_h7_n28 | 3 | 0.2215 ± 0.0099 | 0.2489 | 0.2363 | 17.828 | 0.261 | 0.193 | 0.594 | 0.510 | 0.931 | 0.392 | 3.061 | 0.755 |
| **s4_h7_n16** | 3 | 0.2203 ± 0.0012 | nan | 0.2304 | 17.785 | 0.258 | 0.183 | 0.588 | 0.502 | 0.934 | 0.402 | 2.965 | 0.761 |
| s5_h7_n24 | 3 | 0.2162 ± 0.0026 | nan | 0.2341 | 17.855 | 0.253 | 0.182 | 0.578 | 0.500 | 0.925 | 0.396 | 3.063 | 0.757 |
| s5_h7_n32 | 3 | 0.2131 ± 0.0055 | nan | 0.2368 | 18.063 | 0.248 | 0.177 | 0.571 | 0.495 | 0.918 | 0.396 | 3.016 | 0.758 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0085; configs within it of the best: s5_h7_n40, s5_h7_n20, s5_h7_n28, s4_h7_n16. **Selected: s4_h7_n16** (cheapest within the floor).


## Sprint 4/5 confirmation — 100-epoch schedule (H, N)

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s45_confirm_h3_n20 | 2 | 0.2417 ± 0.0065 | 0.2714 | 0.2070 | 17.240 | 0.295 | 0.225 | 0.644 | 0.556 | 0.945 | 0.401 | 3.011 | 0.778 |
| **s9_dense_S** | 2 | 0.2376 ± 0.0046 | 0.2604 | 0.2162 | 17.434 | 0.291 | 0.218 | 0.643 | 0.574 | 0.930 | 0.406 | 3.052 | 0.792 |
| s45_confirm_h14_n40 | 2 | 0.2349 ± 0.0069 | 0.2646 | 0.2114 | 17.563 | 0.283 | 0.212 | 0.620 | 0.541 | 0.914 | 0.403 | 2.979 | 0.775 |
| s45_confirm_h3_n40 | 2 | 0.2347 ± 0.0016 | 0.2667 | 0.2073 | 17.507 | 0.285 | 0.210 | 0.624 | 0.544 | 0.924 | 0.407 | 2.964 | 0.775 |
| s45_confirm_h14_n16 | 2 | 0.2325 ± 0.0040 | 0.2621 | 0.2011 | 17.488 | 0.284 | 0.221 | 0.629 | 0.531 | 0.929 | 0.411 | 3.039 | 0.792 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0051; configs within it of the best: s45_confirm_h3_n20, s9_dense_S. **Selected: s9_dense_S** (cheapest within the floor).


## Sprint 4/5 second validation season (2021 held out, 2022 in training) — 100 epochs

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s45v21_h3_n40 | 2 | 0.2882 ± 0.0044 | 0.3015 | 0.2298 | 15.231 | 0.272 | 0.237 | 0.635 | 0.774 | 0.916 | 0.373 | 3.063 | 0.691 |
| s45v21_h14_n16 | 2 | 0.2872 ± 0.0002 | 0.3013 | 0.2223 | 15.184 | 0.272 | 0.240 | 0.632 | 0.770 | 0.927 | 0.376 | 3.099 | 0.688 |
| s45v21_h3_n20 | 2 | 0.2855 ± 0.0049 | 0.3027 | 0.2194 | 15.232 | 0.271 | 0.239 | 0.630 | 0.748 | 0.922 | 0.379 | 3.064 | 0.692 |
| **s45v21_h3_n16** | 2 | 0.2847 ± 0.0022 | 0.2998 | 0.2237 | 15.249 | 0.272 | 0.238 | 0.638 | 0.753 | 0.933 | 0.377 | 3.078 | 0.700 |
| GFS-bilinear (reference) | | 0 | | 0 | 15.635 | 0.221 | 0.201 | 0.525 | 0.871 | 1.639 | 0.695 | 7.074 | 1.862 |

Noise floor 0.0050; configs within it of the best: s45v21_h3_n40, s45v21_h14_n16, s45v21_h3_n20, s45v21_h3_n16. **Selected: s45v21_h3_n16** (cheapest within the floor).


## Sprint 6 — deterministic vs diffusion

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **s6_diff** | 2 | 0.2584 ± 0.0003 | nan | 0.2399 | 17.014 | 0.297 | 0.215 | 0.659 | 0.631 | 0.893 | 0.381 | 2.952 | 0.757 |
| s9_dense_S | 2 | 0.2376 ± 0.0046 | 0.2604 | 0.2162 | 17.434 | 0.291 | 0.218 | 0.643 | 0.574 | 0.930 | 0.406 | 3.052 | 0.792 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0050; configs within it of the best: s6_diff. **Selected: s6_diff** (cheapest within the floor).


## Sprint 9 — capacity and MoE

| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | precip_wet_mae | precip_csi15 | precip_csi30 | precip_fss15 | precip_bias_ratio | tmax_mae | tmin_mae | rh_mae | wind_vec_rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s9_dense_L | 2 | 0.2424 ± 0.0055 | 0.2609 | 0.2183 | 17.318 | 0.292 | 0.220 | 0.652 | 0.573 | 0.948 | 0.398 | 3.019 | 0.776 |
| s9_moe_e8k2f50 | 2 | 0.2421 ± 0.0071 | 0.2631 | 0.2145 | 17.127 | 0.299 | 0.233 | 0.660 | 0.598 | 0.971 | 0.406 | 3.063 | 0.792 |
| s9_moe_e8k1f50 | 2 | 0.2392 ± 0.0020 | 0.2634 | 0.2153 | 17.223 | 0.296 | 0.226 | 0.653 | 0.573 | 0.953 | 0.410 | 3.061 | 0.801 |
| **s9_dense_S** | 2 | 0.2376 ± 0.0046 | 0.2604 | 0.2162 | 17.434 | 0.291 | 0.218 | 0.643 | 0.574 | 0.930 | 0.406 | 3.052 | 0.792 |
| s9_dense_M | 2 | 0.2376 ± 0.0015 | 0.2595 | 0.2218 | 17.428 | 0.288 | 0.215 | 0.643 | 0.564 | 0.942 | 0.398 | 3.060 | 0.783 |
| s9_moe_e4k1f50 | 2 | 0.2375 ± 0.0056 | 0.2607 | 0.2231 | 17.488 | 0.293 | 0.217 | 0.650 | 0.567 | 0.949 | 0.404 | 3.037 | 0.775 |
| s9_moe_e8k1f100 | 2 | 0.2354 ± 0.0058 | 0.2600 | 0.2151 | 17.430 | 0.285 | 0.220 | 0.635 | 0.562 | 0.946 | 0.405 | 3.035 | 0.783 |
| s9_moe_e4k2f50 | 2 | 0.2353 ± 0.0006 | 0.2577 | 0.2147 | 17.385 | 0.291 | 0.223 | 0.649 | 0.568 | 0.960 | 0.407 | 3.080 | 0.799 |
| s9_moe_e16k1f50 | 2 | 0.2330 ± 0.0043 | 0.2562 | 0.2191 | 17.482 | 0.285 | 0.215 | 0.640 | 0.551 | 0.947 | 0.410 | 3.071 | 0.784 |
| s9_moe_e8k1f25 | 2 | 0.2305 ± 0.0088 | 0.2566 | 0.2247 | 17.638 | 0.282 | 0.208 | 0.625 | 0.541 | 0.940 | 0.399 | 3.054 | 0.778 |
| s9_moe_e16k2f50 | 2 | 0.2300 ± 0.0185 | 0.2736 | 0.2022 | 17.501 | 0.279 | 0.209 | 0.622 | 0.516 | 0.953 | 0.402 | 3.018 | 0.768 |
| GFS-bilinear (reference) | | 0 | | 0 | 17.826 | 0.278 | 0.174 | 0.624 | 0.807 | 1.501 | 0.687 | 6.412 | 1.863 |

Noise floor 0.0075; configs within it of the best: s9_dense_L, s9_moe_e8k2f50, s9_moe_e8k1f50, s9_dense_S, s9_dense_M, s9_moe_e4k1f50, s9_moe_e8k1f100, s9_moe_e4k2f50. **Selected: s9_dense_S** (cheapest within the floor).
