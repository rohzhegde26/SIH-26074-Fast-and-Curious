# Sprint 8.5 Paired Bootstrap Confidence Intervals Report

**Project:** SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  
**Branch:** `feat/spatiotemporal-diffusion-downscaler`  
**Bootstrap Iterations:** 1,000  
**Resampling Unit:** Complete 7-Day Spatiotemporal Forecast Cube  

---

## 1. Metric Point Estimates and 95% Confidence Intervals

| Condition | Metric | Point Estimate | 95% Confidence Interval |
|---|---|---:|:---:|
| `REF_C_K8_S4_ETA05` | crps | 0.5801 | [0.4076, 0.7735] |
| `REF_C_K8_S4_ETA05` | precip_crps | 1.7995 | [1.5170, 2.1241] |
| `REF_C_K8_S4_ETA05` | wet_mae | 5.1650 | [4.6141, 5.7609] |
| `REF_C_K8_S4_ETA05` | csi30 | 0.4584 | [0.3827, 0.5304] |
| `REF_DET_K8_S4_ETA0` | crps | 0.5806 | [0.4168, 0.7817] |
| `REF_DET_K8_S4_ETA0` | precip_crps | 1.8002 | [1.5048, 2.1145] |
| `REF_DET_K8_S4_ETA0` | wet_mae | 5.1307 | [4.5664, 5.7231] |
| `REF_DET_K8_S4_ETA0` | csi30 | 0.4565 | [0.3804, 0.5297] |
| `REF_A_K2_S16_ETA0` | crps | 0.5697 | [0.4039, 0.7747] |
| `REF_A_K2_S16_ETA0` | precip_crps | 1.7511 | [1.4466, 2.0506] |
| `REF_A_K2_S16_ETA0` | wet_mae | 5.4673 | [4.8976, 6.0338] |
| `REF_A_K2_S16_ETA0` | csi30 | 0.4364 | [0.3602, 0.5071] |
| `REF_B_K4_S8_ETA0` | crps | 0.5728 | [0.4062, 0.7722] |
| `REF_B_K4_S8_ETA0` | precip_crps | 1.7666 | [1.4431, 2.1177] |
| `REF_B_K4_S8_ETA0` | wet_mae | 5.2367 | [4.6695, 5.8296] |
| `REF_B_K4_S8_ETA0` | csi30 | 0.4489 | [0.3763, 0.5242] |

---

## 2. Methodology and Statistical Notes

1. **Cube-Preserving Invariant:** Every bootstrap resample draws entire 7-day cubes with replacement, fully preserving temporal autocorrelation and spatial correlation.
2. **Point Estimate Identity:** The bootstrap point estimate is mathematically identical to the Case-Preserving Weighted Sample Mean.