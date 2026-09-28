# Sprint 9 Implementation Plan: Model Capacity Scaling (Dense -> Larger Dense -> MoE)

## 1. Executive Summary
Sprint 9 tests whether increasing model capacity moves the complete accuracy-uncertainty-spatial-detail Pareto frontier outward before introducing sparse Mixture-of-Experts (MoE).

The frozen control is Candidate 3 from Sprint 6:
- Trainable parameters: 15,685,478
- Base channels: 96
- Prediction type: v-prediction
- Loss weighting: group-tail multi-task loss
- History length: H=14
- Context size: N=24, central target crop M=16
- Timesteps: T=100 (linear beta schedule: beta_start=1e-4, beta_end=0.035)

## 2. Capacity Scaling Ladder
The capacity ladder isolates model width as the sole primary independent variable:
1. **Dense-S**: base_channels = 96 (15,685,478 parameters, 1.00x).
2. **Dense-M**: base_channels = 136 (31,198,518 parameters, 1.99x). Target range [25M, 35M].
3. **Dense-L**: base_channels = 176 (51,997,958 parameters, 3.31x). Target range [45M, 65M].
4. **MoE-4**: base_channels = 96 with 4 bottleneck experts and Top-1 sparse routing (total parameters: 35.8M, active parameters: 15.7M).

## 3. Matched Inference Compute Protocol
Inference compute is fixed across all models at 32 NFE:
- **Point Mode**: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE total).
- **Distribution Mode**: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE total).

## 4. Evaluation Dimensions
1. **Point Accuracy**: Wet-MAE, CSI@15, CSI@30, Tmax MAE, Tmin MAE, RH MAE, Wind Vector RMSE.
2. **Probabilistic Calibration**: Precipitation Fair-CRPS, Spread-Skill Ratio (SSR), 90% Prediction Interval Coverage, Brier Score, and Brier Skill Score.
3. **Spatial Texture & Sharpness**: 2D Laplacian energy ratio and radially-averaged High-Frequency Power Spectral Density (PSD) retention for both single ensemble members and ensemble means.
4. **Calibration Requirement**: Post-hoc spread multiplier alpha* needed to achieve calibration.
5. **Efficiency & Computational Budget**: Trainable parameters, active forward FLOPs, inference latency, and peak GPU memory on Dual Tesla T4 accelerators.

## 5. Execution Strategy
To bypass Kaggle queue delays and the weekly batch accelerator cap, a standalone interactive Jupyter Notebook (`sprint_9_model_capacity_scaling.ipynb`) is placed in the local Downloads folder for direct interactive execution on Kaggle.
