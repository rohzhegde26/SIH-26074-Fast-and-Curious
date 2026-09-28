# Sprint 9 Model Training and Architecture Audit

## 1. Checkpoint Provenance and Weight Invariants
All scaling experiments in Sprint 9 derive from the frozen Sprint 6 Candidate 3 multi-task champion:
- **Baseline Checkpoint**: `models/checkpoints/sprint6_candidate3_multitask_champion.pt`
- **Expected SHA-256**: `f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92`
- **Candidate 3 Parameters**: 15,685,478 (Dense-S)

## 2. Invariant Scientific Constants
The following operational constants were held strictly invariant across all capacity tiers:
- History Length: H = 14 days
- Coarse Spatial Context: N = 24 (0.25 degree resolution)
- Target Output Region: M = 16 (upsampled 5x to 80x80 grid)
- Diffusion Timesteps: T = 100
- Diffusion Parameterization: v-prediction (Salimans & Ho, 2022)
- Noise Schedule: Linear beta schedule (beta_start = 1e-4, beta_end = 0.035)
- Loss Formulation: Variable-aware multi-task group-tail loss:
  - Precipitation focal tail weight: 3.0 on extreme cells (> 15 mm)
  - Group weights: lambda_precip = 1.0, lambda_thermo = 1.2, lambda_wind = 1.1

## 3. Measured Parameter Counts
The capacity ladder architectures were compiled and measured directly in PyTorch:
- **Dense-S**: 15,685,478 parameters (1.00x)
- **Dense-M**: 31,198,518 parameters (1.99x)
- **Dense-L**: 51,997,958 parameters (3.31x)
- **MoE-4**: 35,765,990 total parameters (15,685,478 active parameters)

## 4. Evaluation and Holdout Quarantine
- Model architecture exploration, capacity tuning, and calibration fitting are performed exclusively on the 2015-2021 training set and 2022 validation set.
- The 2023 El Nino holdout year remains quarantined for final verification.
