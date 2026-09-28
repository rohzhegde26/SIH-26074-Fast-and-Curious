# Sprint 9 Walkthrough: Model Capacity Scaling (Dense -> Larger Dense -> MoE)

## 1. Context and Objective
Following Sprint 8.5's bottleneck diagnosis, which proved that post-hoc calibration alone cannot fully overcome conditional under-dispersion, Sprint 9 evaluates whether neural model capacity scaling moves the accuracy-uncertainty-sharpness Pareto frontier outward.

## 2. Capacity Ladder Design
The capacity ladder isolates base channel width while keeping all other architectural and physical invariants locked:
- **Dense-S (Control)**: 15.69M parameters (Candidate 3 baseline).
- **Dense-M**: 31.20M parameters (1.99x scale, base_channels = 136).
- **Dense-L**: 52.00M parameters (3.31x scale, base_channels = 176).
- **MoE-4**: 35.77M total parameters with 15.69M active parameters (sparse Top-1 routing over 4 bottleneck experts).

## 3. Matched 32 NFE Inference Regimes
Each model tier is audited under matched inference compute:
1. **Point Forecast Mode**: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE). Focuses on deterministic Wet-MAE, extreme convective storm recall (CSI@15, CSI@30), and spatial textures.
2. **Distribution Mode**: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE). Focuses on probabilistic Fair-CRPS, Spread-Skill Ratio (SSR), 90% prediction interval coverage, and Brier Skill Scores.

## 4. Interactive Kaggle Notebook
To circumvent weekly batch quota constraints and queue bottlenecks, the entire training, evaluation, and analysis workflow is bundled into a standalone notebook:
- **Location**: `C:\Users\rohit\Downloads\sprint_9_model_capacity_scaling.ipynb`
- Can be uploaded directly to Kaggle and run interactively with Dual Tesla T4 GPUs.
