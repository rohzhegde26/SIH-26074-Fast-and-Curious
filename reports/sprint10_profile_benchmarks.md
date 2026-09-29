# Sprint 10: Operational Profiles Benchmarking Report

Evaluated on: `Kaggle Dual Tesla T4 (16GB)` accelerators.

| Profile Name | Model Tier | Total Params | Members (K) | Steps (S) | Total NFE | Peak VRAM | Latency / Cube | Latency / Lead-Day | Throughput | Target Latency | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACCURATE** | DENSE-L | 51,997,958 | 2 | 16 | 32 | 2.85 GB | **0.926 s** | 132.3 ms | 1.08 cubes/s | 1,500.0 ms | MET (38.2% head-room) |
| **ENSEMBLE** | DENSE-L | 51,997,958 | 8 | 8 | 64 | 3.42 GB | **1.675 s** | 239.4 ms | 0.60 cubes/s | 3,000.0 ms | MET (44.2% head-room) |

## Operational Profile Specialization

### 1. ACCURATE Profile (K=2, S=16, 32 NFE)
- **Primary Use Case**: Rapid state disaster response, district-level emergency operations, and flash-flood alerts.
- **Key Characteristics**: Prioritizes deep temporal iterative refinement (16 DDIM steps) with dual-member dispersion to ensure minimal single-lead artifacts.
- **SLA Fulfillment**: Requires sub-1.5 second turnaround; achieves 926 ms per 7-day multi-channel cube.

### 2. ENSEMBLE Profile (K=8, S=8, 64 NFE)
- **Primary Use Case**: Central IMD / NCMRWF routine numerical weather prediction downscaling, seasonal drought monitoring, and reservoir catchment inflow modeling.
- **Key Characteristics**: Maximizes probabilistic spread and distribution sampling across 8 diverse trajectories, optimizing Fair-CRPS, Spread-Skill Ratio (SSR), and Brier Skill Scores.
- **SLA Fulfillment**: Requires sub-3.0 second turnaround; achieves 1.675 seconds per 7-day multi-channel cube.
