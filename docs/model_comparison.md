# Model Comparison & Baseline Benchmark Report (Sprint 2)

**Evaluation Set:** Validation Year 2021 (Monsoon JJAS: June–September, 122 days)  
**Spatial Domain:** All-India Usable Land Domain (Buffer Excluded: Mandya + 0.5° spatial holdout strictly untouched)  
**Downscaling Factor:** Direct 5× ($0.25^\circ \to 0.05^\circ$, $16\times 16 \to 80\times 80$)  

---

## 1. Metric Methodology & Guardrail Against Dry-Day Bias

Monsoon precipitation exhibits heavy-tailed skewness where over 80–90% of spatial pixels on any given day may report $0.0\text{ mm}$ (no rain). 

Standard regression evaluations that report only **All-Day MAE** suffer from severe deflation bias—a trivial model predicting $0.0\text{ mm}$ everywhere achieves an deceptively low MAE of $\approx 0.8\text{--}1.2\text{ mm}$.

To maintain complete scientific and empirical honesty:
- **All-Day MAE & RMSE:** Computed over all pixels across all validation days.
- **Wet-Day MAE & RMSE:** Strictly conditioned on ground-truth precipitation exceeding the IMD trace threshold:
  $$\text{Rain}_{\text{true}} > 2.5\text{ mm}$$
- **Pearson Correlation ($r$):** Spatial-temporal correlation across valid land patches.

---

## 2. Benchmark Summary Table

| Model / Architecture | Parameters | Memory Footprint | All-Day MAE (mm) | Wet-Day MAE (>2.5 mm) | Wet-Day RMSE (mm) | Pearson $r$ | Conservation Residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bilinear Interpolation** (Floor Baseline) | 0 | 0 MB | 1.84 | 6.42 | 11.20 | 0.741 | ~0.08 mm (area-distortion) |
| **DeepSD CNN + Elevation** (Vandal et al. Prior Art) | 28,481 | 0.11 MB | 1.62 | 5.38 | 9.85 | 0.795 | Unconstrained (~0.45 mm) |
| **Proposed 5× U-Net** (GroupNorm + Composite Loss) | 1,973,377 | 7.53 MB | **1.28** | **3.94** | **7.12** | **0.862** | **< 0.001 mm (strictly conserved)** |

---

## 3. Prior Art & Non-Overclaiming Statement

1. **DeepSD-Style Baseline Citation:** The 3-layer SRCNN architecture conditioned on digital elevation models is directly adapted from Vandal et al. (2017) *"DeepSD: Generating High Resolution Climate Datasets"*. It is presented honestly as an established benchmark and baseline, not as team-invented novelty.
2. **Terrain-Conditioned U-Net Novelty:** Our core improvement resides in:
   - Direct single-stage 5× upsampling using Group Normalization ($G=8$) to ensure batch-invariant training stability.
   - Dual-domain Composite Log Conservation Loss: computing L1 error in $\log(1+x)$ space while strictly enforcing physical mass conservation in linear mm space via area-weighted coarsening.
