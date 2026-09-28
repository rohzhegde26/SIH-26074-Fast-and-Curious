# Sprint 9 Spatial Detail and Texture Scaling Report

## 1. Executive Summary
This report analyzes fine-scale spatial texture retention, gradient preservation, and spectral characteristics across the Sprint 9 capacity scaling ladder:
- Dense-S (15.69M params, base_channels = 96)
- Dense-M (31.20M params, base_channels = 136)
- Dense-L (52.00M params, base_channels = 176)
- MoE-4 (35.77M params total / 15.69M active params)

## 2. Methodology & Mathematical Formulations

### 2.1 Discrete 2D Laplacian Energy
To measure fine-scale gradient sharpness, each 80x80 precipitation field is convolved with the 3x3 discrete Laplacian operator:
$$L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
The Laplacian energy is defined as the mean squared response:
$$E_{\text{Lap}} = \frac{1}{H \cdot W} \sum_{i=1}^H \sum_{j=1}^W \left( (I * L)_{i, j} \right)^2$$
The retention ratio is defined relative to the ground truth verification field:
$$R_{\text{Lap}} = \frac{E_{\text{Lap}}^{\text{pred}}}{E_{\text{Lap}}^{\text{truth}}}$$

### 2.2 Radially-Averaged Power Spectral Density (PSD)
Using 2D Fast Fourier Transform (FFT), the spatial power spectrum is computed:
$$S(u, v) = |\mathcal{F}\{I\}(u, v)|^2$$
Radial averaging over concentric wavenumber bins $k = \sqrt{u^2 + v^2}$ yields 1D power spectra $P(k)$. High-frequency power $P_{\text{HF}}$ is aggregated over wavenumbers $k \ge k_{\text{Nyquist}} / 2$.
The high-frequency retention ratio is:
$$R_{\text{HF}} = \frac{P_{\text{HF}}^{\text{pred}}}{P_{\text{HF}}^{\text{truth}}}$$

## 3. Single-Member vs. Ensemble-Mean Phase Dynamics
A fundamental physical principle governing generative weather downscaling:
1. **Single Ensemble Members**: Retain sharp, physically plausible local storm gradients ($R_{\text{Lap}} \approx 20\% - 30\%$). Individual members place convective cells at specific coordinate locations.
2. **Ensemble Mean**: Averages over stochastic members, which naturally cancels out high-wavenumber phase discrepancies ($R_{\text{Lap}} \approx 7\% - 10\%$). This spatial smoothing is a mathematical property of minimum-MSE expectation, not a defect of the model.

## 4. Multi-Dimensional Spatial Texture Frontier Across 32 NFE

| Model Tier | Base Channels | Status | Single-Member $R_{\text{Lap}}$ | Ensemble-Mean $R_{\text{Lap}}$ | Single-Member $R_{\text{HF}}$ | Spatial Autocorrelation ($r_1$) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Dense-S (Control)** | 96 | **VALIDATED_BASELINE** | 0.215 | 0.076 | 0.248 | 0.884 |
| **Dense-M** | 136 | TARGET_PENDING_KAGGLE | Target > 0.250 | Target > 0.085 | Target > 0.285 | Target ~ 0.865 |
| **Dense-L** | 176 | TARGET_PENDING_KAGGLE | Target > 0.300 | Target > 0.100 | Target > 0.320 | Target ~ 0.850 |
| **MoE-4** | 96 | TARGET_PENDING_KAGGLE | Match Dense-M | Match Dense-M | Match Dense-M | Match Dense-M |

## 5. Architectural Hypotheses
1. **Capacity Directly Restores Spatial Sharpness**: Increasing base channels from 96 to 176 is hypothesized to increase single-member Laplacian energy retention from 21.5% to > 30.0% (> 39% relative improvement).
2. **High-Frequency Power Recovery**: High-frequency spectral energy is hypothesized to exceed 32.0% in Dense-L, proving that larger denoiser width reduces artificial numerical diffusion.
3. **MoE Spatial Quality**: MoE-4 is designed to achieve single-member Laplacian retention comparable to Dense-M while executing at Dense-S active parameter scale.
4. **Validation Grounding**: Empirical numbers for Dense-M and Dense-L will be updated upon execution of `sprint_9_phase1_dense_capacity_scaling.ipynb` on Kaggle.
