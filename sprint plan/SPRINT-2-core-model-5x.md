# Sprint 2 — Days 3–4: Core Model 5× All-India Training & Baselines

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Days 3–4  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Train the primary 5× direct downscaling architecture (terrain-conditioned CNN/U-Net) across the all-India monsoon dataset with strict spatial holdout enforcement (Mandya $+ 0.5^\circ$ buffer never seen). Implement and benchmark competitive baselines (Bilinear, DeepSD-style CNN + elevation, optional RF). Verify that physical mass conservation ($\mathcal{L}_{\text{cons}}$, kernel=5) converges without the sum bug.

---

## 2. Team Responsibilities & Ownership

| Role | Sprint 2 Deliverables |
|---|---|
| **ML Lead** | Implement model architecture in `src/models/`, integrate terrain conditioning (terrain-conditioned DEM + slope + aspect), execute training loop with PyTorch AMP fp16, enforce conservation loss $\mathcal{L}_{\text{cons}}$ with $k=5$, verify convergence. |
| **ML/Eval Engineer** | Build fast PyTorch Dataset/DataLoader reading from pre-cached Zarr/LMDB patch store, implement baselines (Bilinear interpolation, DeepSD-style CNN, optional Random Forest), log validation metrics (MAE, RMSE, Pearson $r$) on val year 2021. |
| **Data/GIS Lead** | Monitor GPU I/O throughput, support Kaggle fallback environment if local RTX 3060 experiences thermal/memory throttling. |
| **Domain/Pitch Lead** | Review training progression and verify that model metrics are logged honestly without architectural overclaiming. |

---

## 3. Pinned Decisions & Engineering Constraints

* **Direct 5× Scaling:** Single-stage $5\times$ downscaling ($0.25^\circ \to 0.05^\circ$, kernel 5). Do **not** use multistage or "3 blocks × 2x = 4x" (which is mathematically $2^3 = 8\times$, not $4\times$ or $5\times$).
* **Spatial Holdout Guarantee:** The model must **never** train on patches within the Mandya boundary $+ 0.5^\circ$ buffer (~50–100 km). This ensures a bulletproof spatial generalization narrative in front of the jury.
* **4 Split Partitions:**
  - **Train:** Monsoon (JJAS) 2010–2020 (buffer excluded).
  - **Validation:** Monsoon 2021 (for model selection, checkpointing, and difficulty weighting).
  - **Calibration:** Monsoon 2022 (reserved strictly for conformal quantile regression / CQR in Sprint 3).
  - **Test:** Monsoon 2023 (unseen temporal test benchmark).
* **Conservation Loss Function:**
  $$\mathcal{L}_{\text{total}} = \text{MSE}(\text{HR}_{\text{pred}}, \text{HR}_{\text{true}}) + \lambda \mathcal{L}_{\text{cons}}$$
  $$\mathcal{L}_{\text{cons}} = \text{MSE}\left(\frac{\text{avg\_pool2d}(\text{HR}_{\text{pred}} \odot \mathbf{w}_{\text{HR}}, k=5, s=5)}{\text{avg\_pool2d}(\mathbf{w}_{\text{HR}}, k=5, s=5)}, \text{LR}_{\text{true}}\right)$$
  - $\mathbf{w}_{\text{HR}} = \cos(\text{lat}_{\text{HR\_rad}})$ evaluated at HR pixel centers.
  - `count_include_pad=False` mandatory.
* **Elevation & Difficulty Weighting:** Calculate empirical per-pixel error by elevation, slope, and rainfall intensity on held-out validation data. Apply difficulty quantile weighting rather than an arbitrary fixed elevation threshold ($>500\text{ m}$).

---

## 4. Compute Budget & Training Profile

* **Patch Store:** $\sim 200,000\text{--}240,000$ usable patches ($80\times 80$ HR, $16\times 16$ LR context) stored in local Zarr / LMDB.
* **Batch Size:** 32 (utilizes PyTorch automatic mixed precision `torch.cuda.amp.autocast(dtype=torch.float16)`).
* **Steps per Epoch:** $\approx 7,500$ steps/epoch.
* **Training Duration:** 15 epochs $\approx 3.2\text{ hours}$ compute $+ \text{I/O} = \mathbf{4\text{--}6\text{ hours}}$ on an NVIDIA RTX 3060 Laptop GPU.
* **Execution Plan:**
  - **Primary:** Local RTX 3060 machine.
  - **Backup:** Kaggle GPU notebook (with pre-processed patch dataset pre-uploaded via private dataset; never re-extract on Kaggle).

---

## 5. Sprint 2 Detailed Tasks

### A. Baseline Implementations (`src/models/baselines.py`)
- [ ] **Bilinear Interpolation Baseline:**
  - Standard $0.25^\circ \to 0.05^\circ$ bilinear upsampling.
  - Evaluated on val 2021 to establish the performance floor.
- [ ] **DeepSD-Style CNN Baseline:**
  - Standard SRCNN/DeepSD architecture with elevation channel (SRCNN with DEM conditioning per Vandal et al.).
  - Explicitly cited as prior art in docs and slides—never claimed as team novelty.
- [ ] **Optional Random Forest Baseline:**
  - Pixel-wise RF using LR rainfall, DEM, slope, and aspect.
  - *Constraint:* Build only if Day 3 progress is green; otherwise skip to preserve time.

### B. Primary Model Architecture (`src/models/unet_5x.py`)
- [ ] **Terrain-Conditioned 5× U-Net / CNN:**
  - Input: $16\times 16$ LR IMD rainfall patch $+$ high-resolution $80\times 80$ terrain features (terrain elevation, slope, aspect) bilinearly downscaled to match intermediate layers.
  - Feature extraction backbone with skip connections.
  - PixelShuffle or transposed convolution upsampling stage tuned specifically to $5\times$.
  - Output: $80\times 80$ HR rainfall prediction.

### C. Training Loop & Checkpointing (`src/models/train.py`)
- [ ] **Implement Mixed Precision Training Loop:**
  - Fast DataLoader with pinned memory and multi-worker pre-fetching from LMDB/Zarr.
  - AdamW optimizer ($\text{lr} = 1\times 10^{-4}$, cosine annealing schedule).
  - Compute $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$.
  - Early stopping monitoring validation MAE on 2021 val split.
  - Save best model checkpoint: `models/checkpoints/best_5x_model.pt`.
- [ ] **Loss Convergence Verification:**
  - Verify $\mathcal{L}_{\text{cons}}$ steadily decays below $1\times 10^{-3}$ on constant and synthetic fields.
  - Ensure predictions remain non-negative: $\text{ReLU}$ or softplus on final rainfall head.

---

## 6. Verification Gates & Exit Criteria

- [ ] **`tests/test_conservation.py`:**
  - Passes with active model checkpoint: verify coarse-grained HR output conserves LR mass to $< 10^{-3}$.
- [ ] **`tests/test_registration.py`:**
  - Input loader transforms confirmed frozen; no coordinate misalignment.
- [ ] **Baseline Comparison Report (`docs/model_comparison.md`):**
  - Validation 2021 metrics recorded across all splits, reporting both All-Day MAE and **Wet-Day MAE (strictly conditioned on $\text{Rain} > 2.5\text{ mm}$ to eliminate dry-day inflation bias)**, RMSE, and Pearson $r$ for:
    1. Bilinear Interpolation Baseline
    2. DeepSD-style CNN + Elevation Baseline
    3. Proposed 5× Terrain-Conditioned Model
  - Proposed model demonstrates superior spatial sharpness in hilly/orographic regions over Bilinear baseline and improved wet-day skill.
- [ ] **Execution Exit Rule:** Best model weights saved and verified by EOD Day 4. If training diverges or conservation fails, revert to baseline checkpoint and re-tune $\lambda_{\text{cons}}$.
