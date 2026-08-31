# Sprint 2 — Sea-Ice Forecasting Model (Days 6–10)

Reference: `00-CANONICAL-SPEC.md`. Depends on Sprint 1's Zarr cube.
Runs in parallel with Sprint 3 (different person, independent workstream).

## Goal
A trained, calibrated, physics-informed sea-ice concentration forecaster that
beats a simple baseline on held-out 2024–2025 data, exported to ONNX and
ready for the backend to call.

## Tasks

### Self-supervised pretrain (`models/sea_ice/pretrain.py`)
- Architecture: encoder-decoder, ResNet34 encoder, attention-gated decoder
  (ANTSIC-UNet style).
- Task: mask 30% of the 10-channel cube (spatially and/or temporally),
  reconstruct the masked portion. Use the Sprint 1 sensor-swap utility so the
  model sees multiple SIC sources for the same day and learns sensor-
  invariant ice representations — this is the sensor-invariance trick from
  the Goldilocks approach; make sure it's actually wired in, not just planned.
- Train on the widest available window per canonical spec.
- Use mixed precision + gradient checkpointing to fit whatever GPU you have.

### Supervised fine-tune (`models/sea_ice/train_forecast.py`)
- Input: past 14 days × 10 channels. Output: next 7-day SIC probability grid
  + uncertainty (MC Dropout, p=0.2).
- Loss: `L_total = BCE + Dice + λ·physics_penalty`, where the physics penalty
  enforces mass conservation and physical bounds (SIC ∈ [0,1]) per the
  canonical spec's loss formulation — implement this as an actual loss term
  computed from the ice-motion/divergence fields, not a placeholder comment.
- Train/val/test split per canonical spec: Train 2018–2023, Test 2024–2025
  strict holdout.
- Log every run to Weights & Biases (or a local equivalent if W&B isn't
  available) — you need these curves for the "how did you validate this"
  judge question.

### Calibration (`models/sea_ice/calibrate.py`)
- Isotonic regression on the MC-Dropout uncertainty outputs, per-pixel, so
  the "90% confidence" number you show in the UI is actually meaningful and
  not just an unscaled dropout variance.

### Baseline for comparison (`models/sea_ice/baseline_convlstm.py`)
- A simple SA-ConvLSTM or plain ConvLSTM trained on SIC history alone (no
  physics channels). You need this to honestly claim "beats baseline X by Y"
  in the pitch — without it, your novelty claim has nothing to compare
  against.

### Evaluation (`models/sea_ice/evaluate.py`)
- Compute on the 2024–2025 test set: RMSE, MAE, IoU, Integrated Ice Edge
  Error — for both your model and the baseline.
- Run the three OOD checks from the canonical spec's dataset table:
  - Time holdout (already the test set by construction)
  - Sensor holdout: feed inputs conditioned on one SIC product, score against
    a different SIC product's ground truth for the same day, to test cross-
    sensor generalization of the forecast itself
  - Space holdout: train/validate on one sub-region of the corridor, test on
    a distinct sub-region — pick two sectors that are actually geographically
    distinct within your corridor (don't reuse a "0-60W Atlantic" label that
    has nothing to do with this corridor)
- Save a results table (`docs/sea_ice_results.md`) and at least one example
  forecast-vs-actual visualization figure for the pitch deck.

### Export
- Export the fine-tuned model to ONNX, quantize to int8 if inference speed on
  your target hardware needs it, and save the checkpoint somewhere retrievable
  (release artifact, Git LFS, or an HF Spaces backup — document which).

## Definition of Done
- [ ] Pretrain stage runs and produces a checkpoint (masked-reconstruction
      loss curve trending down)
- [ ] Fine-tuned model beats the ConvLSTM baseline on at least one of
      RMSE/IoU/Integrated Ice Edge Error on the 2024–2025 test set
- [ ] Isotonic calibration applied and the calibrated uncertainty is what
      gets served to the backend, not the raw dropout variance
- [ ] Sensor-holdout and space-holdout results computed and written to
      `docs/sea_ice_results.md`, with real numbers, not placeholders
- [ ] ONNX export works and produces the same output as the PyTorch model on
      a sanity-check input
- [ ] At least one forecast visualization figure saved for the pitch deck
