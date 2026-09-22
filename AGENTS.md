# Agent Directives: SIH 26074 Weather Downscaling

## 1. Mandatory Compute Constraint: Model Training on Kaggle

> [!IMPORTANT]
> **Strict Operational Rule**: Run **ALL** model training workloads remotely on **Kaggle** accelerators (GPU or TPU) rather than running heavy training iterations on the local CPU or local machine.

### Guidelines for Model Training
1. **Remote Execution Only**:
   - Never run extensive, multi-epoch or multi-batch model training routines locally on the CPU.
   - Local execution is strictly reserved for quick sanity checks (e.g. 1 batch shape validation, unit tests, data ingestion validation).
2. **Kaggle Infrastructure**:
   - **Username**: `rohitajitbharadwaj` (authenticated via API token in `~/.kaggle`).
   - **GPU Quota**: 6.0 hours / week (2× Tesla T4 16GB or 1× Tesla P100 16GB).
   - **TPU Quota**: 20.0 hours / week (TPU VM v3-8, 8 cores, 128GB TPU HBM, 330GB RAM).
3. **Dispatch Workflow**:
   - Use `scripts/kaggle/dispatch_kaggle.py` to package code, push kernels to Kaggle, track training execution, and retrieve checkpoints/output artifacts into `output/kaggle/`.
   - Kernel metadata should specify `"enable_gpu": true` (or `"enable_tpu": true` with `"machine_shape": "Tpu1VmV38"` for TPU runs).
