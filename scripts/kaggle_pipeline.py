"""
scripts/kaggle_pipeline.py

One-Command Kaggle Execution Pipeline:
1. Hardware & Environment Pre-Flight Check (GPU, VRAM, CUDA, PyTorch).
2. Verification Gates (22/22 unit tests passing).
3. Full 14-Year Dense Zarr Store Materialization (220,332 patches, 80x80 HR, 16x16 LR).
4. UNet5x VRAM Footprint & Mixed Precision Benchmark on GPU.
"""

import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def run_cmd(cmd: str, desc: str):
    print(f"\n========================================================")
    print(f"[*] {desc}")
    print(f"    Command: {cmd}")
    print(f"========================================================")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Step failed with return code {res.returncode} after {dt:.2f}s!")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {desc} completed in {dt:.2f}s.")


def main():
    print("""
    ============================================================
       SIH-26074: All-India Downscaling & Zarr Store Generator
       Hardware: Kaggle GPU (T4 / P100) Environment
    ============================================================
    """)

    # 1. Hardware Inspection
    print("[1/4] Inspecting Hardware Environment...")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (No CUDA)"
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_available else 0.0
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available:  {cuda_available}")
    print(f"  GPU Device:      {device_name}")
    print(f"  Total VRAM:      {vram_total:.2f} GB")

    # 2. Run Test Suite
    run_cmd(
        f"{sys.executable} -m pytest tests/ -v",
        "Running All 22 Verification Unit Tests",
    )

    # 3. Materialize Full 14-Year Zarr Store
    zarr_out = Path("data/cache/india_monsoon_patches.zarr")
    run_cmd(
        f"{sys.executable} scripts/build_zarr_store.py --full --output {zarr_out}",
        "Materializing Full 14-Year 220,332-Patch Zarr Array Cube",
    )

    # 4. Benchmark UNet5x Forward Pass and VRAM
    run_cmd(
        f"{sys.executable} -m src.models.unet_5x",
        "Benchmarking UNet5x Memory Footprint on GPU with GroupNorm and AMP",
    )

    print("\n========================================================")
    print(" [ALL GATES PASSED] Kaggle Materialization Complete!")
    print(f" Zarr Store Location: {zarr_out.resolve()}")
    print("========================================================\n")


if __name__ == "__main__":
    main()
