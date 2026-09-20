"""
scripts/run_real_pipeline.py

End-to-End Orchestrator for Real Data Remediation:
    1. Ingests ERA5 coarse & ERA5-Land fine reanalysis -> data/raw/era5_land/era5_land_daily.nc
    2. Materializes paired dataset -> data/cache/multitask_real.npz (1,220 samples)
    3. Audits all datasets -> scripts/audit_real_datasets.py
    4. Trains MultiTaskUNet5x champion checkpoint -> models/checkpoints/multitask_5x_champion.pt
    5. Benchmarks baselines vs MultiTaskUNet5x -> scripts/benchmark_multitask_baselines.py
    6. Verifies full pytest suite
"""

from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.openmeteo_era5_ingestion import build_and_serialize_era5_reanalysis_dataset
from src.data.real_data_ingestion import build_and_cache_real_multitask_dataset
from scripts.audit_real_datasets import run_full_audit
from scripts.benchmark_multitask_baselines import run_benchmark


def run_command_checked(cmd: list):
    print(f"\n[*] Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")
    print(f"[+] Completed in {time.time()-t0:.1f}s")


def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING COMPLETE REAL-DATA WORKFLOW (SIH PROBLEM 26074)")
    print("=" * 80)

    # Step 1: Open-Meteo ERA5 / ERA5-Land Reanalysis Ingestion
    print("\n>>> STEP 1: ECMWF ERA5 & ERA5-Land Reanalysis Ingestion (1,220 days)")
    era5_nc = build_and_serialize_era5_reanalysis_dataset(force_rebuild=False)
    print(f"[+] ERA5 NetCDF ready: {era5_nc}")

    # Step 2: Assemble multitask_real.npz (1,220 samples)
    print("\n>>> STEP 2: Real Multi-Task Dataset Materialization (1,220 samples)")
    cache_npz = build_and_cache_real_multitask_dataset(force_rebuild=True)
    print(f"[+] Multi-task dataset ready: {cache_npz}")

    # Step 3: Full Audit
    print("\n>>> STEP 3: Complete Dataset & Provenance Audit")
    run_full_audit()

    # Step 4: Retrain MultiTaskUNet5x Checkpoint
    print("\n>>> STEP 4: Retrain MultiTaskUNet5x on Authentic Real Data (25 epochs)")
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_multitask_unet.py"),
        "--epochs", "25",
        "--batch_size", "16",
        "--lr", "1e-3",
    ]
    run_command_checked(train_cmd)

    # Step 5: Run Gridded Benchmark on 2023 Held-Out Split
    print("\n>>> STEP 5: Run 3-Way Baseline Benchmark on 122 Test Samples")
    run_benchmark(max_test_samples=122)

    # Step 6: Pytest validation
    print("\n>>> STEP 6: Run Pytest Test Suite")
    test_cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    run_command_checked(test_cmd)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"[+] COMPLETE REAL-DATA REMEDIATION SUCCEEDED IN {total_time/60:.1f} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
