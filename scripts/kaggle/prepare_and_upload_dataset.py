"""
scripts/kaggle/prepare_and_upload_dataset.py

Packages frozen Sprint 2 data artifacts into a Kaggle Dataset:
  - datasets/multitask_temporal_v1.zarr
  - data/sample_index.parquet
  - data/normalization_stats.yaml

Target: rohitajitbharadwaj/sih26074-multitask-temporal-v1
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import get_kaggle_username


def prepare_and_upload():
    print("=" * 65)
    print("Kaggle Dataset Packaging & Upload: sih26074-multitask-temporal-v1")
    print("=" * 65)

    staging_dir = ROOT / "data" / "kaggle_dataset_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    dataset_slug = "sih26074-multitask-temporal-v1"
    dataset_id = f"{username}/{dataset_slug}"

    # 1. Write dataset-metadata.json
    metadata = {
        "title": "sih26074-multitask-temporal-v1",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Copy sample index and normalization stats
    shutil.copy(ROOT / "data" / "sample_index.parquet", staging_dir / "sample_index.parquet")
    shutil.copy(ROOT / "data" / "normalization_stats.yaml", staging_dir / "normalization_stats.yaml")
    print("[+] Staged sample_index.parquet and normalization_stats.yaml")

    # 3. Create a fast zip archive of multitask_temporal_v1.zarr
    zarr_src = ROOT / "datasets" / "multitask_temporal_v1.zarr"
    zarr_archive = staging_dir / "multitask_temporal_v1.zarr.zip"
    print(f"[*] Compressing Zarr store to {zarr_archive.name} using native tar/zip...")
    t0 = time.time()

    # Use native tar if available or shutil
    # tar on Windows can create zip: tar -acf <out.zip> <folder>
    cmd = ["tar", "-acf", str(zarr_archive), "-C", str(ROOT / "datasets"), "multitask_temporal_v1.zarr"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] tar failed ({res.stderr}), falling back to shutil.make_archive...")
        shutil.make_archive(
            str(staging_dir / "multitask_temporal_v1.zarr"),
            "zip",
            root_dir=str(ROOT / "datasets"),
            base_dir="multitask_temporal_v1.zarr",
        )

    dur = time.time() - t0
    size_mb = zarr_archive.stat().st_size / (1024 * 1024)
    print(f"[+] Compressed Zarr in {dur:.1f}s ({size_mb:.1f} MB)")

    # 4. Check if dataset already exists on Kaggle
    print(f"[*] Checking if {dataset_id} exists on Kaggle...")
    check_cmd = ["kaggle", "datasets", "status", dataset_id]
    check_res = subprocess.run(check_cmd, capture_output=True, text=True)

    if check_res.returncode == 0 and "ready" in check_res.stdout.lower():
        print(f"[+] Dataset {dataset_id} already exists. Creating new version...")
        up_cmd = ["kaggle", "datasets", "version", "-p", str(staging_dir), "-m", "Sprint 3 frozen dataset update", "-t"]
    else:
        print(f"[*] Creating new dataset {dataset_id} on Kaggle...")
        up_cmd = ["kaggle", "datasets", "create", "-p", str(staging_dir), "-t"]

    t_up0 = time.time()
    up_res = subprocess.run(up_cmd, capture_output=True, text=True)
    if up_res.returncode != 0:
        print(f"[-] Upload failed: {up_res.stderr}")
        return False

    print(f"[+] Successfully uploaded to Kaggle in {time.time() - t_up0:.1f}s!")
    print(f"    Output: {up_res.stdout.strip()}")
    return True


if __name__ == "__main__":
    prepare_and_upload()
