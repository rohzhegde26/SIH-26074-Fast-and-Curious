"""
scripts/kaggle/dispatch_temporal_training.py

Automated Kaggle Accelerator Dispatcher for Sprint 3 Spatiotemporal Downscaler.
Dispatches:
  - Phase 1: 7-Day Deterministic Multivariate Baseline (TemporalMultiTaskUNet5x)
  - Phase 2: Conditional Spatiotemporal Residual Diffusion (SpatiotemporalResidualDiffusion)
  - Timing Probe: 1-epoch execution benchmarking on Dual Tesla T4 GPUs.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import (
    get_kaggle_username,
    query_kaggle_gpu_quota,
    update_agent_compute_context,
)


def bundle_src_base64() -> str:
    """Zips the src/ package tree into a compact in-memory base64 string."""
    buf = io.BytesIO()
    src_path = ROOT / "src"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_path.rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_temporal_kernel(
    staging_dir: Path,
    slug: str,
    title: str,
    train_args: list,
) -> Path:
    """Packages code, unpacker header, and metadata for remote Kaggle GPU execution."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    src_b64 = bundle_src_base64()
    cli_args_str = " ".join(f'"{a}"' for a in train_args)

    unpacker_header = f"""# Auto-generated Kaggle unpacker header
import base64
import io
import os
import sys
import zipfile
from pathlib import Path

_cur = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(os.getcwd())
os.chdir(str(_cur))

# 0. Ensure dependencies
import subprocess
for _pkg in ["zarr"]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[*] Installing {{_pkg}} in Kaggle environment...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=True)
        print(f"[+] Installed {{_pkg}} successfully.", flush=True)

# 1. Unpack embedded source bundle
if not (_cur / "src").exists():
    print("[*] Unpacking embedded source bundle...", flush=True)
    _data = base64.b64decode(\"\"\"{src_b64}\"\"\")
    with zipfile.ZipFile(io.BytesIO(_data)) as _zf:
        _zf.extractall(_cur)
    print("[+] Unpacked source bundle successfully.", flush=True)

if str(_cur) not in sys.path:
    sys.path.insert(0, str(_cur))

# 2. Extract dataset from Kaggle dataset input if present
_input_dir = Path("/kaggle/input/sih26074-multitask-temporal-v1")
_zarr_zip = _input_dir / "multitask_temporal_v1.zarr.zip"
if not _zarr_zip.exists() and Path("/kaggle/input").exists():
    _candidates = list(Path("/kaggle/input").rglob("multitask_temporal_v1.zarr.zip"))
    if _candidates:
        _zarr_zip = _candidates[0]
        _input_dir = _zarr_zip.parent

_target_zarr = _cur / "datasets" / "multitask_temporal_v1.zarr"

if _zarr_zip.exists() and not _target_zarr.exists():
    print(f"[*] Extracting Zarr archive {{_zarr_zip}} to {{_cur / 'datasets'}}...", flush=True)
    (_cur / "datasets").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_zarr_zip, "r") as _zf:
        _zf.extractall(_cur / "datasets")
    print("[+] Extracted Zarr archive successfully.", flush=True)

# Copy parquet index and normalization stats
(_cur / "data").mkdir(parents=True, exist_ok=True)
for _f in ["sample_index.parquet", "normalization_stats.yaml"]:
    _src_f = _input_dir / _f
    if not _src_f.exists() and Path("/kaggle/input").exists():
        _found = list(Path("/kaggle/input").rglob(_f))
        if _found:
            _src_f = _found[0]
    _dst_f = _cur / "data" / _f
    if _src_f.exists() and not _dst_f.exists():
        import shutil
        shutil.copy(_src_f, _dst_f)

# Override CLI args if invoked directly
if len(sys.argv) <= 1:
    sys.argv = ["train.py"] + [{', '.join(repr(a) for a in train_args)}]
"""

    train_src_code = (ROOT / "scripts" / "train_temporal_downscaler.py").read_text(encoding="utf-8")
    combined_train_code = unpacker_header + "\n" + train_src_code

    with open(staging_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(combined_train_code)

    # Copy src package directly to staging as well
    src_dir = ROOT / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, staging_dir / "src", dirs_exist_ok=True)

    # Kernel metadata
    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": ["rohitajitbharadwaj/sih26074-multitask-temporal-v1"],
    }

    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Prepared self-contained Kaggle kernel bundle in: {staging_dir}", flush=True)
    return staging_dir


def dispatch_temporal_job(
    slug: str,
    title: str,
    train_args: list,
    max_wait_minutes: int = 40,
    poll_interval_sec: int = 20,
) -> bool:
    """Dispatches temporal training job to Kaggle Dual T4 GPUs and monitors execution."""
    quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print("[*] KAGGLE GPU PRE-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {quota['remaining_hours']} hrs ({quota['remaining_minutes']} mins)")
    print(f"    Used GPU Time:      {quota['used_hours']} hrs")
    print(f"    Quota Refresh Time: {quota['refresh_time']}")
    print("=" * 65)
    update_agent_compute_context(quota)

    if quota["remaining_minutes"] < 5.0:
        print("[-] ERROR: Less than 5 minutes of Kaggle GPU quota remaining! Aborting dispatch.")
        return False

    staging_dir = ROOT / ".kaggle_staging_temporal"
    prepare_temporal_kernel(
        staging_dir=staging_dir,
        slug=slug,
        title=title,
        train_args=train_args,
    )

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    print(f"[*] Pushing kernel '{kernel_id}' to Kaggle Dual T4 GPU...", flush=True)
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(staging_dir)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[-] Push failed:\n{res.stderr.strip()}", flush=True)
        return False
    print(f"[+] Successfully pushed! {res.stdout.strip()}", flush=True)

    print(f"[*] Monitoring live execution of '{kernel_id}'...", flush=True)
    start_time = time.time()
    max_sec = max_wait_minutes * 60
    success = False

    while time.time() - start_time < max_sec:
        time.sleep(poll_interval_sec)
        status_res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
        )
        status_line = status_res.stdout.strip()
        elapsed_min = (time.time() - start_time) / 60.0
        print(f"    [{time.strftime('%H:%M:%S')} | Elapsed: {elapsed_min:.1f}m] {status_line}", flush=True)

        if "complete" in status_line.lower():
            print("[+] Remote Kaggle GPU run completed successfully!")
            success = True
            break
        elif "error" in status_line.lower() or "failed" in status_line.lower():
            print(f"[-] Remote Kaggle GPU run failed: {status_line}")
            break

    # Download output artifacts
    output_dir = ROOT / "output" / "kaggle" / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading logs and artifacts to {output_dir}...")
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )

    # Copy checkpoints and reports to canonical repository directories
    for ckpt in output_dir.rglob("*.pt"):
        dst = ROOT / "models" / "checkpoints" / ckpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt, dst)
        print(f"[+] Retrieved checkpoint: {dst.relative_to(ROOT)}")

    for rpt in output_dir.rglob("*.json"):
        dst = ROOT / "reports" / rpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpt, dst)
        print(f"[+] Retrieved report: {dst.relative_to(ROOT)}")

    # Post-run quota check
    time.sleep(5)
    post_quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print("[*] KAGGLE GPU POST-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {post_quota['remaining_hours']} hrs ({post_quota['remaining_minutes']} mins)")
    print(f"    Used GPU Time:      {post_quota['used_hours']} hrs")
    print("=" * 65)
    update_agent_compute_context(post_quota)

    return success


def parse_dispatch_args():
    parser = argparse.ArgumentParser(description="Dispatch Sprint 3 training to Kaggle GPU")
    parser.add_argument("--mode", type=str, default="deterministic", choices=["deterministic", "diffusion"])
    parser.add_argument("--history_len", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count (1 for timing probe)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=8, help="DDIM sampling steps for validation")
    parser.add_argument("--slug", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_dispatch_args()
    slug = args.slug or f"sih26074-s3-{args.mode}-h{args.history_len}-e{args.epochs}"
    title = slug.replace("-", " ")

    train_cli = [
        "--mode", args.mode,
        "--history_len", str(args.history_len),
        "--model_size", args.model_size,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--ddim_steps", str(args.ddim_steps),
    ]

    dispatch_temporal_job(slug=slug, title=title, train_args=train_cli)
