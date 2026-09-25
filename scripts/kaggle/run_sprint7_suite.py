"""
scripts/kaggle/run_sprint7_suite.py

Orchestrates the complete Sprint 7 Diffusion-Step & Sampler Frontier Campaign on Kaggle Dual Tesla T4 GPUs:
  - Phase 1 Gate: Legacy DDIM-32 vs Corrected Standard DDIM-32.
  - Phase 2 Sweep: DDIM at 4, 8, 16, 32, 64 steps.
  - Phase 3 Sweep: Higher-Order Solvers (DPM-Solver++ 2M and PNDM) at matched NFE.
  - Phase 4 Profiling: Precise GPU latency, throughput, and VRAM benchmarking.
  - Phase 5 Confirmatory: Single evaluation of Pareto-optimal champion on quarantined 2023 holdout test set.

Safety Invariants:
  - 100% Inference Mode: Zero retraining or optimizer backpropagation.
  - Safe Quota Floor: Aborts if Kaggle GPU quota < 0.50 hours (30 minutes).
  - Frozen Capacity Invariant: Exactly 15,685,478 parameters.
  - Quarantined 2023 Test Protocol: Evaluated once on selected champion.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
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

SAFE_QUOTA_FLOOR_HOURS = 0.50  # 30-minute safety floor


def bundle_src_base64() -> str:
    """Zips the src/ package tree into a compact in-memory base64 string."""
    buf = io.BytesIO()
    src_path = ROOT / "src"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_path.rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_sprint7_kernel(
    staging_dir: Path,
    slug: str,
    title: str,
    eval_args: list,
) -> Path:
    """Packages code, unpacker header, and metadata for remote Kaggle GPU execution."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"
    src_b64 = bundle_src_base64()

    unpacker_header = f"""# Auto-generated Kaggle unpacker header for Sprint 7 Sampler Frontier
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
_zarr_zip = None
if Path("/kaggle/input").exists():
    for _cand in Path("/kaggle/input").rglob("*.zarr.zip"):
        _zarr_zip = _cand
        break

if _zarr_zip and _zarr_zip.exists():
    _stem_name = _zarr_zip.name.replace(".zip", "")
    _target_zarr = _cur / "datasets" / _stem_name
    if not _target_zarr.exists():
        print(f"[*] Extracting Zarr archive {{_zarr_zip}} to {{_cur / 'datasets'}}...", flush=True)
        (_cur / "datasets").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(_zarr_zip, "r") as _zf:
            _zf.extractall(_cur / "datasets")
        print("[+] Extracted Zarr archive successfully.", flush=True)

# Copy parquet index and normalization stats
(_cur / "data").mkdir(parents=True, exist_ok=True)
if Path("/kaggle/input").exists():
    import shutil
    for _f in Path("/kaggle/input").rglob("*.parquet"):
        shutil.copy(_f, _cur / "data" / _f.name)
    for _f in Path("/kaggle/input").rglob("*.yaml"):
        shutil.copy(_f, _cur / "data" / _f.name)

# Copy checkpoint files if present
(_cur / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
if Path("/kaggle/input").exists():
    import shutil
    for _f in Path("/kaggle/input").rglob("*.pt"):
        shutil.copy(_f, _cur / "models" / "checkpoints" / _f.name)
        print(f"[+] Staged checkpoint: {_f.name} into {_cur / 'models' / 'checkpoints'}", flush=True)

# Override CLI args if invoked directly
if len(sys.argv) <= 1:
    sys.argv = ["train.py"] + [{', '.join(repr(a) for a in eval_args)}]
"""

    eval_src_code = (ROOT / "scripts" / "evaluate_sprint7_samplers.py").read_text(encoding="utf-8")
    combined_code = unpacker_header + "\n" + eval_src_code

    with open(staging_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(combined_code)

    # Copy src package directly to staging as well
    src_dir = ROOT / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, staging_dir / "src", dirs_exist_ok=True)

    metadata = {
        "id": kernel_id,
        "title": title,
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            "rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14",
            "ssachithananthan/sih26074-sprint6-checkpoints",
        ],
    }

    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Prepared self-contained Sprint 7 kernel bundle in: {staging_dir}", flush=True)
    return staging_dir


def dispatch_and_monitor(
    slug: str,
    title: str,
    eval_args: list,
    max_wait_minutes: int = 60,
    poll_interval_sec: int = 25,
) -> bool:
    quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print("[*] KAGGLE GPU PRE-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {quota['remaining_hours']} hrs ({quota['remaining_minutes']} mins)")
    print(f"    Used GPU Time:      {quota['used_hours']} hrs")
    print(f"    Refresh Time:       {quota['refresh_time']}")
    print("=" * 65)

    if quota["remaining_hours"] < SAFE_QUOTA_FLOOR_HOURS:
        print(f"[-] Remaining quota ({quota['remaining_hours']} hrs) below safety floor ({SAFE_QUOTA_FLOOR_HOURS} hrs). Aborting.")
        return False

    staging_dir = ROOT / ".kaggle_staging_sprint7"
    prepare_sprint7_kernel(staging_dir, slug, title, eval_args)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    print(f"[*] Pushing kernel '{kernel_id}' to Kaggle Dual T4 GPU...", flush=True)
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(staging_dir)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[-] Push failed:\n{res.stderr.strip()}", flush=True)
        return False
    print(f"[+] Successfully pushed! {res.stdout.strip()}", flush=True)

    match = re.search(r"kaggle\.com/code/[^/]+/([a-zA-Z0-9\-_]+)", res.stdout)
    if match:
        actual_slug = match.group(1)
        kernel_id = f"{username}/{actual_slug}"

    watch_url = f"https://www.kaggle.com/code/{kernel_id}"
    print(f"\n=======================================================", flush=True)
    print(f"[+] LIVE KAGGLE WATCH LINK: {watch_url}", flush=True)
    print(f"=======================================================\n", flush=True)

    print(f"[*] Monitoring live execution of '{kernel_id}'...", flush=True)
    start_time = time.time()
    max_sec = max_wait_minutes * 60
    success = False

    while time.time() - start_time < max_sec:
        time.sleep(poll_interval_sec)
        status_res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", kernel_id],
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
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    # Copy report JSON files to reports/
    for rpt in output_dir.rglob("*.json"):
        dst = ROOT / "reports" / rpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpt, dst)
        print(f"[+] Retrieved report: {dst.relative_to(ROOT)}")

    # Check post-run quota
    time.sleep(5)
    post_quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print("[*] KAGGLE GPU POST-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {post_quota['remaining_hours']} hrs ({post_quota['remaining_minutes']} mins)")
    print(f"    Used GPU Time:      {post_quota['used_hours']} hrs")
    print("=" * 65)
    update_agent_compute_context(post_quota)

    return success


def run_sprint7_suite():
    parser = argparse.ArgumentParser(description="Run Sprint 7 Sampler Frontier Campaign on Kaggle")
    parser.add_argument("--slug", type=str, default="sih26074-s7-sampler-frontier")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    slug = args.slug
    title = slug

    eval_cli = [
        "--condition", "all",
        "--batch_size", str(args.batch_size),
    ]
    if args.max_batches is not None:
        eval_cli.extend(["--max_batches", str(args.max_batches)])

    print("=" * 70)
    print("STARTING SPRINT 7 SAMPLER FRONTIER KAGGLE CAMPAIGN")
    print(f"Target Accelerator: Kaggle Dual Tesla T4 GPUs | Base Model: Scaled 15.69M Backbone")
    print(f"Kernel Slug: {slug}")
    print("=" * 70)

    success = dispatch_and_monitor(
        slug=slug,
        title=title,
        eval_args=eval_cli,
        max_wait_minutes=90,
        poll_interval_sec=30,
    )

    if not success:
        print("[-] Remote Kaggle run did not complete successfully.")
        sys.exit(1)

    # Benchmark compiler
    bench_script = ROOT / "scripts" / "benchmark_sprint7_samplers.py"
    if bench_script.exists():
        print("\n[*] Compiling Sprint 7 Benchmark Summary, Comparison Tables, and Walkthrough...")
        subprocess.run([sys.executable, str(bench_script)], check=True)
        print("[+] Benchmark compilation complete.")


if __name__ == "__main__":
    run_sprint7_suite()
