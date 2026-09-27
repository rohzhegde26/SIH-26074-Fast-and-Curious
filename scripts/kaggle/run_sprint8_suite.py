"""
scripts/kaggle/run_sprint8_suite.py

Orchestrates the Sprint 8 Ensemble and Test-Time Scaling Campaign on Kaggle GPU Accelerators:
  - Phase 0 Gate: Reproducibility validation on Candidate 3 champion weights.
  - Phase 1 Sweep: Deterministic DDIM baselines (K=1, S in {4, 8, 16, 32, 64}).
  - Phase 2 Sweep: Ensemble size scaling (K in {2, 4, 8, 16}) and trajectory noise eta in {0.25, 0.50, 1.00}.
  - Phase 3 Sweep: Matched-compute budget frontier (Budgets 8, 16, 32, 64 NFE).
  - Phase 4 Confirmatory: Quarantined 2023 holdout test set evaluation on champion configuration.

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
from typing import Any, Dict, List, Optional
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


def prepare_sprint8_kernel(
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
    eval_args_repr = repr(list(eval_args))

    unpacker_header = f"""# Auto-generated Kaggle unpacker header for Sprint 8 Ensemble Scaling
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
        print(f"[+] Staged checkpoint: {{_f.name}} into {{_cur / 'models' / 'checkpoints'}}", flush=True)

# Override CLI args if invoked directly
if len(sys.argv) <= 1:
    sys.argv = ["train.py"] + {eval_args_repr}
"""

    eval_src_code = (ROOT / "scripts" / "evaluate_sprint8_ensemble.py").read_text(encoding="utf-8")
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
            "rohitajitbharadwaj/sih26074-sprint6-checkpoints",
        ],
    }

    with open(staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Prepared self-contained Sprint 8 kernel bundle in: {staging_dir}", flush=True)
    return staging_dir


def dispatch_kernel(
    slug: str,
    title: str,
    eval_args: list,
    staging_name: str = ".kaggle_staging_sprint8",
) -> Optional[str]:
    """Pushes a kernel to Kaggle and returns the live watch URL."""
    quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print("[*] KAGGLE GPU PRE-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {quota['remaining_hours']} hrs ({quota['remaining_minutes']} mins)")
    print(f"    Used GPU Time:      {quota['used_hours']} hrs")
    print(f"    Refresh Time:       {quota['refresh_time']}")
    print("=" * 65)

    if quota["remaining_hours"] < SAFE_QUOTA_FLOOR_HOURS:
        print(f"[-] Remaining quota ({quota['remaining_hours']} hrs) below safety floor ({SAFE_QUOTA_FLOOR_HOURS} hrs). Aborting.")
        return None

    staging_dir = ROOT / staging_name
    prepare_sprint8_kernel(staging_dir, slug, title, eval_args)

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
        return None
    print(f"[+] Successfully pushed! {res.stdout.strip()}", flush=True)

    match = re.search(r"kaggle\.com/code/[^/]+/([a-zA-Z0-9\-_]+)", res.stdout)
    if match:
        actual_slug = match.group(1)
        kernel_id = f"{username}/{actual_slug}"

    watch_url = f"https://www.kaggle.com/code/{kernel_id}"
    print(f"\n=======================================================", flush=True)
    print(f"[+] LIVE KAGGLE WATCH LINK: {watch_url}", flush=True)
    print(f"=======================================================\n", flush=True)

    return watch_url


def monitor_kernel(
    slug: str,
    max_wait_minutes: int = 90,
    poll_interval_sec: int = 30,
) -> bool:
    """Monitors a running kernel until completion and retrieves artifacts."""
    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

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


def run_sprint8_suite():
    parser = argparse.ArgumentParser(description="Run Sprint 8 Ensemble Scaling Campaign on Kaggle")
    parser.add_argument("--slug", type=str, default="sih26074-s8-ensemble-scaling")
    parser.add_argument("--condition", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--monitor", action="store_true", help="Monitor live execution until completion")
    parser.add_argument("--monitor-only", action="store_true", help="Monitor existing kernel without pushing new bundle")
    parser.add_argument("--mode", type=str, choices=["single", "dual"], default="single", help="Execution mode: single kernel or dual concurrent workers")
    args = parser.parse_args()

    if args.monitor_only:
        print("=" * 70)
        print(f"MONITORING RUNNING KERNEL: {args.slug}")
        print("=" * 70)
        success = monitor_kernel(args.slug, max_wait_minutes=90, poll_interval_sec=30)
        if success:
            bench_script = ROOT / "scripts" / "benchmark_sprint8_compute_frontier.py"
            if bench_script.exists():
                print("\n[*] Compiling Sprint 8 Benchmark Summary, Matched Tables, and Frontier...")
                subprocess.run([sys.executable, str(bench_script)], check=True)
                print("[+] Benchmark compilation complete.")
        return

    print("=" * 70)
    print("STARTING SPRINT 8 ENSEMBLE SCALING KAGGLE CAMPAIGN")
    print("Target Accelerator: Kaggle Dual Tesla T4 GPUs | Base Model: Scaled 15.69M Backbone")
    print(f"Mode: {args.mode.upper()} | Condition: {args.condition}")
    print("=" * 70)

    if args.mode == "single":
        slug = args.slug
        title = slug
        eval_cli = [
            "--condition", args.condition,
            "--batch_size", str(args.batch_size),
            "--chunk_size", str(args.chunk_size),
        ]
        if args.max_batches is not None:
            eval_cli.extend(["--max_batches", str(args.max_batches)])

        watch_url = dispatch_kernel(slug, title, eval_cli, staging_name=".kaggle_staging_sprint8")
        if not watch_url:
            print("[-] Dispatch failed.")
            sys.exit(1)

        print(f"[+] Sprint 8 Kernel dispatched successfully! Monitor link: {watch_url}")

        if args.monitor:
            success = monitor_kernel(slug, max_wait_minutes=90, poll_interval_sec=30)
            if success:
                # Compile benchmark reports
                bench_script = ROOT / "scripts" / "benchmark_sprint8_compute_frontier.py"
                if bench_script.exists():
                    print("\n[*] Compiling Sprint 8 Benchmark Summary, Matched Tables, and Frontier...")
                    subprocess.run([sys.executable, str(bench_script)], check=True)
                    print("[+] Benchmark compilation complete.")

    elif args.mode == "dual":
        # Dual-worker Process-Level Isolation (Strategy A)
        # Worker 1: Deterministic baselines + Small budgets (Budgets 8 and 16)
        # Worker 2: Broad ensemble scaling + Large budgets (Budgets 32 and 64)
        slug_w1 = f"{args.slug}-w1-det-small"
        slug_w2 = f"{args.slug}-w2-ens-large"

        cli_w1 = ["--condition", "budget8,budget16,det_s04,det_s08,det_s16,det_s32", "--batch_size", str(args.batch_size), "--chunk_size", str(args.chunk_size)]
        cli_w2 = ["--condition", "budget32,budget64", "--batch_size", str(args.batch_size), "--chunk_size", str(args.chunk_size)]

        if args.max_batches is not None:
            cli_w1.extend(["--max_batches", str(args.max_batches)])
            cli_w2.extend(["--max_batches", str(args.max_batches)])

        url1 = dispatch_kernel(slug_w1, slug_w1, cli_w1, staging_name=".kaggle_staging_s8_w1")
        url2 = dispatch_kernel(slug_w2, slug_w2, cli_w2, staging_name=".kaggle_staging_s8_w2")

        print("\n=======================================================")
        print(f"[+] DUAL WORKER 1 (Det & Small Budgets): {url1}")
        print(f"[+] DUAL WORKER 2 (Flagship & Large Budgets): {url2}")
        print("=======================================================\n")


if __name__ == "__main__":
    run_sprint8_suite()
