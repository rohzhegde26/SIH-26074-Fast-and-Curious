"""
scripts/kaggle/run_sprint8_5_suite.py

Orchestrates the Sprint 8.5 Calibration and Diagnostic Campaign on Kaggle GPU Accelerators:
  1. Checks GPU quota and verifies safety floor (>= 0.50 hrs).
  2. Bundles src/ (including src/models/calibration.py) and unpacker script.
  3. Pushes kernel 'rohitajitbharadwaj/sih26074-s8-5-calibration' to Kaggle Dual T4 GPU.
  4. Returns live watch URL immediately.
  5. Monitors execution, downloads artifacts into output/kaggle/sih26074-s8-5-calibration,
     and syncs validation reports to reports/.
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

SAFE_QUOTA_FLOOR_HOURS = 0.50


def bundle_src_base64() -> str:
    buf = io.BytesIO()
    src_path = ROOT / "src"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_path.rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_sprint8_5_kernel(
    staging_dir: Path,
    slug: str,
    title: str,
    eval_args: list,
) -> Path:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"
    src_b64 = bundle_src_base64()
    eval_args_repr = repr(list(eval_args))

    unpacker_header = f"""# Auto-generated Kaggle unpacker header for Sprint 8.5 Calibration
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
for _pkg in ["zarr", "scikit-learn"]:
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

    eval_src_code = (ROOT / "scripts" / "evaluate_sprint8_5_calibration.py").read_text(encoding="utf-8")
    combined_code = unpacker_header + "\n" + eval_src_code

    with open(staging_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(combined_code)

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

    print(f"[+] Prepared self-contained Sprint 8.5 kernel bundle in: {staging_dir}", flush=True)
    return staging_dir


def dispatch_kernel(
    slug: str,
    title: str,
    eval_args: list,
    staging_name: str = ".kaggle_staging_sprint8_5",
) -> Optional[str]:
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
    prepare_sprint8_5_kernel(staging_dir, slug, title, eval_args)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    print(f"[*] Pushing kernel '{kernel_id}' to Kaggle GPU...", flush=True)
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

    # Copy summary JSON to reports/
    found_summary = False
    for rpt in list(output_dir.rglob("*.json")) + list(output_dir.rglob("*.md")):
        dst = ROOT / "reports" / rpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpt, dst)
        print(f"[+] Retrieved report: {dst.relative_to(ROOT)}")
        if rpt.name == "sprint8_5_validation_summary.json":
            found_summary = True

    # Fallback: check kernel log for BEGIN_JSON_SUMMARY_EXPORT
    if not found_summary:
        print("[*] Checking kernel logs for embedded JSON summary...")
        try:
            import kaggle
            api = kaggle.KaggleApi()
            api.authenticate()
            log_text = api.kernels_logs(kernel_id)
            if "BEGIN_JSON_SUMMARY_EXPORT" in log_text:
                start = log_text.index("BEGIN_JSON_SUMMARY_EXPORT") + len("BEGIN_JSON_SUMMARY_EXPORT")
                end = log_text.index("END_JSON_SUMMARY_EXPORT")
                json_str = log_text[start:end].strip()
                summary_data = json.loads(json_str)
                target_json = ROOT / "reports" / "sprint8_5_validation_summary.json"
                with open(target_json, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)
                print(f"[+] Successfully extracted and saved: {target_json.relative_to(ROOT)}")
                found_summary = True
        except Exception as e:
            print(f"[-] Log fallback extraction error: {e}")

    # Run report generators
    print("[*] Compiling all Sprint 8.5 diagnostic and calibration markdown reports...")
    try:
        from scripts.analyze_sprint8_5_uncertainty import generate_sprint8_5_reports
        from scripts.bootstrap_sprint8_5 import run_bootstrap_sprint8_5
        sum_p = ROOT / "reports" / "sprint8_5_validation_summary.json"
        generate_sprint8_5_reports(sum_p)
        run_bootstrap_sprint8_5(sum_p)
        print("[+] All markdown reports and bootstrap intervals compiled successfully.")
    except Exception as e:
        print(f"[-] Report generation error: {e}")

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


def main():
    parser = argparse.ArgumentParser(description="Run Sprint 8.5 Calibration Campaign on Kaggle")
    parser.add_argument("--slug", type=str, default="sih26074-s85-calibration")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run-holdout", action="store_true", default=True, help="Run 2023 holdout test set")
    parser.add_argument("--monitor", action="store_true", help="Monitor live execution until completion")
    parser.add_argument("--monitor-only", action="store_true", help="Monitor existing kernel without pushing new bundle")
    args = parser.parse_args()

    slug = args.slug
    title = slug
    eval_args = [
        "--split", "val",
        "--batch-size", str(args.batch_size),
        "--calib-split", "61",
    ]
    if args.run_holdout:
        eval_args.append("--run-holdout")

    if args.monitor_only:
        monitor_kernel(slug)
        return

    watch_url = dispatch_kernel(slug, title, eval_args)
    if not watch_url:
        print("[-] Failed to dispatch Kaggle kernel.")
        sys.exit(1)

    if args.monitor:
        success = monitor_kernel(slug)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
