"""
scripts/kaggle/monitor_kernel.py

Polls a running Kaggle kernel until completion, downloads outputs,
and copies checkpoints/reports into canonical repository paths.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import get_kaggle_username, query_kaggle_gpu_quota, update_agent_compute_context


def monitor_kernel(slug: str, max_wait_minutes: int = 40, poll_interval_sec: int = 15):
    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"
    print("=" * 65, flush=True)
    print(f"[*] Monitoring live execution of Kaggle kernel: {kernel_id}", flush=True)
    print("=" * 65, flush=True)

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
            print("[+] Remote Kaggle GPU run completed successfully!", flush=True)
            success = True
            break
        elif "error" in status_line.lower() or "failed" in status_line.lower():
            print(f"[-] Remote Kaggle GPU run failed: {status_line}", flush=True)
            break

    # Download output artifacts
    output_dir = ROOT / "output" / "kaggle" / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading logs and artifacts to {output_dir}...", flush=True)
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )

    # Print log file if available
    for log_f in output_dir.glob("*.log"):
        print(f"\n--- Output Log: {log_f.name} ---", flush=True)
        try:
            print(log_f.read_text(encoding="utf-8")[-2000:], flush=True)
        except Exception:
            pass

    # Copy checkpoints and reports
    for ckpt in output_dir.rglob("*.pt"):
        dst = ROOT / "models" / "checkpoints" / ckpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt, dst)
        print(f"[+] Retrieved checkpoint: {dst.relative_to(ROOT)}", flush=True)

    for rpt in output_dir.rglob("*.json"):
        dst = ROOT / "reports" / rpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rpt, dst)
        print(f"[+] Retrieved report: {dst.relative_to(ROOT)}", flush=True)

    # Query and display post-dispatch quota
    time.sleep(3)
    post_quota = query_kaggle_gpu_quota()
    print("=" * 65, flush=True)
    print("[*] KAGGLE GPU POST-DISPATCH QUOTA:", flush=True)
    print(f"    Remaining GPU Time: {post_quota['remaining_hours']} hrs ({post_quota['remaining_minutes']} mins)", flush=True)
    print(f"    Used GPU Time:      {post_quota['used_hours']} hrs", flush=True)
    print("=" * 65, flush=True)
    update_agent_compute_context(post_quota)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", type=str, required=True, help="Kernel slug to monitor")
    parser.add_argument("--max_wait_minutes", type=int, default=40)
    args = parser.parse_args()
    monitor_kernel(slug=args.slug, max_wait_minutes=args.max_wait_minutes)
