"""
scripts/kaggle/monitor_dual_runs.py

Monitors two queued Kaggle runs:
  - rohitajitbharadwaj/sih26074-s85-calibration
  - rohitajitbharadwaj/sih26074-s85-calibration-run

When either starts running or completes:
  1. Once complete, downloads output artifacts and logs.
  2. Extracts sprint8_5_validation_summary.json to reports/.
  3. Runs scripts/analyze_sprint8_5_uncertainty.py.
  4. Runs scripts/bootstrap_sprint8_5.py.
  5. Verifies zero em dashes in generated reports.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kaggle
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

SLUGS = [
    "sih26074-s85-calibration",
    "sih26074-s85-calibration-run",
]


def check_status(api, owner, slug):
    try:
        res = api.kernels_status(f"{owner}/{slug}")
        status = res.get("status", "") if isinstance(res, dict) else str(getattr(res, "_status", ""))
        return status.upper()
    except Exception as e:
        return f"ERROR: {e}"


def download_output_and_logs(api, owner, winner_slug):
    print(f"[*] Downloading results for winner: {owner}/{winner_slug}...")
    output_dir = ROOT / "output" / "kaggle" / winner_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try kaggle CLI output download
    subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "output", f"{owner}/{winner_slug}", "-p", str(output_dir), "--force"],
        capture_output=True,
        text=True,
    )

    # Copy any generated json/md
    found_summary = False
    for rpt in list(output_dir.rglob("*.json")) + list(output_dir.rglob("*.md")):
        dst = ROOT / "reports" / rpt.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(rpt, dst)
        print(f"[+] Retrieved: {dst.relative_to(ROOT)}")
        if rpt.name == "sprint8_5_validation_summary.json":
            found_summary = True

    # 2. If not found in files, extract from stdout logs
    if not found_summary:
        print("[*] Checking stdout log for BEGIN_JSON_SUMMARY_EXPORT...")
        try:
            log_text = api.kernels_logs(f"{owner}/{winner_slug}")
            if "BEGIN_JSON_SUMMARY_EXPORT" in log_text:
                start = log_text.index("BEGIN_JSON_SUMMARY_EXPORT") + len("BEGIN_JSON_SUMMARY_EXPORT")
                end = log_text.index("END_JSON_SUMMARY_EXPORT")
                json_str = log_text[start:end].strip()
                summary_data = json.loads(json_str)
                dst = ROOT / "reports" / "sprint8_5_validation_summary.json"
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)
                print(f"[+] Successfully extracted and saved: {dst.relative_to(ROOT)}")
                found_summary = True
        except Exception as e:
            print(f"[-] Log extraction error: {e}")

    return found_summary


def post_process():
    print("[*] Running post-processing analysis scripts...")
    # 1. Uncertainty diagnostics
    res1 = subprocess.run([sys.executable, "scripts/analyze_sprint8_5_uncertainty.py"], capture_output=True, text=True, cwd=str(ROOT))
    print(res1.stdout)
    if res1.stderr:
        print(res1.stderr)

    # 2. Bootstrap confidence intervals
    res2 = subprocess.run([sys.executable, "scripts/bootstrap_sprint8_5.py"], capture_output=True, text=True, cwd=str(ROOT))
    print(res2.stdout)
    if res2.stderr:
        print(res2.stderr)

    # 3. Check zero em dashes
    reports_dir = ROOT / "reports"
    em_dash_found = False
    for p in reports_dir.glob("sprint8_5_*.md"):
        txt = p.read_text(encoding="utf-8")
        if "\u2014" in txt or "\u2013" in txt:
            print(f"[!] Warning: Em dash found in {p.name}! Sanitizing...")
            clean = txt.replace("\u2014", " - ").replace("\u2013", " - ")
            p.write_text(clean, encoding="utf-8")
            em_dash_found = True
    if not em_dash_found:
        print("[+] Verified: Strictly ZERO em dashes in all Sprint 8.5 reports.")

    print("[+] All post-processing complete!")


def main():
    api = kaggle.KaggleApi()
    api.authenticate()
    owner = "rohitajitbharadwaj"

    print("=" * 70)
    print("[*] SPRINT 8.5 DUAL-RUN MONITOR STARTED")
    print(f"    Kernel 1: {owner}/{SLUGS[0]}")
    print(f"    Kernel 2: {owner}/{SLUGS[1]}")
    print("=" * 70, flush=True)

    start_time = time.time()
    cancelled_other = False

    while True:
        s0 = check_status(api, owner, SLUGS[0])
        s1 = check_status(api, owner, SLUGS[1])
        elapsed = (time.time() - start_time) / 60.0

        print(f"[{time.strftime('%H:%M:%S')} | {elapsed:.1f}m] {SLUGS[0]}: {s0} | {SLUGS[1]}: {s1}", flush=True)

        # Check for completion
        if "COMPLETE" in s0:
            print(f"[+] {SLUGS[0]} is COMPLETE!")
            if download_output_and_logs(api, owner, SLUGS[0]):
                post_process()
                break
        elif "COMPLETE" in s1:
            print(f"[+] {SLUGS[1]} is COMPLETE!")
            if download_output_and_logs(api, owner, SLUGS[1]):
                post_process()
                break

        # Cancel the second if one starts running to save quota
        if not cancelled_other:
            if "RUNNING" in s0 and "QUEUED" in s1:
                print(f"[*] {SLUGS[0]} is RUNNING! Cancelling queued {SLUGS[1]} to save quota...")
                try:
                    subprocess.run(["kaggle", "kernels", "cancel", f"{owner}/{SLUGS[1]}"], capture_output=True)
                    cancelled_other = True
                except Exception:
                    pass
            elif "RUNNING" in s1 and "QUEUED" in s0:
                print(f"[*] {SLUGS[1]} is RUNNING! Cancelling queued {SLUGS[0]} to save quota...")
                try:
                    subprocess.run(["kaggle", "kernels", "cancel", f"{owner}/{SLUGS[0]}"], capture_output=True)
                    cancelled_other = True
                except Exception:
                    pass

        # Error handling
        if ("ERROR" in s0 or "FAILED" in s0) and ("ERROR" in s1 or "FAILED" in s1):
            print("[-] Both kernels failed! Exiting monitor.")
            break

        time.sleep(25)


if __name__ == "__main__":
    main()
