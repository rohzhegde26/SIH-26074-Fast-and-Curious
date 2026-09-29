"""
scripts/kaggle/dispatch_kaggle.py

Automated Kaggle GPU Dispatcher for AutoResearch.
Features:
1. Live GPU Quota Tracking before and after every dispatch.
2. Dynamic budget injection into program.md and data/cache/kaggle_quota.json.
3. Automated bundling and packaging with enable_gpu: true.
4. Remote polling and output metric retrieval.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def get_kaggle_username() -> str:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        if hasattr(api, "config_values") and "username" in api.config_values:
            return api.config_values["username"]
    except Exception:
        pass

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json, "r") as f:
            data = json.load(f)
        return data.get("username", "rohitajitbharadwaj")
    return "rohitajitbharadwaj"


def query_kaggle_gpu_quota() -> dict:
    """
    Queries Kaggle API for live GPU quota and parses seconds, hours, and refresh time.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        raw_quota = api.quota_view()
        if hasattr(raw_quota, "__dict__"):
            quota_data = json.loads(str(raw_quota)) if isinstance(str(raw_quota), str) else raw_quota
        elif isinstance(raw_quota, str):
            quota_data = json.loads(raw_quota)
        else:
            quota_data = raw_quota

        gpu_q = quota_data.get("gpuQuota", {})
        used_str = str(gpu_q.get("timeUsed", "0s"))
        total_str = str(gpu_q.get("totalTimeAllowed", "21600s"))

        def parse_seconds(s: str) -> float:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
            return float(m.group(1)) if m else 0.0

        used_sec = parse_seconds(used_str)
        total_sec = parse_seconds(total_str)
        rem_sec = max(0.0, total_sec - used_sec)

        status = {
            "total_seconds": total_sec,
            "used_seconds": used_sec,
            "remaining_seconds": rem_sec,
            "total_hours": round(total_sec / 3600.0, 2),
            "used_hours": round(used_sec / 3600.0, 2),
            "remaining_hours": round(rem_sec / 3600.0, 2),
            "remaining_minutes": round(rem_sec / 60.0, 1),
            "refresh_time": quota_data.get("quotaRefreshTime", "Unknown"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return status
    except Exception as e:
        print(f"[!] Warning: Could not query Kaggle quota: {e}")
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
        }


def update_agent_compute_context(quota: dict):
    """
    Saves quota status to data/cache/kaggle_quota.json and updates program.md
    so the autonomous agents are explicitly aware of their remaining GPU time.
    """
    if "error" in quota:
        # Never overwrite the recorded budget with made-up numbers.
        print("[!] Quota unknown; leaving program.md and quota cache untouched.")
        return

    # 1. Save JSON cache
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "kaggle_quota.json"
    with open(cache_file, "w") as f:
        json.dump(quota, f, indent=2)

    # 2. Update program.md with live compute constraints
    program_path = Path("program.md")
    if not program_path.exists():
        return

    content = program_path.read_text(encoding="utf-8")
    section_header = "## Active Compute Budget & Live Kaggle Accelerator Quota"
    
    budget_text = f"""{section_header}
> **LIVE STATUS (Updated {quota['timestamp']}):**
> - **Total GPU Allowance:** {quota['total_hours']} hours ({quota.get('total_seconds', 0):.0f} s)
> - **GPU Time Consumed:** {quota['used_hours']} hours ({quota.get('used_seconds', 0):.0f} s)
> - **GPU TIME REMAINING:** **{quota['remaining_hours']} hours ({quota['remaining_minutes']} minutes)**
> - **Quota Refreshes At:** {quota['refresh_time']}
>
> ⚠️ **HARD CONSTRAINT FOR AGENTS:**
> You have AT MOST **{quota['remaining_hours']} hours** of GPU time remaining for this cycle.
> Every training run depletes this quota. Budget your experiments carefully. If you run 5-minute runs,
> you have at most ~{int(quota['remaining_minutes'] // 5)} candidate iterations left before quota exhaustion!
"""

    if section_header in content:
        # Replace existing section up to next section or end
        pattern = re.compile(rf"{re.escape(section_header)}.*?(?=\n## |\Z)", re.DOTALL)
        updated_content = pattern.sub(budget_text.strip(), content)
    else:
        updated_content = budget_text + "\n" + content

    program_path.write_text(updated_content, encoding="utf-8")
    print(f"[+] Updated program.md with live budget: {quota['remaining_hours']} hrs remaining.")


def prepare_kernel_bundle(
    staging_dir: Path,
    kernel_slug: str,
    title: str,
    train_script: Path,
    dataset_sources: list = None,
) -> Path:
    """Prepares directory containing code and kernel-metadata.json."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    kernel_id = f"{username}/{kernel_slug}"

    shutil.copy(train_script, staging_dir / "train.py")

    src_dir = Path("src")
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
        "dataset_sources": dataset_sources or [],
    }

    with open(staging_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return staging_dir


def run_kaggle_job(
    staging_dir: Path,
    kernel_slug: str,
    poll_interval_sec: int = 20,
    max_wait_minutes: int = 30,
) -> bool:
    """Pushes kernel to Kaggle, monitors status, and downloads outputs."""
    # Pre-run quota check
    quota_pre = query_kaggle_gpu_quota()
    print("=" * 65)
    print(f"[*] PRE-DISPATCH KAGGLE GPU QUOTA:")
    if "error" in quota_pre:
        print("    Unknown (quota query failed); check kaggle.com/me/account")
    else:
        print(f"    Remaining: {quota_pre['remaining_hours']} hrs ({quota_pre['remaining_minutes']} mins)")
        print(f"    Used:      {quota_pre['used_hours']} hrs")
        print(f"    Refresh:   {quota_pre['refresh_time']}")
    print("=" * 65)
    update_agent_compute_context(quota_pre)

    if quota_pre.get("remaining_minutes", float("inf")) < 5.0:
        print("[-] ERROR: Less than 5 minutes of Kaggle GPU quota remaining! Aborting dispatch.")
        return False

    username = get_kaggle_username()
    kernel_id = f"{username}/{kernel_slug}"

    print(f"[*] Pushing kernel '{kernel_id}' to Kaggle GPU...")
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(staging_dir)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[-] Push failed: {res.stderr}")
        return False
    print(f"[+] Successfully pushed! {res.stdout.strip()}")

    print(f"[*] Monitoring execution of '{kernel_id}'...")
    start_time = time.time()
    max_sec = max_wait_minutes * 60

    while time.time() - start_time < max_sec:
        time.sleep(poll_interval_sec)
        status_res = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
        )
        status_line = status_res.stdout.strip()
        print(f"    [Status: {time.strftime('%H:%M:%S')}] {status_line}")

        if "complete" in status_line.lower():
            print("[+] Remote GPU run completed successfully!")
            break
        elif "error" in status_line.lower() or "failed" in status_line.lower():
            print(f"[-] Remote GPU run failed: {status_line}")
            break

    # Download outputs
    output_dir = Path("output") / "kaggle" / kernel_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading logs and artifacts to {output_dir}...")
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )
    print(f"[+] Download complete. Files saved in: {output_dir}")

    # Post-run quota check & update
    time.sleep(5)  # give Kaggle a brief moment to update accounting
    quota_post = query_kaggle_gpu_quota()
    print("=" * 65)
    print(f"[*] POST-DISPATCH KAGGLE GPU QUOTA:")
    if "error" in quota_post:
        print("    Unknown (quota query failed); check kaggle.com/me/account")
    else:
        print(f"    Remaining: {quota_post['remaining_hours']} hrs ({quota_post['remaining_minutes']} mins)")
        print(f"    Used:      {quota_post['used_hours']} hrs")
        print(f"    Refresh:   {quota_post['refresh_time']}")
    print("=" * 65)
    update_agent_compute_context(quota_post)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dispatch training run to Kaggle GPU with live quota tracking.")
    parser.add_argument("--script", default="src/models/train.py", help="Path to training script")
    parser.add_argument("--slug", default="sih26074-autoresearch-worker", help="Kaggle kernel slug")
    parser.add_argument("--title", default="SIH 26074 AutoResearch Worker", help="Kernel title")
    parser.add_argument("--dataset", action="append", default=[], help="Kaggle dataset ID (can repeat)")
    parser.add_argument("--check-quota-only", action="store_true", help="Only check quota and update context")
    args = parser.parse_args()

    if args.check_quota_only:
        q = query_kaggle_gpu_quota()
        print(json.dumps(q, indent=2))
        update_agent_compute_context(q)
        sys.exit(0)

    staging = Path(".kaggle_staging")
    prepare_kernel_bundle(
        staging_dir=staging,
        kernel_slug=args.slug,
        title=args.title,
        train_script=Path(args.script),
        dataset_sources=args.dataset,
    )
    success = run_kaggle_job(staging_dir=staging, kernel_slug=args.slug)
    sys.exit(0 if success else 1)
