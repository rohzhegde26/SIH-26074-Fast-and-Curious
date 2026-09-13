"""
scripts/kaggle/dispatch_multitask_training.py

Dispatches MultiTaskUNet5x training to Kaggle Dual T4 GPUs.
Features:
    1. Checks and tracks live Kaggle GPU quota before dispatch.
    2. Packages source code, models, losses, and training script into .kaggle_staging.
    3. Configures kernel metadata with enable_gpu=True.
    4. Pushes kernel to Kaggle and monitors live execution.
    5. Downloads multitask_5x_champion.pt and training metrics to models/checkpoints/.
"""

import argparse
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

from scripts.kaggle.dispatch_kaggle import (
    get_kaggle_username,
    query_kaggle_gpu_quota,
    update_agent_compute_context,
)


import base64
import io
import zipfile

def bundle_src_base64() -> str:
    """Zips src/ tree into a compact in-memory base64 string."""
    buf = io.BytesIO()
    src_path = ROOT / "src"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_path.rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            zf.write(p, rel)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def prepare_multitask_kernel(staging_dir: Path, slug: str, title: str) -> Path:
    """Packages source code and training entrypoint for Kaggle GPU execution."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    src_b64 = bundle_src_base64()
    unpacker_header = (
        'import base64\n'
        'import io\n'
        'import os\n'
        'import sys\n'
        'import zipfile\n'
        'from pathlib import Path\n\n'
        '_cur = Path(os.getcwd())\n'
        'if not (_cur / "src").exists():\n'
        '    print("[*] Unpacking embedded source bundle...", flush=True)\n'
        f'    _data = base64.b64decode("""{src_b64}""")\n'
        '    with zipfile.ZipFile(io.BytesIO(_data)) as _zf:\n'
        '        _zf.extractall(_cur)\n'
        '    print("[+] Unpacked source bundle successfully.", flush=True)\n\n'
        'if str(_cur) not in sys.path:\n'
        '    sys.path.insert(0, str(_cur))\n\n'
    )

    train_src_code = (ROOT / "scripts" / "train_multitask_unet.py").read_text(encoding="utf-8")
    combined_train_code = unpacker_header + train_src_code

    with open(staging_dir / "train.py", "w", encoding="utf-8") as f:
        f.write(combined_train_code)

    # Copy src package
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
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
    }

    with open(staging_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Prepared self-contained Kaggle kernel bundle in: {staging_dir}", flush=True)
    return staging_dir


def dispatch_and_monitor(
    staging_dir: Path,
    slug: str,
    max_wait_minutes: int = 45,
    poll_interval_sec: int = 20,
) -> bool:
    """Pushes kernel to Kaggle and monitors execution until completion."""
    quota = query_kaggle_gpu_quota()
    print("=" * 65)
    print(f"[*] KAGGLE GPU PRE-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {quota['remaining_hours']} hrs ({quota['remaining_minutes']} mins)")
    print("=" * 65)
    update_agent_compute_context(quota)

    if quota["remaining_minutes"] < 15.0:
        print("[-] Error: Insufficient Kaggle GPU quota (< 15 mins). Aborting.")
        return False

    username = get_kaggle_username()
    kernel_id = f"{username}/{slug}"

    print(f"[*] Pushing kernel '{kernel_id}' to Kaggle GPU (NvidiaTeslaT4)...", flush=True)
    res = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(staging_dir), "--accelerator", "NvidiaTeslaT4"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"[-] Push failed: {res.stderr}")
        return False
    print(f"[+] Successfully pushed! {res.stdout.strip()}")

    print(f"[*] Monitoring live remote execution on Kaggle GPU...")
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
        print(f"    [{time.strftime('%H:%M:%S')}] {status_line}")

        if "complete" in status_line.lower():
            print("[+] Remote GPU training completed successfully!")
            success = True
            break
        elif "error" in status_line.lower() or "failed" in status_line.lower():
            print(f"[-] Remote GPU training failed: {status_line}")
            break

    # Download output artifacts
    output_dir = ROOT / "output" / "kaggle" / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading trained model and logs from Kaggle to {output_dir}...")
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(output_dir)],
        capture_output=True,
        text=True,
    )

    # If multitask_5x_champion.pt was saved, copy to models/checkpoints/
    found_ckpt = None
    for candidate in [output_dir / "multitask_5x_champion.pt"] + list(output_dir.glob("**/multitask_5x_champion.pt")):
        if candidate.exists():
            found_ckpt = candidate
            break

    local_ckpt = ROOT / "models" / "checkpoints" / "multitask_5x_champion.pt"
    if found_ckpt is not None:
        local_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(found_ckpt, local_ckpt)
        print(f"[+] Successfully deployed remote checkpoint from {found_ckpt} to: {local_ckpt}")
    else:
        print(f"[!] Warning: multitask_5x_champion.pt not found in {output_dir}")

    # Post-dispatch quota update
    quota_post = query_kaggle_gpu_quota()
    print("=" * 65)
    print(f"[*] KAGGLE GPU POST-DISPATCH QUOTA:")
    print(f"    Remaining GPU Time: {quota_post['remaining_hours']} hrs ({quota_post['remaining_minutes']} mins)")
    print("=" * 65)
    update_agent_compute_context(quota_post)

    return success


def main():
    parser = argparse.ArgumentParser(description="Dispatch MultiTaskUNet5x training to Kaggle GPU.")
    parser.add_argument("--slug", default="sih-26074-multitask-unet5x-weather-downscaler", help="Kernel slug")
    parser.add_argument("--title", default="SIH 26074 MultiTask UNet5x Weather Downscaler", help="Kernel title")
    parser.add_argument("--max-wait", type=int, default=45, help="Max wait in minutes")
    args = parser.parse_args()

    staging_dir = ROOT / ".kaggle_multitask_staging"
    prepare_multitask_kernel(staging_dir, args.slug, args.title)
    success = dispatch_and_monitor(staging_dir, args.slug, max_wait_minutes=args.max_wait)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
