"""
scripts/kaggle/run_sprint4_suite.py

Orchestrates the complete Sprint 4 History-Length Sweep on Kaggle Dual Tesla T4 GPUs:
  - EXP-H03-REF: H=3 days (Reference, 30 epochs, DDIM-32)
  - EXP-H05:     H=5 days (30 epochs, DDIM-32)
  - EXP-H07:     H=7 days (30 epochs, DDIM-32)
  - EXP-H10:     H=10 days (30 epochs, DDIM-32)
  - EXP-H14:     H=14 days (30 epochs, DDIM-32)

Safety Invariant: Stops immediately if remaining GPU quota drops below 1.5 hours.
"""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import query_kaggle_gpu_quota
from scripts.kaggle.dispatch_temporal_training import dispatch_temporal_job

EXPERIMENTS = [
    {"h": 3, "slug": "sih26074-s4-diff-h03-ref", "title": "sih26074-s4-diff-h03-ref"},
    {"h": 5, "slug": "sih26074-s4-diff-h05",     "title": "sih26074-s4-diff-h05"},
    {"h": 7, "slug": "sih26074-s4-diff-h07",     "title": "sih26074-s4-diff-h07"},
    {"h": 10, "slug": "sih26074-s4-diff-h10",    "title": "sih26074-s4-diff-h10"},
    {"h": 14, "slug": "sih26074-s4-diff-h14",    "title": "sih26074-s4-diff-h14"},
]

SAFE_QUOTA_FLOOR_HOURS = 1.5


def run_suite(epochs: int = 30, ddim_steps: int = 32, patience: int = 7, eval_interval: int = 5):
    print("=" * 70)
    print("STARTING SPRINT 4 HISTORY-LENGTH EXPERIMENTAL SUITE")
    print(f"Total Experiments: {len(EXPERIMENTS)} | Model: Scaled ~16.05M Backbone")
    print(f"Sampler Invariant: Fixed DDIM-{ddim_steps} | Epochs per Run: {epochs}")
    print(f"Validation Sampling Interval: Every {eval_interval} epochs")
    print("=" * 70)

    for exp in EXPERIMENTS:
        h = exp["h"]
        slug = exp["slug"]
        title = exp["title"]

        rpt_path = ROOT / "reports" / f"training_diffusion_h{h:02d}_history.json"
        if rpt_path.exists():
            print(f"\n[*] Experiment H={h} report already exists ({rpt_path.name}). Skipping.")
            continue

        # Check quota safety floor
        quota = query_kaggle_gpu_quota()
        rem_hrs = quota.get("remaining_hours", 0.0)
        print(f"\n[*] Pre-run quota check for H={h}: {rem_hrs:.2f} hrs remaining.")

        if rem_hrs < SAFE_QUOTA_FLOOR_HOURS:
            print(f"[!] Remaining quota ({rem_hrs:.2f} hrs) is below safety floor ({SAFE_QUOTA_FLOOR_HOURS} hrs). Aborting sweep.")
            sys.exit(1)

        train_cli = [
            "--mode", "diffusion",
            "--history_len", str(h),
            "--model_size", "ultra",
            "--epochs", str(epochs),
            "--batch_size", "8",
            "--lr", "3e-4",
            "--ddim_steps", str(ddim_steps),
            "--eval_sampling_interval", str(eval_interval),
            "--early_stopping_patience", str(patience),
        ]

        print(f"\n>>> Dispatching {slug} (H={h}, epochs={epochs}, ddim={ddim_steps})...")
        print(f"    Live Kaggle URL: https://www.kaggle.com/code/rohitajitbharadwaj/{slug}")
        success = dispatch_temporal_job(
            slug=slug,
            title=title,
            train_args=train_cli,
            max_wait_minutes=60,
            poll_interval_sec=30,
        )

        if not success:
            print(f"[-] Experiment {slug} failed. Halting subsequent runs.")
            sys.exit(1)

        print(f"[+] Completed {slug} successfully.")
        time.sleep(10)

    print("\n" + "=" * 70)
    print("[+] SPRINT 4 EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Automatically generate benchmark report
    print("[*] Compiling Sprint 4 Benchmark Evaluation...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_history_experiments.py")], check=True)
    print("[+] Benchmark report generated successfully.")


if __name__ == "__main__":
    run_suite()
