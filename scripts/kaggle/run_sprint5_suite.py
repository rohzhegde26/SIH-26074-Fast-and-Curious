"""
scripts/kaggle/run_sprint5_suite.py

Orchestrates the complete Sprint 5 Spatial-Context (N/M) Sweep on Kaggle Dual Tesla T4 GPUs:
  - EXP-N16: N=16 coarse cells (N/M = 1.00, Baseline Control)
  - EXP-N20: N=20 coarse cells (N/M = 1.25, Coastal Arabian Sea margin)
  - EXP-N24: N=24 coarse cells (N/M = 1.50, Mesoscale context)
  - EXP-N32: N=32 coarse cells (N/M = 2.00, Cross-peninsular synoptic wave)

Safety Invariants:
  - Halts immediately if remaining GPU quota drops below 0.50 hours (30 minutes).
  - Frozen Temporal Invariant: H* = 14 antecedent days across all conditions.
  - Frozen Capacity Invariant: Exactly 15,685,478 parameters across all N.
  - Fixed Sampler Invariant: DDIM-32 (eta=0.0).
"""

from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import query_kaggle_gpu_quota
from scripts.kaggle.dispatch_temporal_training import dispatch_temporal_job

SPATIAL_EXPERIMENTS = [
    {"n": 16, "slug": "sih26074-s5-diff-n16", "title": "sih26074-s5-diff-n16"},
    {"n": 20, "slug": "sih26074-s5-diff-n20", "title": "sih26074-s5-diff-n20"},
    {"n": 24, "slug": "sih26074-s5-diff-n24", "title": "sih26074-s5-diff-n24"},
    {"n": 32, "slug": "sih26074-s5-diff-n32", "title": "sih26074-s5-diff-n32"},
]

SAFE_QUOTA_FLOOR_HOURS = 0.50  # 30 minutes safety buffer


def run_suite(epochs: int = 30, ddim_steps: int = 32, patience: int = 7, eval_interval: int = 5):
    print("=" * 70)
    print("STARTING SPRINT 5 SPATIAL-CONTEXT (N/M) EXPERIMENTAL SUITE")
    print(f"Total Configurations: {len(SPATIAL_EXPERIMENTS)} | Model: Scaled 15.69M Backbone")
    print(f"Antecedent Window: Frozen H* = 14 | Sampler: Fixed DDIM-{ddim_steps}")
    print(f"Epochs per Run: {epochs} | Safety Floor: {SAFE_QUOTA_FLOOR_HOURS * 60:.0f} mins")
    print("=" * 70)

    for exp in SPATIAL_EXPERIMENTS:
        n = exp["n"]
        slug = exp["slug"]
        title = exp["title"]

        rpt_path = ROOT / "reports" / f"training_spatial_n{n:02d}_history.json"
        if rpt_path.exists():
            print(f"\n[*] Experiment N={n} report already exists ({rpt_path.name}). Skipping.")
            continue

        # Check quota safety floor
        quota = query_kaggle_gpu_quota()
        rem_hrs = quota.get("remaining_hours", 0.0)
        print(f"\n[*] Pre-run quota check for N={n}: {rem_hrs:.2f} hrs remaining.")

        if rem_hrs < SAFE_QUOTA_FLOOR_HOURS:
            print(f"[!] Remaining quota ({rem_hrs:.2f} hrs) is below safety floor ({SAFE_QUOTA_FLOOR_HOURS} hrs). Aborting sweep.")
            sys.exit(1)

        train_cli = [
            "--mode", "diffusion",
            "--history_len", "14",
            "--context_size", str(n),
            "--spatial_mode",
            "--model_size", "ultra",
            "--epochs", str(epochs),
            "--batch_size", "8",
            "--lr", "3e-4",
            "--ddim_steps", str(ddim_steps),
            "--eval_sampling_interval", str(eval_interval),
            "--early_stopping_patience", str(patience),
        ]

        print(f"\n>>> Dispatching {slug} (N={n}, N/M={n/16:.2f}, epochs={epochs}, ddim={ddim_steps})...")
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
    print("[+] SPRINT 5 EXPERIMENTAL SUITE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Automatically generate benchmark report
    print("[*] Compiling Sprint 5 Spatial Benchmark Evaluation...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "benchmark_spatial_experiments.py")], check=True)
    print("[+] Benchmark report generated successfully.")


if __name__ == "__main__":
    run_suite()
