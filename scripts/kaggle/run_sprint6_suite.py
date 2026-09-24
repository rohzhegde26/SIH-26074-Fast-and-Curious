"""
scripts/kaggle/run_sprint6_suite.py

Orchestrates the Sprint 6 Residual Diffusion & Meteorological Refinement Experiments on Kaggle Dual Tesla T4 GPUs:
  - EXP-01: Frozen Control (N=24, H=14, eps-prediction, uniform loss - from Sprint 5)
  - EXP-02: Velocity Prediction (N=24, H=14, v-prediction, uniform loss, CMVS checkpointing)
  - EXP-03: Multi-Task Variable-Aware Noise Weighting (N=24, H=14, v-prediction, group_tail loss, CMVS checkpointing)

Safety Invariants:
  - Quota safety floor: minimum 0.50 hours (30 minutes) remaining.
  - Frozen Temporal Invariant: H* = 14 antecedent days.
  - Frozen Spatial Invariant: N* = 24 coarse cells (N/M = 1.50).
  - Frozen Capacity Invariant: Exactly 15,685,478 parameters across all candidates.
  - Fixed Sampler Invariant: DDIM-32 (eta=0.0).
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kaggle.dispatch_kaggle import query_kaggle_gpu_quota, get_kaggle_username
from scripts.kaggle.dispatch_temporal_training import dispatch_temporal_job

SPRINT6_EXPERIMENTS = [
    {
        "id": "exp02_vpred",
        "slug": "sih26074-s6-diff-vpred",
        "title": "sih26074-s6-diff-vpred",
        "exp_name": "sprint6_candidate2_vpred",
        "prediction_type": "v_prediction",
        "loss_weighting": "uniform",
        "checkpoint_criterion": "cmvs",
        "desc": "Candidate 2: Velocity Prediction (v-prediction) with CMVS Checkpoint Selection",
    },
    {
        "id": "exp03_multitask",
        "slug": "sih26074-s6-diff-multitask",
        "title": "sih26074-s6-diff-multitask",
        "exp_name": "sprint6_candidate3_multitask",
        "prediction_type": "v_prediction",
        "loss_weighting": "group_tail",
        "checkpoint_criterion": "cmvs",
        "desc": "Candidate 3: Multi-Task Variable-Aware Noise Weighting + Convective Tail Calibration",
    },
]

SAFE_QUOTA_FLOOR_HOURS = 0.50  # 30 minutes safety buffer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Sprint 6 Residual Diffusion Suite on Kaggle")
    parser.add_argument(
        "--candidate",
        type=str,
        default="all",
        choices=["vpred", "multitask", "all"],
        help="Which candidate experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs per experiment")
    parser.add_argument("--ddim_steps", type=int, default=32, help="Sampling steps for validation/test evaluation")
    parser.add_argument("--eval_interval", type=int, default=5, help="Epoch interval to evaluate DDIM sampling")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    return parser.parse_args()


def run_suite():
    args = parse_args()
    print("=" * 70)
    print("STARTING SPRINT 6 RESIDUAL DIFFUSION EXPERIMENTAL CAMPAIGN")
    print(f"Target Accelerator: Kaggle Dual Tesla T4 GPUs | Base Model: Scaled 15.69M Backbone")
    print(f"Antecedent Window: Frozen H* = 14 | Spatial Context: Frozen N* = 24 (N/M = 1.50)")
    print(f"Epochs per Run: {args.epochs} | Safety Floor: {SAFE_QUOTA_FLOOR_HOURS * 60:.0f} mins")
    print("=" * 70)

    selected = []
    for exp in SPRINT6_EXPERIMENTS:
        if args.candidate == "all":
            selected.append(exp)
        elif args.candidate == "vpred" and exp["id"] == "exp02_vpred":
            selected.append(exp)
        elif args.candidate == "multitask" and exp["id"] == "exp03_multitask":
            selected.append(exp)

    for exp in selected:
        slug = exp["slug"]
        title = exp["title"]
        exp_name = exp["exp_name"]
        pred_type = exp["prediction_type"]
        loss_w = exp["loss_weighting"]
        crit = exp["checkpoint_criterion"]

        rpt_path = ROOT / "reports" / f"{exp_name}_history.json"
        if rpt_path.exists():
            print(f"\n[*] Experiment {exp_name} report already exists ({rpt_path.name}). Skipping.")
            continue

        # Check quota safety floor
        quota = query_kaggle_gpu_quota()
        rem_hrs = quota.get("remaining_hours", 0.0)
        print(f"\n[*] Pre-run quota check for {slug}: {rem_hrs:.2f} hrs remaining.")

        if rem_hrs < SAFE_QUOTA_FLOOR_HOURS:
            print(f"[!] Remaining quota ({rem_hrs:.2f} hrs) is below safety floor ({SAFE_QUOTA_FLOOR_HOURS} hrs). Aborting sweep.")
            sys.exit(1)

        username = get_kaggle_username()
        watch_url = f"https://www.kaggle.com/code/{username}/{slug}"
        print(f"\n=======================================================")
        print(f"[+] DISPATCHING {slug}")
        print(f"    Description: {exp['desc']}")
        print(f"    Live Watch Link: {watch_url}")
        print(f"=======================================================\n")

        train_cli = [
            "--mode", "diffusion",
            "--history_len", "14",
            "--context_size", "24",
            "--spatial_mode",
            "--model_size", "ultra",
            "--epochs", str(args.epochs),
            "--batch_size", "8",
            "--lr", "3e-4",
            "--ddim_steps", str(args.ddim_steps),
            "--eval_sampling_interval", str(args.eval_interval),
            "--early_stopping_patience", str(args.patience),
            "--prediction_type", pred_type,
            "--loss_weighting", loss_w,
            "--checkpoint_criterion", crit,
            "--exp_name", exp_name,
        ]

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
    print("[+] SPRINT 6 EXPERIMENTAL CAMPAIGN COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Benchmark synthesis if script exists
    bench_script = ROOT / "scripts" / "benchmark_sprint6_experiments.py"
    if bench_script.exists():
        print("[*] Compiling Sprint 6 Benchmark Summary and Comparison Tables...")
        subprocess.run([sys.executable, str(bench_script)], check=True)
        print("[+] Benchmark artifacts compiled successfully.")


if __name__ == "__main__":
    run_suite()
