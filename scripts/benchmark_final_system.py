"""
scripts/benchmark_final_system.py

Sprint 10: Multi-Profile Latency, Throughput, and Memory Profiling Engine.
Evaluates both production deployment profiles:
  1. ACCURATE (Dense-L, K=2 members, S=16 DDIM steps, eta=0.50, 32 NFE)
  2. ENSEMBLE (Dense-L, K=8 members, S=8 DDIM steps, eta=0.50, 64 NFE)

Measures:
  - Cold-start load time
  - VRAM allocation (peak GB)
  - Wall-clock inference latency per forecast cube (seconds)
  - Latency per lead-day (7 daily forecast horizons)
  - Throughput (cubes / second)
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.scalable_residual_diffusion import create_scalable_residual_diffusion


def benchmark_profile(
    profile_cfg_path: Path,
    device: torch.device,
    num_warmup: int = 1,
    num_runs: int = 3,
) -> Dict[str, Any]:
    """Runs rigorous latency and memory profiling for a specified deployment profile."""
    assert profile_cfg_path.exists(), f"Profile config not found: {profile_cfg_path}"
    with open(profile_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    profile_name = cfg["profile_name"]
    tier = cfg["model_tier"]
    k_members = int(cfg["num_members"])
    s_steps = int(cfg["denoising_steps"])
    eta = float(cfg["eta"])
    target_ms = float(cfg["target_latency_ms"])

    print(f"\n[*] Benchmarking Profile: {profile_name} (Tier: {tier.upper()}, K={k_members}, S={s_steps}, eta={eta})")

    # 1. Cold-start load timing
    t0_load = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model = create_scalable_residual_diffusion(tier).to(device)
    model.eval()
    load_time_sec = time.time() - t0_load

    prof = model.profile_compute(device=device, batch_size=1)
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        peak_vram_gb = prof.get("estimated_vram_gb", 2.85)

    print(f"    Load time: {load_time_sec:.3f} s | VRAM: {peak_vram_gb:.2f} GB | Params: {prof['total_parameters']:,}")

    # Generate synthetic inputs matching exact operational shapes
    b = 1
    h = torch.randn(b, 14, 6, 16, 16, device=device)
    f = torch.randn(b, 7, 6, 16, 16, device=device)
    terr = torch.randn(b, 5, 80, 80, device=device)

    # 2. Warm-up
    print(f"    Warming up ({num_warmup} iteration)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            for k in range(k_members):
                _ = model.sample(h, f, terr, num_steps=s_steps, eta=eta, seed=42 + k)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # 3. Timed benchmark passes
    print(f"    Timing across {num_runs} benchmark passes...")
    latencies = []
    with torch.no_grad():
        for run_idx in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_start = time.time()

            for k in range(k_members):
                _ = model.sample(h, f, terr, num_steps=s_steps, eta=eta, seed=100 + run_idx * 10 + k)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_cube = time.time() - t_start
            latencies.append(t_cube)

    mean_sec = float(np.mean(latencies))
    std_sec = float(np.std(latencies))
    lead_day_ms = (mean_sec / 7.0) * 1000.0
    mean_ms = mean_sec * 1000.0
    throughput_cubes_sec = 1.0 / mean_sec

    print(f"    Mean latency per cube: {mean_sec:.3f} s (+/- {std_sec:.3f} s)")
    print(f"    Latency per lead day:  {lead_day_ms:.1f} ms")
    print(f"    Throughput:            {throughput_cubes_sec:.2f} cubes/sec")

    return {
        "profile_name": profile_name,
        "model_tier": tier,
        "device": str(device),
        "total_parameters": prof["total_parameters"],
        "active_parameters": prof["active_parameters"],
        "num_members": k_members,
        "denoising_steps": s_steps,
        "total_nfe": k_members * s_steps,
        "cold_start_load_sec": round(load_time_sec, 3),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "mean_latency_sec": round(mean_sec, 3),
        "std_latency_sec": round(std_sec, 3),
        "mean_latency_ms": round(mean_ms, 1),
        "target_latency_ms": target_ms,
        "latency_per_lead_day_ms": round(lead_day_ms, 1),
        "throughput_cubes_per_sec": round(throughput_cubes_sec, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Sprint 10 Profile Benchmarking Suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-json", type=str, default="reports/sprint10_final_benchmarks.json")
    parser.add_argument("--output-md", type=str, default="reports/sprint10_profile_benchmarks.md")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(f"SPRINT 10: OPERATIONAL PROFILES BENCHMARK (Device: {device})")
    print("=" * 80)

    cfg_dir = ROOT / "configs" / "final"
    profiles_to_bench = [
        cfg_dir / "profile_accurate.yaml",
        cfg_dir / "profile_ensemble.yaml",
    ]

    all_results = {}
    for p_cfg in profiles_to_bench:
        if p_cfg.exists():
            res = benchmark_profile(p_cfg, device=device, num_warmup=1, num_runs=args.runs)
            all_results[res["profile_name"].lower()] = res

    # Save JSON report
    out_json = ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved benchmark JSON to {out_json}")

    # Generate Markdown Table
    out_md = ROOT / args.output_md
    lines = [
        "# Sprint 10: Operational Profiles Benchmarking Report",
        "",
        f"Evaluated on device: `{device}` with {args.runs} timed runs per profile.",
        "",
        "| Profile Name | Model Tier | Total Params | Members (K) | Steps (S) | Total NFE | Peak VRAM | Latency / Cube | Latency / Lead-Day | Throughput | Target Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, d in all_results.items():
        lines.append(
            f"| **{d['profile_name']}** | {d['model_tier'].upper()} | {d['total_parameters']:,} | "
            f"{d['num_members']} | {d['denoising_steps']} | {d['total_nfe']} | {d['peak_vram_gb']} GB | "
            f"**{d['mean_latency_sec']} s** | {d['latency_per_lead_day_ms']} ms | "
            f"{d['throughput_cubes_per_sec']} cubes/s | {d['target_latency_ms']} ms |"
        )
    lines.extend([
        "",
        "## Deployment Profile Profiles",
        "- **ACCURATE**: Tailored for State Disaster Emergency Operations and localized cloudburst alerts.",
        "- **ENSEMBLE**: Tailored for Central IMD / NCMRWF High-Performance Computing clusters and flood warnings.",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved benchmark report to {out_md}")


if __name__ == "__main__":
    main()
