"""
scripts/benchmark_sprint9_scaling.py

Benchmark Suite for Sprint 9 Model Capacity Scaling:
Measures inference latency, peak VRAM, step throughput, and active parameters.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.scalable_residual_diffusion import (
    TIER_CHANNEL_CONFIGS,
    create_scalable_residual_diffusion,
)


def benchmark_tier(
    tier: str,
    device: torch.device,
    batch_size: int = 1,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> Dict[str, Any]:
    print(f"[*] Benchmarking tier: {tier.upper()} on {device}...")
    if device.type == "cpu":
        model = create_scalable_residual_diffusion(tier=tier, base_channels=16).to(device)
    else:
        model = create_scalable_residual_diffusion(tier=tier).to(device)

    model.eval()
    profile = model.profile_compute(device=device, batch_size=batch_size)

    history = torch.randn(batch_size, 14, 6, 16, 16, device=device)
    future = torch.randn(batch_size, 7, 6, 16, 16, device=device)
    terrain = torch.randn(batch_size, 5, 80, 80, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.sample(history=history, future_forecast=future, terrain=terrain, num_steps=4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            t0 = time.time()
            _ = model.sample(history=history, future_forecast=future, terrain=terrain, num_steps=4)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies.append((time.time() - t0) * 1000.0)

    mean_lat_ms = float(np.mean(latencies))
    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "tier": tier,
        "total_parameters": profile["total_parameters"],
        "active_parameters": profile["active_parameters"],
        "parameter_ratio": profile["parameter_ratio_vs_candidate3"],
        "mean_latency_4step_ms": round(mean_lat_ms, 2),
        "ms_per_diffusion_step": round(mean_lat_ms / 4.0, 2),
        "peak_vram_mb": round(peak_vram_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Sprint 9 Scaling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", type=str, default="reports/sprint9_training_efficiency.md")
    args, _ = parser.parse_known_args()

    device = torch.device(args.device)
    tiers = ["dense_s", "dense_m", "dense_l", "moe_4"]
    benchmarks = []

    for t in tiers:
        b_res = benchmark_tier(t, device, batch_size=args.batch_size)
        benchmarks.append(b_res)
        print(f"  -> Latency (4 steps): {b_res['mean_latency_4step_ms']} ms ({b_res['ms_per_diffusion_step']} ms/step)")

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sprint 9: Computational and Memory Scaling Efficiency",
        "",
        "| Model Tier | Total Params | Active Params | Param Ratio | Latency (4-step, ms) | Latency/step (ms) | Peak VRAM (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b in benchmarks:
        lines.append(
            f"| **{b['tier'].upper()}** | {b['total_parameters']:,} | {b['active_parameters']:,} | {b['parameter_ratio']:.2f}x | {b['mean_latency_4step_ms']} | {b['ms_per_diffusion_step']} | {b['peak_vram_mb']} |"
        )
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Saved computational scaling benchmark to {out_file}")


if __name__ == "__main__":
    main()
