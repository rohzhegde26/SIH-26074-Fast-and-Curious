"""
scripts/benchmark_temporal_baselines.py

Evaluates parameter-free spatiotemporal baselines on frozen validation split (2022):
  - Baseline 0A: Channel-Aware Coarse-to-Fine (Conservative nearest precip + bilinear others)
  - Baseline 0B: All-Bilinear Interpolation
  - Baseline 0C: Temporal Persistence (D-1 extended across all 7 leads)

Saves results to reports/baselines_benchmark_summary.json.
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import invert_normalization
from src.models.baselines import (
    BilinearTemporalBaseline,
    ChannelAwareTemporalBaseline,
    PersistenceTemporalBaseline,
)
from scripts.train_temporal_downscaler import compute_lead_metrics


def run_baselines_benchmark():
    print("=" * 70)
    print("SIH 26074 Sprint 3: Evaluating Non-Learned Baselines on Val (2022)")
    print("=" * 70)

    zarr_path = ROOT / "datasets" / "multitask_temporal_v1.zarr"
    index_path = ROOT / "data" / "sample_index.parquet"
    stats_path = ROOT / "data" / "normalization_stats.yaml"

    val_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        history_len=3,
        normalize=True,
    )

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"[+] Loaded validation split: {len(val_ds)} samples (2022)")

    models = {
        "baseline_0a_channel_aware": ChannelAwareTemporalBaseline(),
        "baseline_0b_all_bilinear": BilinearTemporalBaseline(),
        "baseline_0c_persistence": PersistenceTemporalBaseline(),
    }

    results = {}

    for name, model in models.items():
        print(f"\n[*] Evaluating {name}...")
        t0 = time.time()
        preds_list = []
        targets_list = []
        coarse_list = []

        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"]  # [B, 3, 6, 16, 16]
                fcst = batch["future_forecast"]  # [B, 7, 6, 16, 16]
                target = batch["target"]  # [B, 7, 6, 80, 80]

                if name == "baseline_0c_persistence":
                    pred = model(history)
                else:
                    pred = model(fcst)

                preds_list.append(pred.numpy())
                targets_list.append(target.numpy())
                coarse_list.append(fcst.numpy())

        preds_all = np.concatenate(preds_list, axis=0)
        targets_all = np.concatenate(targets_list, axis=0)
        coarse_all = np.concatenate(coarse_list, axis=0)

        preds_phys = invert_normalization(preds_all, val_ds.stats)
        targets_phys = invert_normalization(targets_all, val_ds.stats)
        coarse_phys = invert_normalization(coarse_all, val_ds.stats)

        metrics = compute_lead_metrics(preds_phys, targets_phys, coarse_phys)
        dur = time.time() - t0
        print(f"[+] {name} completed in {dur:.2f}s:")
        agg = metrics["aggregate"]
        print(f"    Precip Wet-MAE:   {agg['precip_wet_mae']:.2f} mm")
        print(f"    Precip CSI@15:    {agg['precip_csi15']:.3f}")
        print(f"    Precip CSI@30:    {agg['precip_csi30']:.3f}")
        print(f"    Tmax MAE:         {agg['tmax_mae']:.2f} C")
        print(f"    Tmin MAE:         {agg['tmin_mae']:.2f} C")
        print(f"    Diurnal Viol:     {agg['diurnal_violation_rate']*100:.2f}%")
        print(f"    RH MAE:           {agg['rh_mae']:.2f} %")
        print(f"    Wind Vector RMSE: {agg['wind_vector_rmse']:.2f} m/s")

        results[name] = {
            "name": name,
            "evaluation_duration_sec": dur,
            "metrics": metrics,
        }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "baselines_benchmark_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved baseline benchmarks to: {out_file.relative_to(ROOT)}")
    return results


if __name__ == "__main__":
    run_baselines_benchmark()
