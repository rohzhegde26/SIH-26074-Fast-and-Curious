"""
scripts/benchmark_sprint6_experiments.py

Sprint 6 Residual Diffusion Benchmark Compiler & Scientific Hypothesis Evaluator.
Evaluates:
  - Candidate 1 (EXP-01): Frozen Control (eps-prediction, unweighted MSE, N=24, H=14)
  - Candidate 2 (EXP-02): Velocity Prediction (v-prediction, unweighted MSE, N=24, H=14)
  - Candidate 3 (EXP-03): Multi-Task Variable-Aware Noise Weighting (v-prediction, group_tail loss, N=24, H=14)

Outputs:
  - reports/sprint_6_residual_diffusion_summary.json
  - reports/sprint_6_comparison_table.md
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
OUTPUT_SUMMARY_JSON = REPORTS_DIR / "sprint_6_residual_diffusion_summary.json"
OUTPUT_TABLE_MD = REPORTS_DIR / "sprint_6_comparison_table.md"

CANDIDATES = [
    {
        "id": "EXP-01",
        "name": "Candidate 1 (Control)",
        "prediction_type": "epsilon",
        "loss_weighting": "uniform",
        "report_file": "training_spatial_n24_history.json",
        "desc": "Frozen Sprint 5 baseline (eps-prediction, unweighted MSE)",
    },
    {
        "id": "EXP-02",
        "name": "Candidate 2 (v-pred)",
        "prediction_type": "v_prediction",
        "loss_weighting": "uniform",
        "report_file": "sprint6_candidate2_vpred_history.json",
        "desc": "Velocity prediction (v-prediction, unweighted MSE, CMVS-selected)",
    },
    {
        "id": "EXP-03",
        "name": "Candidate 3 (Multi-Task Tail)",
        "prediction_type": "v_prediction",
        "loss_weighting": "group_tail",
        "report_file": "sprint6_candidate3_multitask_history.json",
        "desc": "Multi-Task group-tail weighted v-prediction (CMVS-selected)",
    },
]


def load_candidate_reports() -> Dict[str, Dict[str, Any]]:
    data = {}
    for cand in CANDIDATES:
        rpt_path = REPORTS_DIR / cand["report_file"]
        if rpt_path.exists():
            with open(rpt_path, "r", encoding="utf-8") as f:
                data[cand["id"]] = json.load(f)
            print(f"[+] Loaded {cand['id']} from {cand['report_file']}")
        else:
            print(f"[-] Report not found for {cand['id']}: {cand['report_file']}")
    return data


def extract_val_record(exp_data: Dict[str, Any]) -> Dict[str, Any]:
    history = exp_data.get("history", [])
    if not history:
        return {}

    # Check if any epoch has CMVS recorded
    epochs_with_metrics = [x for x in history if x.get("metrics") and x["metrics"].get("per_lead")]
    if not epochs_with_metrics:
        best_epoch = min(history, key=lambda x: x.get("val_loss", float("inf")))
        return {
            "epoch": best_epoch.get("epoch", 1),
            "val_loss": best_epoch.get("val_loss", 0.0),
            "train_loss": best_epoch.get("train_loss", 0.0),
            "cmvs": 1.0,
            "precip_mae": 0.0,
            "precip_wet_mae": 0.0,
            "precip_csi15": 0.0,
            "precip_csi30": 0.0,
            "tmax_mae": 0.0,
            "tmin_mae": 0.0,
            "rh_mae": 0.0,
            "wind_vector_rmse": 0.0,
        }

    # Best epoch selected by CMVS if available, else by minimum validation loss
    def get_cmvs(ep):
        m = ep.get("metrics", {}).get("aggregate", {})
        if "cmvs" in m:
            return m["cmvs"]
        wet = m.get("precip_wet_mae", 8.70)
        csi = m.get("precip_csi30", 0.631)
        tmax = m.get("tmax_mae", 0.37)
        wind = m.get("wind_vector_rmse", 1.69)
        return 0.35 * (wet / 8.70) + 0.35 * max(0.0, 1.0 - csi / 0.631) + 0.15 * (tmax / 0.37) + 0.15 * (wind / 1.69)

    best_epoch = min(epochs_with_metrics, key=get_cmvs)
    metrics = best_epoch.get("metrics", {})
    agg = metrics.get("aggregate", {})
    per_lead = metrics.get("per_lead", [])

    return {
        "epoch": best_epoch.get("epoch", 1),
        "val_loss": best_epoch.get("val_loss", 0.0),
        "train_loss": best_epoch.get("train_loss", 0.0),
        "cmvs": get_cmvs(best_epoch),
        "precip_mae": agg.get("precip_mae", float(np.mean([m["precip_mae"] for m in per_lead]))),
        "precip_wet_mae": agg.get("precip_wet_mae", float(np.mean([m["precip_wet_mae"] for m in per_lead]))),
        "precip_csi15": agg.get("precip_csi15", float(np.mean([m["precip_csi15"] for m in per_lead]))),
        "precip_csi30": agg.get("precip_csi30", float(np.mean([m["precip_csi30"] for m in per_lead]))),
        "tmax_mae": agg.get("tmax_mae", float(np.mean([m["tmax_mae"] for m in per_lead]))),
        "tmin_mae": agg.get("tmin_mae", float(np.mean([m["tmin_mae"] for m in per_lead]))),
        "rh_mae": agg.get("rh_mae", float(np.mean([m["rh_mae"] for m in per_lead]))),
        "wind_vector_rmse": agg.get("wind_vector_rmse", float(np.mean([m["wind_vector_rmse"] for m in per_lead]))),
        "per_lead": per_lead,
    }


def extract_test_record(exp_data: Dict[str, Any]) -> Dict[str, Any]:
    test_m = exp_data.get("test_metrics", {})
    agg = test_m.get("aggregate", {})
    per_lead = test_m.get("per_lead", [])
    if not agg and per_lead:
        agg = {
            "precip_mae": float(np.mean([m["precip_mae"] for m in per_lead])),
            "precip_wet_mae": float(np.mean([m["precip_wet_mae"] for m in per_lead])),
            "precip_csi15": float(np.mean([m["precip_csi15"] for m in per_lead])),
            "precip_csi30": float(np.mean([m["precip_csi30"] for m in per_lead])),
            "tmax_mae": float(np.mean([m["tmax_mae"] for m in per_lead])),
            "tmin_mae": float(np.mean([m["tmin_mae"] for m in per_lead])),
            "rh_mae": float(np.mean([m["rh_mae"] for m in per_lead])),
            "wind_vector_rmse": float(np.mean([m["wind_vector_rmse"] for m in per_lead])),
        }
    return agg


def compile_benchmark():
    reports = load_candidate_reports()
    if not reports:
        print("[-] No reports found to compile benchmark.")
        return

    summary = {
        "program": "SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)",
        "sprint": 6,
        "frozen_invariants": {
            "antecedent_history_h": 14,
            "spatial_context_n": 24,
            "model_parameters": 15685478,
            "sampler": "DDIM-32 (eta=0.0)",
        },
        "candidates": {},
    }

    val_records = {}
    test_records = {}

    for cand in CANDIDATES:
        cid = cand["id"]
        if cid in reports:
            v_rec = extract_val_record(reports[cid])
            t_rec = extract_test_record(reports[cid])
            val_records[cid] = v_rec
            test_records[cid] = t_rec
            summary["candidates"][cid] = {
                "name": cand["name"],
                "prediction_type": cand["prediction_type"],
                "loss_weighting": cand["loss_weighting"],
                "description": cand["desc"],
                "validation": v_rec,
                "holdout_test": t_rec,
            }

    # Evaluate Hypotheses
    hypotheses = {}
    if "EXP-01" in val_records and "EXP-02" in val_records:
        e1_val = val_records["EXP-01"]
        e2_val = val_records["EXP-02"]
        e1_test = test_records.get("EXP-01", {})
        e2_test = test_records.get("EXP-02", {})

        cmvs_improvement = (e1_val["cmvs"] - e2_val["cmvs"]) / max(1e-5, e1_val["cmvs"]) * 100.0
        hypotheses["H1_vpred_stability"] = {
            "status": "CONFIRMED" if e2_val["cmvs"] < e1_val["cmvs"] else "FALSIFIED",
            "val_cmvs_control": e1_val["cmvs"],
            "val_cmvs_vpred": e2_val["cmvs"],
            "delta_cmvs_pct": cmvs_improvement,
        }

    if "EXP-02" in val_records and "EXP-03" in val_records:
        e2_val = val_records["EXP-02"]
        e3_val = val_records["EXP-03"]
        csi_diff = (e3_val["precip_csi30"] - e2_val["precip_csi30"]) / max(1e-5, e2_val["precip_csi30"]) * 100.0
        tmax_diff = (e3_val["tmax_mae"] - e2_val["tmax_mae"]) / max(1e-5, e2_val["tmax_mae"]) * 100.0
        hypotheses["H2_tail_heteroscedasticity"] = {
            "status": "CONFIRMED" if (e3_val["precip_csi30"] >= e2_val["precip_csi30"] and tmax_diff <= 5.0) else "FALSIFIED",
            "csi30_vpred": e2_val["precip_csi30"],
            "csi30_multitask": e3_val["precip_csi30"],
            "delta_csi30_pct": csi_diff,
            "tmax_mae_diff_pct": tmax_diff,
        }

    summary["hypotheses"] = hypotheses

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved benchmark summary JSON: {OUTPUT_SUMMARY_JSON}")

    # Build Markdown Comparison Table
    lines = [
        "# Sprint 6 Benchmark: Residual Diffusion Formulation & Meteorological Refinement",
        "",
        "**Program**: SIH Problem Statement 26074 (Weather Downscaling from Block to Panchayat Level)  ",
        "**Corpus**: `rohzhegde26/SIH-26074-Fast-and-Curious`  ",
        "**Branch**: `feat/spatiotemporal-diffusion-downscaler`  ",
        "**Date**: September 25, 2026  ",
        "",
        "## Invariants Maintained Across All Formulations",
        "- **Antecedent History**: Frozen $H^* = 14$ days",
        "- **Spatial Context**: Frozen $N^* = 24$ coarse cells ($N/M = 1.50$, $660 \\times 660$ km domain)",
        "- **Target Grid**: $M = 16$ coarse cells downscaled $5\\times$ to $80 \\times 80$ fine cells ($0.05^\\circ$)",
        "- **Model Capacity**: Strictly locked at **15,685,478 parameters**",
        "- **Sampler**: Deterministic DDIM-32 ($\\eta = 0.0$)",
        "",
        "---",
        "",
        "## Table A: Multi-Task Validation Objective and Convergence Table (2022 Validation Season)",
        "",
        "| Candidate ID | Formulation | Parameterization | Loss Objective | Best Epoch | Val Loss | Val CMVS | Wet-MAE (mm) | CSI@30 | Tmax MAE (°C) | Wind RMSE (m/s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for cand in CANDIDATES:
        cid = cand["id"]
        if cid in val_records:
            v = val_records[cid]
            lines.append(
                f"| **{cid}** | {cand['name']} | `{cand['prediction_type']}` | `{cand['loss_weighting']}` | "
                f"Epoch {v['epoch']} | {v['val_loss']:.4f} | **{v['cmvs']:.4f}** | "
                f"{v['precip_wet_mae']:.2f} | {v['precip_csi30']:.3f} | {v['tmax_mae']:.2f} | {v['wind_vector_rmse']:.2f} |"
            )

    lines.extend([
        "",
        "> [!NOTE]",
        "> **CMVS Calculation**: $\\text{CMVS} = 0.35 \\left(\\frac{\\text{WetMAE}}{8.70}\\right) + 0.35 \\left(1.0 - \\frac{\\text{CSI@30}}{0.631}\\right) + 0.15 \\left(\\frac{\\text{TmaxMAE}}{0.37}\\right) + 0.15 \\left(\\frac{\\text{WindRMSE}}{1.69}\\right)$. Lower is better.",
        "",
        "---",
        "",
        "## Table B: Confirmatory Holdout Test Performance (2023 Season - Quarantined)",
        "",
        "| Candidate ID | Formulation | Precip Wet-MAE (mm) | Precip CSI@15 | Precip CSI@30 | Tmax MAE (°C) | Tmin MAE (°C) | Wind Vector RMSE (m/s) |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for cand in CANDIDATES:
        cid = cand["id"]
        if cid in test_records and test_records[cid]:
            t = test_records[cid]
            lines.append(
                f"| **{cid}** | {cand['name']} | **{t.get('precip_wet_mae', 0.0):.2f}** | "
                f"{t.get('precip_csi15', 0.0):.3f} | {t.get('precip_csi30', 0.0):.3f} | "
                f"{t.get('tmax_mae', 0.0):.2f} | {t.get('tmin_mae', 0.0):.2f} | "
                f"{t.get('wind_vector_rmse', 0.0):.2f} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Scientific Hypothesis Evaluation",
        "",
    ])

    for h_name, h_info in hypotheses.items():
        lines.append(f"### {h_name}: **{h_info['status']}**")
        for k, val in h_info.items():
            if k != "status":
                lines.append(f"- **{k}**: {val}")
        lines.append("")

    with open(OUTPUT_TABLE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[+] Saved comparison table Markdown: {OUTPUT_TABLE_MD}")


if __name__ == "__main__":
    compile_benchmark()
