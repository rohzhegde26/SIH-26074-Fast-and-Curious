"""Sprint 10 open-problem results (all scored on the 2023 test season, which no model or calibration saw)
-> docs/results_s10.md"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sihv3.metrics import composite_skill  # noqa: E402

K = ["precip_crps", "precip_ssr", "precip_cov90", "precip_bias_ratio", "brier30", "precip_csi30", "precip_wet_mae",
     "tmax_mae", "rh_mae", "wind_vec_rmse"]


def res(tag):
    f = glob.glob(str(REPO / "results" / "*" / "out" / tag / "result.json"))
    return json.load(open(f[0])) if f else None


def row(name, r, split="test"):
    a, ref = r[split]["aggregate"], r[split]["reference_gfs_bilinear"]["aggregate"]
    css = composite_skill(a, ref)
    qm = r[split].get("precip_qm")
    cq = f"{composite_skill(qm['aggregate'], ref):.4f}" if qm else ""
    return f"| {name} | {css:.4f} | {cq} | " + " | ".join(f"{a.get(k, float('nan')):.3f}" for k in K) + " |"


def main():
    md = ["# Sprint 10 open problems — results on the 2023 test season\n",
          "All models: H=3, N=40 (N/M 2.5), mm-loss, MoE backbone (E8 top-2, 50 %), trained 2015-2022 for fixed epochs;",
          "diffusion sampled with DPM-Solver++ 24 steps x 8 members. CSS vs GFS-bilinear (ensemble mean for diffusion).\n",
          f"| model | CSS | CSS +precipQM | " + " | ".join(K) + " |", "|" + "---|" * (3 + len(K))]
    for tag, name in [("s10_fin_det_s0", "deterministic MoE (seed 0)"), ("s10_fin_det_s1", "deterministic MoE (seed 1)"),
                      ("s10_diff_ins_S_s0", "diffusion, in-sample residuals (seed 0)"),
                      ("s10_diff_ins_S_s1", "diffusion, in-sample residuals (seed 1)"),
                      ("s10_diff_oof_S_s0", "diffusion, CROSS-FITTED residuals (seed 0)"),
                      ("s10_diff_oof_S_s1", "diffusion, CROSS-FITTED residuals (seed 1)"),
                      ("s10_diff_oof_MoE_s0", "diffusion, cross-fitted, MoE denoiser (seed 0)"),
                      ("s10_diff_oof_S_cal_s0", "diffusion, cross-fitted, trained 2015-21 (calibration model)")]:
        r = res(tag)
        md.append(row(name, r) if r else f"| {name} | _pending_ |")
    for f in sorted(glob.glob(str(REPO / "results" / "*" / "out" / "cal_*" / "calibration.json"))):
        c = json.load(open(f))
        md += [f"\n## Calibration ({Path(f).parent.name}): fitted on {c['fit_years']}, scored on {c['eval_years']}\n",
               "| ensemble | variant | CSS (ens mean) | " + " | ".join(k for k in K if k in ("precip_crps", "precip_ssr", "precip_cov90", "precip_bias_ratio", "brier30")) + " | tmax_crps |",
               "|" + "---|" * 9]
        for grp, label in (("model", "2015-21 model"), ("applied", "final model (params transferred)")):
            for v, m in c.get(grp, {}).items():
                md.append(f"| {label} | {v} | {m['css_ensmean']:.4f} | " + " | ".join(
                    f"{m.get(k, float('nan')):.3f}" for k in ("precip_crps", "precip_ssr", "precip_cov90", "precip_bias_ratio", "brier30"))
                          + f" | {m.get('tmax_crps', float('nan')):.3f} |")
    (REPO / "docs" / "results_s10.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
