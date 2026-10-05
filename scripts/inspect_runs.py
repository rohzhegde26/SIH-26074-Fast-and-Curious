"""
Health check of finished runs: learning curve, stop reason, red flags, and comparison with the
Sprint 3 non-learned baselines.   python scripts/inspect_runs.py [job_name ...]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sihv3.metrics import composite_skill  # noqa: E402
base = json.load(open(REPO / "results" / "baselines_s3.json"))
QM_VAL = base["val"]["gfs_qm_css_vs_trainer_ref"]  # GFS-QM scored against the same reference as the models


def check(f):
    r = json.load(open(f))
    if "val" not in r:  # fixed-epoch runs (Sprint 10 protocol) have no validation season: report test only
        t = r["test"]
        ref = t["reference_gfs_bilinear"]["aggregate"]
        qm = t.get("precip_qm")
        h = r["history"]
        print(f"{r['args']['tag']:22s} {r['args']['mode']} H{r['args']['H']} N{r['args']['N']} train {r['args'].get('train_years')} | "
              f"{len(h)} ep, loss {h[0]['train_loss']:.3f} -> {h[-1]['train_loss']:.3f}, {r['train_minutes']:.0f} min | "
              f"TEST CSS {composite_skill(t['aggregate'], ref):+.4f}"
              + (f" (+precipQM {composite_skill(qm['aggregate'], ref):+.4f})" if qm else "")
              + f" | bias {t['aggregate']['precip_bias_ratio']:.2f} csi30 {t['aggregate']['precip_csi30']:.3f} "
              f"Tmax {t['aggregate']['tmax_mae']:.2f} wind {t['aggregate']['wind_vec_rmse']:.2f}"
              + (f" | crps_p {t['aggregate']['precip_crps']:.3f} ssr_p {t['aggregate'].get('precip_ssr', float('nan')):.2f} cov90 {t['aggregate'].get('precip_cov90', float('nan')):.2f}" if "precip_crps" in t["aggregate"] else ""))
        if any(e["train_loss"] != e["train_loss"] for e in h):
            print(f"{'':22s} !! NaN train loss")
        return
    for sp in ("val", "test"):  # recompute with the current CSS definition (stored values may predate fixes)
        ref = r[sp]["reference_gfs_bilinear"]["aggregate"]
        r[sp]["css"] = composite_skill(r[sp]["aggregate"], ref)
        if "precip_qm" in r[sp]:
            r[sp]["precip_qm"]["css"] = composite_skill(r[sp]["precip_qm"]["aggregate"], ref)
    a, h = r["args"], r["history"]
    tag = a["tag"]
    flags = []
    key = "val_css" if a["mode"] == "det" else "val_vloss"
    curve = [e.get(key) for e in h]
    if any(c is None or c != c for c in curve):
        flags.append("NaN in validation curve")
    if r["best_epoch"] <= 3 and len(h) > 6:
        flags.append(f"best epoch {r['best_epoch']} of {len(h)} (early peak -> overfitting/instability?)")
    stopped = "time budget" if len(h) < a["epochs"] and len(h) - r["best_epoch"] < a["patience"] else (
        "patience" if len(h) < a["epochs"] else "max epochs")
    if stopped in ("time budget", "max epochs") and r["best_epoch"] >= len(h) - 2:
        flags.append(f"still improving when {stopped} stopped it (undertrained)")
    v, t = r["val"], r["test"]
    if v["aggregate"].get("tmax_lt_tmin_rate", 0) > 0.01:
        flags.append(f"Tmax<Tmin on {v['aggregate']['tmax_lt_tmin_rate']:.1%} of pixels")
    if abs(v["css"] - t["css"]) > 0.15:
        flags.append(f"val/test CSS gap {v['css'] - t['css']:+.3f}")
    br = v["aggregate"]["precip_bias_ratio"]
    if abs(br - 1) > abs(v["reference_gfs_bilinear"]["aggregate"]["precip_bias_ratio"] - 1):
        flags.append(f"precip bias ratio {br:.2f} worse than GFS-bilinear")
    pts = ", ".join(f"{e['epoch']}:{e.get(key, float('nan')):.3f}" for e in h[:: max(1, len(h) // 8)])
    print(f"{tag:22s} {a['mode']} {a['size']} H{a['H']} N{a['N']} | {len(h)} ep, best {r['best_epoch']} ({stopped}), "
          f"{r['train_minutes']:.0f} min | val CSS {v['css']:+.4f} (+precipQM {v.get('precip_qm', {}).get('css', float('nan')):+.4f}; GFS-QM {QM_VAL:+.3f}) test {t['css']:+.4f} | "
          f"wetMAE {v['aggregate']['precip_wet_mae']:.2f} csi15 {v['aggregate']['precip_csi15']:.3f} "
          f"csi30 {v['aggregate']['precip_csi30']:.3f} bias {br:.2f} Tmax {v['aggregate']['tmax_mae']:.2f} "
          f"RH {v['aggregate']['rh_mae']:.2f} wind {v['aggregate']['wind_vec_rmse']:.2f}")
    print(f"{'':22s} curve {key}: {pts}")
    for fl in flags:
        print(f"{'':22s} !! {fl}")


def main():
    jobs = sys.argv[1:] or [p.name for p in (REPO / "results").iterdir() if p.is_dir()]
    for j in sorted(jobs):
        files = sorted(glob.glob(str(REPO / "results" / j / "out" / "*" / "result.json")))
        st = REPO / "results" / j / "out" / "job_status.json"
        if st.exists():
            print(f"== {j}: {json.load(open(st))}")
        for f in files:
            check(f)
        logs = glob.glob(str(REPO / "results" / j / "out" / "*.log"))
        for lg in logs:
            txt = Path(lg).read_text(errors="ignore")
            if "Traceback" in txt:
                print(f"   !! Traceback in {Path(lg).name}:\n" + "\n".join(txt.splitlines()[-6:]))


if __name__ == "__main__":
    main()
