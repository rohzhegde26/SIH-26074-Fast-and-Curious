"""
Aggregate results/<job>/out/<tag>/result.json into per-sprint tables and apply the selection rules
of docs/selection_criteria.md (seed mean, noise floor, ties -> cheaper config).

  python scripts/summarize.py            -> docs/results.md (+ prints the tables)
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihv3.metrics import composite_skill  # noqa: E402  (recompute CSS with the current metric definition)

REPO = Path(__file__).resolve().parents[1]
COLS = ["precip_wet_mae", "precip_csi15", "precip_csi30", "precip_fss15", "precip_bias_ratio",
        "tmax_mae", "tmin_mae", "rh_mae", "wind_vec_rmse"]


def load_runs():
    runs = []
    for f in glob.glob(str(REPO / "results" / "*" / "out" / "*" / "result.json")):
        r = json.load(open(f))
        a = r["args"]
        cfg = re.sub(r"_s\d+$", "", a["tag"])
        refv = r["val"]["reference_gfs_bilinear"]["aggregate"]
        reft = r["test"]["reference_gfs_bilinear"]["aggregate"]
        r["val"]["css"] = composite_skill(r["val"]["aggregate"], refv)
        r["test"]["css"] = composite_skill(r["test"]["aggregate"], reft)
        if "precip_qm" in r["val"]:
            r["val"]["precip_qm"]["css"] = composite_skill(r["val"]["precip_qm"]["aggregate"], refv)
        runs.append({"tag": a["tag"], "cfg": cfg, "seed": a["seed"], "mode": a["mode"], "size": a["size"],
                     "H": a["H"], "N": a["N"], "moe": (a["moe_experts"], a["moe_topk"], a["moe_frac"]),
                     "params": r["params_total"], "active": r["params_active"], "val_css": r["val"]["css"],
                     "test_css": r["test"]["css"], "val": r["val"]["aggregate"], "test": r["test"]["aggregate"],
                     "val_css_qm": r["val"].get("precip_qm", {}).get("css", float("nan")),
                     "ref": r["val"]["reference_gfs_bilinear"]["aggregate"], "minutes": r["train_minutes"],
                     "latency": r["val"].get("latency_sec_per_sample"), "vram": r.get("peak_vram_gb"),
                     "best_epoch": r["best_epoch"]})
    return runs


def table(runs, prefix, cost_key):
    sel = [r for r in runs if r["cfg"].startswith(prefix if isinstance(prefix, tuple) else (prefix,))]
    if not sel:
        return f"_no results yet for {prefix}_\n", None
    g = defaultdict(list)
    for r in sel:
        g[r["cfg"]].append(r)
    rows = []
    for cfg, rs in g.items():
        css = np.array([r["val_css"] for r in rs])
        rows.append({"cfg": cfg, "n": len(rs), "css": css.mean(), "sd": css.std(ddof=1) if len(rs) > 1 else np.nan,
                     "test": np.mean([r["test_css"] for r in rs]), "cost": cost_key(rs[0]), "rs": rs,
                     "css_qm": np.mean([r["val_css_qm"] for r in rs]),
                     **{c: np.mean([r["val"][c] for r in rs]) for c in COLS}})
    rows.sort(key=lambda x: -x["css"])
    sds = [r["sd"] for r in rows if np.isfinite(r["sd"])]
    # tie threshold = 1 standard error of the difference of two seed means: pooled_sd * sqrt(2 / n_seeds)
    n_med = float(np.median([r["n"] for r in rows]))
    floor = max(0.005, float(np.sqrt(np.mean(np.square(sds)))) * np.sqrt(2 / max(n_med, 1))) if sds else 0.005
    top = rows[0]
    tied = [r for r in rows if top["css"] - r["css"] < floor]
    winner = min(tied, key=lambda r: r["cost"])
    ref = rows[0]["rs"][0]["ref"]
    hdr = "| config | seeds | val CSS (mean ± sd) | val CSS +precipQM | test CSS | " + " | ".join(COLS) + " |\n|" + "---|" * (5 + len(COLS)) + "\n"
    body = "".join(
        f"| {'**' + r['cfg'] + '**' if r is winner else r['cfg']} | {r['n']} | {r['css']:.4f} ± {r['sd']:.4f} | {r['css_qm']:.4f} | {r['test']:.4f} | "
        + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |\n" for r in rows)
    body += "| GFS-bilinear (reference) | | 0 | | 0 | " + " | ".join(f"{ref[c]:.3f}" for c in COLS) + " |\n"
    note = (f"\nNoise floor {floor:.4f}; configs within it of the best: {', '.join(r['cfg'] for r in tied)}. "
            f"**Selected: {winner['cfg']}** (cheapest within the floor).\n")
    winner["ranking"] = [r["cfg"] for r in rows]
    return hdr + body + note, winner


def main():
    runs = load_runs()
    out = ["# Results (validation 2022; test 2023 shown for reference only)\n"]
    sections = [("Sprint 3 — deterministic baseline + loss variant (H=7, N=16)", ("s4_h7_n16", "lin_"), lambda r: 0),
                ("Sprint 4 — history length (N=16)", "s4_", lambda r: r["H"]),
                ("Sprint 5 — spatial context (H=7; N=16 row shared with Sprint 4)", ("s5_", "s4_h7_n16"), lambda r: r["N"]),
                ("Sprint 4/5 confirmation — 100-epoch schedule (H, N)", ("s45_confirm", "s9_dense_S"), lambda r: (r["H"], r["N"])),
                ("Sprint 4/5 second validation season (2021 held out, 2022 in training) — 100 epochs", "s45v21_", lambda r: (r["H"], r["N"])),
                ("Sprint 6 — deterministic vs diffusion", ("s6_", "s9_dense_S"), lambda r: (r["mode"] == "diff", r["params"])),
                ("Sprint 9 — capacity and MoE", "s9_", lambda r: (r["active"], r["params"]))]
    winners, rankings = {}, {}
    for title, prefix, cost in sections:
        md, w = table(runs, prefix, cost)
        out += [f"\n## {title}\n", md]
        winners[title.split(" —")[0]] = w["cfg"] if w else None
        if w:
            rankings[title.split(" —")[0]] = w["ranking"]
    (REPO / "docs" / "results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("WINNERS", winners)
    print("RANKINGS", rankings)


if __name__ == "__main__":
    main()
