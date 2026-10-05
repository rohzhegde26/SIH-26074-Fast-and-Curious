"""Sprint 11 results -> docs/results_s11.md: history length x forecast history (selection on 2022, test 2023
reported), post-training improvements (seed averaging, probability-matched mean, rain tail cap) and the
averaged-residual denoiser."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sihv3.metrics import composite_skill  # noqa: E402


def res(tag):
    f = glob.glob(str(REPO / "results" / "*" / "out" / tag / "result.json"))
    return json.load(open(f[0])) if f else None


def css(r, sp):
    a = r[sp]
    return composite_skill(a["aggregate"], a["reference_gfs_bilinear"]["aggregate"])


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "_pending_"
    return f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.4f} (n=1)"


def main():
    md = ["# Sprint 11 — history, forecast history and post-training improvements\n",
          "Selection season 2022 (models trained 2015–2021); 2023 test reported only. Recipe = the shipped deterministic",
          "backbone (MoE S, E8 top-2 50 %, N/M 2.5, no mm-loss, 50 epochs). Means ± seed SD over seeds 0–2.\n",
          "## 1. History length H × forecast history\n",
          "`obs` = observed ERA5 history only (as shipped); `+fc` = observed history plus the GFS forecast that was valid",
          "on each history day (from the run issued that day), i.e. GFS's recent errors are visible to the model.\n",
          "| H | input | 2022 val CSS | 2023 test CSS | 2022 Tmax MAE | 2022 rain CSI30 | paired Δ(+fc − obs) 2022 |",
          "|---|---|---|---|---|---|---|"]
    best = {}
    for H in (3, 5, 7, 10, 14):
        per = {}
        for kind in ("obs", "fc"):
            rs = [res(f"s11_h{H}_{kind}_s{s}") for s in range(3)]
            per[kind] = rs
            v = [css(r, "val") if r else None for r in rs]
            t = [css(r, "test") if r else None for r in rs]
            tm = [r["val"]["aggregate"]["tmax_mae"] if r else None for r in rs]
            c30 = [r["val"]["aggregate"]["precip_csi30"] if r else None for r in rs]
            best[(H, kind)] = [x for x in v if x is not None]
            dd = ""
            if kind == "fc":
                d = [css(f, "val") - css(o, "val") for o, f in zip(per["obs"], per["fc"]) if o and f]
                if d:
                    se = np.std(d, ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                    dd = f"{np.mean(d):+.4f} (SE {se:.4f}, n={len(d)})"
            tmv = [x for x in tm if x is not None]
            c3v = [x for x in c30 if x is not None]
            md.append(f"| {H} | {'obs + fc' if kind == 'fc' else 'obs'} | {ms(v)} | {ms(t)} | "
                      f"{np.mean(tmv):.3f} | {np.mean(c3v):.3f} | {dd} |" if tmv else
                      f"| {H} | {'obs + fc' if kind == 'fc' else 'obs'} | _pending_ | | | | |")
    md += ["", "## 2. Post-training improvements on the shipped model (s11-post, s11-post2)\n"]
    for job, tags in (("s11-post", ("post22", "post23")), ("s11-post2", ("post22b", "post23b"))):
        for t in tags:
            f = glob.glob(str(REPO / "results" / job / "out" / t / "post.json"))
            if not f:
                continue
            r = json.load(open(f[0]))
            md += [f"\n**{t}** (season {r['eval_year']})\n", "| variant | CSS | rain CRPS | rain SSR | Brier>30 | CSI30 | CSI64.5 | POD64.5 | rain bias |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for row in r["rows"]:
                a = row["aggregate"]
                g = lambda k, d=3: f"{a[k]:.{d}f}" if k in a else "–"
                md.append(f"| {row['variant']} | {row['css']:.4f} | {g('precip_crps')} | {g('precip_ssr', 2)} | {g('brier30', 4)} | "
                          f"{g('precip_csi30')} | {g('precip_csi64.5')} | {g('precip_pod64.5', 2)} | {g('precip_bias_ratio', 2)} |")
            if "member_tails" in r:
                md.append("\nMember rain tails on land (mm/day): " + "; ".join(
                    f"{k}: p99 {v['0.99']:.0f}, p99.9 {v['0.999']:.0f}, p99.99 {v['0.9999']:.0f}, max {v['max']:.0f}, "
                    f"above training block max {100 * v['frac_above_cap']:.3f} %" for k, v in r["member_tails"].items()))
    md += ["", "## 3. Denoiser trained on the 3-seed averaged backbone (s11-davg)\n",
           "| denoiser trained on | 2022 val CSS | 2022 rain CRPS | 2023 CSS | 2023 rain CRPS |", "|---|---|---|---|---|"]
    for name, cal, fin in (("single-seed residuals (shipped)", "s10f_diff_oof_S_cal_s0", "s10f_diff_oof_S_s0"),
                           ("3-seed averaged residuals", "s11_diff_avg3_cal_s0", "s11_diff_avg3_s0")):
        c, f = res(cal), res(fin)
        if c and f:
            md.append(f"| {name} | {css(c, 'val'):.4f} | {c['val']['aggregate']['precip_crps']:.3f} | {css(f, 'test'):.4f} | "
                      f"{f['test']['aggregate']['precip_crps']:.3f} |")
    # 4. full-pipeline rebuild of the sweep winner (H5 + forecast history)
    md += ["", "## 4. Full-pipeline rebuild of the sweep winner (H = 5 + forecast history, prefix s12)\n",
           "| seed | OOF CSS 2015–22 (H5+fc) | OOF CSS 2015–22 (shipped H3 obs) | final det 2023 (H5+fc) | final det 2023 (H3 obs) |",
           "|---|---|---|---|---|"]
    old_oof = {0: ("s10s0-oofm", "oofm"), 1: ("s10s1-oof", "oof"), 2: ("s10s2-oof", "oof")}
    for s in range(3):
        n = glob.glob(str(REPO / "results" / f"s12s{s}-oof" / "out" / "oof" / "oof_metrics.json"))
        o = glob.glob(str(REPO / "results" / old_oof[s][0] / "out" / old_oof[s][1] / "oof_metrics.json"))
        fn, fo = res(f"s12_fin_det_s{s}"), res(f"s10f_fin_det_s{s}")
        md.append(f"| {s} | {json.load(open(n[0]))['all_oof']['css']:.4f} | {json.load(open(o[0]))['all_oof']['css']:.4f} | "
                  f"{css(fn, 'test'):.4f} | {css(fo, 'test'):.4f} |" if n and o and fn and fo else f"| {s} | _pending_ |")
    fo_, fn_ = REPO / "results" / "fast_seasons_shipped_h3obs.txt", REPO / "results" / "fast_seasons_s12_h5fc.txt"
    if fo_.exists() and fn_.exists():
        import re
        rd = lambda f: {int(m.group(1)): float(m.group(2)) for m in (re.search(r"(\d{4}): CSS ([0-9.]+)", l) for l in open(f)) if m}
        o, n = rd(fo_), rd(fn_)
        d = np.array([n[y] - o[y] for y in sorted(o)])
        md += ["\nFAST (3-seed average + rain QM fitted on the other seasons), every training season scored out of fold:\n",
               "| season | " + " | ".join(str(y) for y in sorted(o)) + " | mean Δ ± SE |", "|---|" + "---|" * (len(o) + 1),
               "| shipped H3 obs | " + " | ".join(f"{o[y]:.4f}" for y in sorted(o)) + " | |",
               "| H5 + fc | " + " | ".join(f"{n[y]:.4f}" for y in sorted(o)) + f" | {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f} |"]
    md += ["\nBALANCED proxy on 2022 (denoiser trained 2015–21, 24 × 8, raw ensemble):\n",
           "| system | seed | 2022 CSS | 2022 rain CRPS | 2023 CSS (2015–22 denoiser) | 2023 rain CRPS |", "|---|---|---|---|---|---|"]
    for name, s, cal, fin in (("shipped H3 obs", 0, "s10f_diff_oof_S_cal_s0", "s10f_diff_oof_S_s0"),
                              ("shipped H3 obs", 1, "s10dcap_S_val_s1", "s10f_diff_oof_S_s1"),
                              ("H5 + fc", 0, "s12_diff_cal_s0", "s12_diff_s0"), ("H5 + fc", 1, "s12_diff_cal_s1", "s12_diff_s1"),
                              ("H5 + fc", 2, "s12_diff_cal_s2", "s12_diff_s2")):
        c, f = res(cal), res(fin)
        if c and f:
            md.append(f"| {name} | {s} | {css(c, 'val'):.4f} | {c['val']['aggregate']['precip_crps']:.3f} | {css(f, 'test'):.4f} | "
                      f"{f['test']['aggregate']['precip_crps']:.3f} |")
    md += ["\n**Decision:** the replacement rule (beat the shipped system on 2022 in both FAST and BALANCED, rain CRPS",
           "≤ +1 %) fails on BALANCED for the selected seed (seed 1). Seed for seed the H5 + fc pipeline is level with the",
           "shipped one, matching the 8-season FAST tie: the history effects seen in the sweep are real but small, and the",
           "sweep's 2022 margin was inflated by choosing the best of 10 configurations on one season.",
           "", "## 5. Sprint 11 outcome\n",
           "| change | evidence (2022 selection) | status |", "|---|---|---|",
           "| FAST = 3-seed averaged backbone + matching rain QM | CSS 0.2558 → 0.2749 (2023: 0.2435 → 0.2659) | **adopted** |",
           "| Rain tail cap min(2× block max, 1.2× domain max) | CSS +0.0003–0.0009, CRPS unchanged; removes 800–1,100 mm members; 0 real 2023 pixel-days clipped | **adopted** |",
           "| Probability-matched rain field | POD ≥ 64.5 mm 0.10 → 0.20 but CSS −0.05 as the mean | **optional output** (`rain_pm`) |",
           "| Denoiser retrained on the averaged backbone | CSS 0.2803 → 0.2600, CRPS +2.5 % | rejected |",
           "| H = 5 + forecast history (full pipeline) | FAST 8-season tie; BALANCED 0.2680 vs 0.2803 | rejected |",
           "| Static Tmax / RH bias correction | season-to-season bias changes sign (Tmax −0.33 … +0.27 °C) | not applicable |"]
    (REPO / "docs" / "results_s11.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
