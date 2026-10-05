"""Final system report (Sprint 10): shipped model, seed selection, 3-seed spread, capacity points, operating modes,
test-time-compute matrix, latency/VRAM and multivariate consistency -> docs/final_system_report.md
Everything is scored on the 2023 test season, which no model, calibration or selection decision used."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sihv3.metrics import composite_skill  # noqa: E402

PENDING = "_pending_"


def res(tag):
    f = glob.glob(str(REPO / "results" / "*" / "out" / tag / "result.json"))
    return json.load(open(f[0])) if f else None


def jload(pattern):
    f = sorted(glob.glob(str(REPO / "results" / pattern)))
    return json.load(open(f[0])) if f else None


def css(r, qm=False):
    t = r["test"]
    ref = t["reference_gfs_bilinear"]["aggregate"]
    a = t["precip_qm"]["aggregate"] if qm and t.get("precip_qm") else t["aggregate"]
    return composite_skill(a, ref)


def f(x, d=3):
    return PENDING if x is None else f"{x:.{d}f}"


def main():
    md = ["# SIH-26074 final system — Sprint 10 report\n",
          "GFS 0.25° → 0.05° downscaling over 11–15°N, 74–78°E, 7 daily leads × 6 variables (precipitation, Tmax, Tmin, RH,",
          "U, V). All numbers below are on the **2023 monsoon test season** (Jun–Sep, 122 forecasts), which no model,",
          "calibration fit or selection decision used. CSS = composite skill vs GFS-bilinear (higher is better; for",
          "ensembles, the ensemble mean). Tie rule: 1 SE of the paired difference.\n",
          "**Summary.**",
          "* One system, four modes (`models/final`, `configs/final/modes.yaml`): FAST 0.244 CSS in 0.06 s; BALANCED 0.282",
          "  (~4 s); ACCURATE 0.284 (5.3 s); ENSEMBLE 0.290 (8.3 s), all on one T4 at batch 1, < 1 GB VRAM.",
          "* The 3-seed expectation for the diffusion pipeline is 0.263 ± 0.018; the shipped seed's 0.281 is the lucky",
          "  end of that range (picked on out-of-fold skill before test was seen).",
          "* What worked: cross-fitted residual diffusion (rain CRPS −19 % vs the standard recipe, spread ratio 0.4 → 1.1),",
          "  ~24 DPM-Solver++ steps, more members for threshold probabilities, spread calibration (Tmax CRPS −9 %).",
          "* What did not: more capacity, in either stage. Stage-1 S/M/L/MoE are tied over 3 seeds; M/L denoisers fit",
          "  better but sample over-confident ensembles. With 8 training seasons the system is data-limited.\n"]

    # 0. Sprint 11 update (post-training improvements now in models/final)
    p23 = jload("s11-post3/out/post23c/post.json")
    p23a = jload("s11-post/out/post23/post.json")
    if p23 and p23a:
        g = lambda rows, name: next((r for r in rows if r["variant"] == name), None)
        fast1, fast3 = g(p23a["rows"], "FAST single seed + rain QM"), g(p23a["rows"], "FAST 3-seed average + rain QM")
        b0, b1 = g(p23["rows"], "diff single seed K=8 spread-cal | ens mean"), g(p23["rows"], "diff single seed K=8 spread-cal + tail cap | ens mean")
        e0, e1 = g(p23["rows"], "diff single seed K=16 spread-cal | ens mean"), g(p23["rows"], "diff single seed K=16 spread-cal + tail cap | ens mean")
        md += ["## Sprint 11 update (shipped in `models/final`; details in `results_s11.md`)\n",
               "| mode | Sprint 10 | Sprint 11 | change |", "|---|---|---|---|",
               f"| FAST | single seed + QM: CSS {fast1['css']:.4f} | **3-seed average + QM: {fast3['css']:.4f}** | backbone averaging (3 forward passes, ~0.17 s) |",
               f"| BALANCED (24 × 8) | CSS {b0['css']:.4f}, rain SSR {b0['aggregate']['precip_ssr']:.2f} | CSS {b1['css']:.4f}, rain SSR {b1['aggregate']['precip_ssr']:.2f} | rain tail cap |",
               f"| ENSEMBLE (24 × 16) | CSS {e0['css']:.4f}, rain SSR {e0['aggregate']['precip_ssr']:.2f} | CSS {e1['css']:.4f}, rain SSR {e1['aggregate']['precip_ssr']:.2f} | rain tail cap |",
               "",
               "* All four modes now come from one bundle: FAST averages the three seeds' backbones; the diffusion modes keep the",
               "  seed-0 backbone and denoiser (a denoiser retrained on the averaged backbone lost 0.020 CSS on 2022).",
               "* Rain tail cap: individual members reached 1,100 mm/day; each member's rain is now capped at min(2 × the wettest",
               "  training day in its 5×5 block, 1.2 × the wettest training day anywhere). Chosen leave-one-season-out so that no",
               "  held-out season, and no 2023 day, exceeds it.",
               "* New optional output `rain_pm` (probability-matched rain) for heavy-rain maps: doubles detection of ≥ 64.5 mm days.",
               "* Tested and rejected: longer history and forecast history (8-season tie after the full rebuild), denoiser on the",
               "  averaged backbone, static temperature bias correction (the bias changes sign between seasons).",
               "* Sprint 10 tables below are unchanged and describe the Sprint 10 evaluation.\n"]

    # 1. shipped system
    md += ["## 1. The shipped system\n",
           "* **Backbone (FAST):** spatiotemporal transformer, S size with an MoE FFN (8 experts, top-2, MoE in 50 % of",
           "  blocks), history H = 3 days, context N = 40 coarse cells (N/M = 2.5), trained 2015–2022, 50 epochs, no mm-loss.",
           "  Rain quantile mapping per lead and 5×5 block, fitted on the multi-season out-of-fold predictions (2015–2022).",
           "* **Generative stage (BALANCED / ACCURATE / ENSEMBLE):** CorrDiff-style residual diffusion (S denoiser,",
           "  v-prediction, cosine schedule, 80 epochs) trained on **cross-fitted** residuals: every training season's",
           "  residual comes from a deterministic model that never saw that season. Sampled with DPM-Solver++(2M), then",
           "  mean-preserving spread calibration per lead × variable, fitted on 2022 by a model trained without 2022.",
           "* **Mode settings:** FAST = 1 member; BALANCED = 24 steps × 8 members; ACCURATE = 32 × 8; ENSEMBLE = 24 × 16.",
           "  Pre-registered from Sprint 7/8 validation (BALANCED was 16 × 8), then BALANCED's step count re-selected on 2022",
           "  with a rule written down before that run (section 5); 2023 never chose a setting.",
           "* Entry point: `sihv3.predict.FinalDownscaler(\"models/final\").predict(history, forecast, mode=...)`.\n"]

    # 2. seed selection
    md += ["## 2. Seed selection (out-of-fold, 2023 not used) and 3-seed spread\n",
           "| seed | OOF CSS 2015–22 (selection score) | final det test CSS | + rain QM | cross-fitted diffusion test CSS | rain CRPS | rain SSR |",
           "|---|---|---|---|---|---|---|"]
    for s, job, tag in [(0, "s10s0-oofm", "oofm"), (1, "s10s1-oof", "oof"), (2, "s10s2-oof", "oof")]:
        o = jload(f"{job}/out/{tag}/oof_metrics.json")
        d, g = res(f"s10f_fin_det_s{s}"), res(f"s10f_diff_oof_S_s{s}")
        ga = g["test"]["aggregate"] if g else {}
        md.append(f"| {s}{' **(shipped)**' if s == 0 else ''} | {f(o['all_oof']['css'], 4) if o else PENDING} | "
                  f"{f(css(d), 4) if d else PENDING} | {f(css(d, True), 4) if d else PENDING} | "
                  f"{f(css(g), 4) if g else PENDING} | {f(ga.get('precip_crps'))} | {f(ga.get('precip_ssr'), 2)} |")
    md += ["\nSeed 0 has the best OOF score but the seeds are tied within noise (paired season SE ≈ 0.004–0.006).",
           "The spread across seeds in the test column is the honest uncertainty of any single shipped model.\n"]

    # 3. cross-fitting
    md += ["## 3. Cross-fitted vs in-sample residual diffusion (seed 0, final recipe)\n",
           "| diffusion trained on | test CSS | rain CRPS | rain SSR | rain cov90 | rain bias | CSI30 |", "|---|---|---|---|---|---|---|"]
    for tag, name in [("s10f_diff_ins_S_s0", "in-sample residuals (standard)"), ("s10f_diff_oof_S_s0", "**cross-fitted residuals (shipped)**")]:
        r = res(tag)
        if not r:
            md.append(f"| {name} | {PENDING} |")
            continue
        a = r["test"]["aggregate"]
        md.append(f"| {name} | {css(r):.4f} | {a['precip_crps']:.3f} | {a['precip_ssr']:.2f} | {a['precip_cov90']:.2f} | "
                  f"{a['precip_bias_ratio']:.2f} | {a['precip_csi30']:.3f} |")

    # 4. capacity
    md += ["\n## 4. Capacity under the final recipe (deterministic, H3 N40, 50 ep, 2015–22)\n",
           "| model | params (active) | test CSS | + rain QM | final train loss |", "|---|---|---|---|---|"]
    groups = {"dense S": [f"s10cap_S_s{s}" for s in range(3)], "dense M": [f"s10cap_M_s{s}" for s in range(3)],
              "dense L": [f"s10cap_L_s{s}" for s in range(3)], "MoE S (shipped arch.)": [f"s10f_fin_det_s{s}" for s in range(3)]}
    for name, tags in groups.items():
        cs = [css(r) for r in (res(t) for t in tags) if r]
        if cs:
            sd = f" ± {np.std(cs, ddof=1):.4f}" if len(cs) > 1 else ""
            md.insert(len(md) - 0, f"| **{name}: mean of {len(cs)} seed(s)** | | **{np.mean(cs):.4f}{sd}** | | |")
    for tag, name in [(t, f"{g}, seed {t[-1]}") for g, ts in groups.items() for t in ts]:
        r = res(tag)
        if not r:
            md.append(f"| {name} | | {PENDING} | | |")
            continue
        pa = f"{r['params_total'] / 1e6:.1f}M ({r['params_active'] / 1e6:.1f}M)" if r.get("params_total") else ""
        hist = r.get("history") or []
        tl = f"{hist[-1]['train_loss']:.3f}" if hist and "train_loss" in hist[-1] else ""
        md.append(f"| {name} | {pa} | {css(r):.4f} | {css(r, True):.4f} | {tl} |")
    md += ["\nMoE train losses include the Switch balance term (0.01 × ≈1 per MoE layer × 3 layers ≈ 0.03), i.e. a data loss",
           "of ≈ 0.027. Train loss falls monotonically with capacity (S 0.034 → M 0.029 → L 0.026), so the larger models do",
           "use their capacity, but 3-seed test skill is flat (M's dip is ~1.5 SE: noise). Caveat on the setup: every size",
           "used the same learning rate, weight decay and 50 epochs; larger models may need more regularisation to turn",
           "lower train loss into skill. The MoE has the same mean as dense S but twice the seed variance (±0.014 vs",
           "±0.007), so a single MoE model is less predictable; dense S would be an equally good, simpler backbone."]

    # 4a. diffusion-denoiser capacity
    md += ["\n## 4a. Diffusion-denoiser capacity (cross-fitted residuals, 80 ep, DPM-Solver++ 24 × 8)\n",
           "Selection evidence = 2022 validation (trained 2015–21, seed 1); 2023 test pairs share the OOF file and seed.\n",
           "| denoiser | params | 2022 val CSS | 2022 val rain CRPS | 2023 test CSS (s1 / s2) | 2023 rain CRPS (s1 / s2) | rain SSR (s1 / s2) |",
           "|---|---|---|---|---|---|---|"]
    for sz, tags in (("S", ["s10f_diff_oof_S_s1", "s10f_diff_oof_S_s2"]), ("M", ["s10dcap_M_s1", "s10dcap_M_s2"]),
                     ("L", ["s10dcap_L_s1", "s10dcap_L_s2"])):
        v = res(f"s10dcap_{sz}_val_s1")
        va = (v or {}).get("val", {}) if v else {}
        vcss = composite_skill(va["aggregate"], va["reference_gfs_bilinear"]["aggregate"]) if va.get("aggregate") else None
        rs = [res(t) for t in tags]
        pr = next((r for r in rs + [v] if r and r.get("params_total")), None)
        md.append(f"| {sz} | {pr['params_total'] / 1e6:.1f}M | " if pr else f"| {sz} | | ")
        md[-1] += (f"{f(vcss, 4)} | {f(va.get('aggregate', {}).get('precip_crps'))} | "
                   + " / ".join(f(css(r), 4) if r else PENDING for r in rs) + " | "
                   + " / ".join(f(r['test']['aggregate']['precip_crps']) if r else PENDING for r in rs) + " | "
                   + " / ".join(f(r['test']['aggregate']['precip_ssr'], 2) if r else PENDING for r in rs) + " |")
    s0 = [res(f"s10dcap_{sz}_s0") for sz in ("M", "L")]
    if any(s0):
        md.append("\nSeed-0 runs (record only, finished after the decision): " + "; ".join(
            f"{sz} test CSS {css(r):.4f}, rain CRPS {r['test']['aggregate']['precip_crps']:.3f}, SSR {r['test']['aggregate']['precip_ssr']:.2f}"
            for sz, r in zip("ML", s0) if r) + " (seed-0 S: 0.2814 / 4.431 / 1.11).")
    md += ["\n**Decision: ship the S denoiser.** Bigger denoisers fit the training residuals better (final loss S 0.125,",
           "M 0.113, L 0.105) but sample *narrower* ensembles (2023 rain SSR ≈ 0.6 vs 0.85–1.02 for S): with ~970 training",
           "samples they overfit the residual distribution, the same mechanism that made in-sample residuals under-dispersed.",
           "L leads on 2022 validation (+0.014 CSS, −1 % rain CRPS) but that is one seed and inside the denoiser-to-denoiser",
           "noise shown by M (−0.008 vs S); every 2023 comparison has L's rain CRPS 1.8–2.2 % worse. Pooled over the four",
           "paired comparisons L − S = +0.006 CSS (1.4 SE) and +1.2 % rain CRPS, at ~2.2× the compute (training 179 vs 81 min per run).",
           "Next experiment: spread-calibrate L (its under-dispersion is exactly what the calibration corrects), and",
           "revisit denoiser capacity when more training seasons are available."]

    # 4b. calibration
    c = jload("s10-calfit/out/calf/calibration.json")
    md += ["\n## 4b. Calibration: fitted on 2022 (model trained 2015–21), applied to the shipped model on 2023 (24 steps × 8)\n"]
    if c:

        al = np.array(c["alpha_spread"])
        md += ["Spread factors (mean over leads) P/Tmax/Tmin/RH/U/V: " + " / ".join(f"{x:.2f}" for x in al.mean(0))
               + f" (range {al.min():.1f}–{al.max():.1f}).\n",
               "| model | variant | CSS | rain CRPS | rain SSR | rain cov90 (ideal 0.70 for K=8) | rain bias | Brier>30 | Tmax CRPS |",
               "|---|---|---|---|---|---|---|---|---|"]
        for grp, label in (("model", "2015–21 calibration model, 2023"), ("applied", "shipped model, 2023")):
            for v, m in c.get(grp, {}).items():
                md.append(f"| {label} | {v} | {m['css_ensmean']:.4f} | {m['precip_crps']:.3f} | {m['precip_ssr']:.2f} | "
                          f"{m['precip_cov90']:.2f} | {m['precip_bias_ratio']:.2f} | {m['brier30']:.4f} | {m['tmax_crps']:.3f} |")
        md += ["\nSpread calibration lowers rain CRPS (−3 %) and Tmax CRPS (−9 %) with CSS and bias unchanged, but rain",
               "coverage overshoots (0.84 vs ideal 0.70): slightly over-dispersed for rain. Rain QM on the ensemble is not",
               "shipped: it worsens Brier>30 for both models on 2023 and pushes the calibration model's bias from 1.22 to 1.36."]
    else:
        md.append(PENDING)

    # 5. modes
    modes = {}
    for t in ("modes_a", "modes_b"):
        m = jload(f"s10-modes/out/{t}/modes.json")
        if m:
            modes[t] = m
    rows = [c for m in modes.values() for c in m["configs"]]

    def find(K, S, cal="spread-calibrated"):
        for c in rows:
            if c["members"] == K and c["steps"] == S and c.get("calibration", cal) == cal:
                return c

    md += ["\n## 5. Operating modes (2023 test)\n",
           "| mode | setting | CSS | rain CRPS | rain SSR | Brier>30 | Tmax CRPS | rain bias | latency s/forecast (T4, batch 1) | peak VRAM GB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    fast = [c for c in rows if c["members"] == 1]
    spec = [("FAST", "det + rain QM", next((c for c in fast if "QM" in c["mode"]), None)),
            ("FAST (no QM)", "det only", next((c for c in fast if "raw" in c["mode"]), None)),
            ("BALANCED", "24 steps × 8", find(8, 24)), ("ACCURATE", "32 steps × 8", find(8, 32)),
            ("ENSEMBLE", "24 steps × 16", find(16, 24))]
    for name, setting, c in spec:
        if not c:
            md.append(f"| {name} | {setting} | {PENDING} |")
            continue
        a = c["aggregate"]
        g = (lambda x, d=3: "–") if c["members"] == 1 else f  # probabilistic scores do not exist for one member
        md.append(f"| {name} | {setting} | {c['css']:.4f} | {g(a.get('precip_crps'))} | {g(a.get('precip_ssr'), 2)} | "
                  f"{g(a.get('brier30'), 4)} | {g(a.get('tmax_crps'))} | {a['precip_bias_ratio']:.2f} | "
                  f"{c['latency_s_one_forecast']:.2f} | {f(c.get('peak_vram_gb'), 1) if c.get('peak_vram_gb') else ''} |")
    if rows:
        md += ["\n* Skill rises with test-time compute: FAST < BALANCED ≤ ACCURATE < ENSEMBLE.",
               "* BALANCED steps re-selected on **2022** (fold det trained 2015–20 + diffusion trained 2015–21, K=8, raw;",
               "  rule pre-registered: smallest S within 0.005 CSS and 1 % rain CRPS of the best):"]
        s22 = {}
        for t in ("steps22_a", "steps22_b"):
            m = jload(f"s10-steps22/out/{t}/modes.json")
            for cfg in (m or {}).get("configs", []):
                if cfg.get("calibration") == "raw":
                    s22[cfg["steps"]] = cfg
        if s22:
            md += ["", "  | steps (K=8) | 2022 CSS | 2022 rain CRPS |", "  |---|---|---|"]
            for S in sorted(s22):
                md.append(f"  | {S} | {s22[S]['css']:.4f} | {s22[S]['aggregate']['precip_crps']:.3f} |")
            md += ["", "  16 steps (the Sprint 7/8 choice, made with in-sample residuals) is short of the knee: the wider",
                   "  cross-fitted residuals need ~24 steps. 32 steps adds nothing on 2022 (ACCURATE ≈ BALANCED there);",
                   "  the real accuracy upgrade is ENSEMBLE's extra members."]
        md += ["", "* Spread factors were fitted at K=8; at K=16 rain is somewhat over-dispersed (SSR 1.27–1.33).",
               "* Latency above is with members sampled sequentially (as evaluated). The shipped `predict()` batches all",
               "  members into one pass with the same initial noise (identical samples, max |diff| 5e-6); see the table below.",
               "* Bundle acceptance test: `models/final` FAST mode on CPU fp32 reproduces the T4 result exactly (CSS 0.2435,",
               "  bias 1.09)."]

    bench = jload("s10-bench/out/bench/bench.json")
    if bench:
        md += [f"\nShipped-mode latency, one forecast (batch 1), {bench['device']}, fp16:\n",
               "| mode | benchmarked setting | sequential members s | batched members s (shipped) | peak VRAM GB (shipped) |",
               "|---|---|---|---|---|"]
        for mode in ("FAST", "BALANCED", "ACCURATE", "ENSEMBLE"):
            sq, bt = bench["modes"].get(f"{mode} (sequential)"), bench["modes"].get(f"{mode} (batched)")
            shp = bt or sq
            st = f"{sq['steps']} × {sq['members']}" if sq["members"] > 1 else "1 member"
            md.append(f"| {mode} | {st} | {sq['latency_s']:.2f} | {bt['latency_s']:.2f} | {shp['peak_vram_gb']:.2f} |" if bt else
                      f"| {mode} | {st} | {sq['latency_s']:.3f} | – | {shp['peak_vram_gb']:.2f} |")
        acc = bench["modes"].get("ACCURATE (batched)")
        md.append("\nBatching gives only 1.3–1.4×: at 80×80 the denoiser already keeps a T4 fairly busy at batch 1. The bench ran"
                  " before BALANCED moved to 24 steps; latency is linear in steps, so shipped BALANCED (24 × 8, batched) ≈ "
                  + (f"{acc['latency_s'] * 24 / 32:.1f} s." if acc else "n/a."))

    # 5b. full accuracy / extremes / spatial structure / probabilistic quality per variable
    md += ["\n## 5b. Per-variable accuracy, extremes, spatial structure and probabilistic quality (2023 test)\n"]
    shown = [(n, c) for n, _, c in spec if c]
    if shown:
        A = lambda key, d=3: [(f"{c['aggregate'][key]:.{d}f}" if key in c["aggregate"] else "–") for _, c in shown]
        groups = [
            ("**Accuracy** (deterministic or ensemble mean)", [
                ("rain MAE / RMSE (mm/day)", None), ("Tmax MAE / RMSE / bias (°C)", None), ("Tmin MAE / RMSE / bias (°C)", None),
                ("RH MAE / RMSE / bias (%)", None), ("wind vector RMSE (m/s)", "wind_vec_rmse"), ("rain bias ratio", "precip_bias_ratio"),
                ("rain wet-day MAE (mm/day)", "precip_wet_mae")]),
            ("**Precipitation events and extremes**", [
                ("CSI ≥ 5 / 15 / 30 / 64.5 mm", None), ("POD / FAR ≥ 64.5 mm (very heavy)", None)]),
            ("**Spatial structure**", [("rain FSS (15 mm)", "precip_fss15"), ("rain spatial correlation", "precip_spatial_corr")]),
            ("**Probabilistic quality** (diffusion modes)", [
                ("CRPS rain / Tmax / Tmin / RH", None), ("spread-skill ratio rain / Tmax / RH (ideal 1)", None),
                ("90 % coverage rain / Tmax (ideal 0.70 at K=8, 0.79 at K=16)", None), ("Brier ≥ 15 / ≥ 30 mm", None)])]
        combo = {
            "rain MAE / RMSE (mm/day)": ("precip_mae", "precip_rmse"), "Tmax MAE / RMSE / bias (°C)": ("tmax_mae", "tmax_rmse", "tmax_bias"),
            "Tmin MAE / RMSE / bias (°C)": ("tmin_mae", "tmin_rmse", "tmin_bias"), "RH MAE / RMSE / bias (%)": ("rh_mae", "rh_rmse", "rh_bias"),
            "CSI ≥ 5 / 15 / 30 / 64.5 mm": ("precip_csi5", "precip_csi15", "precip_csi30", "precip_csi64.5"),
            "POD / FAR ≥ 64.5 mm (very heavy)": ("precip_pod64.5", "precip_far64.5"),
            "CRPS rain / Tmax / Tmin / RH": ("precip_crps", "tmax_crps", "tmin_crps", "rh_crps"),
            "spread-skill ratio rain / Tmax / RH (ideal 1)": ("precip_ssr", "tmax_ssr", "rh_ssr"),
            "90 % coverage rain / Tmax (ideal 0.70 at K=8, 0.79 at K=16)": ("precip_cov90", "tmax_cov90"),
            "Brier ≥ 15 / ≥ 30 mm": ("brier15", "brier30")}
        md += ["| metric | " + " | ".join(n for n, _ in shown) + " |", "|---|" + "---|" * len(shown)]
        for title, items in groups:
            md.append(f"| {title} |" + " |" * len(shown))
            for label, key in items:
                if key:
                    md.append(f"| {label} | " + " | ".join(A(key)) + " |")
                else:
                    cols = [A(k, 4 if "brier" in k else (2 if any(s in k for s in ("ssr", "cov90", "csi", "pod", "far")) else 3)) for k in combo[label]]
                    md.append(f"| {label} | " + " | ".join(" / ".join(col[i] for col in cols) for i in range(len(shown))) + " |")
        md += ["\n* **Very heavy rain (≥ 64.5 mm): use FAST + rain QM for a yes/no warning.** It detects 31 % of events (CSI",
               "  0.18) vs ~10 % (CSI 0.07–0.09) for the ensemble *mean* of the diffusion modes, which averages extremes away.",
               "  The diffusion modes should instead be used through their exceedance probabilities (`prob[\"precip>64.5\"]`),",
               "  where they are better calibrated (Brier ≥ 30 mm 0.037–0.039).",
               "* Everywhere else the diffusion modes improve the mean forecast too: rain RMSE, Tmax/Tmin/RH MAE, wind",
               "  RMSE, FSS (0.50 → 0.57) and spatial correlation all improve over FAST.",
               "* Spread after calibration: rain is somewhat over-dispersed (SSR ~1.25), Tmax and RH are still under-dispersed",
               "  (SSR 0.78–0.87, Tmax 90 % coverage 0.60–0.68 vs ideal 0.70–0.79). The 2022-fitted factors transfer only",
               "  partly to 2023; per-variable factors fitted on more seasons are the fix."]

    # 6. test-time compute matrix
    md += ["\n## 6. Test-time compute matrix (spread-calibrated; cell = CSS / rain CRPS / latency s)\n"]
    Ks = sorted({c["members"] for c in rows if c["members"] > 1})
    Ss = sorted({c["steps"] for c in rows if c["members"] > 1})
    if Ks:
        md += ["| members \\ steps | " + " | ".join(str(s) for s in Ss) + " |", "|---|" + "---|" * len(Ss)]
        for K in Ks:
            cells = []
            for S in Ss:
                c = find(K, S)
                cells.append(f"{c['css']:.4f} / {c['aggregate']['precip_crps']:.3f} / {c['latency_s_one_forecast']:.1f}" if c else PENDING)
            md.append(f"| K={K} | " + " | ".join(cells) + " |")
        md += ["\nRaw (uncalibrated) vs spread-calibrated rain CRPS / SSR:\n",
               "| K | S | raw CRPS | cal CRPS | raw SSR | cal SSR |", "|---|---|---|---|---|---|"]
        for K in Ks:
            for S in Ss:
                r0, r1 = find(K, S, "raw"), find(K, S)
                if r0 and r1:
                    md.append(f"| {K} | {S} | {r0['aggregate']['precip_crps']:.3f} | {r1['aggregate']['precip_crps']:.3f} | "
                              f"{r0['aggregate'].get('precip_ssr', float('nan')):.2f} | {r1['aggregate'].get('precip_ssr', float('nan')):.2f} |")
    else:
        md.append(PENDING)

    # 7. consistency
    md += ["\n## 7. Multivariate physical consistency (per member; observed = ERA5-Land/CHIRPS targets)\n"]
    obs = next((m["observed_consistency"] for m in modes.values()), None)
    if obs:
        keys = list(obs)
        md += ["| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys),
               "| observed | " + " | ".join(f"{obs[k]:.3f}" for k in keys) + " |"]
        for name, _, c in spec:
            if c and "consistency" in c:
                md.append(f"| {name} | " + " | ".join(f"{c['consistency']['model'][k]:.3f}" for k in keys) + " |")
    else:
        md.append(PENDING)
    fm = next((m.get("fast_vs_oof_file_maxdiff") for m in modes.values() if "fast_vs_oof_file_maxdiff" in m), None)
    if fm is not None:
        md.append(f"\nArtefact check: FAST output vs the OOF file's 2023 rows, max |diff| = {fm:.4f} (normalised units, fp16).")
        md += ["Mean |diff| is ~6e-4 and only 0.04 % of values differ by > 0.05; the rare large local differences come from",
               "MoE top-2 routing flips under reduced precision (0.3–0.7 % of tokens switch expert between fp32 and bf16).",
               "Aggregate scores are unaffected (identical CSS to 4 decimals)."]
    md += ["\n## 8. Headline numbers and caveats\n",
           "* **Expected skill of the pipeline** (3 seeds, cross-fitted diffusion, 24 × 8): test CSS "
           + (lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}" if len(v) == 3 else PENDING)(
               [css(r) for r in (res(f"s10f_diff_oof_S_s{s}") for s in range(3)) if r])
           + ". The shipped seed (0) scores above this mean; it was chosen on out-of-fold skill before test was seen,",
           "  so its higher test score is luck, not selection.",
           "* Seed variance is dominated by heavy-rain detection (CSI30 0.13–0.19 across seeds). Averaging the three seeds'",
           "  deterministic backbones is the obvious next upgrade (3× stage-1 cost, ~0.2 s/forecast).",
           "* One test season (2023, 122 forecasts) and 8 training seasons: differences under ~0.01 CSS are noise."]

    # 9. compute statistics + reproducibility
    walls = {}
    for js in glob.glob(str(REPO / "results" / "*" / "out" / "job_status.json")):
        job = Path(js).parents[1].name
        walls[job] = json.load(open(js)).get("wall_min", 0) / 60
    s10 = {k: v for k, v in walls.items() if k.startswith("s10")}
    md += ["\n## 9. Compute, data and reproducibility\n",
           f"* Kaggle T4×2 session-hours (wall clock of each job's session): all v3 jobs {sum(walls.values()):.1f} h over "
           f"{len(walls)} jobs; Sprint 10 alone {sum(s10.values()):.1f} h over {len(s10)} jobs (4 team accounts, ≤ 2 sessions each).",
           "* Dataset: `multitask_temporal_v3_realgfs.zarr` (real GFS 0.25° 00Z 2015–2023, ERA5 / ERA5-Land, CHIRPS; 1096",
           "  monsoon initialisations; splits 2015–22 train, 2023 test; 2022 or 2021 held out for selection).",
           "* Experiment IDs = job names in `kaggle/queue.json` / `kaggle/registry.json`; run tags in `results/<job>/out/<tag>/`.",
           "* Every decision with its time and evidence: `docs/decision_log.md`. Scaling matrix: `results/final_scaling_matrix.csv`.",
           "* Checkpoints: `models/final/{det,diff}.pt` (186 MB, outside git); rebuild with `scripts/export_final.py` from the",
           "  Kaggle outputs of `s10f-det-b` and `s10f-diff` (account rohitajitbharadwaj)."]

    text = "\n".join(md) + "\n"
    (REPO / "docs" / "final_system_report.md").write_text(text, encoding="utf-8")
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "final_system_report.md").write_text(text, encoding="utf-8")
    write_matrix(rows, bench)
    print(text)


def write_matrix(mode_rows, bench):
    """results/final_scaling_matrix.csv: one row per evaluated operating point on 2023 (capacity x denoising steps x
    ensemble size x quality x cost)."""
    import csv
    out = []

    def add(stage, model, size, seed, r=None, steps=0, members=1, calib="", agg=None, css_=None, lat=None, vram=None, train_min=None):
        a = agg if agg is not None else (r["test"]["aggregate"] if r else {})
        out.append({"stage": stage, "model": model, "size": size, "seed": seed,
                    "params_total_M": round(r["params_total"] / 1e6, 2) if r and r.get("params_total") else "",
                    "params_active_M": round(r["params_active"] / 1e6, 2) if r and r.get("params_active") else "",
                    "steps": steps, "members": members, "calibration": calib,
                    "css": round(css_ if css_ is not None else css(r), 4),
                    **{k: (round(a[k], 4) if k in a else "") for k in (
                        "precip_crps", "precip_ssr", "precip_cov90", "brier30", "precip_bias_ratio", "precip_csi30",
                        "precip_fss15", "tmax_mae", "tmax_crps", "rh_mae", "wind_vec_rmse")},
                    "latency_s_one_forecast": round(lat, 3) if lat is not None else "",
                    "peak_vram_gb": round(vram, 2) if vram else "", "train_min": round(train_min, 1) if train_min else ""})
    for sz in "SML":
        for s in range(3):
            r = res(f"s10cap_{sz}_s{s}")
            if r:
                add("deterministic", "dense", sz, s, r, train_min=r.get("train_minutes"))
    for s in range(3):
        r = res(f"s10f_fin_det_s{s}")
        if r:
            add("deterministic", "MoE E8 top-2 50%", "S", s, r, train_min=r.get("train_minutes"))
    for sz, tags in (("S", [f"s10f_diff_oof_S_s{s}" for s in range(3)]), ("M", [f"s10dcap_M_s{s}" for s in range(3)]),
                     ("L", [f"s10dcap_L_s{s}" for s in range(3)])):
        for t in tags:
            r = res(t)
            if r:
                add("diffusion denoiser", "cross-fitted residual diffusion", sz, int(t[-1]), r, steps=24, members=8,
                    calib="raw", train_min=r.get("train_minutes"))
    for c in mode_rows:
        add("operating point (shipped seed 0)", c["mode"], "S", 0, None, steps=c["steps"], members=c["members"],
            calib=c.get("calibration", "rain QM" if "QM" in c["mode"] else "raw"), agg=c["aggregate"], css_=c["css"],
            lat=c.get("latency_s_one_forecast"), vram=c.get("peak_vram_gb"))
    for k, b in (bench or {}).get("modes", {}).items():
        out.append({"stage": "latency bench (T4, batch 1)", "model": k, "size": "S", "seed": 0, "steps": b["steps"],
                    "members": b["members"], "latency_s_one_forecast": round(b["latency_s"], 3),
                    "peak_vram_gb": round(b["peak_vram_gb"], 2) if b.get("peak_vram_gb") else ""})
    keys = list(out[0].keys())
    with open(REPO / "results" / "final_scaling_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in out:
            w.writerow({k: row.get(k, "") for k in keys})


if __name__ == "__main__":
    main()
