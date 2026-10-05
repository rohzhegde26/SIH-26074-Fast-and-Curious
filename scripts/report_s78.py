"""
Sprint 7 (steps x sampler) and Sprint 8 (members vs steps at matched NFE) tables from
results/d-*/out/*/rows.json  ->  docs/results_s7_s8.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def rows(study):
    out = []
    for f in glob.glob(str(REPO / "results" / f"d-{study}-*" / "out" / f"{study}_p*" / "rows.json")):
        out += json.load(open(f))
    return out


def fmt(r):
    a = r["aggregate"]
    bins = r.get("precip_by_intensity", {})
    heavy = bins.get("30-64.5", {}).get("crps")
    return (f"{r['css_ensmean']:.4f} | {a.get('precip_crps', float('nan')):.3f} | {a.get('precip_ssr', float('nan')):.2f} | "
            f"{a.get('precip_cov90', float('nan')):.2f} (ideal {0.9 * (r['members'] - 1) / (r['members'] + 1):.2f}) | "
            f"{a.get('brier30', float('nan')):.4f} | "
            f"{heavy if heavy is None else round(heavy, 2)} | {a.get('tmax_crps', float('nan')):.3f} | "
            f"{r['sec_per_sample']:.2f} | {r['peak_vram_gb'] or float('nan'):.1f}")


HDR = ("ens-mean CSS | precip CRPS | precip SSR | precip cov90 (ideal for K members) | Brier>30 | CRPS obs 30-64.5 mm | Tmax CRPS | "
       "s/sample | VRAM GB")


def main():
    md = ["# Sprint 7-8 results (validation 2022, residual diffusion on the deterministic transformer)\n"]
    s7 = sorted(rows("s7"), key=lambda r: (r["sampler"], r["steps"]))
    md += ["## Sprint 7 — sampler x denoising steps (K = 8 members)\n",
           f"| sampler | steps | NFE | {HDR} |", "|" + "---|" * 12]
    md += [f"| {r['sampler']} | {r['steps']} | {r['nfe']} | {fmt(r)} |" for r in s7]
    s8 = sorted(rows("s8"), key=lambda r: (r["nfe"], r["members"]))
    md += ["\n## Sprint 8 — ensemble members vs steps at matched NFE (DDIM)\n",
           f"| NFE | members K | steps S | {HDR} |", "|" + "---|" * 12]
    md += [f"| {r['nfe']} | {r['members']} | {r['steps']} | {fmt(r)} |" for r in s8]
    # per-lead CRPS for the best S8 configuration at each budget
    for nfe in sorted({r["nfe"] for r in s8}):
        best = min((r for r in s8 if r["nfe"] == nfe), key=lambda r: r["aggregate"].get("precip_crps", 9e9))
        md.append(f"\nNFE {nfe}: best precip CRPS with K={best['members']}, S={best['steps']}; per-lead CRPS "
                  + ", ".join(f"D+{i}: {p.get('precip_crps', float('nan')):.2f}" for i, p in enumerate(best["per_lead"])))
    (REPO / "docs" / "results_s7_s8.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
