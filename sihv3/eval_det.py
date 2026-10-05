"""
Re-evaluate finished deterministic checkpoints: raw vs post-hoc precipitation quantile mapping
(fitted on train), val + test.   python -m sihv3.eval_det --ckpts '**/s4_h14_n16_s0/best.pt' ... --tag e1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sihv3.data import V3Data
from sihv3.train import apply_precip_qm, build, det_predict, evaluate, fit_precip_qm, resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--tag", default="eval_det")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    res, data, key = {}, None, None
    for c in a.ckpts:
        ck = torch.load(resolve(c), map_location=dev, weights_only=False)
        ar = argparse.Namespace(**ck["args"])
        if key != (ar.H, ar.N):
            data, key = V3Data(history_len=ar.H, context=ar.N), (ar.H, ar.N)
        m = build(ar, diffusion=False, size=ar.size).to(dev)
        m.load_state_dict(ck["ema"])
        m.eval()

        def pf(b):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                return det_predict(m, b)[None].float()
        qm = fit_precip_qm(data, pf, dev)
        r = {}
        for split in ("val", "test"):
            raw = evaluate(data, split, pf, dev)
            cal = evaluate(data, split, pf, dev, post=lambda P: apply_precip_qm(P, qm))
            r[split] = {"raw_css": raw["css"], "qm_css": cal["css"], "raw": raw["aggregate"], "qm": cal["aggregate"]}
            print(f"[{ar.tag}] {split}: CSS raw {raw['css']:.4f} -> +precipQM {cal['css']:.4f} | bias {raw['aggregate']['precip_bias_ratio']:.2f}"
                  f" -> {cal['aggregate']['precip_bias_ratio']:.2f} | csi15 {raw['aggregate']['precip_csi15']:.3f} -> {cal['aggregate']['precip_csi15']:.3f}"
                  f" | csi30 {raw['aggregate']['precip_csi30']:.3f} -> {cal['aggregate']['precip_csi30']:.3f}"
                  f" | wetMAE {raw['aggregate']['precip_wet_mae']:.2f} -> {cal['aggregate']['precip_wet_mae']:.2f}", flush=True)
        res[ar.tag] = r
        json.dump(res, open(out / "eval_det.json", "w"), indent=1, default=float)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
