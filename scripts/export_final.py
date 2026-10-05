"""Export the shipped SIH-26074 system into models/final.

  python scripts/export_final.py --det ckpts/.../s10f_fin_det_s0/best.pt --diff ckpts/.../s10f_diff_oof_S_s0/best.pt \
      --alpha_json results/s10-calfit/out/calf/calibration.json \
      [--extra_dets <seed-1 final det> <seed-2 final det> --fast_oof ckpts/ds_oof_avg3/oof_avg3.npz] \
      [--diff_backbone avg]

FAST uses the average of all bundled seeds when --extra_dets is given; its rain quantile mapping is then fitted on
--fast_oof (the matching averaged out-of-fold predictions, 2015-2022). Without --extra_dets the single-seed QM from
--cal (sihv3.modes calibration_params.npz) is used, as in the Sprint 10 bundle.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihv3.data import V3Data, find  # noqa: E402
from sihv3.modes import fit_qm_arrays  # noqa: E402
from sihv3.post import tail_cap  # noqa: E402
from sihv3.predict import export_bundle  # noqa: E402

MODES = {  # pre-registered in docs/decision_log.md before the 2023 modes grid was seen
    "FAST": {"members": 1, "rain_qm": True},
    "BALANCED": {"steps": 24, "members": 8, "spread_calibration": True},  # 16 -> 24: re-selected on 2022 (decision_log)
    "ACCURATE": {"steps": 32, "members": 8, "spread_calibration": True},
    "ENSEMBLE": {"steps": 24, "members": 16, "spread_calibration": True},
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--det", required=True)
    p.add_argument("--diff", required=True)
    p.add_argument("--alpha_json", default=None, help="calibration.json from sihv3.calibrate")
    p.add_argument("--cal", default=None, help="calibration_params.npz from sihv3.modes (alpha + single-seed QM)")
    p.add_argument("--extra_dets", nargs="*", default=[])
    p.add_argument("--fast_oof", default=None, help="averaged OOF file matching det + extra_dets (for the FAST QM)")
    p.add_argument("--diff_backbone", choices=["single", "avg"], default="single")
    p.add_argument("--out", default="models/final")
    a = p.parse_args()
    data = V3Data(history_len=3, context=40)
    if a.alpha_json:
        alpha = np.array(json.load(open(a.alpha_json))["alpha_spread"], np.float32).ravel()
    else:
        alpha = np.load(a.cal)["alpha"]
    modes = json.loads(json.dumps(MODES))
    if a.extra_dets:
        assert a.fast_oof, "averaged FAST needs the matching averaged OOF file for its rain QM"
        z = np.load(a.fast_oof)
        assert list(z["dates"].astype(str)) == list(data.dates) and str(z["kind"]) == "oof"
        years = np.array([int(d[:4]) for d in data.dates])
        fit = np.where(years != 2023)[0]
        qm = fit_qm_arrays(data.norm.inv(z["ydet"][fit].astype(np.float32), axis=2), data.norm.inv(data.targ[fit], axis=2),
                           data.mask[fit])
        modes["FAST"]["backbone"] = "avg"
    else:
        c = np.load(a.cal)
        qm = (c["qm_pq"], c["qm_oq"])
    for m in ("BALANCED", "ACCURATE", "ENSEMBLE"):
        modes[m]["backbone"] = a.diff_backbone
        modes[m]["rain_cap"] = True  # adopted in Sprint 11 (decision_log)
    yrs = np.array([int(d[:4]) for d in data.dates])
    tr = np.where(yrs != 2023)[0]
    cap = tail_cap(data.norm.inv(data.targ[tr], axis=2), data.mask[tr])  # training seasons 2015-2022 only
    export_bundle(a.out, a.det, a.diff, alpha, qm, data.static, find("normalization_stats_v3.yaml"), modes,
                  extra_dets=a.extra_dets, rain_cap=cap)
    print("bundle written to", a.out, json.dumps(modes))
