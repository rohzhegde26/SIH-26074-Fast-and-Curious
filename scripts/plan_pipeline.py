"""Generate the stage-1 jobs of the final pipeline (3 year-folds + final model + out-of-fold file) for a winning
configuration, one account per seed, audited against the shipped recipe, and append them to kaggle/inbox.json.

  python scripts/plan_pipeline.py --prefix s12 --flags "--H 5 --fc_history" --seeds 0 1 2 --accounts a2 a3 a4 [--dry]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import sihv3.train as T  # noqa: E402

BASE = ("--mode det --size S --epochs 50 --patience 15 --time_budget_min 90 --N 40 --moe_experts 8 --moe_topk 2 "
        "--moe_frac 0.5")
FOLDS = {"f0": ("2018-2022", "2015-2017"), "f1": ("2015-2017,2021,2022", "2018-2020"), "f2": ("2015-2020", "2021-2022")}
SHIPPED = BASE + " --H 3 --train_years 2015-2022 --seed 0 --tag x"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", required=True)
    p.add_argument("--flags", required=True, help="the winning change, e.g. '--H 5 --fc_history'")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--accounts", nargs="+", required=True)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    assert len(a.accounts) == len(a.seeds)
    flags = a.flags if "--H" in a.flags else a.flags + " --H 3"
    det = lambda s, tr, tag: f"{BASE} {flags} --train_years {tr} --seed {s} --tag {a.prefix}_{tag}_s{s}"
    jobs = []
    for s, acct in zip(a.seeds, a.accounts):
        n = lambda k: f"{a.prefix}s{s}-{k}"
        jobs += [
            {"name": n("det-a"), "account": acct, "expected_min": 65, "needs": [],
             "lanes": [[det(s, FOLDS["f0"][0], "cf_f0")], [det(s, FOLDS["f1"][0], "cf_f1")]]},
            {"name": n("det-b"), "account": acct, "expected_min": 70, "needs": [],
             "lanes": [[det(s, FOLDS["f2"][0], "cf_f2")], [det(s, "2015-2022", "fin_det")]]},
            {"name": n("oof"), "account": acct, "expected_min": 15, "needs": [n("det-a"), n("det-b")],
             "kernel_sources": [f"sih26074-v3-{n('det-a')}", f"sih26074-v3-{n('det-b')}"],
             "lanes": [[f"@oof --mode oof --base **/{a.prefix}_fin_det_s{s}/best.pt "
                        + " ".join(f"--folds **/{a.prefix}_cf_{f}_s{s}/best.pt={FOLDS[f][1]}" if i == 0 else
                                   f"**/{a.prefix}_cf_{f}_s{s}/best.pt={FOLDS[f][1]}" for i, f in enumerate(FOLDS))
                        + " --tag oof"]]},
        ]
    for j in jobs:
        j["sprint"] = f"{a.prefix} pipeline ({a.flags})"
    # audit: every training command parses and differs from the shipped recipe only where intended
    sys.argv = ["x"] + SHIPPED.split()
    ref = vars(T.parse())
    allowed = {"H", "fc_history", "seed", "train_years", "tag"} | {k.lstrip("-") for k in a.flags.split() if k.startswith("--")}
    for j in jobs:
        assert all(j["lanes"]), j["name"]
        for lane in j["lanes"]:
            for c in lane:
                if c.startswith("@"):
                    continue
                sys.argv = ["x"] + c.split()
                v = vars(T.parse())
                d = {k: v[k] for k in v if v[k] != ref[k]}
                assert set(d) <= allowed, (j["name"], d)
    # fold coverage: every training season is held out by exactly one fold
    held = sorted(sum((T.parse_years(FOLDS[f][1]) for f in FOLDS), []))
    assert held == list(range(2015, 2023)), held
    for f, (tr, ho) in FOLDS.items():
        assert not set(T.parse_years(tr)) & set(T.parse_years(ho)), f
    print(json.dumps(jobs, indent=1) if a.dry else f"{len(jobs)} jobs audited")
    if not a.dry:
        json.dump([{"append": j} for j in jobs], open(REPO / "kaggle" / "inbox.json", "w"), indent=1)


if __name__ == "__main__":
    main()
