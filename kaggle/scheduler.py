"""
Autonomous overnight scheduler.

* kaggle/queue.json holds job specs {name, account|None, lanes, expected_min, sprint, kernel_sources, needs}.
* Every 2 min: refresh job states, download finished results, retry failed jobs once, and launch pending
  jobs whose dependencies are complete onto accounts with a free slot (MAX_PER_ACCOUNT sessions each).
* Planner hooks append dependent phases automatically:
    - Phase A complete -> pick H*, N* (scripts/summarize.py rules) -> Sprint 6 (det + diffusion) and
      Sprint 9 (dense M/L, MoE grid) at (H*, N*)
    - Sprint 6 diffusion complete -> Sprint 7/8 evaluation jobs
* Every action is appended to kaggle/scheduler.log and decisions to docs/decision_log.md.
Run:  python kaggle/scheduler.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dashboard import FINAL, STATUS, load, refresh, render  # noqa: E402
from launch import REGISTRY, REPO, USERS  # noqa: E402

QUEUE = REPO / "kaggle" / "queue.json"
LOG = REPO / "kaggle" / "scheduler.log"
DLOG = REPO / "docs" / "decision_log.md"
STATE = REPO / "kaggle" / "scheduler_state.json"
INBOX = REPO / "kaggle" / "inbox.json"
HEART = REPO / "kaggle" / "scheduler_heartbeat.json"
MAX_PER_ACCOUNT = 2
IST = timezone(timedelta(hours=5, minutes=30))
S_ARGS = "--mode det --size S --epochs 50 --time_budget_min 55 --patience 10"


def now():
    return datetime.now(IST).strftime("%H:%M")


def log(msg):
    line = f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def decide(msg, why):
    with open(DLOG, "a", encoding="utf-8") as f:
        f.write(f"| {now()} | {msg} | {why} |\n")
    log(f"DECISION {msg} -- {why}")


def save(q):
    QUEUE.write_text(json.dumps(q, indent=1))


def job_state(name, status, reg):
    """latest registry entry for this job name -> (status, entry)"""
    es = [e for e in reg if e["name"] == name]
    if not es:
        return None, None
    e = max(es, key=lambda x: x["launched_utc"])
    st = status.get(f"{e['name']}@{e['launched_utc']:.0f}", {}).get("status", e["status"])
    return st, e


def tag_owner(tag, reg):
    """(account, slug) of the job whose lanes produced run `tag`."""
    for e in sorted(reg, key=lambda x: -x["launched_utc"]):
        if any(f"--tag {tag}" in c for lane in e["lanes"] for c in lane):
            return e["account"], e["slug"]
    return None, None


def launch(job):
    cmd = [sys.executable, str(REPO / "kaggle" / "launch.py"), "--account", job["account"], "--name", job["name"],
           "--expected_min", str(job["expected_min"]), "--sprint", job.get("sprint", "")]
    for lane in job["lanes"]:
        cmd += ["--lane", *lane]
    if job.get("kernel_sources"):
        cmd += ["--kernel_sources", *job["kernel_sources"]]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    ok = r.returncode == 0
    log(f"LAUNCH {job['name']} on {job['account']} -> {'ok' if ok else 'FAILED ' + (r.stdout + r.stderr)[-300:]}")
    return ok


def summarize():
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize.py")], capture_output=True, text=True,
                       cwd=REPO, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    m = re.search(r"WINNERS (\{.*\})", r.stdout)
    k = re.search(r"RANKINGS (\{.*\})", r.stdout)
    w = eval(m.group(1)) if m else {}
    w["_rankings"] = eval(k.group(1)) if k else {}
    return w


# ------------------------------------------------------------------------------- planners
def plan_phase_bc(q, reg, st):
    w = summarize()
    h_cfg, n_cfg, l_cfg = w.get("Sprint 4"), w.get("Sprint 5"), w.get("Sprint 3")
    H = int(re.search(r"_h(\d+)", h_cfg).group(1)) if h_cfg else 7
    N = int(re.search(r"_n(\d+)", n_cfg).group(1)) if n_cfg else 16
    lin = True  # fixed at 22:15 (see decision log): mm-loss beat plain loss on CSS (+0.005) and CSI@30 (+0.03)
    extra = " --precip_lin_w 1.0" if lin else ""
    decide(f"H* = {H}, N* = {N} (N/M = {N / 16:.2f}), precip mm-loss = {lin}",
           f"Sprint 4 winner {h_cfg}, Sprint 5 winner {n_cfg}, Sprint 3 winner {l_cfg} (seed-mean val CSS, ties to cheaper)")
    st.update(H=H, N=N, lin=lin)
    base = f"--H {H} --N {N}{extra}"
    # Sprint 6 + Sprint 9 "S" rung: fresh deterministic S at (H*, N*) on the long schedule (Phase A runs were
    # 50 epochs and still improving), seeds 0/1; the diffusion stage builds on seed 0 (same account).
    LONG = "--epochs 100 --patience 15"
    det_tag, det_acct, det_job = "s9_dense_S_s0", "a1", "b-s9-S"
    q.append({"name": det_job, "account": det_acct, "sprint": "S6/S9", "expected_min": 110, "needs": [],
              "lanes": [[f"--mode det --size S {LONG} --time_budget_min 95 {base} --seed 0 --tag s9_dense_S_s0"],
                        [f"--mode det --size S {LONG} --time_budget_min 95 {base} --seed 1 --tag s9_dense_S_s1"]]})
    D = f"--mode diff --size S --epochs 60 --patience 10 --time_budget_min 110 {base} --det_ckpt **/{det_tag}/best.pt"
    q.append({"name": "b-s6-diff", "account": det_acct, "sprint": "S6", "expected_min": 150, "needs": [det_job],
              "kernel_sources": ["sih26074-v3-" + det_job],
              "lanes": [[f"{D} --seed 0 --tag s6_diff_s0"], [f"{D} --seed 1 --tag s6_diff_s1"]]})
    M = f"--mode det --size M {LONG} --time_budget_min 115 {base}"
    L_ = f"--mode det --size L {LONG} --time_budget_min 160 {base}"
    q.append({"name": "c-s9-M", "account": None, "sprint": "S9", "expected_min": 130, "needs": [],
              "lanes": [[f"{M} --seed 0 --tag s9_dense_M_s0"], [f"{M} --seed 1 --tag s9_dense_M_s1"]]})
    q.append({"name": "c-s9-L", "account": None, "sprint": "S9", "expected_min": 175, "needs": [],
              "lanes": [[f"{L_} --seed 0 --tag s9_dense_L_s0"], [f"{L_} --seed 1 --tag s9_dense_L_s1"]]})
    # Confirmation: runner-up H (at N*) and runner-up N (at H*) on the same long schedule, so the Phase A
    # choice (50 epochs, all runs still improving) is checked under converged training.
    rk = w.get("_rankings", {})
    alt = []
    h2 = next((c for c in rk.get("Sprint 4", []) if c != h_cfg), None)
    n2 = next((c for c in rk.get("Sprint 5", []) if c != n_cfg), None)
    if h2:
        alt.append(("H", int(re.search(r"_h(\d+)", h2).group(1)), N))
    if n2:
        alt.append(("N", H, int(re.search(r"_n(\d+)", n2).group(1))))
    # always give the long-history / wide-context hypotheses a fair (converged) test as well
    alt += [("Hmax", 14, N), ("Nmax", H, 40), ("Hmax+Nmax", 14, 40)]
    seen = {(H, N)}
    for kind, hh, nn in alt:
        if (hh, nn) in seen:
            continue
        seen.add((hh, nn))
        cb = f"--H {hh} --N {nn}{extra}"
        tg = f"s45_confirm_h{hh}_n{nn}"
        if any(j.get("name") == f"c-confirm-h{hh}-n{nn}" for j in q):  # pre-queued already
            continue
        q.append({"name": f"c-confirm-h{hh}-n{nn}", "account": None, "sprint": "S4/5 confirm", "expected_min": 140, "needs": [],
                  "lanes": [[f"--mode det --size S {LONG} --time_budget_min 130 {cb} --seed 0 --tag {tg}_s0"],
                            [f"--mode det --size S {LONG} --time_budget_min 130 {cb} --seed 1 --tag {tg}_s1"]]})
        decide(f"Long-schedule confirmation run for runner-up {kind}: H={hh}, N={nn}", "Phase A runs were capped at 50 epochs while still improving")
    moe = [(4, 1, 0.5), (8, 1, 0.5), (16, 1, 0.5), (4, 2, 0.5), (8, 2, 0.5), (16, 2, 0.5), (8, 1, 0.25), (8, 1, 1.0)]
    for e, k, fr in moe:
        tag = f"s9_moe_e{e}k{k}f{int(fr * 100)}"
        A = f"--mode det --size S {LONG} --time_budget_min 95 {base} --moe_experts {e} --moe_topk {k} --moe_frac {fr}"
        q.append({"name": f"c-{tag.replace('_', '-')}", "account": None, "sprint": "S9", "expected_min": 110, "needs": [],
                  "lanes": [[f"{A} --seed 0 --tag {tag}_s0"], [f"{A} --seed 1 --tag {tag}_s1"]]})
    q.append({"planner": "phase_d", "needs": ["b-s6-diff"]})
    log(f"PLANNED Phase B/C: {sum(1 for j in q if j.get('name', '').startswith(('b-', 'c-')))} jobs")


def plan_phase_d(q, reg, st):
    acct = next(j["account"] for j in q if j.get("name") == "b-s6-diff")
    det_tag = "s9_dense_S_s0"
    srcs = ["sih26074-v3-b-s9-S", "sih26074-v3-b-s6-diff"]
    E = f"--diff_ckpt **/s6_diff_s0/best.pt --det_ckpt **/{det_tag}/best.pt"
    for study, parts, exp in (("s7", 4, 110), ("s8", 2, 110)):
        for p in range(0, parts, 2):
            q.append({"name": f"d-{study}-p{p}", "account": acct, "sprint": study.upper(), "expected_min": exp,
                      "needs": [], "kernel_sources": srcs, "module": "eval_diff",
                      "lanes": [[f"{E} --study {study} --part {p} --nparts {parts} --tag {study}_p{p}"],
                                [f"{E} --study {study} --part {p + 1} --nparts {parts} --tag {study}_p{p + 1}"]]})
    decide(f"Sprint 7/8 evaluation on s6_diff_s0 (det {det_tag})", "diffusion seed 0 trained; seed 1 kept for training-noise check")
    log("PLANNED Phase D")


PLANNERS = {"phase_bc": plan_phase_bc, "phase_d": plan_phase_d}


# ------------------------------------------------------------------------------- main loop
def main():
    st = load(STATE, {})
    status = load(STATUS, {})
    quota = {}
    log("scheduler started")
    while True:
        try:
            HEART.write_text(json.dumps({"phase": "working", "t": time.time()}))
            refresh(status, quota)
            reg = load(REGISTRY, [])
            q = load(QUEUE, [])
            # queue edits arrive via kaggle/inbox.json so the scheduler never has to be stopped for them
            if INBOX.exists():
                for op in load(INBOX, []):
                    if "append" in op and not any(j.get("name") == op["append"]["name"] for j in q):
                        q.insert(len([j for j in q if not j.get("planner")]), op["append"])
                        log(f"INBOX append {op['append']['name']}")
                    elif "update" in op:
                        for j in q:
                            if j.get("name") == op["update"] and j.get("state", "pending") == "pending":
                                j.update(op["fields"])
                                log(f"INBOX update {op['update']} {list(op['fields'])}")
                INBOX.unlink()
            done = {j["name"] for j in q if j.get("name") and "complete" in (job_state(j["name"], status, reg)[0] or "")}
            # retries + bookkeeping
            for j in q:
                if not j.get("name") or j.get("state") != "launched":
                    continue
                s, _ = job_state(j["name"], status, reg)
                if s and ("error" in s or "cancel" in s or s == "push_failed"):
                    if j.get("retries", 0) < 1:
                        j["retries"] = j.get("retries", 0) + 1
                        j["state"] = "pending"
                        log(f"RETRY {j['name']} after status {s}")
                    else:
                        j["state"] = "failed"
                        log(f"GAVE UP {j['name']} after status {s}")
                elif s and "complete" in s and j["state"] != "done":
                    j["state"] = "done"
                    log(f"DONE {j['name']}")
            # planners whose needs are met
            for j in list(q):
                if j.get("planner") and not j.get("planned") and all(n in done for n in j["needs"]):
                    j["planned"] = True
                    PLANNERS[j["planner"]](q, reg, st)
            # launches
            running = {a: 0 for a in USERS}
            for e in reg:
                s = status.get(f"{e['name']}@{e['launched_utc']:.0f}", {}).get("status", e["status"])
                if not any(f in s for f in FINAL) and s != "push_failed":
                    running[e["account"]] += 1
            for j in q:
                if not j.get("name") or j.get("state", "pending") != "pending":
                    continue
                if not all(n in done for n in j.get("needs", [])):
                    continue
                accts = [j["account"]] if j.get("account") else sorted(USERS, key=lambda a: running[a])
                acct = next((a for a in accts if running[a] < MAX_PER_ACCOUNT), None)
                if acct is None:
                    continue
                j["account"] = acct
                if j.get("module"):
                    j["lanes"] = [[f"@{j['module']} {c}" for c in lane] for lane in j["lanes"]]
                    j.pop("module")
                if launch(j):
                    j["state"] = "launched"
                    running[acct] += 1
                else:
                    j["retries"] = j.get("retries", 0) + 1
                    if j["retries"] > 2:
                        j["state"] = "failed"
            save(q)
            STATE.write_text(json.dumps(st))
            STATUS.write_text(json.dumps(status, indent=1))
            pend = sum(1 for j in q if j.get("name") and j.get("state", "pending") == "pending")
            tail = LOG.read_text(encoding="utf-8").splitlines()[-8:] if LOG.exists() else []
            nl = chr(10)
            screen = (render(status, quota) + nl * 2 + f"queued (waiting for slot/dependency): {pend}" + nl * 2
                      + "recent scheduler events:" + nl + nl.join(tail) + nl)
            sys.stdout.write(chr(27) + "[2J" + chr(27) + "[H" + screen)
            sys.stdout.flush()
        except Exception:
            log("ERROR " + traceback.format_exc()[-800:])
        HEART.write_text(json.dumps({"phase": "sleeping", "t": time.time(), "until": time.time() + 120}))
        time.sleep(120)


if __name__ == "__main__":
    main()
