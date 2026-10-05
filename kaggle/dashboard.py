"""
Live overnight dashboard: status / elapsed / ETA of every Kaggle job on the three team accounts,
remaining GPU quota, and a countdown to the 10:00 IST target. Finished jobs' results (json + logs)
are downloaded automatically into results/<job>/.  Run:  python kaggle/dashboard.py   (Ctrl+C to quit)
Also usable headless:  python kaggle/dashboard.py --once   (one refresh, prints table)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from launch import REGISTRY, REPO, USERS, kg  # noqa: E402

STATUS = REPO / "kaggle" / "status.json"
RESULTS = REPO / "results"
IST = timezone(timedelta(hours=5, minutes=30))
TARGET = datetime(2026, 10, 2, 10, 0, tzinfo=IST)
FINAL = ("complete", "error", "cancel")


def load(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def refresh(status: dict, quota_cache: dict) -> None:
    reg = load(REGISTRY, [])
    for e in reg:
        st = status.setdefault(f"{e['name']}@{e['launched_utc']:.0f}", {})
        if any(f in st.get("status", "") for f in FINAL) and st.get("downloaded"):
            continue
        if e["status"] == "push_failed":
            st["status"] = "push_failed"
            continue
        out = kg(e["account"], "kernels", "status", f"{e['user']}/{e['slug']}")
        s = out.split('status "')[-1].split('"')[0].replace("KernelWorkerStatus.", "").lower() if 'status "' in out else "?"
        if s != st.get("status"):
            st["status"] = s
            st["since"] = time.time()
            if s == "running" and "started" not in st:
                st["started"] = time.time()
        if any(f in s for f in FINAL) and not st.get("downloaded"):
            st.setdefault("finished", time.time())
            dst = RESULTS / e["name"]
            dst.mkdir(parents=True, exist_ok=True)
            kg(e["account"], "kernels", "output", f"{e['user']}/{e['slug']}", "-p", str(dst),
               "--file-pattern", r"(\.json|\.log)$", "--force")
            st["downloaded"] = True
    if time.time() - quota_cache.get("t", 0) > 600:
        for a in USERS:
            line = next((l for l in kg(a, "quota").splitlines() if l.startswith("GPU")), "")
            quota_cache[a] = line.split()[2] if line else "?"
        quota_cache["t"] = time.time()
    STATUS.write_text(json.dumps(status, indent=1))


def render(status: dict, quota_cache: dict) -> str:
    now = datetime.now(IST)
    left = TARGET - now
    reg = load(REGISTRY, [])
    lines = [f"SIH-26074 overnight experiments  |  now {now:%H:%M} IST  |  target 10:00 IST in "
             f"{int(left.total_seconds() // 3600)}h {int(left.total_seconds() % 3600 // 60):02d}m",
             "GPU quota left: " + "  ".join(f"{USERS[a]} {quota_cache.get(a, '?')}" for a in USERS), "",
             f"{'job':28s} {'acct':4s} {'sprint':8s} {'status':10s} {'elapsed':>8s} {'expect':>7s} {'ETA (IST)':>10s}"]
    counts = {}
    for e in reg:
        st = status.get(f"{e['name']}@{e['launched_utc']:.0f}", {})
        s = st.get("status", e["status"])
        counts[s] = counts.get(s, 0) + 1
        start = st.get("started", e["launched_utc"])
        end = st.get("finished", time.time())
        el = (end - start) / 60
        eta = "" if any(f in s for f in FINAL) else (datetime.fromtimestamp(start, IST) + timedelta(minutes=e["expected_min"])).strftime("%H:%M")
        bar_n = min(10, int(10 * el / max(e["expected_min"], 1)))
        bar = "done" if "complete" in s else ("#" * bar_n + "-" * (10 - bar_n))
        lines.append(f"{e['name'][:28]:28s} {e['account']:4s} {e.get('sprint', '')[:8]:8s} {s[:10]:10s} "
                     f"{el:7.0f}m {e['expected_min']:6.0f}m {eta:>10s}  {bar}")
    lines += ["", "totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    return "\n".join(lines)


def main():
    status, quota = load(STATUS, {}), {}
    once = "--once" in sys.argv
    while True:
        try:
            refresh(status, quota)
        except Exception as ex:  # never die overnight
            print("refresh error:", ex)
        txt = render(status, quota)
        if once:
            print(txt)
            return
        sys.stdout.write("\x1b[2J\x1b[H" + txt + "\n")
        sys.stdout.flush()
        time.sleep(60)


if __name__ == "__main__":
    main()
