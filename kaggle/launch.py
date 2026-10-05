"""
Package sihv3 into a single Kaggle script kernel and push it to one of the team accounts.

Each kernel runs on "GPU T4 x2" and executes two independent *lanes* in parallel (lane i -> cuda:i);
each lane runs its list of training commands sequentially. Same quota cost as one GPU, ~2x throughput.

  python kaggle/launch.py --account a2 --name s4-h-sweep --expected_min 140 \
      --lane "--mode det --H 1 --N 16 --seed 0 --tag s4_h1_s0" "--mode det --H 3 --N 16 --seed 0 --tag s4_h3_s0" \
      --lane "--mode det --H 7 --N 16 --seed 0 --tag s4_h7_s0" "--mode det --H 14 --N 16 --seed 0 --tag s4_h14_s0"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "kaggle" / "runs"
REGISTRY = REPO / "kaggle" / "registry.json"
USERS = {"a1": "rohitajitbharadwaj", "a2": "ssachithananthan", "a3": "rohithphegde", "a4": "saketmeda"}


def kaggle_env(acct: str) -> dict:
    """Environment that makes the Kaggle CLI act as one team account (a1 = default user token)."""
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    home = Path.home()
    if acct in ("a2", "a4"):
        env["KAGGLE_API_TOKEN"] = (home / ".kaggle" / f"acct{acct[1]}" / "access_token").read_text().strip()
    elif acct == "a3":
        env.pop("KAGGLE_API_TOKEN", None)
        h = str(home / ".kaggle" / "acct3_home")
        env.update(USERPROFILE=h, HOME=h, KAGGLE_CONFIG_DIR=str(Path(h) / ".kaggle"))
    return env


def kg(acct: str, *args: str) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=kaggle_env(acct))
    return (r.stdout + r.stderr).strip()

BOOT = r'''
import base64, io, os, subprocess, sys, threading, time, zipfile, glob, json
T0 = time.time()
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall("/kaggle/working/code")
os.makedirs("/kaggle/working/out", exist_ok=True)
try:
    import zarr; assert int(zarr.__version__.split(".")[0]) >= 3
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr>=3"], check=True)
# record the input layout, then unpack the dataset if it arrived as a zip
with open("/kaggle/working/out/input_tree.txt", "w") as f:
    for root, dirs, files in os.walk("/kaggle/input"):
        if root.count(os.sep) <= 6:
            f.write(root + "  dirs=" + str(dirs[:8]) + " files=" + str(files[:8]) + chr(10))
print(open("/kaggle/working/out/input_tree.txt").read()[:3000], flush=True)
if not [z for z in glob.glob("/kaggle/input/**/zarr.json", recursive=True) if os.path.isdir(os.path.join(os.path.dirname(z), "future_forecast"))]:
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        zipfile.ZipFile(z).extractall("/kaggle/tmp/data")
    os.environ["SIH_DATA"] = "/kaggle/tmp/data"
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout, flush=True)
ngpu = max(1, int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1))
status = {}
def lane(i, cmds):
    for c in cmds:
        tag = c.split("--tag")[-1].split()[0]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % ngpu), PYTHONPATH="/kaggle/working/code")
        with open(f"/kaggle/working/out/{tag}.log", "w") as log:
            mod, args = ("sihv3." + c.split()[0][1:], c.split()[1:]) if c.startswith("@") else ("sihv3.train", c.split())
            p = subprocess.Popen([sys.executable, "-u", "-m", mod] + args, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                log.write(line); log.flush(); print(f"[gpu{i}] " + line, end="", flush=True)
            status[tag] = p.wait()
ths = [threading.Thread(target=lane, args=(i, l)) for i, l in enumerate(LANES)]
[t.start() for t in ths]; [t.join() for t in ths]
json.dump({"status": status, "wall_min": (time.time() - T0) / 60}, open("/kaggle/working/out/job_status.json", "w"))
print("JOB DONE", status, f"{(time.time() - T0) / 60:.1f} min", flush=True)
'''


def code_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (REPO / "sihv3").glob("*.py"):
            z.write(f, f"sihv3/{f.name}")
    return base64.b64encode(buf.getvalue()).decode()


def registry() -> list:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, choices=list(USERS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--lane", nargs="+", action="append", required=True, help="commands for one GPU lane")
    ap.add_argument("--expected_min", type=float, required=True)
    ap.add_argument("--kernel_sources", nargs="*", default=[], help="slugs (same account) to mount, e.g. det ckpts")
    ap.add_argument("--sprint", default="")
    a = ap.parse_args()
    user = USERS[a.account]
    slug = f"sih26074-v3-{a.name}"
    d = RUNS / a.name
    d.mkdir(parents=True, exist_ok=True)
    script = f"CODE = {code_zip()!r}\nLANES = {a.lane!r}\n" + BOOT
    (d / "job.py").write_text(script)
    meta = {"id": f"{user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            # "dataset:<slug>" entries in --kernel_sources mount an extra private dataset of the same account
            "dataset_sources": [f"{user}/sih26074-v3-realgfs"] + [f"{user}/{s[8:]}" for s in a.kernel_sources if s.startswith("dataset:")],
            "competition_sources": [],
            "kernel_sources": [f"{user}/{s}" if "/" not in s else s for s in a.kernel_sources if not s.startswith("dataset:")]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    live = kg(a.account, "kernels", "status", f"{user}/{slug}")
    if any(w in live for w in ("RUNNING", "QUEUED")):  # never overwrite a job that is still running
        print(f"REFUSED: {user}/{slug} is still active ({live[-80:]})")
        raise SystemExit(3)
    out = kg(a.account, "kernels", "push", "-p", str(d))
    print(out)
    ok = "successfully pushed" in out
    reg = [e for e in registry() if e["name"] != a.name]
    reg.append({"name": a.name, "account": a.account, "user": user, "slug": slug, "sprint": a.sprint,
                "lanes": a.lane, "expected_min": a.expected_min, "launched_utc": time.time(),
                "status": "queued" if ok else "push_failed", "push_msg": out[-300:]})
    REGISTRY.write_text(json.dumps(reg, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
