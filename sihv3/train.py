"""
Unified trainer for every sprint experiment.

  python -m sihv3.train --mode det  --size S --H 7 --N 24 --seed 0 --tag s4_h7_s0
  python -m sihv3.train --mode diff --size S --H 7 --N 24 --det_ckpt out/s3/best.pt --tag s6_diff_s0
  python -m sihv3.train --mode det  --size S --moe_experts 8 --moe_topk 1 --moe_frac 0.5 ...

Deterministic: predicts y = upsample(GFS) + f(.) ; masked Huber loss in normalised space.
Diffusion:     r = (y - y_det) / sigma_c with y_det from a frozen deterministic model; v-prediction with
               a cosine schedule on continuous t in [0,1]; the denoiser sees y_det and z_t.
Checkpoint selection: det -> best validation Composite Skill Score (CSS); diff -> best validation v-loss.
Final evaluation of the selected (EMA) weights on val and test, written to <out>/<tag>/result.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sihv3.data import V3Data
from sihv3.metrics import composite_skill, det_metrics, prob_metrics
from sihv3.model import STTConfig, SpatiotemporalTransformer, PRESETS


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["det", "diff"], default="det")
    p.add_argument("--size", default="S")
    p.add_argument("--H", type=int, default=7)
    p.add_argument("--N", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--time_budget_min", type=float, default=150.0, help="stop training after this (minutes)")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--moe_experts", type=int, default=0)
    p.add_argument("--moe_topk", type=int, default=1)
    p.add_argument("--moe_frac", type=float, default=0.0)
    p.add_argument("--fc_history", action="store_true", help="add the forecast history (GFS valid on each history day) as input")
    p.add_argument("--moe_aux", type=float, default=0.01)
    p.add_argument("--precip_lin_w", type=float, default=0.0,
                   help="extra weight on a precipitation L1 term in mm/10 (counters log-space median bias)")
    p.add_argument("--det_ckpt", default=None, help="frozen deterministic model for --mode diff")
    p.add_argument("--eval_members", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=16)
    p.add_argument("--val_year", type=int, default=None, help="alternative validation season (default 2022)")
    p.add_argument("--train_years", default=None, help="e.g. 2015-2022 or 2017,2018 (explicit year split)")
    p.add_argument("--val_years", default="", help="years held out for validation with --train_years ('' = none)")
    p.add_argument("--eval_sampler", default="ddim", choices=["ddim", "dpmpp2m"])
    p.add_argument("--det_pred_file", default=None,
                   help="npz with precomputed y_det for every sample (cross-fitted OOF); replaces --det_ckpt in diff mode")
    p.add_argument("--tag", default="run")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    p.add_argument("--max_batches", type=int, default=0, help="smoke test: limit batches per epoch")
    return p.parse_args()


# ------------------------------------------------------------------------------- utilities
def parse_years(spec):
    if spec is None or spec == "":
        return None if spec is None else []
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def resolve(path: str) -> str:
    """Allow glob patterns (e.g. '**/s4_h7_n16_s0/best.pt') searched under /kaggle/input and cwd."""
    if "*" not in path:
        return path
    import glob
    for root in ("/kaggle/input", "/kaggle/working", "."):
        hits = sorted(glob.glob(os.path.join(root, path), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(path)


def masked_huber(pred, targ, mask, ch_w=None):
    l = F.smooth_l1_loss(pred.float(), targ.float(), reduction="none", beta=1.0) * mask
    per_ch = l.sum((0, 1, 3, 4)) / mask.sum((0, 1, 3, 4)).clamp_min(1)
    return per_ch.mean() if ch_w is None else (per_ch * ch_w).sum() / ch_w.sum()


def build(args, diffusion: bool, size=None) -> SpatiotemporalTransformer:
    cfg = STTConfig(**{**PRESETS[size or args.size], "diffusion": diffusion, "moe_experts": args.moe_experts,
                       "moe_topk": args.moe_topk, "moe_frac": args.moe_frac,
                       "hist_fc": getattr(args, "fc_history", False)})
    return SpatiotemporalTransformer(cfg)


def cosine_ab(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@torch.no_grad()
def det_predict(model, batch):
    y, up, _ = model(batch["history"], batch["forecast"], batch["static"])
    return y + up


@torch.no_grad()
def _denoise(model, batch, y_det, z, steps, sampler, eta, gen):
    """Integrate one batch of initial noise z from t=0.999 to 0; returns the final residual z."""
    ts = torch.linspace(0.999, 0.0, steps + 1, device=y_det.device)  # training uses t in [1e-3, 0.999]
    x0_prev = None
    for i in range(steps):
        t, tn = ts[i], ts[i + 1]
        a, s = cosine_ab(t)
        an, sn = cosine_ab(tn)
        v, _, _ = model(batch["history"], batch["forecast"], batch["static"], t=t.expand(len(z)),
                        fine_extra=torch.cat([y_det, z], 2))
        v = v.float()
        x0 = a * z - s * v
        eps = s * z + a * v
        if sampler == "dpmpp2m" and x0_prev is not None and i < steps - 1:
            # DPM-Solver++(2M) in data-prediction form (Lu et al. 2022) for the VP cosine schedule
            lam = lambda aa, ss: torch.log(aa / ss.clamp_min(1e-8))
            h = lam(an, sn) - lam(a, s)
            tp = ts[i - 1]
            ap, sp = cosine_ab(tp)
            h_prev = lam(a, s) - lam(ap, sp)
            r = h_prev / h
            d = (1 + 1 / (2 * r)) * x0 - (1 / (2 * r)) * x0_prev
            z = (sn / s) * z - an * torch.expm1(-h) * d
        else:  # DDIM (eta=0 deterministic, eta>0 stochastic)
            if tn > 0:
                c = eta * torch.sqrt((sn ** 2 / s ** 2) * (1 - a ** 2 / an ** 2).clamp_min(0)) if eta > 0 else 0.0
                dir_ = torch.sqrt((sn ** 2 - c ** 2).clamp_min(0)) if eta > 0 else sn
                noise = torch.randn(z.shape, device=z.device, generator=gen) if eta > 0 else 0.0
                z = an * x0 + dir_ * eps + c * noise
            else:
                z = x0
        x0_prev = x0
    return z


def sample(model, batch, y_det, sigma, steps=16, members=1, sampler="ddim", eta=0.0, gen=None, batch_members=False):
    """Returns [K, B, 7, 6, 80, 80] normalised samples of y = y_det + sigma * r.
    batch_members: all K members in one batched pass (deterministic samplers only). The initial noise is drawn in the
    same order as the sequential path, so both give the same samples up to floating-point reordering."""
    if batch_members and members > 1 and eta == 0.0:
        z = torch.cat([torch.randn(y_det.shape, device=y_det.device, generator=gen) for _ in range(members)])
        rep = lambda v: v.repeat(members, *([1] * (v.dim() - 1)))
        big = {k: rep(batch[k]) for k in ("history", "forecast", "static")}
        z = _denoise(model, big, rep(y_det), z, steps, sampler, eta, gen)
        return (y_det[None] + sigma * z.view(members, *y_det.shape))
    outs = []
    for _ in range(members):
        z = torch.randn(y_det.shape, device=y_det.device, generator=gen)
        z = _denoise(model, batch, y_det, z, steps, sampler, eta, gen)
        outs.append(y_det + sigma * z)
    return torch.stack(outs)


def fit_precip_qm(data, predict_fn, device, nq=100, block=5):
    """Quantile map predicted -> observed precipitation (mm) per lead and per coarse block (block x block fine
    pixels, land only), fitted on the train split. Returns (pq, oq) arrays of shape [7, nb, nb, nq]."""
    qs = np.concatenate([np.linspace(0, 0.98, nq - 10), np.linspace(0.985, 0.9995, 10)])
    P, T, M = [], [], []
    for b in data.batches("train", 8, False, device):
        P.append(data.norm.inv(predict_fn(b)[0], axis=2)[:, :, 0].cpu().numpy())
        T.append(data.norm.inv(b["target"], axis=2)[:, :, 0].cpu().numpy())
        M.append(b["mask"][:, :, 0].cpu().numpy().astype(bool))
    P, T, M = np.concatenate(P), np.concatenate(T), np.concatenate(M)  # [n,7,80,80]
    nb = 80 // block
    pq = np.zeros((7, nb, nb, nq), np.float32)
    oq = np.zeros_like(pq)
    for l in range(7):
        for i in range(nb):
            for j in range(nb):
                sl = (slice(None), l, slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
                m = M[sl]
                if m.sum() < 50:  # (almost) all sea: identity map
                    pq[l, i, j] = oq[l, i, j] = qs * 100
                    continue
                pq[l, i, j] = np.quantile(P[sl][m], qs)
                oq[l, i, j] = np.quantile(T[sl][m], qs)
    return pq, oq


def apply_precip_qm(P, qm, block=5):
    """P physical [K,B,7,6,80,80] -> precipitation channel quantile-mapped (per lead, per block) in place."""
    pq, oq = qm
    nb = pq.shape[1]
    for l in range(7):
        for i in range(nb):
            for j in range(nb):
                sl = (slice(None), slice(None), l, 0, slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
                x = P[sl]
                y = np.interp(x, pq[l, i, j], oq[l, i, j])
                top = x > pq[l, i, j, -1]  # beyond the fitted range: keep the top-quantile ratio
                y[top] = x[top] * oq[l, i, j, -1] / max(pq[l, i, j, -1], 1e-3)
                P[sl] = y
    return P


def evaluate(data, split, predict_fn, device, members=1, hook=None, post=None):
    """predict_fn(batch) -> [K,B,7,6,80,80] normalised. Returns metrics dict incl. reference + CSS."""
    preds, refs, targs, masks = [], [], [], []
    for b in data.batches(split, 8, False, device):
        p = predict_fn(b)
        n = data.norm
        preds.append(n.inv(p, axis=3).cpu().numpy())
        if post is not None:
            preds[-1] = post(preds[-1])
        lo = (b["forecast"].shape[-1] - 16) // 2
        up = F.interpolate(b["forecast"][:, :, :, lo:lo + 16, lo:lo + 16].flatten(0, 1), size=(80, 80),
                           mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80)
        refs.append(n.inv(up, axis=2).cpu().numpy())
        targs.append(n.inv(b["target"], axis=2).cpu().numpy())
        masks.append(b["mask"].cpu().numpy())
        if hook is not None:
            hook(preds[-1], targs[-1], masks[-1])
    P = np.concatenate(preds, 1)  # [K, n, ...]
    R, T, M = np.concatenate(refs), np.concatenate(targs), np.concatenate(masks)
    ref = det_metrics(R, T, M)
    res = prob_metrics(P.transpose(1, 0, 2, 3, 4, 5), T, M) if P.shape[0] > 1 else det_metrics(P[0], T, M)
    res["css"] = composite_skill(res["aggregate"], ref["aggregate"])
    res["reference_gfs_bilinear"] = ref
    return res


# ------------------------------------------------------------------------------- main
def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"[{args.tag}] device={dev} gpus={torch.cuda.device_count()} args={vars(args)}", flush=True)

    data = V3Data(history_len=args.H, context=args.N, val_year=args.val_year,
                  train_years=parse_years(args.train_years), val_years=parse_years(args.val_years) or [],
                  fc_history=args.fc_history)
    has_val = len(data.idx["val"]) > 0
    print(f"[{args.tag}] data: train {len(data.idx['train'])} val {len(data.idx['val'])} test {len(data.idx['test'])}"
          f" | load {time.time() - t_start:.0f}s", flush=True)

    diff = args.mode == "diff"
    model = build(args, diffusion=diff).to(dev)
    n_total, n_active = model.n_params(), model.n_params(active=True)
    print(f"[{args.tag}] params total {n_total / 1e6:.2f}M active {n_active / 1e6:.2f}M", flush=True)

    det_model, sigma, det_pred = None, None, None
    if diff and args.det_pred_file:
        z = np.load(resolve(args.det_pred_file))
        assert list(z["dates"].astype(str)) == list(data.dates), "det_pred_file rows must align with the dataset"
        det_pred = torch.from_numpy(z["ydet"].astype(np.float32))
        print(f"[{args.tag}] using precomputed y_det {tuple(det_pred.shape)} from {args.det_pred_file} ({z['kind']})", flush=True)

    def y_det_of(b):
        if det_pred is not None:
            return det_pred[b["index"]].to(dev)
        return det_predict(det_model, b).float()

    if diff:
        if det_pred is None:
            ck = torch.load(resolve(args.det_ckpt), map_location=dev, weights_only=False)
            a2 = argparse.Namespace(**{**vars(args), **{k: ck["args"][k] for k in ("moe_experts", "moe_topk", "moe_frac")}})
            det_model = build(a2, diffusion=False, size=ck["args"]["size"]).to(dev)
            det_model.load_state_dict(ck["ema"])
            det_model.eval().requires_grad_(False)
            assert ck["args"]["H"] == args.H and ck["args"]["N"] == args.N, "det model must share H/N"
        # per-channel residual scale over the training set
        ss, cnt = torch.zeros(6, device=dev), torch.zeros(6, device=dev)
        for b in data.batches("train", 16, False, dev):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                r = (b["target"] - y_det_of(b)) * b["mask"]
            ss += (r.float() ** 2).sum((0, 1, 3, 4))
            cnt += b["mask"].sum((0, 1, 3, 4))
        sigma = torch.sqrt(ss / cnt).view(1, 1, 6, 1, 1)
        print(f"[{args.tag}] residual sigma per channel {sigma.flatten().tolist()}", flush=True)

    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))
    steps_per_epoch = math.ceil(len(data.idx["train"]) / args.bs)
    total_steps, warm = steps_per_epoch * args.epochs, steps_per_epoch * 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total_steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ch_w = torch.tensor([1.5, 1, 1, 1, 1, 1], device=dev)  # precipitation is the primary advisory variable
    pmean, pstd = float(data.norm.mean[0]), float(data.norm.std[0])

    def loss_fn(net, b):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            if not diff:
                y, up, aux = net(b["history"], b["forecast"], b["static"])
                yhat = y + up
                loss = masked_huber(yhat, b["target"], b["mask"], ch_w) + args.moe_aux * aux.mean()
                if args.precip_lin_w > 0:
                    pm = b["mask"][:, :, 0]
                    pp = torch.expm1((yhat.float()[:, :, 0] * pstd + pmean).clamp(max=7.0)).clamp_min(0)
                    pt = torch.expm1((b["target"][:, :, 0] * pstd + pmean).clamp(max=7.0)).clamp_min(0)
                    loss = loss + args.precip_lin_w * ((pp - pt).abs() * pm).sum() / pm.sum().clamp_min(1) / 10.0
                return loss
            with torch.no_grad():
                y_det = y_det_of(b)
            r0 = (b["target"] - y_det) / sigma * b["mask"]
            t = torch.rand(len(r0), device=dev).clamp(1e-3, 1 - 1e-3)
            a, s = cosine_ab(t.view(-1, 1, 1, 1, 1))
            eps = torch.randn_like(r0)
            zt = a * r0 + s * eps
            v_tgt = a * eps - s * r0
            v, _, aux = net(b["history"], b["forecast"], b["static"], t=t, fine_extra=torch.cat([y_det, zt], 2))
            l = ((v.float() - v_tgt) ** 2 * b["mask"]).sum((0, 1, 3, 4)) / b["mask"].sum((0, 1, 3, 4)).clamp_min(1)
            return (l * ch_w).sum() / ch_w.sum() + args.moe_aux * aux.mean()

    best, best_ep, hist_log, step = -1e9, 0, [], 0
    budget = args.time_budget_min * 60
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tl, nb = 0.0, 0
        for b in data.batches("train", args.bs, True, dev, rng):
            loss = loss_fn(model, b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            decay = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            tl += loss.item()
            nb += 1
            if args.max_batches and nb >= args.max_batches:
                break
        # validation (skipped when there is no validation season: fixed-epoch training, last epoch kept)
        ema.eval()
        if not has_val:
            score, metric = float(ep), {}
        elif diff:
            vl, vn = 0.0, 0
            g = torch.Generator(device=dev).manual_seed(123)
            torch.manual_seed(123)
            with torch.no_grad():
                for b in data.batches("val", 8, False, dev):
                    vl += loss_fn(ema, b).item()
                    vn += 1
            score, metric = -vl / max(vn, 1), {"val_vloss": vl / max(vn, 1)}
        else:
            r = evaluate(data, "val", lambda b: det_predict(ema, b)[None].float(), dev)
            score, metric = r["css"], {"val_css": r["css"], **{k: r["aggregate"][k] for k in ("precip_wet_mae", "precip_csi15", "tmax_mae", "wind_vec_rmse")}}
        ep_t = time.time() - t0
        elapsed = time.time() - t_start
        rec = {"epoch": ep, "train_loss": tl / max(nb, 1), "epoch_sec": ep_t, "elapsed_min": elapsed / 60, **metric}
        hist_log.append(rec)
        improved = score > best
        if improved:
            best, best_ep = score, ep
            torch.save({"ema": ema.state_dict(), "args": vars(args), "cfg": ema.cfg.to_dict(), "epoch": ep,
                        "sigma": None if sigma is None else sigma.flatten().tolist()}, out / "best.pt")
        eta = (min(args.epochs, ep + args.patience) - ep) * ep_t
        print(f"[{args.tag}] ep {ep:3d} loss {rec['train_loss']:.4f} " + " ".join(f"{k} {v:.4f}" for k, v in metric.items())
              + f" | {ep_t:.0f}s/ep elapsed {elapsed / 60:.1f}m{' *' if improved else ''}", flush=True)
        json.dump(hist_log, open(out / "history.json", "w"), indent=1)
        if (has_val and ep - best_ep >= args.patience) or elapsed + ep_t * 1.3 > budget:
            print(f"[{args.tag}] stop at ep {ep} (best {best_ep}, {'patience' if ep - best_ep >= args.patience else 'time budget'})", flush=True)
            break

    # final evaluation with best EMA weights
    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    ema.load_state_dict(ck["ema"])
    result = {"args": vars(args), "params_total": n_total, "params_active": n_active, "best_epoch": best_ep,
              "train_minutes": (time.time() - t_start) / 60, "history": hist_log}
    qm = None
    for split in [sp for sp in ("val", "test") if len(data.idx[sp]) > 0]:
        t0 = time.time()
        if diff:
            g = torch.Generator(device=dev).manual_seed(7)

            def pf(b):
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    yd = y_det_of(b)
                    return sample(ema, b, yd, sigma, steps=args.eval_steps, members=args.eval_members,
                                  sampler=args.eval_sampler, gen=g)
            r = evaluate(data, split, pf, dev)
        else:
            def pf(b):
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    return det_predict(ema, b)[None].float()
            r = evaluate(data, split, pf, dev)
        r["latency_sec_per_sample"] = (time.time() - t0) / len(data.idx[split])
        if not diff:  # post-hoc precipitation quantile mapping fitted on train (fixes log-loss dry bias)
            if qm is None:  # fitted once, on the train split
                qm = fit_precip_qm(data, pf, dev)
            rq = evaluate(data, split, pf, dev, post=lambda P: apply_precip_qm(P, qm))
            r["precip_qm"] = {"css": rq["css"], "aggregate": rq["aggregate"]}
            print(f"[{args.tag}] FINAL {split} +precipQM: CSS {rq['css']:.4f} | bias {rq['aggregate']['precip_bias_ratio']:.2f} "
                  f"csi15 {rq['aggregate']['precip_csi15']:.3f} csi30 {rq['aggregate']['precip_csi30']:.3f} "
                  f"wetMAE {rq['aggregate']['precip_wet_mae']:.2f}", flush=True)
        result[split] = r
        print(f"[{args.tag}] FINAL {split}: CSS {r['css']:.4f} | " + " ".join(
            f"{k} {r['aggregate'][k]:.3f}" for k in ("precip_wet_mae", "precip_csi15", "precip_csi30", "precip_fss15",
                                                      "tmax_mae", "tmin_mae", "rh_mae", "wind_vec_rmse") if k in r["aggregate"])
              + (f" crps_p {r['aggregate']['precip_crps']:.3f} ssr_p {r['aggregate'].get('precip_ssr', float('nan')):.2f}" if diff else ""),
              flush=True)
    if torch.cuda.is_available():
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(result, open(out / "result.json", "w"), indent=1, default=float)
    print(f"[{args.tag}] DONE total {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
