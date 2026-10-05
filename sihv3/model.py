"""
Spatiotemporal transformer downscaler (STT) for SIH-26074.

One network covers every experiment in the sprint plan:
  * deterministic downscaler  (Sprints 3-5):  predicts y - upsample(GFS)
  * conditional residual diffusion (Sprint 6+): denoises r = y - y_det, conditioned on the frozen
    deterministic prediction y_det (regression + diffusion, CorrDiff-style), v-prediction
  * capacity scaling (dense S/M/L) and sparse MoE feed-forward layers (Sprint 9)

Architecture (all attention is factorised so cost stays linear in days x locations):

  history  [B,H,6,N,N] --patch 2--> axial blocks (space within day, time per location)  --> hist tokens
  forecast [B,7,6,N,N] --patch 2--> axial blocks + cross-attention to history (same location) --> ctx tokens
  fine     [B,7,C,80,80] (upsampled GFS centre, terrain, land mask, [y_det, r_t]) --patch 4-->
           decoder blocks: space attn (per lead day) -> time attn (per patch) -> cross-attn to that day's
           N x N context tokens -> FFN | MoE, every sub-layer adaLN-zero conditioned on the noise level
  output   unpatchify -> [B,7,6,80,80]

Positions are 2-D sin-cos of the true lat/lon offset (degrees) from the domain centre, so coarse and
fine tokens share one coordinate frame and any context size N (16..40) works without retraining shapes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_VAR = 6
N_LEAD = 7
FINE = 80
COARSE_RES, FINE_RES = 0.25, 0.05


@dataclass
class STTConfig:
    dim: int = 256
    heads: int = 8
    depth_hist: int = 2
    depth_fut: int = 2
    depth_dec: int = 6
    mlp_ratio: float = 4.0
    patch_fine: int = 4
    patch_coarse: int = 2
    diffusion: bool = False
    moe_experts: int = 0      # 0 = dense FFN everywhere
    moe_topk: int = 1
    moe_frac: float = 0.0     # fraction of decoder blocks (the last ones) whose FFN is MoE
    dropout: float = 0.0
    hist_fc: bool = False     # history carries 12 channels: observed + the GFS forecast that was valid on each day

    def to_dict(self):
        return asdict(self)


PRESETS = {  # ~ S 15M, M 2x, L 3.3x (matches the Sprint 9 capacity ladder)
    "S": dict(dim=256, heads=8, depth_hist=2, depth_fut=2, depth_dec=6),
    "M": dict(dim=320, heads=8, depth_hist=2, depth_fut=2, depth_dec=8),
    "L": dict(dim=384, heads=8, depth_hist=2, depth_fut=2, depth_dec=10),
}


# ----------------------------------------------------------------------------------- helpers
def sincos_2d(lat: torch.Tensor, lon: torch.Tensor, dim: int) -> torch.Tensor:
    """lat/lon offsets in degrees, shape [T] -> [T, dim]."""
    d4 = dim // 4
    freq = torch.exp(torch.arange(d4, device=lat.device, dtype=torch.float32) * (-math.log(500.0) / d4)) * 20.0
    a, b = lat[:, None] * freq, lon[:, None] * freq
    return torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], dim=1)


def grid_offsets(n: int, res: float, patch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre offsets (deg) of patch tokens on an n x n grid of cells of size `res`, centred on 0."""
    m = n // patch
    c = (torch.arange(m, device=device, dtype=torch.float32) + 0.5) * res * patch - n * res / 2
    lat = (-c)[:, None].expand(m, m).reshape(-1)  # row 0 = north
    lon = c[None, :].expand(m, m).reshape(-1)
    return lat, lon


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, cross=False):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)
        self.drop = dropout

    def forward(self, x, ctx=None):
        b, n, d = x.shape
        ctx = x if ctx is None else ctx
        q = self.q(x).view(b, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(b, n, d))


class FFN(nn.Module):
    def __init__(self, dim, ratio, dropout=0.0):
        super().__init__()
        h = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Dropout(dropout), nn.Linear(h, dim))

    def forward(self, x):
        return self.net(x), x.new_zeros(())


class MoEFFN(nn.Module):
    """Top-k routed mixture of FFN experts (each expert = one dense FFN), Switch-style balance loss."""

    def __init__(self, dim, ratio, n_exp, topk, dropout=0.0):
        super().__init__()
        self.n, self.k = n_exp, topk
        self.router = nn.Linear(dim, n_exp, bias=False)
        self.experts = nn.ModuleList([FFN(dim, ratio, dropout) for _ in range(n_exp)])
        self.last_load = None

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        logits = self.router(x2).float()
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.k, dim=-1)
        topv = (topv / topv.sum(-1, keepdim=True)).to(x2.dtype)
        out = torch.zeros_like(x2)
        load = torch.zeros(self.n, device=x.device)
        for e in range(self.n):
            hit = (topi == e)
            rows = hit.any(-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            w = (topv * hit.to(topv.dtype)).sum(-1)[rows]
            y, _ = self.experts[e](x2[rows])
            out.index_add_(0, rows, y * w[:, None])
            load[e] = rows.numel()
        frac = load / load.sum().clamp_min(1)
        aux = self.n * (frac * probs.mean(0)).sum()  # Switch Transformer load-balancing loss
        self.last_load = frac.detach()
        return out.reshape(shp), aux


class AxialBlock(nn.Module):
    """Encoder block on tokens [B, T, S, D]: attention over S within each T, then over T per S,
    optional cross-attention (per location) to a memory [B, Tm, S, D]."""

    def __init__(self, dim, heads, ratio, cross=False, dropout=0.0):
        super().__init__()
        self.n1, self.a_s = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n2, self.a_t = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.cross = cross
        if cross:
            self.n3, self.a_c = nn.LayerNorm(dim), Attention(dim, heads, dropout)
        self.n4, self.ffn = nn.LayerNorm(dim), FFN(dim, ratio, dropout)

    def forward(self, x, mem=None):
        b, t, s, d = x.shape
        x = x + self.a_s(self.n1(x).reshape(b * t, s, d)).reshape(b, t, s, d)
        xt = self.n2(x).transpose(1, 2).reshape(b * s, t, d)
        x = x + self.a_t(xt).reshape(b, s, t, d).transpose(1, 2)
        if self.cross:
            q = self.n3(x).transpose(1, 2).reshape(b * s, t, d)
            m = mem.transpose(1, 2).reshape(b * s, mem.shape[1], d)
            x = x + self.a_c(q, m).reshape(b, s, t, d).transpose(1, 2)
        return x + self.ffn(self.n4(x))[0]


class DecoderBlock(nn.Module):
    """Fine-grid block on [B, 7, S, D]; adaLN-zero conditioned on c [B, D]."""

    def __init__(self, dim, heads, ratio, moe=None, dropout=0.0):
        super().__init__()
        self.ns, self.a_s = nn.LayerNorm(dim, elementwise_affine=False), Attention(dim, heads, dropout)
        self.nt, self.a_t = nn.LayerNorm(dim, elementwise_affine=False), Attention(dim, heads, dropout)
        self.nc, self.a_c = nn.LayerNorm(dim, elementwise_affine=False), Attention(dim, heads, dropout)
        self.nf = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = MoEFFN(dim, ratio, *moe, dropout) if moe else FFN(dim, ratio, dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, ctx, c):
        b, t, s, d = x.shape
        p = self.ada(c)[:, None, None, :].chunk(12, dim=-1)
        h = modulate(self.ns(x), p[0], p[1]).reshape(b * t, s, d)
        x = x + p[2] * self.a_s(h).reshape(b, t, s, d)
        h = modulate(self.nt(x), p[3], p[4]).transpose(1, 2).reshape(b * s, t, d)
        x = x + p[5] * self.a_t(h).reshape(b, s, t, d).transpose(1, 2)
        h = modulate(self.nc(x), p[6], p[7]).reshape(b * t, s, d)
        x = x + p[8] * self.a_c(h, ctx.reshape(b * t, ctx.shape[2], d)).reshape(b, t, s, d)
        y, aux = self.ffn(modulate(self.nf(x), p[9], p[10]))
        return x + p[11] * y, aux


# ----------------------------------------------------------------------------------- model
class SpatiotemporalTransformer(nn.Module):
    def __init__(self, cfg: STTConfig, n_static: int = 6):
        super().__init__()
        self.cfg = cfg
        d, pc, pf = cfg.dim, cfg.patch_coarse, cfg.patch_fine
        self.coarse_in = nn.Conv2d(N_VAR, d, pc, pc)
        self.hist_in = nn.Conv2d(N_VAR * (2 if cfg.hist_fc else 1), d, pc, pc)
        self.day_emb = nn.Embedding(14 + N_LEAD, d)  # 0..13 = D-14..D-1, 14..20 = D..D+6
        self.hist_blocks = nn.ModuleList([AxialBlock(d, cfg.heads, cfg.mlp_ratio, False, cfg.dropout)
                                          for _ in range(cfg.depth_hist)])
        self.fut_blocks = nn.ModuleList([AxialBlock(d, cfg.heads, cfg.mlp_ratio, True, cfg.dropout)
                                         for _ in range(cfg.depth_fut)])
        self.ctx_norm = nn.LayerNorm(d)
        fine_ch = N_VAR + n_static + (2 * N_VAR if cfg.diffusion else 0)
        self.fine_in = nn.Conv2d(fine_ch, d, pf, pf)
        n_moe = int(round(cfg.moe_frac * cfg.depth_dec)) if cfg.moe_experts > 0 else 0
        self.dec = nn.ModuleList([
            DecoderBlock(d, cfg.heads, cfg.mlp_ratio,
                         (cfg.moe_experts, cfg.moe_topk) if i >= cfg.depth_dec - n_moe else None, cfg.dropout)
            for i in range(cfg.depth_dec)])
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.out_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, pf * pf * N_VAR)
        for m in (self.out_ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def n_params(self, active: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not active:
            return total
        inactive = 0
        for m in self.modules():
            if isinstance(m, MoEFFN):
                per = sum(p.numel() for p in m.experts[0].parameters())
                inactive += per * (m.n - m.k)
        return total - inactive

    def _coarse_tokens(self, x, conv, day_ids):
        b, t, c, n, _ = x.shape
        tok = conv(x.reshape(b * t, c, n, n)).flatten(2).transpose(1, 2)  # [b*t, S, d]
        s = tok.shape[1]
        lat, lon = grid_offsets(n, COARSE_RES, self.cfg.patch_coarse, x.device)
        tok = tok.reshape(b, t, s, -1) + sincos_2d(lat, lon, self.cfg.dim)[None, None]
        return tok + self.day_emb(day_ids)[None, :, None, :]

    def forward(self, history, forecast, fine_static, t=None, fine_extra=None):
        """
        history [B,H,6,N,N], forecast [B,7,6,N,N] (normalised), fine_static [B,6+?,80,80] per-sample static
        fine inputs broadcast over leads (terrain 5 + land mask 1) concatenated with the upsampled GFS centre
        per lead: we pass upsampled GFS separately inside fine_extra-free path, see build_fine().
        t: [B] diffusion time in [0,1] (None -> deterministic, uses t=0 embedding)
        fine_extra: [B,7,12,80,80] = (y_det, r_t) for diffusion
        """
        cfg = self.cfg
        b, H = history.shape[:2]
        dev = history.device
        hist = self._coarse_tokens(history, self.hist_in, torch.arange(14 - H, 14, device=dev))
        for blk in self.hist_blocks:
            hist = blk(hist)
        ctx = self._coarse_tokens(forecast, self.coarse_in, torch.arange(14, 14 + N_LEAD, device=dev))
        for blk in self.fut_blocks:
            ctx = blk(ctx, hist)
        ctx = self.ctx_norm(ctx)

        n = forecast.shape[-1]
        lo = (n - 16) // 2
        up = F.interpolate(forecast[:, :, :, lo:lo + 16, lo:lo + 16].reshape(b * N_LEAD, N_VAR, 16, 16),
                           size=(FINE, FINE), mode="bilinear", align_corners=False).reshape(b, N_LEAD, N_VAR, FINE, FINE)
        parts = [up, fine_static[:, None].expand(-1, N_LEAD, -1, -1, -1)]
        if cfg.diffusion:
            parts.append(fine_extra)
        fine = torch.cat(parts, dim=2)
        x = self.fine_in(fine.reshape(b * N_LEAD, -1, FINE, FINE)).flatten(2).transpose(1, 2)
        s = x.shape[1]
        lat, lon = grid_offsets(FINE, FINE_RES, cfg.patch_fine, dev)
        x = x.reshape(b, N_LEAD, s, -1) + sincos_2d(lat, lon, cfg.dim)[None, None]
        x = x + self.day_emb(torch.arange(14, 14 + N_LEAD, device=dev))[None, :, None, :]

        tt = torch.zeros(b, device=dev) if t is None else t
        c = self.t_mlp(timestep_embedding(tt, cfg.dim))
        aux = x.new_zeros(())
        for blk in self.dec:
            x, a = blk(x, ctx, c)
            aux = aux + a
        sh, sc = self.out_ada(c)[:, None, None, :].chunk(2, dim=-1)
        x = self.out(modulate(self.out_norm(x), sh, sc))  # [B,7,S,p*p*6]
        p, m = cfg.patch_fine, FINE // cfg.patch_fine
        x = x.reshape(b, N_LEAD, m, m, p, p, N_VAR).permute(0, 1, 6, 2, 4, 3, 5).reshape(b, N_LEAD, N_VAR, FINE, FINE)
        return x, up, aux


def build_model(size: str = "S", **overrides) -> SpatiotemporalTransformer:
    cfg = STTConfig(**{**PRESETS[size], **overrides})
    return SpatiotemporalTransformer(cfg)


if __name__ == "__main__":
    for size in ("S", "M", "L"):
        m = build_model(size)
        print(size, f"{m.n_params() / 1e6:.2f}M")
    m = build_model("S", moe_experts=8, moe_topk=1, moe_frac=0.5)
    print("S+MoE8 top1 50%", f"total {m.n_params() / 1e6:.2f}M active {m.n_params(True) / 1e6:.2f}M")
    h, f, st = torch.randn(2, 7, 6, 24, 24), torch.randn(2, 7, 6, 24, 24), torch.randn(2, 6, 80, 80)
    y, up, aux = build_model("S")(h, f, st)
    print("det out", tuple(y.shape))
    md = build_model("S", diffusion=True)
    y, _, _ = md(h, f, st, t=torch.rand(2), fine_extra=torch.randn(2, 7, 12, 80, 80))
    print("diff out", tuple(y.shape))
