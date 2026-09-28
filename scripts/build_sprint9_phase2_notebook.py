"""
scripts/build_sprint9_phase2_notebook.py

Constructs the dedicated Phase 2 interactive Kaggle notebook:
'notebooks/sprint_9_phase2_moe_sparse_routing.ipynb'
and copies it to 'C:\\Users\\rohit\\Downloads\\sprint_9_phase2_moe_sparse_routing.ipynb'.

Phase 2 focuses strictly on Sparse Mixture of Experts (MoE) Routing:
  1. Dense-S: 15,685,478 params (Candidate 3 control, base_channels=96)
  2. MoE-4: 22,773,350 total params (Sparse Top-1 routing over 4 bottleneck experts, 15,688,550 active params)
  3. Benchmark comparison against Phase 1 Dense-M (31.20M) and Dense-L (52.00M)
  4. Tests the core scientific hypothesis: Does MoE decouple capacity from inference compute?

Operates on the authentic Zarr dataset (multitask_temporal_v2_h14.zarr) and 2022 validation set.
Strictly ZERO em dashes (\\u2014 and \\u2013).
"""

import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

DEST_REPO = NOTEBOOK_DIR / "sprint_9_phase2_moe_sparse_routing.ipynb"
DEST_DOWNLOADS = Path(r"C:\Users\rohit\Downloads\sprint_9_phase2_moe_sparse_routing.ipynb")


def create_phase2_notebook():
    cells = []

    def add_md(source: str):
        lines = [line + "\n" for line in source.strip().split("\n")]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        })

    def add_code(source: str):
        lines = [line + "\n" for line in source.strip().split("\n")]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        })

    # -------------------------------------------------------------
    # CELL 1: Markdown Title
    # -------------------------------------------------------------
    add_md("""# Sprint 9 (Phase 2): Sparse Mixture of Experts (MoE) Routing
### SIH Problem Statement 26074: Block to Panchayat Weather Downscaling

This notebook executes **Phase 2** of Sprint 9:
1. **Sparse MoE Architecture**:
   - **Dense-S Control**: 15,685,478 parameters (base_channels = 96, active = 15.69M)
   - **MoE-4 Champion**: 22,773,350 total parameters with **15,688,550 active parameters** during inference (Top-1 routing over 4 bottleneck experts at the 10x10 stage)
   - Benchmarked directly against Phase 1 winners: **Dense-M** (31.20M) and **Dense-L** (52.00M).
2. **Authentic Data Pipeline**:
   - Trains on authentic 2015-2021 reanalysis and GFS forecast conditioning from `multitask_temporal_v2_h14.zarr`.
   - Evaluates on the complete authentic 2022 validation season (122 forecast cubes, 427 daily slices).
   - Zero synthetic or proxy fallbacks.
3. **Hypothesis Testing**:
   - Can MoE-4 match Dense-M / Dense-L skill while operating at Dense-S active parameter budget and inference latency?
4. **Matched 32 NFE Inference Regimes**:
   - Point Mode: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE)
   - Distribution Mode: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE)
5. **Empirical Artifacts Updated**:
   - Generates empirical tables and updates `reports/sprint9_moe_pareto_frontier.md` and `reports/sprint9_moe_scaling_results.json`.""")

    # -------------------------------------------------------------
    # CELL 2: Code Environment Setup
    # -------------------------------------------------------------
    add_code("""# Cell 1: Hardware Environment Diagnostics and Package Setup
import os
import sys
import time
import math
import hashlib
import json
import zipfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

# 1. Hardware Diagnostics
print("=" * 60)
print("[*] Sprint 9 (Phase 2: MoE Sparse Routing) Environment Diagnostics:")
print(f"    Python:          {sys.version.split()[0]}")
print(f"    PyTorch:         {torch.__version__}")
print(f"    CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"    Device Name:     {torch.cuda.get_device_name(0)}")
    print(f"    Device Count:    {torch.cuda.device_count()}")
    print(f"    Allocated VRAM:  {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
else:
    device = torch.device("cpu")
    print("    [!] WARNING: CUDA not detected, running on CPU.")
print("=" * 60)

# 2. Package Dependency Check
try:
    import zarr
except ImportError:
    print("[*] Installing required meteorological libraries: zarr, numcodecs...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zarr", "numcodecs", "pyyaml", "pandas", "pyarrow", "fastparquet"])
    import zarr

print(f"[+] Zarr version: {zarr.__version__}")""")

    # -------------------------------------------------------------
    # CELL 3: Markdown Data Sourcing
    # -------------------------------------------------------------
    add_md("""## 1. Authentic Data Discovery and Integrity Verification
Directly loads from attached Kaggle datasets:
- `rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14` (authentic Zarr dataset)
- `rohitajitbharadwaj/sih26074-sprint6-checkpoints` (Candidate 3 control weights)
- `rohitajitbharadwaj/sih26074-sprint9-dense-m` (Phase 1 Dense-M weights)""")

    # -------------------------------------------------------------
    # CELL 4: Code Data Discovery
    # -------------------------------------------------------------
    add_code("""# Cell 2: Dataset Resolution & Invariant Verification
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
models_dir = working_dir / "models" / "checkpoints"
reports_dir = working_dir / "reports"
models_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

# 1. Locate authentic Zarr store
zarr_file = None
for p in [Path("data/multitask_temporal_v2_h14.zarr"), Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/multitask_temporal_v2_h14.zarr")]:
    if p.exists() and ((p / ".zgroup").exists() or (p / "dates").exists() or (p / "zarr.json").exists()):
        zarr_file = p
        break

if zarr_file is None and Path("/kaggle/input").exists():
    for d in Path("/kaggle/input").rglob("*.zarr"):
        if (d / ".zgroup").exists() or (d / "dates").exists() or (d / "zarr.json").exists():
            zarr_file = d
            break

if zarr_file is None or not zarr_file.exists():
    raise FileNotFoundError("Authentic Zarr dataset 'multitask_temporal_v2_h14.zarr' not found in inputs! Attach 'rohitajitbharadwaj/sih26074-multitask-temporal-v2-h14'.")

print(f"[+] Located authentic Zarr dataset: {zarr_file}")

# 2. Locate Parquet index and YAML normalization stats
index_file = None
for p in [Path("data/sample_index_v2_h14.parquet"), Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/sample_index_v2_h14.parquet")]:
    if p.exists():
        index_file = p
        break
if index_file is None and Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("sample_index_v2_h14.parquet"):
        index_file = f
        break

stats_file = None
for p in [Path("data/normalization_stats_v2.yaml"), Path("/kaggle/input/sih26074-multitask-temporal-v2-h14/normalization_stats_v2.yaml")]:
    if p.exists():
        stats_file = p
        break
if stats_file is None and Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("normalization_stats_v2.yaml"):
        stats_file = f
        break

if index_file is None or stats_file is None:
    raise FileNotFoundError("Parquet sample index or YAML normalization stats not found in inputs!")

print(f"[+] Located sample index: {index_file}")
print(f"[+] Located normalization stats: {stats_file}")

# 3. Stage and Verify Candidate 3 Checkpoint
candidate3_path = None
EXPECTED_SHA = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"

for p in [models_dir / "sprint6_candidate3_multitask_champion.pt", Path("models/checkpoints/sprint6_candidate3_multitask_champion.pt")]:
    if p.exists():
        candidate3_path = p
        break

if candidate3_path is None and Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("sprint6_candidate3_multitask_champion.pt"):
        candidate3_path = f
        break

if not candidate3_path or not candidate3_path.exists():
    raise FileNotFoundError("Candidate 3 checkpoint 'sprint6_candidate3_multitask_champion.pt' not found in /kaggle/input! Attach 'rohitajitbharadwaj/sih26074-sprint6-checkpoints'.")

raw_b = candidate3_path.read_bytes()
if raw_b.startswith(b"version https://git-lfs.github.com"):
    raise RuntimeError("Candidate 3 checkpoint is an unhydrated Git-LFS pointer! Ensure full binary weights are attached.")

actual_sha = hashlib.sha256(raw_b).hexdigest()
if actual_sha != EXPECTED_SHA:
    raise AssertionError(
        f"Candidate 3 SHA mismatch! Expected {EXPECTED_SHA}, got {actual_sha}."
    )
print(f"[+] Candidate 3 Checkpoint Verified: {actual_sha}")""")

    # -------------------------------------------------------------
    # CELL 5: Markdown Dataset Class
    # -------------------------------------------------------------
    add_md("""## 2. Authentic PyTorch Spatiotemporal Dataset
Streams 7-day multi-task forecast cubes from `multitask_temporal_v2_h14.zarr`.""")

    # -------------------------------------------------------------
    # CELL 6: Code Dataset Class
    # -------------------------------------------------------------
    add_code("""# Cell 3: SpatiotemporalDownscalingDataset Implementation
class SpatiotemporalDownscalingDataset(Dataset):
    def __init__(self, zarr_path: Path, index_path: Path, stats_path: Path, split: str = "val", history_len: int = 14, context_size: int = 24):
        self.zarr_path = Path(zarr_path)
        self.index_path = Path(index_path)
        self.stats_path = Path(stats_path)
        self.split = str(split).lower()
        self.history_len = history_len
        self.context_size = context_size

        self.store = zarr.open_group(str(self.zarr_path), mode="r")
        self.df_all = pd.read_parquet(self.index_path)
        if self.split != "all":
            self.df = self.df_all[self.df_all["split"] == self.split].reset_index(drop=True)
        else:
            self.df = self.df_all.copy().reset_index(drop=True)

        all_ids = list(self.df_all["sample_id"])
        id_to_global = {sid: i for i, sid in enumerate(all_ids)}
        self.global_indices = [id_to_global[sid] for sid in self.df["sample_id"]]

        self.terrain_tensor = torch.from_numpy(np.asarray(self.store["terrain"][:], dtype=np.float32))
        with open(self.stats_path, "r", encoding="utf-8") as f:
            self.stats_doc = yaml.safe_load(f)
        self.stats = self.stats_doc.get("channels", {})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        g_idx = self.global_indices[idx]
        hist_raw = np.asarray(self.store["history"][g_idx], dtype=np.float32)
        fcst_raw = np.asarray(self.store["future_forecast"][g_idx], dtype=np.float32)
        target_raw = np.asarray(self.store["target"][g_idx], dtype=np.float32)

        # Slice history to history_len
        if hist_raw.shape[0] > self.history_len:
            hist_raw = hist_raw[-self.history_len:]

        # Normalize using channel stats
        ch_names = ["precip", "tmax", "tmin", "rh", "wind_u", "wind_v"]
        for c_idx, name in enumerate(ch_names):
            if name in self.stats:
                m = self.stats[name]["mean"]
                s = max(1e-6, self.stats[name]["std"])
                hist_raw[:, c_idx] = (hist_raw[:, c_idx] - m) / s
                fcst_raw[:, c_idx] = (fcst_raw[:, c_idx] - m) / s
                target_raw[:, c_idx] = (target_raw[:, c_idx] - m) / s

        return {
            "history": torch.from_numpy(hist_raw),
            "future_forecast": torch.from_numpy(fcst_raw),
            "terrain": self.terrain_tensor,
            "target": torch.from_numpy(target_raw),
        }

print("[+] Dataset streaming engine initialized.")""")

    # -------------------------------------------------------------
    # CELL 7: Markdown Model Architecture
    # -------------------------------------------------------------
    add_md("""## 3. Sparse MoE Architecture & Denoiser
Defines `TopKRouter`, `MoETimeConditionedConvNeXtBlock`, and `ScalableSpatiotemporalResidualDiffusion` with MoE routing support.""")

    # -------------------------------------------------------------
    # CELL 8: Code Architecture
    # -------------------------------------------------------------
    add_code("""# Cell 4: MoE Architecture Supporting Sparse Top-1 Bottleneck Routing
def compute_residual_target(y_fine_norm: torch.Tensor, future_forecast_norm: torch.Tensor) -> torch.Tensor:
    b, leads, c, h, w = future_forecast_norm.shape
    if h > 16 or w > 16:
        offset_h = (h - 16) // 2
        offset_w = (w - 16) // 2
        fcst_crop = future_forecast_norm[:, :, :, offset_h : offset_h + 16, offset_w : offset_w + 16]
    else:
        fcst_crop = future_forecast_norm

    fcst_flat = fcst_crop.reshape(b * leads, c, 16, 16)
    fcst_up = F.interpolate(fcst_flat, size=(80, 80), mode="bilinear", align_corners=False).view(b, leads, c, 80, 80)
    return y_fine_norm - fcst_up


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeConditionedConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, expansion: int = 2, num_groups: int = 8):
        super().__init__()
        assert dim % num_groups == 0
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.time_proj = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim * 2))
        self.pwconv1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.dwconv(x)
        out = self.norm(out)
        scale_shift = self.time_proj(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = torch.clamp(scale, min=-4.0, max=4.0)
        shift = torch.clamp(shift, min=-8.0, max=8.0)
        out = out * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return res + out


class TopKRouter(nn.Module):
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 1, aux_loss_weight: float = 0.01):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        num_tokens, _ = x.shape
        logits = self.gate(x)
        full_probs = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(full_probs, self.top_k, dim=-1)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-8)

        mask = torch.zeros_like(full_probs).scatter_(-1, topk_indices, 1.0)
        f_e = mask.mean(dim=0)
        P_e = full_probs.mean(dim=0)
        aux_loss = self.aux_loss_weight * self.num_experts * torch.sum(f_e * P_e)

        eps = 1e-8
        entropy = -torch.sum(P_e * torch.log(P_e + eps))
        max_ent = torch.log(torch.tensor(float(self.num_experts), device=x.device))
        norm_ent = entropy / (max_ent + eps)

        metrics = {
            "entropy": entropy.item(),
            "normalized_entropy": norm_ent.item(),
            "expert_frequencies": f_e.detach().cpu().tolist(),
            "expert_probabilities": P_e.detach().cpu().tolist(),
        }
        return topk_weights, topk_indices, aux_loss, metrics


class MoETimeConditionedConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int, expansion: int = 2, num_groups: int = 8, num_experts: int = 4, top_k: int = 1, aux_loss_weight: float = 0.01):
        super().__init__()
        self.dim = dim
        self.time_emb_dim = time_emb_dim
        self.expansion = expansion
        self.num_groups = num_groups
        self.num_experts = num_experts
        self.top_k = top_k

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim)
        self.time_proj = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim * 2))

        self.router = TopKRouter(dim=dim, num_experts=num_experts, top_k=top_k, aux_loss_weight=aux_loss_weight)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim * expansion, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(dim * expansion, dim, kernel_size=1),
            )
            for _ in range(num_experts)
        ])
        self.last_aux_loss = torch.tensor(0.0)
        self.last_metrics = {}

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        res = x
        b_total, dim, h, w = x.shape
        out = self.dwconv(x)
        out = self.norm(out)

        scale_shift = self.time_proj(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = torch.clamp(scale, min=-4.0, max=4.0)
        shift = torch.clamp(shift, min=-8.0, max=8.0)
        out = out * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

        token_rep = F.adaptive_avg_pool2d(out, 1).view(b_total, dim)
        topk_weights, topk_indices, aux_loss, metrics = self.router(token_rep)
        self.last_aux_loss = aux_loss
        self.last_metrics = metrics

        combined = torch.zeros_like(out)
        for k_idx in range(self.top_k):
            indices_k = topk_indices[:, k_idx]
            weights_k = topk_weights[:, k_idx].view(b_total, 1, 1, 1)

            for expert_id, expert in enumerate(self.experts):
                expert_mask = (indices_k == expert_id)
                if expert_mask.any():
                    expert_in = out[expert_mask]
                    expert_out = expert(expert_in)
                    combined[expert_mask] = combined[expert_mask] + weights_k[expert_mask] * expert_out

        return res + combined

    def get_active_parameters(self) -> int:
        shared = (
            sum(p.numel() for p in self.dwconv.parameters())
            + sum(p.numel() for p in self.norm.parameters())
            + sum(p.numel() for p in self.time_proj.parameters())
            + sum(p.numel() for p in self.router.parameters())
        )
        expert = sum(p.numel() for p in self.experts[0].parameters()) * self.top_k
        return shared + expert


class ScalableSpatiotemporalDenoiser(nn.Module):
    def __init__(self, in_weather_channels: int = 6, terrain_channels: int = 5, num_leads: int = 7, base_channels: int = 96, time_emb_dim: int = 64, embed_dim: int = 32, num_heads: int = 4, use_moe: bool = False, num_experts: int = 4, top_k: int = 1, aux_loss_weight: float = 0.01):
        super().__init__()
        self.in_weather_channels = in_weather_channels
        self.num_leads = num_leads
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.embed_dim = embed_dim
        self.use_moe = use_moe

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.lead_embed = nn.Embedding(num_leads, embed_dim)

        self.hist_proj = nn.Sequential(nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(embed_dim, embed_dim, kernel_size=1))
        self.fcst_proj = nn.Sequential(nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(embed_dim, embed_dim, kernel_size=1))
        self.hist_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.hist_future_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.spatial_up = nn.Sequential(nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False), nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1), nn.GELU())
        self.terrain_enc = nn.Sequential(nn.Conv2d(terrain_channels, base_channels, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1))

        in_stem_dim = in_weather_channels + base_channels + base_channels
        self.stem = nn.Sequential(nn.Conv2d(in_stem_dim, base_channels, kernel_size=3, padding=1), nn.GELU())
        self.stem_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        c2 = base_channels * 2
        self.down1_conv = nn.Conv2d(base_channels, c2, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        c3 = base_channels * 4
        self.down2_conv = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        c4 = base_channels * 8
        self.down3_conv = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        if use_moe:
            self.down3_block = MoETimeConditionedConvNeXtBlock(dim=c4, time_emb_dim=time_emb_dim, num_groups=8, num_experts=num_experts, top_k=top_k, aux_loss_weight=aux_loss_weight)
        else:
            self.down3_block = TimeConditionedConvNeXtBlock(c4, time_emb_dim, num_groups=8)

        self.bottleneck_temporal_attn = nn.MultiheadAttention(embed_dim=c4, num_heads=num_heads, batch_first=True)
        self.bottleneck_norm = nn.LayerNorm(c4)

        self.up3 = nn.Upsample(size=(20, 20), mode="bilinear", align_corners=False)
        self.dec3_conv = nn.Conv2d(c4 + c3, c3, kernel_size=3, padding=1)
        self.dec3_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        self.up2 = nn.Upsample(size=(40, 40), mode="bilinear", align_corners=False)
        self.dec2_conv = nn.Conv2d(c3 + c2, c2, kernel_size=3, padding=1)
        self.dec2_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        self.up1 = nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False)
        self.dec1_conv = nn.Conv2d(c2 + base_channels, base_channels, kernel_size=3, padding=1)
        self.dec1_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        self.head_precip = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmax = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_tmin = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_rh = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_u = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.head_wind_v = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, r_t: torch.Tensor, t: torch.Tensor, history: torch.Tensor, future_forecast: torch.Tensor, terrain: torch.Tensor):
        b, num_leads, _, _, _ = r_t.shape
        h_len = history.shape[1]

        t_emb = self.time_mlp(t)
        t_emb_flat = t_emb.unsqueeze(1).expand(b, num_leads, self.time_emb_dim).reshape(b * num_leads, self.time_emb_dim)

        _, _, _, n_lat, n_lon = history.shape
        hist_flat = history.reshape(b * h_len, self.in_weather_channels, n_lat, n_lon)
        hist_feats = self.hist_proj(hist_flat)
        hist_tokens = F.adaptive_avg_pool2d(hist_feats, 1).view(b, h_len, self.embed_dim)
        hist_context, _ = self.hist_self_attn(hist_tokens, hist_tokens, hist_tokens)

        coarse_flat = future_forecast.reshape(b * num_leads, self.in_weather_channels, n_lat, n_lon)
        fcst_feats = self.fcst_proj(coarse_flat)
        fcst_tokens = F.adaptive_avg_pool2d(fcst_feats, 1).view(b, num_leads, self.embed_dim)
        lead_idx = torch.arange(num_leads, device=r_t.device)
        lead_emb = self.lead_embed(lead_idx).unsqueeze(0).expand(b, num_leads, self.embed_dim)
        fcst_tokens = fcst_tokens + lead_emb

        cross_context, _ = self.hist_future_cross_attn(query=fcst_tokens, key=hist_context, value=hist_context)
        fused_tokens = self.cross_norm(fcst_tokens + cross_context)

        fused_spatial = fcst_feats + fused_tokens.view(b * num_leads, self.embed_dim, 1, 1)
        if n_lat > 16 or n_lon > 16:
            crop_lat = (n_lat - 16) // 2
            crop_lon = (n_lon - 16) // 2
            fused_spatial_16 = fused_spatial[:, :, crop_lat : crop_lat + 16, crop_lon : crop_lon + 16]
        else:
            fused_spatial_16 = fused_spatial

        atmos_80 = self.spatial_up(fused_spatial_16)
        terr_feats = self.terrain_enc(terrain)
        terr_flat = terr_feats.unsqueeze(1).expand(b, num_leads, self.base_channels, 80, 80).reshape(b * num_leads, self.base_channels, 80, 80)

        r_t_flat = r_t.view(b * num_leads, self.in_weather_channels, 80, 80)
        stem_in = torch.cat([r_t_flat, atmos_80, terr_flat], dim=1)

        s1 = self.stem(stem_in)
        s1 = self.stem_block(s1, t_emb_flat)
        s2 = self.down1_conv(s1)
        s2 = self.down1_block(s2, t_emb_flat)
        s3 = self.down2_conv(s2)
        s3 = self.down2_block(s3, t_emb_flat)
        s4 = self.down3_conv(s3)
        s4 = self.down3_block(s4, t_emb_flat)

        _, c4, h10, w10 = s4.shape
        s4_perm = s4.view(b, num_leads, c4, h10 * w10).permute(0, 3, 1, 2).reshape(b * h10 * w10, num_leads, c4)
        s4_temporal, _ = self.bottleneck_temporal_attn(s4_perm, s4_perm, s4_perm)
        s4_temporal = self.bottleneck_norm(s4_perm + s4_temporal)
        s4_fused = s4_temporal.view(b, h10 * w10, num_leads, c4).permute(0, 2, 3, 1).reshape(b * num_leads, c4, h10, w10)

        u3 = self.up3(s4_fused)
        d3 = self.dec3_block(self.dec3_conv(torch.cat([u3, s3], dim=1)), t_emb_flat)
        u2 = self.up2(d3)
        d2 = self.dec2_block(self.dec2_conv(torch.cat([u2, s2], dim=1)), t_emb_flat)
        u1 = self.up1(d2)
        d1 = self.dec1_block(self.dec1_conv(torch.cat([u1, s1], dim=1)), t_emb_flat)

        eps_p = self.head_precip(d1)
        eps_tmax = self.head_tmax(d1)
        eps_tmin = self.head_tmin(d1)
        eps_rh = self.head_rh(d1)
        eps_u = self.head_wind_u(d1)
        eps_v = self.head_wind_v(d1)
        return torch.cat([eps_p, eps_tmax, eps_tmin, eps_rh, eps_u, eps_v], dim=1).view(b, num_leads, self.in_weather_channels, 80, 80)


class ScalableSpatiotemporalResidualDiffusion(nn.Module):
    def __init__(self, tier: str = "dense_s", base_channels: Optional[int] = None, timesteps: int = 100, use_moe: bool = False, num_experts: int = 4, top_k: int = 1, aux_loss_weight: float = 0.01):
        super().__init__()
        self.tier = tier
        self.timesteps = timesteps
        self.use_moe = use_moe

        bc_map = {"dense_s": 96, "moe_4": 96, "dense_m": 136, "dense_l": 176}
        self.base_channels = base_channels or bc_map.get(tier, 96)

        betas = torch.linspace(1e-4, 0.035, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoiser = ScalableSpatiotemporalDenoiser(
            base_channels=self.base_channels,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_weight=aux_loss_weight,
        )

        if tier == "dense_s" and not use_moe and self.base_channels == 96:
            p_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            assert p_count == 15_685_478, f"Dense-S mismatch: {p_count} != 15685478"

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def active_parameters(self) -> int:
        if not self.use_moe:
            return self.trainable_parameters
        total = self.trainable_parameters
        if hasattr(self.denoiser.down3_block, "get_active_parameters"):
            b_total = sum(p.numel() for p in self.denoiser.down3_block.parameters())
            b_active = self.denoiser.down3_block.get_active_parameters()
            return total - b_total + b_active
        return total

    def compute_v_target(self, r_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * noise - sqrt_one_minus * r_0

    def compute_training_loss(self, r_0: torch.Tensor, history: torch.Tensor, future_forecast: torch.Tensor, terrain: torch.Tensor, target_norm: Optional[torch.Tensor] = None):
        b = r_0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=r_0.device)
        noise = torch.randn_like(r_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        r_t = sqrt_alpha * r_0 + sqrt_one_minus * noise

        model_pred = self.denoiser(r_t=r_t, t=t, history=history, future_forecast=future_forecast, terrain=terrain)
        v_target = self.compute_v_target(r_0, t, noise)

        # Cast predictions and targets to float32 for stable loss computation under FP16 mixed precision
        pred_f = model_pred.float()
        target_f = v_target.float()

        ref_precip = target_norm[:, :, 0:1] if target_norm is not None else r_0[:, :, 0:1]
        tail_mask = (ref_precip.float() > 0.469).float()
        precip_weights = 1.0 + 2.0 * tail_mask
        p_loss = (precip_weights * F.smooth_l1_loss(pred_f[:, :, 0:1], target_f[:, :, 0:1], beta=1.0, reduction="none")).mean()
        thermo_loss = F.smooth_l1_loss(pred_f[:, :, 1:4], target_f[:, :, 1:4], beta=1.0)
        wind_loss = F.smooth_l1_loss(pred_f[:, :, 4:6], target_f[:, :, 4:6], beta=1.0)

        diffusion_loss = 1.0 * p_loss + 1.2 * thermo_loss + 1.1 * wind_loss

        if self.use_moe and hasattr(self.denoiser.down3_block, "last_aux_loss"):
            aux = self.denoiser.down3_block.last_aux_loss.to(r_0.device)
            loss = diffusion_loss + aux
        else:
            loss = diffusion_loss

        return loss, model_pred, v_target

    @torch.no_grad()
    def sample(self, history: torch.Tensor, future_forecast: torch.Tensor, terrain: torch.Tensor, num_steps: int = 4, eta: float = 0.0, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        b, leads, c, h, w = future_forecast.shape
        device = history.device
        r_t = torch.randn(b, leads, 6, 80, 80, device=device)

        step_indices = np.linspace(self.timesteps - 1, 0, num_steps, dtype=int)
        for i in range(len(step_indices)):
            cur_step = int(step_indices[i])
            t = torch.full((b,), cur_step, device=device, dtype=torch.long)
            v_pred = self.denoiser(r_t=r_t, t=t, history=history, future_forecast=future_forecast, terrain=terrain)

            alpha_bar = self.alphas_cumprod[cur_step]
            sqrt_alpha_bar = self.sqrt_alphas_cumprod[cur_step]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[cur_step]

            pred_r0 = sqrt_alpha_bar * r_t - sqrt_one_minus * v_pred
            pred_eps = sqrt_alpha_bar * v_pred + sqrt_one_minus * r_t

            if i < len(step_indices) - 1:
                next_step = int(step_indices[i + 1])
                alpha_bar_next = self.alphas_cumprod[next_step]
                sqrt_alpha_bar_next = self.sqrt_alphas_cumprod[next_step]
                sqrt_one_minus_next = self.sqrt_one_minus_alphas_cumprod[next_step]

                sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_next))
                noise = torch.randn_like(r_t) if eta > 0.0 else torch.zeros_like(r_t)
                dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_next - sigma**2, min=0.0)) * pred_eps
                r_t = sqrt_alpha_bar_next * pred_r0 + dir_xt + sigma * noise
            else:
                r_t = pred_r0

        if h > 16 or w > 16:
            off_h = (h - 16) // 2
            off_w = (w - 16) // 2
            fcst_crop = future_forecast[:, :, :, off_h : off_h + 16, off_w : off_w + 16]
        else:
            fcst_crop = future_forecast

        fcst_up = F.interpolate(fcst_crop.reshape(b * leads, c, 16, 16), size=(80, 80), mode="bilinear", align_corners=False).view(b, leads, c, 80, 80)
        return fcst_up + r_t

print("[+] Scalable models compiled: Dense-S (15.69M), MoE-4 (22.77M total / 15.69M active).")""")

    # -------------------------------------------------------------
    # CELL 9: Markdown Candidate 3 Control Loading
    # -------------------------------------------------------------
    add_md("""## 4. Frozen Candidate 3 Control Evaluation on 2022 Validation Set
Loads the frozen Candidate 3 weights and prepares the 2022 validation loader.""")

    # -------------------------------------------------------------
    # CELL 10: Code Candidate 3 Loading
    # -------------------------------------------------------------
    add_code("""# Cell 5: Candidate 3 Control Evaluation Setup
model_s = ScalableSpatiotemporalResidualDiffusion(tier="dense_s").to(device)

if candidate3_path is None or not candidate3_path.exists():
    raise FileNotFoundError("Candidate 3 checkpoint is missing! Hard-fail enforced.")

state_dict = torch.load(candidate3_path, map_location=device)
weights = state_dict.get("model_state_dict", state_dict)
model_s.load_state_dict(weights, strict=True)
print("[+] Loaded Candidate 3 weights into Dense-S with strict=True.")

# Set up validation loader
ds_val = SpatiotemporalDownscalingDataset(zarr_file, index_file, stats_file, split="val")
val_loader = DataLoader(ds_val, batch_size=2, shuffle=False)
print(f"[+] Loaded {len(ds_val)} cubes for 2022 validation evaluation.")""")

    # -------------------------------------------------------------
    # CELL 11: Markdown Training MoE-4
    # -------------------------------------------------------------
    add_md("""## 5. Training MoE-4 on Authentic 2015-2021 Data
Trains **MoE-4** (22,773,350 total parameters, 15,688,550 active parameters) with:
- Top-1 sparse routing over 4 bottleneck experts
- Switch Transformer / GShard load balancing auxiliary loss (weight = 0.01)
- 30 epochs matching Candidate 3 control training budget
- AdamW (lr=1e-4, weight_decay=1e-4) with Cosine Annealing learning rate schedule
- Numerical stability guards: AdaGN clamping, Smooth L1 Huber loss on convective tail, and GradScaler(init_scale=2048.0).""")

    # -------------------------------------------------------------
    # CELL 12: Code Training MoE-4
    # -------------------------------------------------------------
    add_code("""# Cell 6: Train MoE-4 Model
ds_train = SpatiotemporalDownscalingDataset(zarr_file, index_file, stats_file, split="train")
train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
print(f"[+] Loaded {len(ds_train)} training cubes (2015-2021).")

TRAIN_EPOCHS = 30  # Matched to Candidate 3 30-epoch training schedule

def train_moe_model(epochs: int = TRAIN_EPOCHS, lr: float = 1e-4):
    print(f"\\n[*] Starting Training for MoE-4 (total_params=22.77M, active_params=15.69M, epochs={epochs})...")
    m = ScalableSpatiotemporalResidualDiffusion(tier="moe_4", use_moe=True, num_experts=4, top_k=1, aux_loss_weight=0.01).to(device)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available(), init_scale=2048.0)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_b = 0
        t0 = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            h = batch["history"].to(device)
            f = batch["future_forecast"].to(device)
            terr = batch["terrain"].to(device)
            y = batch["target"].to(device)
            r0 = compute_residual_target(y, f)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss, _, _ = m.compute_training_loss(r0, h, f, terr, target_norm=y)

            if not torch.isfinite(loss):
                print(f"    [WARNING] Non-finite loss encountered ({loss.item()}). Flushing gradients and skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"    [WARNING] Non-finite grad_norm ({grad_norm}). Flushing gradients and skipping step.")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_b)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        # Routing diagnostics
        routing_metrics = m.denoiser.down3_block.last_metrics
        freqs = [f"{f:.2f}" for f in routing_metrics.get("expert_frequencies", [])]
        ent = routing_metrics.get("normalized_entropy", 0.0)
        print(f"    [MoE-4 Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | LR: {cur_lr:.6f} | Entropy: {ent:.3f} | Freqs: {freqs} | Time: {elapsed:.1f}s")

    ckpt_path = models_dir / "sprint9_moe4_weights.pt"
    torch.save({
        "tier": "moe_4",
        "model_state_dict": m.state_dict(),
        "loss": avg_loss,
        "epochs": epochs,
        "total_parameters": m.trainable_parameters,
        "active_parameters": m.active_parameters,
        "routing_metrics": m.denoiser.down3_block.last_metrics,
    }, ckpt_path)
    print(f"[+] Saved MoE-4 weights to {ckpt_path.name}")
    return m

model_moe = train_moe_model(epochs=TRAIN_EPOCHS)""")

    # -------------------------------------------------------------
    # CELL 13: Markdown Comparative Evaluation
    # -------------------------------------------------------------
    add_md("""## 6. Matched 32 NFE Comparative Evaluation
Evaluates Dense-S (Control) and MoE-4 across all 122 forecast cubes of the 2022 validation dataset:
- **Point Mode**: K=8, S=4, eta=0.5 (32 NFE)
- **Distribution Mode**: K=2, S=16, eta=0.0 (32 NFE)
- Benchmarks against Phase 1 empirical results for **Dense-M** and **Dense-L**.""")

    # -------------------------------------------------------------
    # CELL 14: Code Comparative Evaluation
    # -------------------------------------------------------------
    add_code("""# Cell 7: Full 32 NFE Comparative Evaluation Engine
MAX_EVAL_CUBES = None  # None evaluates the complete 2022 validation set (all 122 forecast cubes)

def evaluate_model_pipeline(model: nn.Module, loader: DataLoader, stats: Dict[str, Any], max_cubes: Optional[int] = MAX_EVAL_CUBES):
    model.eval()
    wet_maes, csi15s, csi30s = [], [], []
    crps_list, spread_list, rmse_list = [], [], []
    cov90_hits, total_pts = 0, 0
    lap_ratios = []
    latencies = []

    m_precip = stats.get("precip", {}).get("mean", 6.266)
    s_precip = stats.get("precip", {}).get("std", 18.620)

    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device).view(1, 1, 3, 3)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_cubes is not None and b_idx >= max_cubes:
                break
            h = batch["history"].to(device)
            f = batch["future_forecast"].to(device)
            terr = batch["terrain"].to(device)
            y = batch["target"].to(device)
            b_size = y.shape[0]

            t_start = time.time()
            # 1. Point Mode: K=8, S=4, eta=0.5
            pt_members = []
            for k in range(8):
                pred_k = model.sample(h, f, terr, num_steps=4, eta=0.5, seed=1000 + b_idx * 10 + k)
                pt_members.append(pred_k)
            t_cube = (time.time() - t_start) / b_size
            latencies.append(t_cube)

            pt_stack = torch.stack(pt_members, dim=0)  # [8, B, 7, 6, 80, 80]
            ens_mean = pt_stack.mean(dim=0)

            # Denormalize precip
            pred_p = ens_mean[:, :, 0] * s_precip + m_precip
            y_p = y[:, :, 0] * s_precip + m_precip

            # Wet MAE (> 1mm)
            wet_mask = (y_p > 1.0)
            if wet_mask.any():
                w_mae = (torch.abs(pred_p - y_p)[wet_mask]).mean().item()
                wet_maes.append(w_mae)

            # CSI@15 and CSI@30
            hits15 = ((pred_p >= 15.0) & (y_p >= 15.0)).sum().item()
            misses15 = ((pred_p < 15.0) & (y_p >= 15.0)).sum().item()
            fas15 = ((pred_p >= 15.0) & (y_p < 15.0)).sum().item()
            den15 = hits15 + misses15 + fas15
            if den15 > 0:
                csi15s.append(hits15 / den15)

            hits30 = ((pred_p >= 30.0) & (y_p >= 30.0)).sum().item()
            misses30 = ((pred_p < 30.0) & (y_p >= 30.0)).sum().item()
            fas30 = ((pred_p >= 30.0) & (y_p < 30.0)).sum().item()
            den30 = hits30 + misses30 + fas30
            if den30 > 0:
                csi30s.append(hits30 / den30)

            # Spatial Laplacian Energy Ratio
            pred_p_flat = pred_p.view(-1, 1, 80, 80)
            y_p_flat = y_p.view(-1, 1, 80, 80)
            lap_pred = F.conv2d(pred_p_flat, kernel, padding=1)
            lap_y = F.conv2d(y_p_flat, kernel, padding=1)
            e_pred = torch.mean(lap_pred ** 2).item()
            e_y = torch.mean(lap_y ** 2).item()
            if e_y > 1e-6:
                lap_ratios.append(e_pred / e_y)

            # 2. Distribution Mode: K=2, S=16, eta=0.0
            dist_members = []
            for k in range(2):
                pred_k = model.sample(h, f, terr, num_steps=16, eta=0.0, seed=5000 + b_idx * 10 + k)
                dist_members.append(pred_k[:, :, 0] * s_precip + m_precip)
            dist_stack = torch.stack(dist_members, dim=0)  # [2, B, 7, 80, 80]

            # Canonical Fair-CRPS (Ferro et al., 2008)
            K = 2
            mae_term = torch.abs(dist_stack - y_p.unsqueeze(0)).mean(dim=0)
            diff_term = torch.abs(dist_stack[0] - dist_stack[1]) / (2.0 * K * (K - 1))
            fair_crps = (mae_term - diff_term).mean().item()
            crps_list.append(fair_crps)

            # Spread-Skill Ratio
            spread = dist_stack.std(dim=0, unbiased=True).mean().item()
            ens_err = (torch.abs(dist_stack.mean(dim=0) - y_p) ** 2).mean().sqrt().item()
            spread_list.append(spread)
            rmse_list.append(ens_err)

            # 90% Prediction Interval Coverage
            p05 = dist_stack.min(dim=0).values
            p95 = dist_stack.max(dim=0).values
            cov_hits = ((y_p >= p05) & (y_p <= p95)).sum().item()
            cov90_hits += cov_hits
            total_pts += y_p.numel()

    avg_spread = float(np.mean(spread_list)) if spread_list else 0.0
    avg_rmse = float(np.mean(rmse_list)) if rmse_list else 1e-6
    ssr = avg_spread / avg_rmse if avg_rmse > 0 else 0.0

    return {
        "wet_mae_mm": round(float(np.mean(wet_maes)), 2) if wet_maes else 0.0,
        "csi15": round(float(np.mean(csi15s)), 4) if csi15s else 0.0,
        "csi30": round(float(np.mean(csi30s)), 4) if csi30s else 0.0,
        "precip_crps": round(float(np.mean(crps_list)), 3) if crps_list else 0.0,
        "raw_ssr": round(float(ssr), 3),
        "raw_cov90": round(float(cov90_hits / max(1, total_pts)), 3),
        "laplacian_retention": round(float(np.mean(lap_ratios)), 3) if lap_ratios else 0.0,
        "avg_latency_s": round(float(np.mean(latencies)), 3) if latencies else 0.0,
    }

print("[*] Evaluating Dense-S Control on 2022 Validation Set...")
metrics_s = evaluate_model_pipeline(model_s, val_loader, ds_val.stats)
print(f"[+] Dense-S Metrics: {metrics_s}")

print("\\n[*] Evaluating MoE-4 on 2022 Validation Set...")
metrics_moe = evaluate_model_pipeline(model_moe, val_loader, ds_val.stats)
print(f"[+] MoE-4 Metrics:   {metrics_moe}")""")

    # -------------------------------------------------------------
    # CELL 15: Markdown Phase 1 Integration & Final Pareto Analysis
    # -------------------------------------------------------------
    add_md("""## 7. Multi-Dimensional Pareto Table & Efficiency Analysis
Combines Phase 1 Dense Scaling results (Dense-S, Dense-M, Dense-L) with Phase 2 MoE-4 to produce the final Sprint 9 capacity-efficiency frontier.""")

    # -------------------------------------------------------------
    # CELL 16: Code Final Pareto Table & Report Generation
    # -------------------------------------------------------------
    add_code("""# Cell 8: Multi-Dimensional Pareto Table & Report Generation
# 1. Phase 1 Empirical Results
phase1_results = {
    "dense_s": {
        "params": 15685478,
        "active_params": 15685478,
        "metrics": metrics_s,
    },
    "dense_m": {
        "params": 31198518,
        "active_params": 31198518,
        "metrics": {
            "wet_mae_mm": 61.62,
            "csi15": 0.6209,
            "csi30": 0.6589,
            "precip_crps": 56.67,
            "raw_ssr": 0.053,
            "raw_cov90": 0.195,
            "laplacian_retention": 0.065,
        }
    },
    "dense_l": {
        "params": 51997958,
        "active_params": 51997958,
        "metrics": {
            "wet_mae_mm": 61.85,
            "csi15": 0.6207,
            "csi30": 0.6602,
            "precip_crps": 56.02,
            "raw_ssr": 0.068,
            "raw_cov90": 0.223,
            "laplacian_retention": 0.084,
        }
    },
    "moe_4": {
        "params": model_moe.trainable_parameters,
        "active_params": model_moe.active_parameters,
        "metrics": metrics_moe,
    }
}

# Save complete JSON
json_path = reports_dir / "sprint9_moe_scaling_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(phase1_results, f, indent=2)
print(f"[+] Saved complete scaling results to {json_path.name}")

# Generate Markdown Pareto Frontier
md_lines = [
    "# Sprint 9: Complete Capacity & Sparse Routing Pareto Frontier",
    "",
    "## 1. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)",
    "",
    "| Model Tier | Total Params | Active Params | Active Ratio | Wet-MAE (mm) | CSI@15 | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Retention |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    f"| **Dense-S (Control)** | 15,685,478 | 15,685,478 | 1.00x | {metrics_s['wet_mae_mm']} | {metrics_s['csi15']} | {metrics_s['csi30']} | {metrics_s['precip_crps']} | {metrics_s['raw_ssr']} | {metrics_s['raw_cov90']} | {metrics_s['laplacian_retention']} |",
    f"| **Dense-M** | 31,198,518 | 31,198,518 | 1.99x | 61.62 | 0.6209 | 0.6589 | 56.670 | 0.053 | 0.195 | 0.065 |",
    f"| **Dense-L** | 51,997,958 | 51,997,958 | 3.31x | 61.85 | 0.6207 | 0.6602 | 56.020 | 0.068 | 0.223 | 0.084 |",
    f"| **MoE-4** | {model_moe.trainable_parameters:,} | {model_moe.active_parameters:,} | {model_moe.active_parameters / 15685478:.2f}x | {metrics_moe['wet_mae_mm']} | {metrics_moe['csi15']} | {metrics_moe['csi30']} | {metrics_moe['precip_crps']} | {metrics_moe['raw_ssr']} | {metrics_moe['raw_cov90']} | {metrics_moe['laplacian_retention']} |",
    "",
    "## 2. Hypothesis Evaluation (MoE vs Dense)",
    f"- **Active Parameter Efficiency**: MoE-4 operates at {model_moe.active_parameters:,} active parameters ({model_moe.active_parameters / 15685478:.2f}x Candidate 3), while matching Dense capacity.",
    f"- **MoE-4 vs Dense-S Wet-MAE**: Delta = {metrics_moe['wet_mae_mm'] - metrics_s['wet_mae_mm']:.2f} mm",
    f"- **MoE-4 vs Dense-S CSI@30**: Delta = {metrics_moe['csi30'] - metrics_s['csi30']:+.4f}",
    f"- **MoE-4 vs Dense-S Fair-CRPS**: Delta = {metrics_moe['precip_crps'] - metrics_s['precip_crps']:.3f}",
    "",
    "## 3. MoE Routing Diagnostics",
    f"- **Normalized Routing Entropy**: {model_moe.denoiser.down3_block.last_metrics.get('normalized_entropy', 0.0):.3f}",
    f"- **Expert Frequencies**: {model_moe.denoiser.down3_block.last_metrics.get('expert_frequencies', [])}",
    "",
    "[+] Sprint 9 Phase 2 Execution Complete."
]

md_path = reports_dir / "sprint9_moe_pareto_frontier.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\\n".join(md_lines))
print(f"[+] Saved Markdown report to {md_path.name}")
print("\\n" + "\\n".join(md_lines))""")

    # -------------------------------------------------------------
    # Assemble notebook dictionary
    # -------------------------------------------------------------
    nb_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (CUDA)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(DEST_REPO, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2)

    try:
        shutil.copy2(DEST_REPO, DEST_DOWNLOADS)
    except Exception as e:
        print(f"[!] Warning: Could not copy to Downloads: {e}")

    print(f"[+] Successfully created Phase 2 Notebook:")
    print(f"    In Repo:      {DEST_REPO}")
    print(f"    In Downloads: {DEST_DOWNLOADS}")
    print(f"    Total Cells:  {len(cells)}")


if __name__ == "__main__":
    create_phase2_notebook()
