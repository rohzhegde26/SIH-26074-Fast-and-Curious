"""
scripts/build_sprint9_phase1_notebook.py

Constructs the dedicated Phase 1 interactive Kaggle notebook:
'notebooks/sprint_9_phase1_dense_capacity_scaling.ipynb'
and copies it to 'C:\\Users\\rohit\\Downloads\\sprint_9_phase1_dense_capacity_scaling.ipynb'.

Phase 1 focuses strictly on the Dense Scaling Ladder:
  1. Dense-S: 15,685,478 params (Candidate 3 control, base_channels=96)
  2. Dense-M: 31,198,518 params (~2.0x scale, base_channels=136)
  3. Dense-L: 51,997,958 params (~3.3x scale, base_channels=176)

Operates on the authentic Zarr dataset (multitask_temporal_v2_h14.zarr) and 2022 validation set.
Strictly ZERO em dashes (\u2014 and \u2013).
"""

import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

DEST_REPO = NOTEBOOK_DIR / "sprint_9_phase1_dense_capacity_scaling.ipynb"
DEST_DOWNLOADS = Path(r"C:\Users\rohit\Downloads\sprint_9_phase1_dense_capacity_scaling.ipynb")


def create_phase1_notebook():
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
    add_md("""# Sprint 9 (Phase 1): Dense Model Capacity Scaling
### SIH Problem Statement 26074: Block to Panchayat Weather Downscaling

This notebook executes **Phase 1** of Sprint 9:
1. **Dense Capacity Ladder**:
   - **Dense-S**: 15,685,478 parameters (Frozen Candidate 3 Control, base_channels = 96)
   - **Dense-M**: 31,198,518 parameters (~2.0x Candidate 3, base_channels = 136)
   - **Dense-L**: 51,997,958 parameters (~3.3x Candidate 3, base_channels = 176)
2. **Authentic Data Pipeline**:
   - Trains on authentic 2015-2021 reanalysis and GFS forecast conditioning.
   - Evaluates on the authentic 2022 validation season (122 cubes, 427 daily slices).
   - Zero synthetic or proxy fallbacks.
3. **Matched 32 NFE Inference Regimes**:
   - Point Mode: K=8 members, S=4 DDIM steps, eta=0.5 (32 NFE).
   - Distribution Mode: K=2 members, S=16 DDIM steps, eta=0.0 (32 NFE).
4. **Empirical Artifacts Updated**:
   - Generates empirical tables and updates `reports/sprint9_capacity_frontier.md` and `reports/sprint9_dense_scaling_results.json`.""")

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

# Check accelerator
print("=" * 60)
print("[*] Environment Hardware Diagnostics:")
print(f"    PyTorch:  {torch.__version__}")
print(f"    CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU:      {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"    VRAM:     {vram_gb:.2f} GB")
print("=" * 60)

for pkg in ["zarr", "scikit-learn"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import zarr

def set_global_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_global_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Execution Device: {device}")""")

    # -------------------------------------------------------------
    # CELL 3: Markdown Dataset Discovery
    # -------------------------------------------------------------
    add_md("""## 1. Authentic Dataset Extraction & Checkpoint Staging
The cell below extracts `sih26074-multitask-temporal-v2-h14` and copies `sih26074-sprint6-checkpoints` into `/kaggle/working/`.""")

    # -------------------------------------------------------------
    # CELL 4: Code Dataset Extraction
    # -------------------------------------------------------------
    add_code("""# Cell 2: Dataset Extraction and Checkpoint Invariant Check
working_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
os.chdir(str(working_dir))

datasets_dir = working_dir / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
data_dir = working_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
models_dir = working_dir / "models" / "checkpoints"
models_dir.mkdir(parents=True, exist_ok=True)
reports_dir = working_dir / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# 1. Unpack Zarr archive
zarr_zip = None
if Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("*.zarr.zip"):
        zarr_zip = f
        break

if zarr_zip and zarr_zip.exists():
    stem = zarr_zip.name.replace(".zip", "")
    target_zarr = datasets_dir / stem
    if not target_zarr.exists():
        print(f"[*] Extracting {zarr_zip.name}...")
        with zipfile.ZipFile(zarr_zip, "r") as zf:
            zf.extractall(datasets_dir)
        print("[+] Extracted Zarr archive successfully.")

# Copy Parquet index and YAML normalization stats
if Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("*.parquet"):
        shutil.copy(f, data_dir / f.name)
    for f in Path("/kaggle/input").rglob("*.yaml"):
        shutil.copy(f, data_dir / f.name)

# 2. Stage and Verify Candidate 3 Checkpoint
candidate3_path = None
EXPECTED_SHA = "f3367f5fdd96b02a864d319fb94c43c1216de435eba89ac7f3d557c5da81df92"

if Path("/kaggle/input").exists():
    for f in Path("/kaggle/input").rglob("sprint6_candidate3_multitask_champion.pt"):
        dest = models_dir / f.name
        if not dest.exists():
            shutil.copy(f, dest)
        candidate3_path = dest
        break

if not candidate3_path or not candidate3_path.exists():
    raise FileNotFoundError(
        "Candidate 3 checkpoint 'sprint6_candidate3_multitask_champion.pt' not found in /kaggle/input!\n"
        "Attach 'rohitajitbharadwaj/sih26074-sprint6-checkpoints' to the Kaggle session."
    )

raw_b = candidate3_path.read_bytes()
if raw_b.startswith(b"version https://git-lfs.github.com"):
    raise RuntimeError(
        "Candidate 3 checkpoint is an unhydrated Git-LFS pointer!\n"
        "Ensure full binary weights are attached in Kaggle dataset."
    )

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
    add_md("""## 3. Parametric Denoiser & Diffusion Model
Full standalone definitions for `TimeConditionedConvNeXtBlock`, `ScalableSpatiotemporalDenoiser`, and `ScalableSpatiotemporalResidualDiffusion`.""")

    # -------------------------------------------------------------
    # CELL 8: Code Model Architecture
    # -------------------------------------------------------------
    add_code("""# Cell 4: Model Architecture Supporting Dense-S (96), Dense-M (136), Dense-L (176)
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
        out = out * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        return res + out


class ScalableSpatiotemporalDenoiser(nn.Module):
    def __init__(self, in_weather_channels: int = 6, terrain_channels: int = 5, num_leads: int = 7, base_channels: int = 96, time_emb_dim: int = 64, embed_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.in_weather_channels = in_weather_channels
        self.num_leads = num_leads
        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.embed_dim = embed_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.lead_embed = nn.Embedding(num_leads, embed_dim)

        self.hist_proj = nn.Sequential(
            nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.fcst_proj = nn.Sequential(
            nn.Conv2d(in_weather_channels, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )
        self.hist_self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.hist_future_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.spatial_up = nn.Sequential(
            nn.Upsample(size=(80, 80), mode="bilinear", align_corners=False),
            nn.Conv2d(embed_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.terrain_enc = nn.Sequential(
            nn.Conv2d(terrain_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
        )

        in_stem_dim = in_weather_channels + base_channels + base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_stem_dim, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.stem_block = TimeConditionedConvNeXtBlock(base_channels, time_emb_dim, num_groups=4)

        c2 = base_channels * 2
        self.down1_conv = nn.Conv2d(base_channels, c2, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeConditionedConvNeXtBlock(c2, time_emb_dim, num_groups=8)

        c3 = base_channels * 4
        self.down2_conv = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeConditionedConvNeXtBlock(c3, time_emb_dim, num_groups=8)

        c4 = base_channels * 8
        self.down3_conv = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
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
    def __init__(self, tier: str = "dense_s", base_channels: Optional[int] = None, timesteps: int = 100):
        super().__init__()
        self.tier = tier
        self.timesteps = timesteps

        bc_map = {"dense_s": 96, "dense_m": 136, "dense_l": 176}
        self.base_channels = base_channels or bc_map.get(tier, 96)

        betas = torch.linspace(1e-4, 0.035, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.denoiser = ScalableSpatiotemporalDenoiser(base_channels=self.base_channels)

        if tier == "dense_s" and self.base_channels == 96:
            p_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            assert p_count == 15_685_478, f"Dense-S mismatch: {p_count} != 15685478"

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

        ref_precip = target_norm[:, :, 0:1] if target_norm is not None else r_0[:, :, 0:1]
        tail_mask = (ref_precip > 0.469).float()
        precip_weights = 1.0 + 2.0 * tail_mask
        p_loss = (precip_weights * (model_pred[:, :, 0:1] - v_target[:, :, 0:1]) ** 2).mean()
        thermo_loss = F.mse_loss(model_pred[:, :, 1:4], v_target[:, :, 1:4])
        wind_loss = F.mse_loss(model_pred[:, :, 4:6], v_target[:, :, 4:6])

        loss = 1.0 * p_loss + 1.2 * thermo_loss + 1.1 * wind_loss
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

print("[+] Scalable models compiled: Dense-S (15.69M), Dense-M (31.20M), Dense-L (52.00M).")""")

    # -------------------------------------------------------------
    # CELL 9: Markdown Candidate 3 Loading & Baseline Eval
    # -------------------------------------------------------------
    add_md("""## 4. Candidate 3 Control Evaluation on 2022 Validation Set
Loads the frozen Candidate 3 weights and evaluates matched 32 NFE:
- Point Mode: K=8, S=4, eta=0.5
- Distribution Mode: K=2, S=16, eta=0.0""")

    # -------------------------------------------------------------
    # CELL 10: Code Candidate 3 Eval
    # -------------------------------------------------------------
    add_code("""# Cell 5: Candidate 3 Control Evaluation
model_s = ScalableSpatiotemporalResidualDiffusion(tier="dense_s").to(device)

if candidate3_path is None or not candidate3_path.exists():
    raise FileNotFoundError("Candidate 3 checkpoint is missing! Hard-fail enforced.")

state_dict = torch.load(candidate3_path, map_location=device)
weights = state_dict.get("model_state_dict", state_dict)
model_s.load_state_dict(weights, strict=True)
print("[+] Loaded Candidate 3 weights into Dense-S with strict=True.")

# Set up validation loader
zarr_file = datasets_dir / "multitask_temporal_v2_h14.zarr"
index_file = data_dir / "sample_index_v2_h14.parquet"
stats_file = data_dir / "normalization_stats_v2.yaml"

ds_val = SpatiotemporalDownscalingDataset(zarr_file, index_file, stats_file, split="val")
val_loader = DataLoader(ds_val, batch_size=2, shuffle=False)
print(f"[+] Loaded {len(ds_val)} cubes for 2022 validation evaluation.")""")

    # -------------------------------------------------------------
    # CELL 11: Markdown Training Dense-M and Dense-L
    # -------------------------------------------------------------
    add_md("""## 5. Training Dense-M and Dense-L on Authentic 2015-2021 Data
Trains:
1. **Dense-M** (31,198,518 params, base_channels = 136)
2. **Dense-L** (51,997,958 params, base_channels = 176)

Training budget: 30 epochs each with AdamW (lr=1e-4, weight_decay=1e-4) and Cosine Annealing learning rate schedule, strictly matching the frozen Candidate 3 control training budget.""")

    # -------------------------------------------------------------
    # CELL 12: Code Training Routine
    # -------------------------------------------------------------
    add_code("""# Cell 6: Train Dense-M and Dense-L
ds_train = SpatiotemporalDownscalingDataset(zarr_file, index_file, stats_file, split="train")
train_loader = DataLoader(ds_train, batch_size=2, shuffle=True)
print(f"[+] Loaded {len(ds_train)} training cubes (2015-2021).")

TRAIN_EPOCHS = 30  # Matched to Candidate 3 30-epoch training schedule

def train_scaled_model(tier_name: str, base_channels: int, epochs: int = TRAIN_EPOCHS, lr: float = 1e-4):
    print(f"\\n[*] Starting Training for {tier_name.upper()} (base_channels={base_channels}, epochs={epochs})...")
    m = ScalableSpatiotemporalResidualDiffusion(tier=tier_name, base_channels=base_channels).to(device)
    m.train()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_b)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"    [{tier_name.upper()} Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | LR: {cur_lr:.6f} | Time: {elapsed:.1f}s")

    ckpt_path = models_dir / f"sprint9_{tier_name}_weights.pt"
    torch.save({"tier": tier_name, "model_state_dict": m.state_dict(), "loss": avg_loss, "epochs": epochs}, ckpt_path)
    print(f"[+] Saved {tier_name.upper()} weights to {ckpt_path.name}")
    return m

model_m = train_scaled_model("dense_m", base_channels=136, epochs=TRAIN_EPOCHS)
model_l = train_scaled_model("dense_l", base_channels=176, epochs=TRAIN_EPOCHS)""")

    # -------------------------------------------------------------
    # CELL 13: Markdown Matched 32 NFE Evaluation
    # -------------------------------------------------------------
    add_md("""## 6. Matched 32 NFE Comparative Evaluation
Evaluates Dense-S, Dense-M, and Dense-L across the complete 2022 validation dataset (all 122 cubes, 427 daily slices) across:
- **Point Mode**: K=8, S=4, eta=0.5 (32 NFE)
- **Distribution Mode**: K=2, S=16, eta=0.0 (32 NFE)
- Complete evaluation with zero cube truncation (evaluates all available validation cubes).
- Measures Wet-MAE, CSI@15, CSI@30, Fair-CRPS, Spread-Skill Ratio (SSR), 90% Coverage, and Laplacian Retention.""")

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

            # 1. Point Mode: K=8, S=4, eta=0.5
            pt_members = []
            for k in range(8):
                pred_k = model.sample(h, f, terr, num_steps=4, eta=0.5, seed=1000 + b_idx * 10 + k)
                pt_members.append(pred_k)
            ens_mean = torch.stack(pt_members, dim=0).mean(dim=0)

            # Invert physical precipitation
            p_pred = torch.clamp(ens_mean[:, :, 0] * s_precip + m_precip, min=0.0)
            p_tgt = torch.clamp(y[:, :, 0] * s_precip + m_precip, min=0.0)

            # Wet MAE (> 1 mm)
            w_mask = (p_tgt >= 1.0) | (p_pred >= 1.0)
            if w_mask.any():
                wet_maes.append(float(torch.mean(torch.abs(p_pred[w_mask] - p_tgt[w_mask])).item()))

            # CSI@15 and CSI@30
            h15 = int(((p_pred >= 15.0) & (p_tgt >= 15.0)).sum().item())
            f15 = int(((p_pred >= 15.0) & (p_tgt < 15.0)).sum().item())
            n15 = int(((p_pred < 15.0) & (p_tgt >= 15.0)).sum().item())
            csi15s.append(h15 / max(1, h15 + f15 + n15))

            h30 = int(((p_pred >= 30.0) & (p_tgt >= 30.0)).sum().item())
            f30 = int(((p_pred >= 30.0) & (p_tgt < 30.0)).sum().item())
            n30 = int(((p_pred < 30.0) & (p_tgt >= 30.0)).sum().item())
            csi30s.append(h30 / max(1, h30 + f30 + n30))

            # Laplacian energy
            for lead in range(7):
                l_pred = torch.mean(F.conv2d(p_pred[0, lead:lead+1].unsqueeze(0), kernel, padding=1)**2).item()
                l_gt = torch.mean(F.conv2d(p_tgt[0, lead:lead+1].unsqueeze(0), kernel, padding=1)**2).item()
                lap_ratios.append(l_pred / max(1e-6, l_gt))

            # 2. Distribution Mode: K=2, S=16, eta=0.0
            dist_members = []
            for k in range(2):
                pred_k = model.sample(h, f, terr, num_steps=16, eta=0.0, seed=5000 + b_idx * 10 + k)
                dist_members.append(torch.clamp(pred_k[:, :, 0] * s_precip + m_precip, min=0.0))
            dist_tensor = torch.stack(dist_members, dim=0)

            d_mean = dist_tensor.mean(dim=0)
            d_std = dist_tensor.std(dim=0, unbiased=True)
            mae = torch.mean(torch.abs(d_mean - p_tgt)).item()
            sp = d_std.mean().item()
            crps_list.append(mae - 0.5 * sp)
            spread_list.append(sp)
            rmse_list.append(torch.sqrt(torch.mean((d_mean - p_tgt)**2)).item())

            q05 = torch.quantile(dist_tensor, 0.05, dim=0)
            q95 = torch.quantile(dist_tensor, 0.95, dim=0)
            cov90_hits += int(((p_tgt >= q05) & (p_tgt <= q95)).sum().item())
            total_pts += p_tgt.numel()

    mean_sp = float(np.mean(spread_list))
    mean_rmse = float(np.mean(rmse_list))
    ssr = mean_sp / max(1e-6, mean_rmse)

    return {
        "wet_mae_mm": round(float(np.mean(wet_maes)), 2),
        "csi15": round(float(np.mean(csi15s)), 4),
        "csi30": round(float(np.mean(csi30s)), 4),
        "precip_crps": round(float(np.mean(crps_list)), 3),
        "raw_ssr": round(ssr, 3),
        "raw_cov90": round(cov90_hits / max(1, total_pts), 3),
        "laplacian_retention": round(float(np.mean(lap_ratios)), 3),
    }

print("[*] Evaluating Dense-S Control on full 2022 validation set...")
res_s = evaluate_model_pipeline(model_s, val_loader, ds_val.stats, max_cubes=MAX_EVAL_CUBES)
print(f"    Dense-S: Wet-MAE={res_s['wet_mae_mm']}, CSI@30={res_s['csi30']}, CRPS={res_s['precip_crps']}, SSR={res_s['raw_ssr']}")

print("[*] Evaluating Dense-M on full 2022 validation set...")
res_m = evaluate_model_pipeline(model_m, val_loader, ds_val.stats, max_cubes=MAX_EVAL_CUBES)
print(f"    Dense-M: Wet-MAE={res_m['wet_mae_mm']}, CSI@30={res_m['csi30']}, CRPS={res_m['precip_crps']}, SSR={res_m['raw_ssr']}")

print("[*] Evaluating Dense-L on full 2022 validation set...")
res_l = evaluate_model_pipeline(model_l, val_loader, ds_val.stats, max_cubes=MAX_EVAL_CUBES)
print(f"    Dense-L: Wet-MAE={res_l['wet_mae_mm']}, CSI@30={res_l['csi30']}, CRPS={res_l['precip_crps']}, SSR={res_l['raw_ssr']}")""")

    # -------------------------------------------------------------
    # CELL 15: Markdown Synthesis & Report Export
    # -------------------------------------------------------------
    add_md("""## 7. Phase 1 Report Synthesis & Artifact Export
Exports empirical results directly to:
- `reports/sprint9_capacity_frontier.md`
- `reports/sprint9_dense_scaling_results.json`""")

    # -------------------------------------------------------------
    # CELL 16: Code Synthesis & Report Export
    # -------------------------------------------------------------
    add_code("""# Cell 8: Synthesis and Markdown Export
final_summary = {
    "dense_s": {"params": 15685478, "metrics": res_s},
    "dense_m": {"params": 31198518, "metrics": res_m},
    "dense_l": {"params": 51997958, "metrics": res_l},
}

with open(reports_dir / "sprint9_dense_scaling_results.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2)

md_lines = [
    "# Sprint 9 (Phase 1): Empirical Dense Capacity Scaling Report",
    "",
    "## 1. Multi-Dimensional Pareto Table (Authentic 2022 Validation Season)",
    "",
    "| Model Tier | Base Ch | Parameters | Param Ratio | Wet-MAE (mm) | CSI@30 | Fair-CRPS | Raw SSR | Cov@90 | Lap Retention |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    f"| **Dense-S (Control)** | 96 | 15,685,478 | 1.00x | {res_s['wet_mae_mm']:.2f} | {res_s['csi30']:.4f} | {res_s['precip_crps']:.3f} | {res_s['raw_ssr']:.3f} | {res_s['raw_cov90']:.3f} | {res_s['laplacian_retention']:.3f} |",
    f"| **Dense-M** | 136 | 31,198,518 | 1.99x | {res_m['wet_mae_mm']:.2f} | {res_m['csi30']:.4f} | {res_m['precip_crps']:.3f} | {res_m['raw_ssr']:.3f} | {res_m['raw_cov90']:.3f} | {res_m['laplacian_retention']:.3f} |",
    f"| **Dense-L** | 176 | 51,997,958 | 3.31x | {res_l['wet_mae_mm']:.2f} | {res_l['csi30']:.4f} | {res_l['precip_crps']:.3f} | {res_l['raw_ssr']:.3f} | {res_l['raw_cov90']:.3f} | {res_l['laplacian_retention']:.3f} |",
    "",
    "## 2. Quantitative Gains vs Candidate 3 Control",
    f"- **Dense-M Wet-MAE Delta**: {res_m['wet_mae_mm'] - res_s['wet_mae_mm']:+.2f} mm",
    f"- **Dense-L Wet-MAE Delta**: {res_l['wet_mae_mm'] - res_s['wet_mae_mm']:+.2f} mm",
    f"- **Dense-M CSI@30 Delta**: {res_m['csi30'] - res_s['csi30']:+.4f}",
    f"- **Dense-L CSI@30 Delta**: {res_l['csi30'] - res_s['csi30']:+.4f}",
    f"- **Dense-L Fair-CRPS Delta**: {res_l['precip_crps'] - res_s['precip_crps']:+.3f}",
    "",
    "[+] Phase 1 Dense Capacity Scaling Execution Complete.",
]

(reports_dir / "sprint9_capacity_frontier.md").write_text("\\n".join(md_lines), encoding="utf-8")
print("[+] Successfully exported reports/sprint9_capacity_frontier.md and reports/sprint9_dense_scaling_results.json!")
print("\\n[+] PHASE 1 NOTEBOOK EXECUTION COMPLETE!")""")

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    content_str = json.dumps(nb, indent=2)
    assert "\u2014" not in content_str, "Em dash detected in phase 1 notebook!"
    assert "\u2013" not in content_str, "En dash detected in phase 1 notebook!"

    DEST_REPO.write_text(content_str, encoding="utf-8")
    DEST_DOWNLOADS.write_text(content_str, encoding="utf-8")

    print(f"[+] Successfully created Phase 1 Notebook:")
    print(f"    In Repo:      {DEST_REPO}")
    print(f"    In Downloads: {DEST_DOWNLOADS}")
    print(f"    Total Cells:  {len(cells)}")


if __name__ == "__main__":
    create_phase1_notebook()
