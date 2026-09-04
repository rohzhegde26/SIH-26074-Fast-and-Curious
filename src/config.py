"""
src/config.py

Configuration and hardware memory guards tailored for 4GB VRAM / 16GB RAM.
Enforces:
    - Micro-batch size: 16 (or 8 for extra safety margin)
    - Gradient accumulation: 4 steps -> Effective batch size = 64
    - Automatic Mixed Precision (AMP): FP16 / BF16
    - Windows memory guard: pin_memory=False, num_workers=2
    - Direct 5x resolution scaling: 16x16 LR -> 80x80 HR
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class HardwareConfig:
    # 4GB VRAM Guards
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # 16 * 4 = 64 effective batch size
    use_amp: bool = True  # torch.cuda.amp.autocast
    vram_target_gb: float = 2.5  # Keep peak allocation < 2.5 GB
    num_workers: int = 2
    pin_memory: bool = False  # Avoid Windows WDDM CPU-GPU paging thrashing


@dataclass
class DataConfig:
    # Resolution & Patch dimensions
    scale_factor: int = 5
    hr_patch_size: int = 80
    lr_patch_size: int = 16
    hr_res_deg: float = 0.05
    lr_res_deg: float = 0.25
    stride_pixels: int = 40
    land_fraction_threshold: float = 0.70

    # Domain bounds (India monsoon domain)
    min_lon: float = 68.0
    max_lon: float = 97.0
    min_lat: float = 8.0
    max_lat: float = 37.0

    # Zarr chunk policy (Prevents monolithic 14-year RAM loading)
    zarr_chunks: Tuple[int, int, int] = (1, 80, 80)

    # Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    mandya_full_geojson: Path = Path("data/processed/mandya_full.geojson")
    mandya_simplified_topojson: Path = Path("data/processed/mandya_simplified.topojson")
    mandya_holdout_geojson: Path = Path("data/processed/mandya_holdout_buffer.geojson")
    spatial_patch_index_path: Path = Path("data/cache/spatial_patch_index.parquet")
    registration_json_path: Path = Path("src/data/registration_transform.json")


@dataclass
class ModelConfig:
    in_channels: int = 1  # IMD LR rainfall (optionally + DEM/slope channels later)
    out_channels: int = 1  # CHIRPS HR rainfall
    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)  # 32, 64, 128, 256 (all divisible by 8)
    num_groups: int = 8  # GroupNorm group count
    dropout: float = 0.05
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    loss_lambda_cons: float = 0.1  # Weight for physical conservation loss


@dataclass
class AppConfig:
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


default_config = AppConfig()
