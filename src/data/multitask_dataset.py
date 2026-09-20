"""
src/data/multitask_dataset.py

PyTorch Dataset for Multi-Task Weather Downscaling.
Delivers paired tensors:
    - coarse_nwp: [5, 16, 16] (Rain, Tmax, Tmin, RH, Wind)
    - fine_terrain: [5, 80, 80] (Elev, Slope, Aspect, Curvature, Windward Lift)
    - fine_targets: [5, 80, 80] (Rain, Tmax, Tmin, RH, Wind)

Canonical Single-Domain 1,220-Sample Partition (11.0°N–15.0°N, 74.0°E–78.0°E):
    - Training: 8 monsoon seasons (2014-2021, 976 samples, 80.0%)
    - Validation: 1 monsoon season (2022, 122 samples, 10.0%)
    - Test: 1 monsoon season (2023 held-out, 122 samples, 10.0%)
    Total: exactly 1,220 spatiotemporal daily samples.
Zero synthetic proxy generation; strict authentic data pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.real_data_ingestion import build_and_cache_real_multitask_dataset, REAL_CACHE_FILE

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
DEFAULT_CACHE_FILE = REAL_CACHE_FILE


def build_and_cache_multitask_dataset(
    cache_path: Path = DEFAULT_CACHE_FILE,
    force_rebuild: bool = False,
) -> Path:
    """
    Constructs and serializes the authentic 1,220 spatiotemporal sample dataset
    spanning 10 monsoon seasons (2014-2023, 122 days/season).
    Strictly raises FileNotFoundError / RuntimeError if required raw datasets are missing.
    Zero synthetic fallback.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_rebuild:
        return cache_path

    return build_and_cache_real_multitask_dataset(
        cache_path=cache_path,
        force_rebuild=force_rebuild,
    )


class MultiTaskPanchayatDataset(Dataset):
    """
    PyTorch Dataset serving paired atmospheric-terrain downscaling tensors.
    """

    def __init__(
        self,
        split: str = "train",
        cache_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
        tile_id: Optional[int] = None,
    ):
        if cache_path is None:
            cache_path = DEFAULT_CACHE_FILE
        if not cache_path.exists():
            build_and_cache_multitask_dataset(cache_path)

        data = np.load(cache_path)
        splits = data["splits"]

        split_map = {"train": 0, "val": 1, "test": 2}
        if split not in split_map:
            raise ValueError(f"Unknown split: {split}. Choose from {list(split_map.keys())}")
        mask = splits == split_map[split]

        self.coarse_nwp = data["coarse_nwp"][mask]
        self.fine_terrain = data["fine_terrain"][mask]
        self.fine_targets = data["fine_targets"][mask]
        self.years = data["years"][mask] if "years" in data else None
        self.day_indices = data["day_indices"][mask] if "day_indices" in data else None
        self.dates = data["dates"][mask] if "dates" in data else None

        if max_samples is not None and max_samples < len(self.coarse_nwp):
            self.coarse_nwp = self.coarse_nwp[:max_samples]
            self.fine_terrain = self.fine_terrain[:max_samples]
            self.fine_targets = self.fine_targets[:max_samples]
            if self.years is not None:
                self.years = self.years[:max_samples]
            if self.day_indices is not None:
                self.day_indices = self.day_indices[:max_samples]
            if self.dates is not None:
                self.dates = self.dates[:max_samples]

    def __len__(self) -> int:
        return len(self.coarse_nwp)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.coarse_nwp[idx]),
            torch.from_numpy(self.fine_terrain[idx]),
            torch.from_numpy(self.fine_targets[idx]),
        )


def get_multitask_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_path: Optional[Path] = None,
    distributed: bool = False,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, val, and test splits.
    Supports PyTorch DistributedSampler when distributed=True.
    """
    train_ds = MultiTaskPanchayatDataset("train", cache_path=cache_path, max_samples=max_samples)
    val_ds = MultiTaskPanchayatDataset("val", cache_path=cache_path, max_samples=max_samples)
    test_ds = MultiTaskPanchayatDataset("test", cache_path=cache_path, max_samples=max_samples)

    train_sampler = None
    if distributed and torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
