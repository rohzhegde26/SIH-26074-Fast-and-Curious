"""
src/data/temporal_dataset.py

PyTorch Dataset & Streaming DataLoader Interface for Sprint 2.
Exposes SpatiotemporalDownscalingDataset backed directly by datasets/multitask_temporal_v1.zarr.
Supports forward statistical normalization and lossless physical inversion.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
import zarr

from src.data.tensor_builder import apply_normalization, invert_normalization

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZARR_PATH = ROOT / "datasets" / "multitask_temporal_v1.zarr"
DEFAULT_SAMPLE_INDEX_PATH = ROOT / "data" / "sample_index.parquet"
DEFAULT_STATS_PATH = ROOT / "data" / "normalization_stats.yaml"


class SpatiotemporalDownscalingDataset(Dataset):
    """
    High-performance PyTorch Dataset streaming spatiotemporal weather downscaling samples from Zarr.
    Each sample yields:
      - history: [3, 6, 16, 16] (antecedent completed observations D-3, D-2, D-1)
      - future_forecast: [7, 6, 16, 16] (coarse forecast conditioning for D through D+6)
      - terrain: [5, 80, 80] (static GLO-30 DSM terrain prior)
      - target: [7, 6, 80, 80] (fine reference supervision targets D through D+6)
    """

    def __init__(
        self,
        zarr_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        stats_path: Optional[Path] = None,
        split: str = "train",
        normalize: bool = True,
        history_len: int = 3,
        transform: Optional[Callable] = None,
    ):
        self.zarr_path = Path(zarr_path or DEFAULT_ZARR_PATH)
        self.index_path = Path(index_path or DEFAULT_SAMPLE_INDEX_PATH)
        self.stats_path = Path(stats_path or DEFAULT_STATS_PATH)
        self.split = str(split).lower()
        self.normalize = bool(normalize)
        self.history_len = int(history_len)
        self.transform = transform

        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr store not found at {self.zarr_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index parquet not found at {self.index_path}")

        self.store = zarr.open_group(str(self.zarr_path), mode="r")
        self.store_history_len = int(self.store["history"].shape[1])
        if not (1 <= self.history_len <= self.store_history_len):
            raise ValueError(
                f"history_len must be between 1 and {self.store_history_len}, got {history_len}"
            )

        self.df_all = pd.read_parquet(self.index_path)

        if self.split != "all":
            self.df = self.df_all[self.df_all["split"] == self.split].reset_index(drop=True)
        else:
            self.df = self.df_all.copy().reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in index")

        # Map local index in self.df to global index in Zarr
        all_ids = list(self.df_all["sample_id"])
        id_to_global = {sid: i for i, sid in enumerate(all_ids)}
        self.global_indices = [id_to_global[sid] for sid in self.df["sample_id"]]

        # Pre-cache terrain prior [5, 80, 80]
        self.terrain_tensor = torch.from_numpy(np.asarray(self.store["terrain"][:], dtype=np.float32))

        # Load normalization stats if requested
        if self.normalize:
            if not self.stats_path.exists():
                raise FileNotFoundError(f"Normalization stats file not found at {self.stats_path}")
            with open(self.stats_path, "r", encoding="utf-8") as f:
                self.stats_doc = yaml.safe_load(f)
            self.stats = self.stats_doc.get("channels", {})
        else:
            self.stats = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = self.global_indices[idx]
        meta = self.df.iloc[idx]

        # Read contiguous sample arrays from Zarr store
        raw_hist = np.asarray(self.store["history"][global_idx], dtype=np.float32)  # [store_h, 6, 16, 16]
        if self.history_len < raw_hist.shape[0]:
            raw_hist = raw_hist[-self.history_len:]  # [H, 6, 16, 16] (most recent antecedent days ending at D-1)
        raw_fcst = np.asarray(self.store["future_forecast"][global_idx], dtype=np.float32)  # [7, 6, 16, 16]
        raw_targ = np.asarray(self.store["target"][global_idx], dtype=np.float32)  # [7, 6, 80, 80]

        if self.normalize:
            # Apply train-fitted normalization per channel
            norm_hist = apply_normalization(raw_hist, self.stats)
            norm_fcst = apply_normalization(raw_fcst, self.stats)
            norm_targ = apply_normalization(raw_targ, self.stats)
        else:
            norm_hist = raw_hist
            norm_fcst = raw_fcst
            norm_targ = raw_targ

        t_hist = torch.from_numpy(norm_hist)
        t_fcst = torch.from_numpy(norm_fcst)
        t_targ = torch.from_numpy(norm_targ)
        t_terr = self.terrain_tensor.clone()

        sample = {
            "history": t_hist,
            "future_forecast": t_fcst,
            "terrain": t_terr,
            "target": t_targ,
            "sample_id": meta["sample_id"],
            "init_date": meta["init_date"],
            "split": meta["split"],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def unnormalize_target(self, normalized_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Inverts normalized target tensor back to physical units (mm, deg C, %, m/s).
        Input shape: [7, 6, 80, 80] or [B, 7, 6, 80, 80].
        """
        if not self.normalize:
            return np.asarray(normalized_tensor, dtype=np.float32)

        is_torch = isinstance(normalized_tensor, torch.Tensor)
        arr = normalized_tensor.cpu().numpy() if is_torch else np.asarray(normalized_tensor)

        if arr.ndim == 4:
            # [7, 6, 80, 80]
            unnorm = invert_normalization(arr, self.stats)
        elif arr.ndim == 5:
            # [B, 7, 6, 80, 80]
            batches = [invert_normalization(arr[b], self.stats) for b in range(arr.shape[0])]
            unnorm = np.stack(batches, axis=0)
        else:
            raise ValueError(f"Unsupported target shape for inversion: {arr.shape}")

        return unnorm.astype(np.float32)


def get_temporal_dataloaders(
    batch_size: int = 4,
    num_workers: int = 0,
    zarr_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for train, val, and test partitions.
    """
    train_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="train",
        normalize=normalize,
    )
    val_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="val",
        normalize=normalize,
    )
    test_ds = SpatiotemporalDownscalingDataset(
        zarr_path=zarr_path,
        index_path=index_path,
        stats_path=stats_path,
        split="test",
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
