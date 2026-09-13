"""
src/models/dataset.py

PyTorch Dataset and DataLoader abstractions for Zarr patch cubes.
Supports:
1. Direct streaming from Zarr 3.x stores (zero-copy memory mapping where possible).
2. Four-way temporal partition filtering (train, val, cal, test).
3. Log1p pre-transform for numeric stability with expm1 inversion.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
try:
    import zarr
except ImportError:
    zarr = None

from src.data.terrain_features import build_terrain_tensor_5ch


class MonsoonPatchDataset(Dataset):
    """
    Dataset streaming paired (LR 16x16, HR 80x80) monsoon patches from Zarr array stores.
    
    Each item returns:
        - lr: [1, 16, 16] float32 tensor
        - hr: [1, 80, 80] float32 tensor
        - lat: float32 central latitude
        - lon: float32 central longitude
        - date: string YYYY-MM-DD
    """

    def __init__(
        self,
        zarr_path: str,
        split: str = "train",
        log_transform: bool = False,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Args:
            zarr_path: path to .zarr array store directory.
            split: 'train' (2010-2020), 'val' (2021), 'cal' (2022), 'test' (2023), or 'all'.
            log_transform: If True, applies log1p to lr and hr tensors.
            indices: Optional pre-filtered array indices.
        """
        self.zarr_path = zarr_path
        self.split = split
        self.log_transform = log_transform

        # Open root group in read-only mode
        self.root = zarr.open_group(zarr_path, mode="r")
        self.hr_array = self.root["hr_patches"]
        self.lr_array = self.root["lr_patches"]
        self.lats = self.root["center_lats"][:]
        self.lons = self.root["center_lons"][:]
        self.dates = self.root["dates"][:]
        self.dem_path = Path("data/raw/dem/synthetic_terrain.nc")
        self.dem_ds = None
        if self.dem_path.exists():
            try:
                self.dem_ds = xr.open_dataset(self.dem_path)
            except Exception:
                self.dem_ds = None

        # Determine indices for split
        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
        else:
            self.indices = self._get_split_indices(split)

    def _get_split_indices(self, split: str) -> np.ndarray:
        """Filter patch indices by year based on 4-way temporal partition."""
        if split == "all":
            return np.arange(len(self.dates), dtype=np.int64)

        # Extract years from date strings
        years = np.array([int(str(d).split("-")[0]) for d in self.dates])

        if split == "train":
            mask = (years >= 2010) & (years <= 2020)
        elif split == "val":
            mask = years == 2021
        elif split == "cal":
            mask = years == 2022
        elif split == "test":
            mask = years == 2023
        else:
            raise ValueError(f"Unknown split: {split}. Choose 'train', 'val', 'cal', 'test', or 'all'.")

        indices = np.where(mask)[0]
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        real_idx = int(self.indices[idx])

        lr_raw = self.lr_array[real_idx].astype(np.float32)
        hr_raw = self.hr_array[real_idx].astype(np.float32)

        # Apply log1p transform if requested
        if self.log_transform:
            lr_val = np.log1p(np.maximum(lr_raw, 0.0))
            hr_val = np.log1p(np.maximum(hr_raw, 0.0))
        else:
            lr_val = np.maximum(lr_raw, 0.0)
            hr_val = np.maximum(hr_raw, 0.0)

        # Add channel dimension: [1, H, W]
        lr_tensor = torch.from_numpy(lr_val).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_val).unsqueeze(0)

        c_lat = float(self.lats[real_idx])
        c_lon = float(self.lons[real_idx])
        date_str = str(self.dates[real_idx])
        month = int(date_str.split("-")[1]) if "-" in date_str else 7

        # Extract or construct 5-channel HR terrain tensor [5, 80, 80]
        terrain_extracted = False
        if self.dem_ds is not None:
            try:
                sub = self.dem_ds.sel(
                    lat=slice(c_lat - 1.975, c_lat + 2.025),
                    lon=slice(c_lon - 1.975, c_lon + 2.025),
                )
                if sub.sizes.get("lat", 0) >= 80 and sub.sizes.get("lon", 0) >= 80:
                    elev_patch = sub["elevation"].values[:80, :80]
                    slope_patch = sub["slope"].values[:80, :80]
                    aspect_patch = sub["aspect"].values[:80, :80]
                    terrain_hr = build_terrain_tensor_5ch(
                        elev_patch, slope_patch, aspect_patch, center_lat_deg=c_lat, month=month
                    )
                    terrain_extracted = True
            except Exception:
                terrain_extracted = False

        if not terrain_extracted:
            x = np.linspace(-2.0, 2.0, 80)
            y = np.linspace(-2.0, 2.0, 80)
            xx, yy = np.meshgrid(x, y)
            elev_syn = 600.0 + 300.0 * np.exp(-(xx**2 + yy**2) / 2.0)
            slope_syn = np.clip(np.sqrt(xx**2 + yy**2) * 5.0, 0.0, 30.0)
            aspect_syn = (np.degrees(np.arctan2(yy, xx)) + 360.0) % 360.0
            terrain_hr = build_terrain_tensor_5ch(
                elev_syn, slope_syn, aspect_syn, center_lat_deg=c_lat, month=month
            )

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "terrain_hr": terrain_hr,
            "lr_phys": torch.from_numpy(np.maximum(lr_raw, 0.0)).unsqueeze(0),
            "hr_phys": torch.from_numpy(np.maximum(hr_raw, 0.0)).unsqueeze(0),
            "lat": c_lat,
            "lon": c_lon,
            "date": date_str,
        }


def get_dataloaders(
    zarr_path: str,
    batch_size: int = 32,
    num_workers: int = 2,
    log_transform: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience factory to create Train, Validation, and Calibration DataLoaders.
    """
    train_ds = MonsoonPatchDataset(zarr_path, split="train", log_transform=log_transform)
    val_ds = MonsoonPatchDataset(zarr_path, split="val", log_transform=log_transform)
    cal_ds = MonsoonPatchDataset(zarr_path, split="cal", log_transform=log_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    cal_loader = DataLoader(
        cal_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, cal_loader
