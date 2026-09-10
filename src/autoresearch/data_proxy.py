"""
src/autoresearch/data_proxy.py

Generates and caches an authentic proxy dataset for fast AutoResearch iterations.
Extracts 80x80 HR and 16x16 LR patches from CHIRPS and Copernicus GLO-30 DEM.
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr

from src.data.terrain_features import build_terrain_tensor_5ch
from src.losses.conservation import coarsen_hr_to_lr_torch


def generate_proxy_data(cache_path: Path = Path("data/cache/autoresearch_proxy.pt")) -> Dict[str, torch.Tensor]:
    """Generates and caches paired spatial patches for fast AutoResearch cycles."""
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    print("[*] Generating AutoResearch proxy dataset from authentic NetCDFs...")
    chirps_path = Path("data/raw/chirps/chirps_sample.nc")
    dem_path = Path("data/raw/dem/glo30_terrain.nc")

    assert chirps_path.exists() and dem_path.exists(), "NetCDF files missing in data/raw/"

    ds_chirps = xr.open_dataset(chirps_path)
    ds_dem = xr.open_dataset(dem_path)

    precip = ds_chirps["precip"].values  # [5, 580, 580]
    elev = ds_dem["elevation"].values    # [580, 580]
    slope = ds_dem["slope"].values      # [580, 580]
    aspect = ds_dem["aspect"].values    # [580, 580]
    lats = ds_dem["lat"].values         # [580]

    # Pre-build 5ch terrain tensor for whole India domain
    t_elev = torch.from_numpy(np.nan_to_num(elev, nan=0.0)).float()
    t_slope = torch.from_numpy(np.nan_to_num(slope, nan=0.0)).float()
    t_aspect = torch.from_numpy(np.nan_to_num(aspect, nan=0.0)).float()
    terrain_5ch_full = build_terrain_tensor_5ch(t_elev, t_slope, t_aspect, center_lat_deg=15.0, month=7)  # [1, 5, 580, 580]
    if terrain_5ch_full.ndim == 4:
        terrain_5ch_full = terrain_5ch_full[0]  # [5, 580, 580]

    lr_list = []
    hr_list = []
    terrain_list = []
    lats_list = []

    # Extract 80x80 patches across multiple dates with stride 60
    patch_sz = 80
    stride = 60
    for t in range(precip.shape[0]):
        p_day = np.nan_to_num(precip[t], nan=0.0)
        for r in range(0, 580 - patch_sz, stride):
            for c in range(0, 580 - patch_sz, stride):
                hr_patch = p_day[r : r + patch_sz, c : c + patch_sz]
                if np.mean(hr_patch) < 0.05 and np.max(hr_patch) < 1.0:
                    continue  # Skip totally dry / ocean patches

                hr_t = torch.from_numpy(hr_patch).float().unsqueeze(0)  # [1, 80, 80]
                terrain_patch = terrain_5ch_full[:, r : r + patch_sz, c : c + patch_sz]
                lat_patch = torch.from_numpy(lats[r : r + patch_sz]).float()

                # Generate registered coarse LR input (16x16) via area-weighted pooling
                lr_t = coarsen_hr_to_lr_torch(hr_t.unsqueeze(0), lat_patch).squeeze(0)  # [1, 16, 16]

                hr_list.append(hr_t)
                lr_list.append(lr_t)
                terrain_list.append(terrain_patch)
                lats_list.append(lat_patch)

    hr_all = torch.stack(hr_list)
    lr_all = torch.stack(lr_list)
    terrain_all = torch.stack(terrain_list)
    lats_all = torch.stack(lats_list)

    print(f"[+] Extracted {len(hr_all)} valid authentic spatial patches.")

    # Shuffle deterministically
    torch.manual_seed(42)
    perm = torch.randperm(len(hr_all))
    hr_all = hr_all[perm]
    lr_all = lr_all[perm]
    terrain_all = terrain_all[perm]
    lats_all = lats_all[perm]

    # Split into 160 train, 64 validation
    n_train = min(160, int(len(hr_all) * 0.7))
    data_dict = {
        "train_lr": lr_all[:n_train],
        "train_hr": hr_all[:n_train],
        "train_terrain": terrain_all[:n_train],
        "train_lats": lats_all[:n_train],
        "val_lr": lr_all[n_train : n_train + 64],
        "val_hr": hr_all[n_train : n_train + 64],
        "val_terrain": terrain_all[n_train : n_train + 64],
        "val_lats": lats_all[n_train : n_train + 64],
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_dict, cache_path)
    print(f"[+] Cached proxy dataset at {cache_path}")
    return data_dict


def get_proxy_dataloaders(batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader) for fast AutoResearch proxy training."""
    data = generate_proxy_data()
    train_ds = TensorDataset(
        data["train_lr"], data["train_hr"], data["train_terrain"], data["train_lats"]
    )
    val_ds = TensorDataset(
        data["val_lr"], data["val_hr"], data["val_terrain"], data["val_lats"]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    t_loader, v_loader = get_proxy_dataloaders()
    for lr, hr, terrain, lats in t_loader:
        print("Batch LR:", lr.shape, "HR:", hr.shape, "Terrain:", terrain.shape)
        break
