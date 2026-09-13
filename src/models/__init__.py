"""
src/models/__init__.py
"""
from src.models.unet_5x import UNet5x
from src.models.baselines import BilinearInterpolationBaseline, DeepSDBaseline
try:
    from src.models.dataset import MonsoonPatchDataset, get_dataloaders
except ImportError:
    MonsoonPatchDataset = None
    get_dataloaders = None

from src.models.multitask_unet import MultiTaskUNet5x

__all__ = [
    "UNet5x",
    "BilinearInterpolationBaseline",
    "DeepSDBaseline",
    "MonsoonPatchDataset",
    "get_dataloaders",
    "MultiTaskUNet5x",
]
