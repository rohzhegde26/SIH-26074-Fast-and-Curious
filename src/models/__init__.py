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

from src.models.residual_diffusion import SpatiotemporalResidualDiffusion
from src.models.scalable_residual_diffusion import (
    ScalableSpatiotemporalResidualDiffusion,
    create_scalable_residual_diffusion,
    TIER_CHANNEL_CONFIGS,
)
from src.models.moe import (
    TopKRouter,
    MoETimeConditionedConvNeXtBlock,
)

__all__ = [
    "UNet5x",
    "BilinearInterpolationBaseline",
    "DeepSDBaseline",
    "MonsoonPatchDataset",
    "get_dataloaders",
    "SpatiotemporalResidualDiffusion",
    "ScalableSpatiotemporalResidualDiffusion",
    "create_scalable_residual_diffusion",
    "TIER_CHANNEL_CONFIGS",
    "TopKRouter",
    "MoETimeConditionedConvNeXtBlock",
]
