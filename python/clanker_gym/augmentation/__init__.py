"""Sim-to-real image augmentation via Stable Diffusion + ControlNet.

Transforms simulation segmentation images into photorealistic camera images
suitable for training perception models.

Requires: diffusers, transformers, torch, accelerate, Pillow
Install: pip install diffusers transformers torch accelerate Pillow
"""

from clanker_gym.augmentation.palette import PaletteRemapper, CLANKERS_PALETTE, ADE20K_MAPPING
from clanker_gym.augmentation.prompts import PromptBuilder, SceneType
from clanker_gym.augmentation.pipeline import Sim2RealPipeline

__all__ = [
    "PaletteRemapper",
    "CLANKERS_PALETTE",
    "ADE20K_MAPPING",
    "PromptBuilder",
    "SceneType",
    "Sim2RealPipeline",
]
