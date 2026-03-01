"""Sim-to-real image augmentation via Stable Diffusion + ControlNet.

Transforms simulation segmentation images into photorealistic camera images
suitable for training perception models.

Requires: diffusers, transformers, torch, accelerate, Pillow
Install: pip install diffusers transformers torch accelerate Pillow
"""

from clankers.augmentation.mcap_augmentor import McapAugmentor
from clankers.augmentation.palette import ADE20K_MAPPING, CLANKERS_PALETTE, PaletteRemapper
from clankers.augmentation.pipeline import Sim2RealPipeline
from clankers.augmentation.prompts import PromptBuilder, SceneType

__all__ = [
    "PaletteRemapper",
    "CLANKERS_PALETTE",
    "ADE20K_MAPPING",
    "PromptBuilder",
    "SceneType",
    "Sim2RealPipeline",
    "McapAugmentor",
]
