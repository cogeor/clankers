"""Sim-to-real image augmentation via Stable Diffusion + ControlNet.

Transforms simulation segmentation images into photorealistic camera images
suitable for training perception models.

Requires: diffusers, transformers, torch, accelerate, Pillow
Install: pip install diffusers transformers torch accelerate Pillow
"""

from clanker_gym.augmentation.mcap_augmentor import McapAugmentor
from clanker_gym.augmentation.palette import ADE20K_MAPPING, CLANKERS_PALETTE, PaletteRemapper
from clanker_gym.augmentation.pipeline import Sim2RealPipeline
from clanker_gym.augmentation.prompts import PromptBuilder, SceneType

__all__ = [
    "PaletteRemapper",
    "CLANKERS_PALETTE",
    "ADE20K_MAPPING",
    "PromptBuilder",
    "SceneType",
    "Sim2RealPipeline",
    "McapAugmentor",
]
