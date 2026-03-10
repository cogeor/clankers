"""Cosmos-Transfer2.5 sim-to-real video augmentation pipeline.

Provides preparation and inference scripts for transforming
Bevy-rendered robot replays into photorealistic video using NVIDIA
Cosmos-Transfer2.5-2B.
"""

from __future__ import annotations

# HuggingFace repository for the model
COSMOS_HF_REPO = "nvidia/Cosmos-Transfer2.5-2B"

# Cosmos native constraints
COSMOS_FPS = 16
COSMOS_CHUNK_FRAMES = 93  # frames per inference pass
COSMOS_480P = (854, 480)

# Segmentation class palette (matches clankers-render)
SEG_PALETTE = {
    "ground": (255, 0, 0),
    "wall": (0, 255, 0),
    "robot": (0, 0, 255),
    "obstacle": (255, 255, 0),
    "table": (128, 128, 128),
}

__all__ = [
    "COSMOS_HF_REPO",
    "COSMOS_FPS",
    "COSMOS_CHUNK_FRAMES",
    "COSMOS_480P",
    "SEG_PALETTE",
]
