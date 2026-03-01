"""Segmentation palette remapping: Clankers palette -> ADE20K colors.

The ControlNet segmentation model (lllyasviel/control_v11p_sd15_seg) was trained
on ADE20K-style color-coded segmentation maps. This module remaps our custom
simulation palette to match ADE20K conventions for best results.

ADE20K class palette reference:
https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Our simulation segmentation palette: class_name -> RGB uint8
CLANKERS_PALETTE: dict[str, tuple[int, int, int]] = {
    "ground": (255, 0, 0),
    "wall": (0, 255, 0),
    "robot": (0, 0, 255),
    "obstacle": (255, 255, 0),
    "table": (128, 128, 128),
    "object": (128, 0, 128),
}

# ADE20K palette: verified against mmseg PALETTE used by ControlNet-seg
# (lllyasviel/control_v11p_sd15_seg).  The mmseg palette is 0-indexed;
# ADE20K CSV class N corresponds to mmseg palette index N-1.
# Reference: https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/annotator/uniformer/mmseg/datasets/ade.py
ADE20K_MAPPING: dict[str, tuple[int, int, int]] = {
    "ground": (80, 50, 50),  # ADE20K "floor" (class 4, palette idx 3)
    "wall": (120, 120, 120),  # ADE20K "wall"  (class 1, palette idx 0)
    "robot": (140, 140, 140),  # ADE20K "road"  (class 7, palette idx 6) — neutral gray surface
    "obstacle": (0, 255, 20),  # ADE20K "box"   (class 42, palette idx 41)
    "table": (255, 6, 82),  # ADE20K "table" (class 16, palette idx 15)
    "object": (255, 0, 163),  # ADE20K "ball"  (class 120, palette idx 119)
}


class PaletteRemapper:
    """Remaps segmentation images from one color palette to another.

    Parameters
    ----------
    source_palette : dict[str, tuple[int, int, int]]
        Source palette mapping class names to RGB tuples.
    target_palette : dict[str, tuple[int, int, int]]
        Target palette mapping class names to RGB tuples.
    tolerance : int
        Color matching tolerance (L-inf distance per channel). Default 10.
    background_color : tuple[int, int, int]
        Color for pixels that don't match any source class. Default (0, 0, 0).

    Examples
    --------
    >>> remapper = PaletteRemapper(CLANKERS_PALETTE, ADE20K_MAPPING)
    >>> ade20k_img = remapper.remap(clankers_seg_img)
    """

    def __init__(
        self,
        source_palette: dict[str, tuple[int, int, int]] | None = None,
        target_palette: dict[str, tuple[int, int, int]] | None = None,
        tolerance: int = 10,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.source = source_palette or CLANKERS_PALETTE
        self.target = target_palette or ADE20K_MAPPING
        self.tolerance = tolerance
        self.background_color = background_color

        # Build lookup: source_rgb -> target_rgb
        self._mapping: list[tuple[NDArray, NDArray]] = []
        for class_name in self.source:
            if class_name in self.target:
                src = np.array(self.source[class_name], dtype=np.uint8)
                tgt = np.array(self.target[class_name], dtype=np.uint8)
                self._mapping.append((src, tgt))

    def remap(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Remap a segmentation image from source palette to target palette.

        Parameters
        ----------
        image : NDArray[np.uint8], shape (H, W, 3)
            Input segmentation image with source palette colors.

        Returns
        -------
        NDArray[np.uint8], shape (H, W, 3)
            Remapped image with target palette colors.
        """
        result = np.full_like(image, self.background_color, dtype=np.uint8)

        for src_color, tgt_color in self._mapping:
            # Find pixels matching source color within tolerance
            diff = np.abs(image.astype(np.int16) - src_color.astype(np.int16))
            mask = np.all(diff <= self.tolerance, axis=-1)
            result[mask] = tgt_color

        return result
