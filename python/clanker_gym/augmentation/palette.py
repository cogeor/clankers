"""Segmentation palette remapping: Clankers palette -> ADE20K colors.

The ControlNet segmentation model (lllyasviel/control_v11p_sd15_seg) was trained
on ADE20K-style color-coded segmentation maps. This module remaps our custom
simulation palette to match ADE20K conventions for best results.

ADE20K class palette reference:
https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

# Our simulation segmentation palette: class_name -> RGB uint8
CLANKERS_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "ground":   (255, 0, 0),
    "wall":     (0, 255, 0),
    "robot":    (0, 0, 255),
    "obstacle": (255, 255, 0),
    "table":    (128, 128, 128),
    "object":   (128, 0, 128),
}

# ADE20K palette: class indices and their standard colors.
# Selected ADE20K classes that best match our simulation classes.
# Reference: https://docs.google.com/spreadsheets/d/1se8YEtb2detS7OuPE86fXGyD269pMycAWe2mtKUj2W8
ADE20K_MAPPING: Dict[str, Tuple[int, int, int]] = {
    "ground":   (80, 50, 50),      # ADE20K class 3 "floor"
    "wall":     (120, 120, 120),   # ADE20K class 0 "wall"
    "robot":    (100, 100, 100),   # ADE20K metallic gray (custom -- no exact ADE class for robot arm)
    "obstacle": (255, 245, 238),   # ADE20K class 93 "box"
    "table":    (4, 200, 3),       # ADE20K class 15 "table"
    "object":   (8, 255, 51),      # ADE20K class 37 "bottle/object"
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
        source_palette: Dict[str, Tuple[int, int, int]] | None = None,
        target_palette: Dict[str, Tuple[int, int, int]] | None = None,
        tolerance: int = 10,
        background_color: Tuple[int, int, int] = (0, 0, 0),
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
