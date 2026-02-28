"""Tests for segmentation palette remapping."""

from __future__ import annotations

import numpy as np
import pytest

from clanker_gym.augmentation.palette import (
    ADE20K_MAPPING,
    CLANKERS_PALETTE,
    PaletteRemapper,
)

# ---------------------------------------------------------------------------
# Palette dictionaries
# ---------------------------------------------------------------------------

_EXPECTED_CLASSES = {"ground", "wall", "robot", "obstacle", "table", "object"}


class TestPaletteDicts:
    def test_clankers_palette_has_all_classes(self):
        assert set(CLANKERS_PALETTE.keys()) == _EXPECTED_CLASSES

    def test_ade20k_mapping_has_all_classes(self):
        assert set(ADE20K_MAPPING.keys()) == _EXPECTED_CLASSES


# ---------------------------------------------------------------------------
# PaletteRemapper
# ---------------------------------------------------------------------------


class TestPaletteRemapper:
    def test_remap_single_color(self):
        """Solid red (ground) image should remap to ADE20K ground color."""
        remapper = PaletteRemapper()
        img = np.full((2, 2, 3), CLANKERS_PALETTE["ground"], dtype=np.uint8)
        result = remapper.remap(img)

        expected = np.array(ADE20K_MAPPING["ground"], dtype=np.uint8)
        assert np.all(result == expected)

    def test_remap_multi_color(self):
        """Image with two different palette colors remaps each region correctly."""
        remapper = PaletteRemapper()
        img = np.zeros((2, 2, 3), dtype=np.uint8)

        # Top row = ground, bottom row = wall
        img[0, :] = CLANKERS_PALETTE["ground"]
        img[1, :] = CLANKERS_PALETTE["wall"]

        result = remapper.remap(img)

        expected_ground = np.array(ADE20K_MAPPING["ground"], dtype=np.uint8)
        expected_wall = np.array(ADE20K_MAPPING["wall"], dtype=np.uint8)

        assert np.all(result[0, :] == expected_ground)
        assert np.all(result[1, :] == expected_wall)

    def test_remap_unknown_color_becomes_background(self):
        """Pixels with colors not in the palette become (0,0,0) background."""
        remapper = PaletteRemapper()
        img = np.full((2, 2, 3), (42, 42, 42), dtype=np.uint8)
        result = remapper.remap(img)
        assert np.all(result == 0)

    def test_remap_tolerance(self):
        """Slightly off-palette colors match within tolerance but not without it."""
        # Near-red: (252, 3, 2) is close to ground (255, 0, 0)
        near_red = np.full((2, 2, 3), (252, 3, 2), dtype=np.uint8)

        # With tolerance=10 -> should match ground
        remapper_tol = PaletteRemapper(tolerance=10)
        result_tol = remapper_tol.remap(near_red)
        expected_ground = np.array(ADE20K_MAPPING["ground"], dtype=np.uint8)
        assert np.all(result_tol == expected_ground)

        # With tolerance=0 -> should become background
        remapper_strict = PaletteRemapper(tolerance=0)
        result_strict = remapper_strict.remap(near_red)
        assert np.all(result_strict == 0)

    def test_remap_preserves_shape(self):
        """Output shape matches input shape for various sizes."""
        remapper = PaletteRemapper()
        for h, w in [(1, 1), (4, 8), (64, 64), (100, 200)]:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            result = remapper.remap(img)
            assert result.shape == (h, w, 3), f"Shape mismatch for ({h}, {w})"

    def test_custom_palettes(self):
        """PaletteRemapper works with custom source and target palettes."""
        custom_src = {"foo": (10, 20, 30)}
        custom_tgt = {"foo": (100, 200, 250)}

        remapper = PaletteRemapper(
            source_palette=custom_src,
            target_palette=custom_tgt,
            tolerance=0,
        )

        img = np.full((3, 3, 3), (10, 20, 30), dtype=np.uint8)
        result = remapper.remap(img)
        assert np.all(result == np.array([100, 200, 250], dtype=np.uint8))

    def test_remap_all_classes(self):
        """Each palette class remaps to its corresponding ADE20K color."""
        remapper = PaletteRemapper(tolerance=0)

        for class_name in CLANKERS_PALETTE:
            src_color = CLANKERS_PALETTE[class_name]
            tgt_color = ADE20K_MAPPING[class_name]

            img = np.full((1, 1, 3), src_color, dtype=np.uint8)
            result = remapper.remap(img)
            expected = np.array(tgt_color, dtype=np.uint8)
            assert np.all(result[0, 0] == expected), (
                f"Class {class_name}: expected {tgt_color}, got {tuple(result[0, 0])}"
            )

    def test_custom_background_color(self):
        """Custom background_color is used for unmatched pixels."""
        bg = (50, 60, 70)
        remapper = PaletteRemapper(background_color=bg)

        img = np.full((2, 2, 3), (42, 42, 42), dtype=np.uint8)
        result = remapper.remap(img)
        assert np.all(result == np.array(bg, dtype=np.uint8))
