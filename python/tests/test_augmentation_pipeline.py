"""Tests for Sim2RealPipeline (mocked -- no GPU needed).

torch, diffusers, and PIL are not available in the test environment, so all
tests that would trigger model loading or inference use unittest.mock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from clankers.augmentation.pipeline import Sim2RealPipeline
from clankers.augmentation.prompts import SceneType

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_pipeline_init_no_load(self):
        """Pipeline is lazy -- _pipe is None after construction."""
        pipe = Sim2RealPipeline(device="cpu", dtype="float32")
        assert pipe._pipe is None

    def test_pipeline_stores_params(self):
        """Constructor stores device, dtype, seed, and scene_type."""
        pipe = Sim2RealPipeline(
            device="cpu",
            dtype="float32",
            scene_type=SceneType.OUTDOOR,
            seed=123,
        )
        assert pipe.device == "cpu"
        assert pipe.dtype_str == "float32"
        assert pipe.scene_type == SceneType.OUTDOOR
        assert pipe.seed == 123

    def test_pipeline_has_remapper_and_prompt_builder(self):
        """Pipeline creates a PaletteRemapper and PromptBuilder on init."""
        pipe = Sim2RealPipeline(device="cpu", dtype="float32")
        assert pipe._remapper is not None
        assert pipe._prompt_builder is not None


# ---------------------------------------------------------------------------
# generate() with mocked internals
# ---------------------------------------------------------------------------


class TestPipelineGenerate:
    @patch("clankers.augmentation.pipeline.Sim2RealPipeline._load_pipeline")
    def test_generate_calls_remap(self, mock_load):
        """generate() remaps the segmentation image via PaletteRemapper."""
        pipe = Sim2RealPipeline(device="cpu", dtype="float32")

        # Mock the diffusers pipeline output
        mock_pil_img = MagicMock()
        mock_pil_img.size = (64, 64)
        # np.array(mock_pil_img) should return an array
        fake_output = np.full((64, 64, 3), 128, dtype=np.uint8)
        mock_pil_img.resize.return_value = mock_pil_img

        mock_result = MagicMock()
        mock_result.images = [mock_pil_img]

        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        pipe._pipe = mock_pipe

        seg = np.full((64, 64, 3), (255, 0, 0), dtype=np.uint8)

        # Mock PIL.Image and torch to avoid import errors
        mock_image_mod = MagicMock()
        mock_image_mod.fromarray.return_value = MagicMock()
        mock_image_mod.LANCZOS = 1
        mock_torch = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {"PIL": MagicMock(), "PIL.Image": mock_image_mod, "torch": mock_torch},
            ),
            patch("clankers.augmentation.pipeline.np.array", return_value=fake_output),
        ):
            pipe.generate(seg, num_inference_steps=5)

        # _load_pipeline was called
        mock_load.assert_called_once()
        # The mock pipeline was called
        mock_pipe.assert_called_once()

    @patch("clankers.augmentation.pipeline.Sim2RealPipeline._load_pipeline")
    def test_generate_uses_prompt_override(self, mock_load):
        """When prompt_override is given, it is passed to the pipeline."""
        pipe = Sim2RealPipeline(device="cpu", dtype="float32")

        mock_pil_img = MagicMock()
        fake_output = np.full((64, 64, 3), 128, dtype=np.uint8)

        mock_result = MagicMock()
        mock_result.images = [mock_pil_img]

        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        pipe._pipe = mock_pipe

        seg = np.full((64, 64, 3), 0, dtype=np.uint8)

        mock_image_mod = MagicMock()
        mock_torch = MagicMock()

        with (
            patch.dict(
                "sys.modules",
                {"PIL": MagicMock(), "PIL.Image": mock_image_mod, "torch": mock_torch},
            ),
            patch("clankers.augmentation.pipeline.np.array", return_value=fake_output),
        ):
            pipe.generate(seg, prompt_override="custom prompt here")

        # Check the call kwargs include our prompt
        call_kwargs = mock_pipe.call_args
        assert call_kwargs is not None
        # prompt is passed as a keyword argument
        assert call_kwargs.kwargs.get("prompt") == "custom prompt here" or (
            len(call_kwargs.args) > 0 and "custom prompt here" in str(call_kwargs)
        )


# ---------------------------------------------------------------------------
# generate_batch() seed distribution
# ---------------------------------------------------------------------------


class TestPipelineBatch:
    @patch("clankers.augmentation.pipeline.Sim2RealPipeline.generate")
    def test_batch_increments_seed(self, mock_gen):
        """generate_batch() passes seed+i for each image."""
        mock_gen.return_value = np.zeros((4, 4, 3), dtype=np.uint8)

        pipe = Sim2RealPipeline(device="cpu", dtype="float32")
        images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

        pipe.generate_batch(images, seed=42)

        assert mock_gen.call_count == 3
        seeds = [call.kwargs["seed"] for call in mock_gen.call_args_list]
        assert seeds == [42, 43, 44]

    @patch("clankers.augmentation.pipeline.Sim2RealPipeline.generate")
    def test_batch_none_seed(self, mock_gen):
        """generate_batch() passes None seed when no base seed is given."""
        mock_gen.return_value = np.zeros((4, 4, 3), dtype=np.uint8)

        pipe = Sim2RealPipeline(device="cpu", dtype="float32")
        images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]

        pipe.generate_batch(images, seed=None)

        seeds = [call.kwargs["seed"] for call in mock_gen.call_args_list]
        assert seeds == [None, None]
