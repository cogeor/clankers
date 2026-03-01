"""Tests for McapAugmentor with real MCAP I/O and a mock pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

try:
    import mcap  # noqa: F401

    _HAS_MCAP = True
except ImportError:
    _HAS_MCAP = False

pytestmark = pytest.mark.skipif(not _HAS_MCAP, reason="mcap not installed")


# Only import when mcap is available -- the module itself guards the import.
if _HAS_MCAP:
    from mcap.reader import make_reader

    # Import CompressionType so we can use NONE (zstandard may not be installed).
    from mcap.writer import CompressionType as _CompressionType
    from mcap.writer import Writer as McapWriter

    from clankers.augmentation.mcap_augmentor import McapAugmentor


@pytest.fixture(autouse=True)
def _patch_mcap_writer():
    """Patch McapWriter in the augmentor module to use NONE compression.

    The default ZSTD compression requires the ``zstandard`` package which
    may not be installed in the test environment.
    """
    if not _HAS_MCAP:
        yield
        return

    _OrigWriter = McapWriter

    class _NoCompressWriter(_OrigWriter):
        def __init__(self, output, **kwargs):
            kwargs.setdefault("compression", _CompressionType.NONE)
            super().__init__(output, **kwargs)

    with patch("clankers.augmentation.mcap_augmentor.McapWriter", _NoCompressWriter):
        yield


# ---------------------------------------------------------------------------
# Mock pipeline
# ---------------------------------------------------------------------------


class MockPipeline:
    """Minimal stand-in for Sim2RealPipeline.generate()."""

    def __init__(self, fill_value: int = 128) -> None:
        self.fill_value = fill_value
        self.call_count = 0
        self.call_seeds: list[int | None] = []

    def generate(
        self,
        segmentation_image: np.ndarray,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int | None = None,
    ) -> np.ndarray:
        self.call_count += 1
        self.call_seeds.append(seed)
        return np.full_like(segmentation_image, self.fill_value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_image(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """Build raw RGB bytes for a solid-color image."""
    pixel = bytes(color)
    return pixel * (width * height)


def _write_test_mcap_raw(
    path: Path,
    width: int,
    height: int,
    num_frames: int,
    color: tuple[int, int, int] = (255, 0, 0),
    camera_topic: str = "/camera/seg",
    extra_channels: dict[str, list[bytes]] | None = None,
) -> None:
    """Write a synthetic MCAP file with raw-bytes camera frames."""
    with open(path, "wb") as f:
        writer = McapWriter(f, compression=_CompressionType.NONE)
        writer.start()

        binary_schema = writer.register_schema(
            name="binary", encoding="application/octet-stream", data=b""
        )
        json_schema = writer.register_schema(name="json", encoding="jsonschema", data=b"")

        cam_channel = writer.register_channel(
            schema_id=binary_schema,
            topic=camera_topic,
            message_encoding="application/octet-stream",
            metadata={
                "width": str(width),
                "height": str(height),
                "channels": "3",
            },
        )

        raw_data = _make_raw_image(width, height, color)
        for i in range(num_frames):
            writer.add_message(
                channel_id=cam_channel,
                log_time=i * 1_000_000_000,
                publish_time=i * 1_000_000_000,
                data=raw_data,
            )

        if extra_channels:
            for topic, messages in extra_channels.items():
                ch = writer.register_channel(
                    schema_id=json_schema,
                    topic=topic,
                    message_encoding="application/json",
                )
                for j, msg_data in enumerate(messages):
                    writer.add_message(
                        channel_id=ch,
                        log_time=j * 1_000_000_000,
                        publish_time=j * 1_000_000_000,
                        data=msg_data,
                    )

        writer.finish()


def _write_test_mcap_json(
    path: Path,
    camera_frames: list[dict[str, Any]],
    camera_topic: str = "/camera/seg",
) -> None:
    """Write a synthetic MCAP file with JSON-encoded camera frames (legacy)."""
    with open(path, "wb") as f:
        writer = McapWriter(f, compression=_CompressionType.NONE)
        writer.start()

        schema_id = writer.register_schema(name="json", encoding="jsonschema", data=b"")

        cam_channel = writer.register_channel(
            schema_id=schema_id,
            topic=camera_topic,
            message_encoding="application/json",
        )

        for i, frame in enumerate(camera_frames):
            writer.add_message(
                channel_id=cam_channel,
                log_time=i * 1_000_000_000,
                publish_time=i * 1_000_000_000,
                data=json.dumps(frame).encode("utf-8"),
            )

        writer.finish()


def _read_messages(path: Path) -> list[tuple[str, bytes, dict[str, str]]]:
    """Read all (topic, data, metadata) triples from an MCAP file."""
    messages = []
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            meta = dict(channel.metadata) if channel.metadata else {}
            messages.append((channel.topic, message.data, meta))
    return messages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMcapAugmentorInit:
    def test_init_stores_attributes(self):
        mock_pipe = MockPipeline()
        aug = McapAugmentor(
            pipeline=mock_pipe,
            input_channel_prefix="/camera/",
            output_suffix="_real",
            copy_non_image_channels=False,
        )
        assert aug._pipeline is mock_pipe
        assert aug._input_channel_prefix == "/camera/"
        assert aug._output_suffix == "_real"
        assert aug._copy_non_image is False


class TestMcapAugmentorRawBytes:
    """Tests using the new raw-bytes format (matching clankers-record output)."""

    def test_process_raw_bytes_mcap(self, tmp_path):
        """Process 3 raw-bytes camera frames: output has 3 augmented images."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        _write_test_mcap_raw(input_mcap, width=4, height=4, num_frames=3)

        mock_pipe = MockPipeline(fill_value=200)
        aug = McapAugmentor(pipeline=mock_pipe)

        summary = aug.process(input_mcap, output_mcap)

        assert summary["images_processed"] == 3
        assert output_mcap.exists()

        messages = _read_messages(output_mcap)
        # Should have 3 augmented messages on /camera/seg_realistic
        aug_msgs = [(t, d, m) for t, d, m in messages if t == "/camera/seg_realistic"]
        assert len(aug_msgs) == 3

        # Verify the output is raw bytes, not JSON
        for _, data, meta in aug_msgs:
            assert meta.get("width") == "4"
            assert meta.get("height") == "4"
            assert meta.get("channels") == "3"
            # Raw bytes: 4*4*3 = 48 bytes
            assert len(data) == 48
            # All pixel values should be fill_value=200
            arr = np.frombuffer(data, dtype=np.uint8)
            assert np.all(arr == 200)

    def test_non_image_channels_copied(self, tmp_path):
        """Non-camera channels are copied through when copy_non_image_channels=True."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        joint_data = [json.dumps({"joint_positions": [0.1, 0.2]}).encode("utf-8")]
        _write_test_mcap_raw(
            input_mcap,
            width=4,
            height=4,
            num_frames=1,
            extra_channels={"/joint_states": joint_data},
        )

        mock_pipe = MockPipeline()
        aug = McapAugmentor(pipeline=mock_pipe, copy_non_image_channels=True)
        aug.process(input_mcap, output_mcap)

        messages = _read_messages(output_mcap)
        topics = {t for t, _, _ in messages}

        assert "/joint_states" in topics
        assert "/camera/seg_realistic" in topics

    def test_non_image_channels_not_copied_when_disabled(self, tmp_path):
        """Non-camera channels are dropped when copy_non_image_channels=False."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        joint_data = [json.dumps({"joint_positions": [0.1, 0.2]}).encode("utf-8")]
        _write_test_mcap_raw(
            input_mcap,
            width=4,
            height=4,
            num_frames=1,
            extra_channels={"/joint_states": joint_data},
        )

        mock_pipe = MockPipeline()
        aug = McapAugmentor(pipeline=mock_pipe, copy_non_image_channels=False)
        aug.process(input_mcap, output_mcap)

        messages = _read_messages(output_mcap)
        topics = {t for t, _, _ in messages}

        assert "/joint_states" not in topics
        assert "/camera/seg_realistic" in topics

    def test_max_images_limit(self, tmp_path):
        """max_images limits the number of images processed."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        _write_test_mcap_raw(input_mcap, width=4, height=4, num_frames=5)

        mock_pipe = MockPipeline()
        aug = McapAugmentor(pipeline=mock_pipe)
        summary = aug.process(input_mcap, output_mcap, max_images=2)

        assert summary["images_processed"] == 2
        assert mock_pipe.call_count == 2

    def test_seed_incremented_per_image(self, tmp_path):
        """Each image gets seed+i for deterministic reproducibility."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        _write_test_mcap_raw(input_mcap, width=4, height=4, num_frames=3)

        mock_pipe = MockPipeline()
        aug = McapAugmentor(pipeline=mock_pipe)
        aug.process(input_mcap, output_mcap, seed=100)

        assert mock_pipe.call_seeds == [100, 101, 102]

    def test_summary_keys(self, tmp_path):
        """process() returns a summary dict with the expected keys."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        _write_test_mcap_raw(input_mcap, width=4, height=4, num_frames=1)

        mock_pipe = MockPipeline()
        aug = McapAugmentor(pipeline=mock_pipe)
        summary = aug.process(input_mcap, output_mcap)

        assert "images_processed" in summary
        assert "channels_found" in summary
        assert "output_path" in summary
        assert "duration_secs" in summary
        assert isinstance(summary["duration_secs"], float)


class TestMcapAugmentorJsonFallback:
    """Tests that JSON-encoded camera frames (legacy) still work."""

    def test_process_json_mcap(self, tmp_path):
        """Process JSON-encoded ImageFrame messages (legacy format)."""
        input_mcap = tmp_path / "input.mcap"
        output_mcap = tmp_path / "output.mcap"

        frames = []
        for _ in range(2):
            data = list((255, 0, 0)) * (4 * 4)
            frames.append(
                {
                    "timestamp_ns": 0,
                    "width": 4,
                    "height": 4,
                    "label": "seg",
                    "data": data,
                }
            )

        _write_test_mcap_json(input_mcap, frames)

        mock_pipe = MockPipeline(fill_value=150)
        aug = McapAugmentor(pipeline=mock_pipe)
        summary = aug.process(input_mcap, output_mcap)

        assert summary["images_processed"] == 2
