"""MCAP-based augmentation: reads segmentation images, generates realistic images, writes output.

Reads simulation segmentation images from MCAP recordings, transforms them
into photorealistic images using the Sim2RealPipeline, and writes the results
to a new MCAP file using raw bytes encoding (matching ``clankers-record``
output format).

Supports both raw-bytes camera channels (written by ``clankers-record``) and
legacy JSON-encoded ``ImageFrame`` messages.

Requires: mcap>=1.0.0
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clankers.augmentation.pipeline import Sim2RealPipeline

import numpy as np
from numpy.typing import NDArray

try:
    from mcap.reader import make_reader  # type: ignore[import-untyped, import-not-found]
    from mcap.writer import Writer as McapWriter  # type: ignore[import-untyped, import-not-found]

    _MCAP_AVAILABLE = True
except ImportError:
    _MCAP_AVAILABLE = False


logger = logging.getLogger(__name__)

# How often to log progress (every N images).
_LOG_INTERVAL = 10


def _require_mcap() -> None:
    if not _MCAP_AVAILABLE:
        raise ImportError(
            "mcap is required for McapAugmentor. Install with: pip install mcap>=1.0.0"
        )


class McapAugmentor:
    """Augments MCAP recordings by transforming segmentation images to realistic ones.

    Reads ``/camera/*`` channels from an input MCAP file, runs each image
    through a :class:`~clankers.augmentation.pipeline.Sim2RealPipeline`,
    and writes the augmented images to a new MCAP file with a configurable
    topic suffix.

    Supports two image encodings:

    - **Raw bytes** (``application/octet-stream``): width/height/channels
      stored in MCAP channel metadata.  This is the format produced by
      ``clankers-record`` and is the default output format.
    - **JSON** (``application/json``): legacy ``ImageFrame`` dicts with
      ``width``, ``height``, ``label``, and ``data`` (flat uint8 list).

    Parameters
    ----------
    pipeline : Sim2RealPipeline
        The inference pipeline to use for image generation.
    input_channel_prefix : str
        Topic prefix to match for segmentation images. Default ``"/camera/"``.
    output_suffix : str
        Suffix appended to channel names for output. Default ``"_realistic"``.
    copy_non_image_channels : bool
        If ``True``, copy non-image channels (joints, actions, rewards) to
        the output. Default ``True``.
    """

    def __init__(
        self,
        pipeline: Sim2RealPipeline,
        input_channel_prefix: str = "/camera/",
        output_suffix: str = "_realistic",
        copy_non_image_channels: bool = True,
    ) -> None:
        _require_mcap()
        self._pipeline = pipeline
        self._input_channel_prefix = input_channel_prefix
        self._output_suffix = output_suffix
        self._copy_non_image = copy_non_image_channels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        max_images: int | None = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Process an MCAP file: read segmentation images, augment, write output.

        Parameters
        ----------
        input_path : str | Path
            Path to the input ``.mcap`` file.
        output_path : str | Path
            Path for the output ``.mcap`` file.
        max_images : int, optional
            Maximum number of images to process. ``None`` means all.
        num_inference_steps : int
            Number of denoising steps per image. Default 20.
        guidance_scale : float
            Classifier-free guidance scale. Default 7.5.
        controlnet_conditioning_scale : float
            ControlNet conditioning strength. Default 1.0.
        seed : int, optional
            Base random seed. Each image gets ``seed + image_index`` for
            deterministic reproducibility.

        Returns
        -------
        dict[str, Any]
            Summary with keys: ``images_processed``, ``channels_found``,
            ``output_path``, ``duration_secs``.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info("Processing %s -> %s", input_path, output_path)
        t0 = time.monotonic()

        images_processed = 0
        channels_found: set[str] = set()

        # Channel ID -> output channel ID mapping (lazily populated).
        _out_channel_ids: dict[int, int] = {}
        # Collect channel metadata for image dimensions.
        _channel_meta: dict[int, dict[str, str]] = {}

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            reader = make_reader(fin)  # type: ignore[possibly-undefined]
            writer = McapWriter(fout)  # type: ignore[possibly-undefined]
            writer.start()

            # Register schemas for output channels.
            json_schema_id = writer.register_schema(name="json", encoding="jsonschema", data=b"")
            binary_schema_id = writer.register_schema(
                name="binary", encoding="application/octet-stream", data=b""
            )

            # Pre-scan channel metadata for image dimensions.
            summary = reader.get_summary()
            if summary and summary.channels:
                for ch in summary.channels.values():
                    if ch.metadata:
                        _channel_meta[ch.id] = dict(ch.metadata)

            for _schema, channel, message in reader.iter_messages():
                topic: str = channel.topic
                is_camera = topic.startswith(self._input_channel_prefix)

                # --- Non-image channel: optionally copy through ---
                if not is_camera:
                    if self._copy_non_image:
                        out_cid = self._ensure_channel(
                            _out_channel_ids,
                            writer,
                            json_schema_id,
                            channel.id,
                            topic,
                            message_encoding=channel.message_encoding,
                        )
                        writer.add_message(
                            channel_id=out_cid,
                            log_time=message.log_time,
                            publish_time=message.publish_time,
                            data=message.data,
                        )
                    continue

                # --- Camera channel ---
                channels_found.add(topic)

                # Respect max_images limit.
                if max_images is not None and images_processed >= max_images:
                    continue

                # Decode image — handle both raw bytes and JSON formats.
                seg_image = self._decode_camera_message(
                    message.data,
                    channel,
                    _channel_meta.get(channel.id, {}),
                )
                if seg_image is None:
                    logger.warning(
                        "Skipping undecodable message on %s at t=%d",
                        topic,
                        message.log_time,
                    )
                    continue

                # Compute per-image seed for determinism.
                img_seed = (seed + images_processed) if seed is not None else None

                # Run augmentation pipeline.
                augmented = self._pipeline.generate(
                    seg_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    seed=img_seed,
                )

                # Write augmented image as raw bytes to output channel.
                out_topic = topic + self._output_suffix
                h, w = augmented.shape[:2]
                channels = augmented.shape[2] if augmented.ndim == 3 else 1
                out_cid = self._ensure_image_channel(
                    _out_channel_ids,
                    writer,
                    binary_schema_id,
                    channel.id,
                    out_topic,
                    width=w,
                    height=h,
                    channels=channels,
                )
                writer.add_message(
                    channel_id=out_cid,
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    data=augmented.tobytes(),
                )

                images_processed += 1

                if images_processed % _LOG_INTERVAL == 0:
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "Processed %d images (%.1fs elapsed, %.2fs/image)",
                        images_processed,
                        elapsed,
                        elapsed / images_processed,
                    )

            writer.finish()

        duration = time.monotonic() - t0
        summary_dict: dict[str, Any] = {
            "images_processed": images_processed,
            "channels_found": sorted(channels_found),
            "output_path": str(output_path),
            "duration_secs": round(duration, 2),
        }
        logger.info(
            "Done: %d images in %.1fs (%s)",
            images_processed,
            duration,
            output_path,
        )
        return summary_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_camera_message(
        data: bytes,
        channel: Any,
        meta: dict[str, str],
    ) -> NDArray[np.uint8] | None:
        """Decode a camera message from either raw bytes or JSON format.

        Raw bytes format: message is a flat H*W*C uint8 buffer with
        width/height/channels in channel metadata.

        JSON format (legacy): message is a JSON dict with ``width``,
        ``height``, ``data`` (flat uint8 list).
        """
        encoding = getattr(channel, "message_encoding", "")

        # Try raw bytes first (preferred format from clankers-record).
        if encoding == "application/octet-stream" or (
            meta and "width" in meta and "height" in meta
        ):
            width = int(meta.get("width", 0))
            height = int(meta.get("height", 0))
            channels = int(meta.get("channels", 3))
            if width > 0 and height > 0:
                expected = height * width * channels
                raw = bytes(data)
                if len(raw) == expected:
                    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, channels)

        # Fall back to JSON ImageFrame format.
        try:
            frame = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        width = int(frame.get("width", 0))
        height = int(frame.get("height", 0))
        pixel_data = frame.get("data", [])

        if width <= 0 or height <= 0 or not pixel_data:
            return None

        channels = 3
        expected = height * width * channels
        if len(pixel_data) != expected:
            return None

        return np.array(pixel_data, dtype=np.uint8).reshape(height, width, channels)

    @staticmethod
    def _ensure_channel(
        cache: dict[int, int],
        writer: McapWriter,
        schema_id: int,
        source_channel_id: int,
        topic: str,
        message_encoding: str = "application/json",
    ) -> int:
        """Return (and lazily register) the output channel id for a source channel."""
        if source_channel_id not in cache:
            cid = writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding=message_encoding,
            )
            cache[source_channel_id] = cid
        return cache[source_channel_id]

    @staticmethod
    def _ensure_image_channel(
        cache: dict[int, int],
        writer: McapWriter,
        schema_id: int,
        source_channel_id: int,
        topic: str,
        width: int,
        height: int,
        channels: int,
    ) -> int:
        """Register an output channel for raw-bytes image data with metadata."""
        # Use a distinct cache key for image output channels.
        cache_key = source_channel_id + 1_000_000
        if cache_key not in cache:
            cid = writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding="application/octet-stream",
                metadata={
                    "width": str(width),
                    "height": str(height),
                    "channels": str(channels),
                },
            )
            cache[cache_key] = cid
        return cache[cache_key]
