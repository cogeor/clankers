"""MCAP-based augmentation: reads segmentation images, generates realistic images, writes output.

Reads simulation segmentation images from MCAP recordings, transforms them
into photorealistic images using the Sim2RealPipeline, and writes the results
to a new MCAP file in the same format as camera recordings.

Requires: mcap>=1.0.0
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from clanker_gym.augmentation.pipeline import Sim2RealPipeline

import numpy as np
from numpy.typing import NDArray

try:
    from mcap.reader import make_reader  # type: ignore[import-untyped]
    from mcap.writer import Writer as McapWriter  # type: ignore[import-untyped]

    _MCAP_AVAILABLE = True
except ImportError:
    _MCAP_AVAILABLE = False


logger = logging.getLogger(__name__)

# How often to log progress (every N images).
_LOG_INTERVAL = 10


def _require_mcap() -> None:
    if not _MCAP_AVAILABLE:
        raise ImportError(
            "mcap is required for McapAugmentor. "
            "Install with: pip install mcap>=1.0.0"
        )


class McapAugmentor:
    """Augments MCAP recordings by transforming segmentation images to realistic ones.

    Reads ``/camera/*`` channels from an input MCAP file, runs each image
    through a :class:`~clanker_gym.augmentation.pipeline.Sim2RealPipeline`,
    and writes the augmented images to a new MCAP file with a configurable
    topic suffix.

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
        pipeline: "Sim2RealPipeline",
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
        max_images: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
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

        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            reader = make_reader(fin)  # type: ignore[possibly-undefined]
            writer = McapWriter(fout)  # type: ignore[possibly-undefined]
            writer.start()

            # Register a single JSON schema for all output channels.
            schema_id = writer.register_schema(
                name="json", encoding="jsonschema", data=b""
            )

            for _schema, channel, message in reader.iter_messages():
                topic: str = channel.topic
                is_camera = topic.startswith(self._input_channel_prefix)

                # --- Non-image channel: optionally copy through ---
                if not is_camera:
                    if self._copy_non_image:
                        out_cid = self._ensure_channel(
                            _out_channel_ids,
                            writer,
                            schema_id,
                            channel.id,
                            topic,
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

                # Parse ImageFrame JSON.
                try:
                    frame = json.loads(message.data.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning(
                        "Skipping non-JSON message on %s at t=%d",
                        topic,
                        message.log_time,
                    )
                    continue

                width = int(frame.get("width", 0))
                height = int(frame.get("height", 0))
                pixel_data = frame.get("data", [])

                if width <= 0 or height <= 0 or not pixel_data:
                    logger.warning(
                        "Skipping invalid ImageFrame on %s (w=%d, h=%d, len=%d)",
                        topic,
                        width,
                        height,
                        len(pixel_data),
                    )
                    continue

                # Reconstruct numpy image from flat uint8 list.
                seg_image = self._decode_image(pixel_data, height, width)
                if seg_image is None:
                    logger.warning(
                        "Could not decode image on %s at t=%d (expected %d bytes, got %d)",
                        topic,
                        message.log_time,
                        height * width * 3,
                        len(pixel_data),
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

                # Build output ImageFrame.
                out_frame = self._encode_image_frame(
                    augmented, frame, message.log_time
                )

                # Write to output channel.
                out_topic = topic + self._output_suffix
                out_cid = self._ensure_channel(
                    _out_channel_ids,
                    writer,
                    schema_id,
                    channel.id,
                    out_topic,
                )
                writer.add_message(
                    channel_id=out_cid,
                    log_time=message.log_time,
                    publish_time=message.publish_time,
                    data=json.dumps(out_frame).encode("utf-8"),
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
        summary: dict[str, Any] = {
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
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(
        pixel_data: list[int], height: int, width: int, channels: int = 3
    ) -> NDArray[np.uint8] | None:
        """Convert a flat list of uint8 values to a (H, W, C) numpy array."""
        expected = height * width * channels
        if len(pixel_data) != expected:
            return None
        arr = np.array(pixel_data, dtype=np.uint8)
        return arr.reshape(height, width, channels)

    @staticmethod
    def _encode_image_frame(
        image: NDArray[np.uint8],
        original_frame: dict[str, Any],
        timestamp_ns: int,
    ) -> dict[str, Any]:
        """Build an ImageFrame dict from an augmented numpy image."""
        h, w = image.shape[:2]
        return {
            "timestamp_ns": original_frame.get("timestamp_ns", timestamp_ns),
            "width": w,
            "height": h,
            "label": original_frame.get("label", "augmented"),
            "data": image.flatten().tolist(),
        }

    @staticmethod
    def _ensure_channel(
        cache: dict[int, int],
        writer: "McapWriter",
        schema_id: int,
        source_channel_id: int,
        topic: str,
    ) -> int:
        """Return (and lazily register) the output channel id for a source channel."""
        if source_channel_id not in cache:
            cid = writer.register_channel(
                schema_id=schema_id,
                topic=topic,
                message_encoding="application/json",
            )
            cache[source_channel_id] = cid
        return cache[source_channel_id]
