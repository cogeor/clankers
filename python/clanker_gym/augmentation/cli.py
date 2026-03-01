"""Command-line interface for MCAP sim-to-real augmentation.

Usage::

    python -m clanker_gym.augmentation input.mcap output.mcap [options]

    Options:
        --steps N         Inference steps (default: 20)
        --guidance N      CFG scale (default: 7.5)
        --controlnet N    ControlNet strength (default: 1.0)
        --seed N          Random seed
        --scene TYPE      Scene type: manipulation, indoor_nav, outdoor, generic
                          (default: manipulation)
        --device DEVICE   Torch device: cuda, cpu, mps (default: cuda)
        --dtype DTYPE     Model precision: float16, float32 (default: float16)
        --max-images N    Max images to process (for testing)
        --no-copy         Don't copy non-image channels to output
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clanker_gym.augmentation",
        description="Transform simulation segmentation MCAP recordings into "
        "photorealistic images using Stable Diffusion + ControlNet.",
    )

    parser.add_argument(
        "input",
        help="Path to the input .mcap file containing segmentation images.",
    )
    parser.add_argument(
        "output",
        help="Path for the output .mcap file with augmented images.",
    )

    # Pipeline parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        metavar="N",
        help="Number of inference steps per image (default: 20).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        metavar="N",
        help="Classifier-free guidance scale (default: 7.5).",
    )
    parser.add_argument(
        "--controlnet",
        type=float,
        default=1.0,
        metavar="N",
        help="ControlNet conditioning strength (default: 1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Base random seed for reproducibility.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="manipulation",
        choices=["manipulation", "indoor_nav", "outdoor", "generic"],
        help="Scene type for prompt generation (default: manipulation).",
    )

    # Model/device parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device: cuda, cpu, mps (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Model precision (default: float16).",
    )

    # Processing options
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of images to process (for testing).",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy non-image channels (joints, actions, rewards) to output.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the augmentation CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy imports -- only load heavy dependencies when actually running
    from clanker_gym.augmentation.mcap_augmentor import McapAugmentor
    from clanker_gym.augmentation.pipeline import Sim2RealPipeline
    from clanker_gym.augmentation.prompts import SceneType

    # Map scene string to enum
    scene_map = {
        "manipulation": SceneType.MANIPULATION,
        "indoor_nav": SceneType.INDOOR_NAV,
        "outdoor": SceneType.OUTDOOR,
        "generic": SceneType.GENERIC,
    }
    scene_type = scene_map[args.scene]

    logger.info(
        "Initializing pipeline (device=%s, dtype=%s, scene=%s)", args.device, args.dtype, args.scene
    )

    try:
        pipeline = Sim2RealPipeline(
            device=args.device,
            dtype=args.dtype,
            scene_type=scene_type,
            seed=args.seed,
        )

        augmentor = McapAugmentor(
            pipeline=pipeline,
            copy_non_image_channels=not args.no_copy,
        )

        t0 = time.monotonic()
        summary = augmentor.process(
            input_path=args.input,
            output_path=args.output,
            max_images=args.max_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.controlnet,
            seed=args.seed,
        )
        elapsed = time.monotonic() - t0

        # Print summary
        print("\n--- Augmentation Summary ---")
        print(f"  Images processed : {summary['images_processed']}")
        print(f"  Channels found   : {', '.join(summary['channels_found']) or '(none)'}")
        print(f"  Output file      : {summary['output_path']}")
        print(f"  Total time       : {elapsed:.1f}s")
        if summary["images_processed"] > 0:
            print(f"  Per image        : {elapsed / summary['images_processed']:.2f}s")
        print("---")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Augmentation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
