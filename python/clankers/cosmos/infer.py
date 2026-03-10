"""Run Cosmos-Transfer2.5 inference using diffusers.

Uses the HuggingFace diffusers pipeline, which handles model download
and caching automatically.

Usage:
    python -m clankers.cosmos infer --spec output/cosmos/spec.json
    python -m clankers.cosmos infer --spec spec.json --control edge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoModel, Cosmos2_5_TransferPipeline
from diffusers.utils import export_to_video, load_video
from PIL import Image

from clankers.cosmos import COSMOS_HF_REPO

# Valid controlnet variants
CONTROL_VARIANTS = ("edge", "depth", "seg", "blur")


class _NoOpSafetyChecker:
    """Stub safety checker that skips the cosmos_guardrail dependency."""

    def to(self, *_args: object, **_kwargs: object) -> _NoOpSafetyChecker:
        return self

    def check_text_safety(self, _text: object) -> bool:
        return True

    def check_video_safety(self, video: object) -> object:
        return video


def _load_controls_from_video(
    video_path: str,
    control_type: str,
    num_frames: int,
) -> list[Image.Image]:
    """Load a video and optionally extract control signals."""
    frames = load_video(video_path)[:num_frames]

    if control_type == "edge":
        import cv2

        edge_maps = [
            cv2.Canny(cv2.cvtColor(np.array(f.convert("RGB")), cv2.COLOR_RGB2BGR), 100, 200)
            for f in frames
        ]
        edge_np = np.stack(edge_maps)[None]  # (T, H, W) -> (1, T, H, W)
        controls_tensor = torch.from_numpy(edge_np).expand(3, -1, -1, -1)  # -> (3, T, H, W)
        return [Image.fromarray(x.numpy()) for x in controls_tensor.permute(1, 2, 3, 0)]

    # For depth/seg/blur, frames are used directly as control images
    return [f.convert("RGB") for f in frames]


def infer(
    spec_path: Path,
    model_id: str = COSMOS_HF_REPO,
    control_type: str = "edge",
) -> Path:
    """Run Cosmos-Transfer2.5 inference via diffusers.

    Parameters
    ----------
    spec_path : Path
        Path to spec.json (produced by prepare.py).
    model_id : str
        HuggingFace model ID (default: nvidia/Cosmos-Transfer2.5-2B).
    control_type : str
        ControlNet variant: edge, depth, seg, or blur (default: edge).

    Returns
    -------
    Path
        Path to the output directory containing generated video.
    """
    if control_type not in CONTROL_VARIANTS:
        raise ValueError(f"control_type must be one of {CONTROL_VARIANTS}, got '{control_type}'")

    spec = json.loads(spec_path.read_text())
    output_dir = Path(spec.get("output_dir", spec_path.parent / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_data = json.loads(Path(spec["prompt_path"]).read_text())
    prompt = prompt_data["prompt"]
    negative_prompt = prompt_data.get("negative_prompt", "")

    num_frames = 93  # Cosmos native chunk size

    # Load controlnet and pipeline
    print(f"Loading controlnet: {control_type}")
    controlnet = AutoModel.from_pretrained(
        model_id,
        revision=f"diffusers/controlnet/general/{control_type}",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading pipeline: {model_id}")
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        revision="diffusers/general",
        torch_dtype=torch.bfloat16,
        safety_checker=_NoOpSafetyChecker(),
    )
    pipe.enable_model_cpu_offload()

    # Determine control input source
    # Prefer the specific control video (depth.mp4, seg.mp4), fall back to rgb video
    control_cfg = spec.get(control_type, {})
    control_video_path = control_cfg.get("control_path", spec.get("video_path"))
    if not control_video_path:
        raise ValueError("No video_path or control_path found in spec")

    conditioning_scale = control_cfg.get("control_weight", 1.0)

    print(f"Loading controls from: {control_video_path}")
    controls = _load_controls_from_video(control_video_path, control_type, num_frames)

    print("Running inference...")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Control: {control_type} (scale={conditioning_scale})")
    print(f"  Frames: {len(controls)}")
    print(f"  Output: {output_dir}")

    result = pipe(
        controls=controls,
        controls_conditioning_scale=conditioning_scale,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=len(controls),
        guidance_scale=spec.get("guidance", 3.0),
        num_inference_steps=spec.get("num_steps", 36),
    )

    output_path = output_dir / "output.mp4"
    export_to_video(result.frames[0], str(output_path), fps=16)

    print(f"\nInference complete. Output: {output_path}")
    return output_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Cosmos-Transfer2.5 inference")
    parser.add_argument("--spec", type=Path, required=True, help="Path to spec.json")
    parser.add_argument(
        "--model-id",
        type=str,
        default=COSMOS_HF_REPO,
        help=f"HuggingFace model ID (default: {COSMOS_HF_REPO})",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="edge",
        choices=CONTROL_VARIANTS,
        help="ControlNet variant (default: edge)",
    )
    args = parser.parse_args()

    infer(
        spec_path=args.spec,
        model_id=args.model_id,
        control_type=args.control,
    )


if __name__ == "__main__":
    main()
