"""Convert Bevy-exported PNG frames to Cosmos-Transfer2.5 input.

Expects a directory with:
  rgb_00000.png, rgb_00001.png, ...       (required)
  depth_00000.png, depth_00001.png, ...   (optional, grayscale u8)
  seg_00000.png, seg_00001.png, ...       (optional, clankers palette)

Produces:
  output_dir/
    rgb.mp4          RGB video
    depth.mp4        Depth video (grayscale)
    seg.mp4          Binary mask video (white=transform, black=keep)
    prompt.json      Text prompt for Cosmos
    spec.json        Cosmos inference spec

Usage:
    python -m clankers.cosmos prepare --input-dir output/frames --output-dir output/cosmos
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from clankers.cosmos import COSMOS_FPS, SEG_PALETTE

# Classes whose pixels become WHITE (transform region) in the binary mask.
# Robot and manipulated objects get re-rendered photorealistically.
TRANSFORM_CLASSES = {"robot", "obstacle", "table"}

# Default prompt for indoor manipulation scenes
DEFAULT_PROMPT = (
    "Photorealistic video of a robot arm manipulating objects on a table in an "
    "indoor workshop. Natural lighting with soft shadows, slightly dusty "
    "environment, visible surface textures on metal and wood. Captured by a "
    "fixed overhead camera."
)
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, CGI, illustration, painting, oversaturated, text, "
    "watermark, blurry, low quality, perfect lighting, unrealistic"
)


def _find_frames(input_dir: Path, prefix: str) -> list[Path]:
    """Find numbered PNG frames with the given prefix, sorted."""
    frames = sorted(input_dir.glob(f"{prefix}_*.png"))
    if not frames:
        frames = sorted(input_dir.glob(f"{prefix}*.png"))
    return frames


def _pngs_to_mp4(
    png_paths: list[Path],
    output_mp4: Path,
    fps: int = COSMOS_FPS,
) -> None:
    """Convert a sequence of PNGs to an MP4 using ffmpeg."""
    if not png_paths:
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("Error: ffmpeg not found in PATH.", file=sys.stderr)
        sys.exit(1)

    # Create a temporary file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in png_paths:
            # ffmpeg concat format: file '/path/to/frame.png'
            # Use absolute path so concat works regardless of temp file location.
            f.write(f"file '{p.resolve().as_posix()}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
        filelist = f.name

    try:
        cmd = [
            ffmpeg,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", filelist,
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(output_mp4),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
    finally:
        Path(filelist).unlink(missing_ok=True)


def _remap_seg_to_binary(seg_path: Path, output_path: Path) -> None:
    """Remap a clankers segmentation PNG to a binary mask.

    White (255,255,255) = transform region (robot, obstacle, table).
    Black (0,0,0) = keep region (ground, wall, unknown).

    Uses nearest-palette-color matching (L2 distance) to handle minor
    dithering/anti-aliasing artifacts from the GPU render pipeline.
    """
    img = np.array(Image.open(seg_path).convert("RGB"), dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for cls_name in TRANSFORM_CLASSES:
        color = SEG_PALETTE.get(cls_name)
        if color is None:
            continue
        diff = img - np.array(color, dtype=np.int32)
        dist_sq = np.sum(diff * diff, axis=-1)
        # Match pixels within tolerance (handles ±2 per channel dithering)
        match = dist_sq < 16  # ~4 per channel
        mask[match] = 255

    # Save as RGB (all channels same) for MP4 compatibility
    binary = np.stack([mask, mask, mask], axis=-1)
    Image.fromarray(binary).save(output_path)


def prepare(
    input_dir: Path,
    output_dir: Path,
    fps: int = COSMOS_FPS,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    guidance: float = 3.0,
    num_steps: int = 35,
    depth_weight: float = 0.5,
    seg_weight: float = 1.0,
) -> Path:
    """Prepare Cosmos input from Bevy-exported frames.

    Parameters
    ----------
    input_dir : Path
        Directory containing rgb_*.png (and optionally depth_*.png, seg_*.png).
    output_dir : Path
        Output directory for MP4s and spec files.
    fps : int
        Video framerate (default: 16, Cosmos native).
    prompt : str
        Text prompt for generation.
    negative_prompt : str
        Negative prompt.
    guidance : float
        Classifier-free guidance scale (default: 3.0).
    num_steps : int
        Diffusion sampling steps (default: 35 for full model).
    depth_weight : float
        Depth control weight (default: 0.5).
    seg_weight : float
        Segmentation control weight (default: 1.0).

    Returns
    -------
    Path
        Path to the generated spec.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- RGB ---
    rgb_frames = _find_frames(input_dir, "rgb")
    if not rgb_frames:
        # Fallback: try frame_*.png (current arm_pick_replay format)
        rgb_frames = _find_frames(input_dir, "frame")
    if not rgb_frames:
        print(f"Error: No RGB frames found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rgb_frames)} RGB frames")
    rgb_mp4 = output_dir / "rgb.mp4"
    _pngs_to_mp4(rgb_frames, rgb_mp4, fps)
    print(f"  → {rgb_mp4}")

    # --- Depth ---
    depth_frames = _find_frames(input_dir, "depth")
    depth_mp4 = None
    if depth_frames:
        print(f"Found {len(depth_frames)} depth frames")
        depth_mp4 = output_dir / "depth.mp4"
        _pngs_to_mp4(depth_frames, depth_mp4, fps)
        print(f"  → {depth_mp4}")
    else:
        print("No depth frames found — Cosmos will auto-generate via DepthAnything2")

    # --- Segmentation → binary mask ---
    seg_frames = _find_frames(input_dir, "seg")
    seg_mp4 = None
    if seg_frames:
        print(f"Found {len(seg_frames)} segmentation frames")
        # Remap to binary masks in a temp directory
        mask_dir = output_dir / "_seg_masks"
        mask_dir.mkdir(exist_ok=True)
        for seg_path in seg_frames:
            mask_path = mask_dir / seg_path.name
            _remap_seg_to_binary(seg_path, mask_path)

        seg_mp4 = output_dir / "seg.mp4"
        mask_paths = sorted(mask_dir.glob("*.png"))
        _pngs_to_mp4(mask_paths, seg_mp4, fps)

        # Clean up temp masks
        shutil.rmtree(mask_dir)
        print(f"  → {seg_mp4}")
    else:
        print("No segmentation frames found — seg control disabled")

    # --- Prompt ---
    prompt_path = output_dir / "prompt.json"
    prompt_data = {"prompt": prompt, "negative_prompt": negative_prompt}
    prompt_path.write_text(json.dumps(prompt_data, indent=2))
    print(f"  → {prompt_path}")

    # --- Spec ---
    spec: dict[str, object] = {
        "prompt_path": str(prompt_path),
        "video_path": str(rgb_mp4),
        "output_dir": str(output_dir / "output"),
        "guidance": guidance,
        "num_steps": num_steps,
    }

    if depth_mp4 is not None:
        spec["depth"] = {
            "control_path": str(depth_mp4),
            "control_weight": depth_weight,
        }

    # Edge: omit control_path → Cosmos auto-generates Canny
    spec["edge"] = {"control_weight": 1.0}

    if seg_mp4 is not None:
        spec["seg"] = {
            "control_path": str(seg_mp4),
            "control_weight": seg_weight,
        }

    spec_path = output_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"  → {spec_path}")

    n_frames = len(rgb_frames)
    chunks = (n_frames + 92) // 93
    print(f"\nReady for Cosmos inference:")
    print(f"  Frames: {n_frames} ({n_frames / fps:.1f}s at {fps}fps)")
    print(f"  Chunks: {chunks} (93 frames each)")
    print(f"  Run: python -m clankers.cosmos infer --spec {spec_path}")

    return spec_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prepare Cosmos input from Bevy frames")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with PNG frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=COSMOS_FPS, help=f"Framerate (default: {COSMOS_FPS})")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Generation prompt")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG scale (default: 3.0)")
    parser.add_argument("--num-steps", type=int, default=35, help="Diffusion steps (default: 35)")
    parser.add_argument("--depth-weight", type=float, default=0.5, help="Depth control weight")
    parser.add_argument("--seg-weight", type=float, default=1.0, help="Seg control weight")
    args = parser.parse_args()

    prepare(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        prompt=args.prompt,
        guidance=args.guidance,
        num_steps=args.num_steps,
        depth_weight=args.depth_weight,
        seg_weight=args.seg_weight,
    )


if __name__ == "__main__":
    main()
