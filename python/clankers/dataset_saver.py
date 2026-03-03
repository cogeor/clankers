"""Save arm pick dataset images to disk from MCAP episodes.

Loads MCAP episodes from a directory, filters by grip success
(red_cube.z >= threshold), and saves camera images at a configurable
interval to disk as PNG files.

Usage::

    python -m clankers.dataset_saver \\
        --input-dir recordings/ \\
        --output-dir dataset/ \\
        --image-interval 5 \\
        --success-threshold 0.525

Requires ``mcap>=1.0.0`` and ``Pillow>=9.0.0``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clankers.mcap_loader import McapEpisodeLoader


@dataclass
class DatasetStats:
    """Statistics from a dataset save run."""

    total_episodes: int = 0
    successful_episodes: int = 0
    failed_episodes: int = 0
    skipped_no_images: int = 0
    total_images_saved: int = 0
    episode_details: list[dict[str, Any]] = field(default_factory=list)


def check_grip_success(
    body_poses: list[dict[str, Any]],
    object_name: str = "red_cube",
    z_threshold: float = 0.525,
) -> bool:
    """Check if the episode achieved a successful grip.

    Examines the final body pose frame to see if the target object's
    z-position exceeds the threshold (indicating it was lifted).

    Parameters
    ----------
    body_poses : list[dict]
        List of BodyPoseFrame dicts from MCAP (each has ``poses`` key).
    object_name : str
        Name of the object to check.
    z_threshold : float
        Minimum z-position for success (default 0.525 = 10cm above table).

    Returns
    -------
    bool
        True if the object was lifted above the threshold at the final step.
    """
    if not body_poses:
        return False

    final_frame = body_poses[-1]
    poses = final_frame.get("poses", {})
    obj_pose = poses.get(object_name)

    if obj_pose is None or len(obj_pose) < 3:
        return False

    # obj_pose is [x, y, z, qx, qy, qz, qw]
    return float(obj_pose[2]) >= z_threshold


def save_episode_images(
    images: NDArray[np.uint8],
    output_dir: str,
    episode_name: str,
    interval: int = 1,
) -> int:
    """Save images from an episode to disk at the specified interval.

    Parameters
    ----------
    images : NDArray[np.uint8]
        Image array of shape ``(T, H, W, C)`` in HWC uint8 format.
    output_dir : str
        Root output directory.
    episode_name : str
        Name for the episode subdirectory (e.g. ``ep_0001``).
    interval : int
        Save every Nth frame (1 = every frame, 5 = every 5th frame).

    Returns
    -------
    int
        Number of images saved.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        # Fallback: save as raw numpy .npy files
        ep_dir = os.path.join(output_dir, episode_name)
        os.makedirs(ep_dir, exist_ok=True)
        saved = 0
        for i in range(0, len(images), interval):
            np.save(os.path.join(ep_dir, f"frame_{i:04d}.npy"), images[i])
            saved += 1
        return saved

    ep_dir = os.path.join(output_dir, episode_name)
    os.makedirs(ep_dir, exist_ok=True)

    saved = 0
    for i in range(0, len(images), interval):
        img = images[i]  # (H, W, C)
        channels = img.shape[2] if img.ndim == 3 else 1

        if channels == 4:
            pil_img = PILImage.fromarray(img, mode="RGBA")
        elif channels == 3:
            pil_img = PILImage.fromarray(img, mode="RGB")
        else:
            pil_img = PILImage.fromarray(img.squeeze(), mode="L")

        pil_img.save(os.path.join(ep_dir, f"frame_{i:04d}.png"))
        saved += 1

    return saved


def process_dataset(
    input_dir: str,
    output_dir: str,
    image_interval: int = 1,
    success_threshold: float = 0.525,
    object_name: str = "red_cube",
    joint_dim: int | None = None,
    save_joint_data: bool = True,
) -> DatasetStats:
    """Process MCAP episodes: filter by success and save images to disk.

    Parameters
    ----------
    input_dir : str
        Directory containing ``.mcap`` episode files.
    output_dir : str
        Directory to save filtered images and metadata.
    image_interval : int
        Save every Nth image frame (1 = all, 5 = every 5th).
    success_threshold : float
        Minimum z-position for the object to count as successfully gripped.
    object_name : str
        Name of the target object in body_poses.
    joint_dim : int | None
        Number of joints to extract. None = all available.
    save_joint_data : bool
        If True, also save joint positions alongside images.

    Returns
    -------
    DatasetStats
        Statistics about the processing run.
    """
    mcap_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mcap")
    )

    if not mcap_files:
        raise ValueError(f"No .mcap files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    stats = DatasetStats()

    for mcap_path in mcap_files:
        stats.total_episodes += 1
        ep_name = os.path.splitext(os.path.basename(mcap_path))[0]

        loader = McapEpisodeLoader(mcap_path)
        data = loader.load()

        # Check for images
        images = data.get("images")
        if images is None:
            stats.skipped_no_images += 1
            stats.episode_details.append(
                {
                    "episode": ep_name,
                    "status": "skipped_no_images",
                }
            )
            continue

        # Check grip success via body_poses
        body_poses = data.get("body_poses")
        success = check_grip_success(
            body_poses if body_poses else [],
            object_name=object_name,
            z_threshold=success_threshold,
        )

        if not success:
            stats.failed_episodes += 1
            stats.episode_details.append(
                {
                    "episode": ep_name,
                    "status": "failed_grip",
                    "reason": f"{object_name} not lifted above {success_threshold}m",
                }
            )
            continue

        # Success: save images at interval
        stats.successful_episodes += 1
        n_saved = save_episode_images(images, output_dir, ep_name, interval=image_interval)
        stats.total_images_saved += n_saved

        detail: dict[str, Any] = {
            "episode": ep_name,
            "status": "success",
            "total_frames": len(images),
            "images_saved": n_saved,
        }

        # Optionally save joint data
        if save_joint_data and data.get("joint_positions") is not None:
            positions = data["joint_positions"]
            if joint_dim is not None and positions.shape[1] > joint_dim:
                positions = positions[:, :joint_dim]
            ep_dir = os.path.join(output_dir, ep_name)
            os.makedirs(ep_dir, exist_ok=True)
            np.save(os.path.join(ep_dir, "joint_positions.npy"), positions)
            detail["joint_frames"] = len(positions)

        stats.episode_details.append(detail)

    # Write summary metadata
    summary = {
        "total_episodes": stats.total_episodes,
        "successful_episodes": stats.successful_episodes,
        "failed_episodes": stats.failed_episodes,
        "skipped_no_images": stats.skipped_no_images,
        "total_images_saved": stats.total_images_saved,
        "image_interval": image_interval,
        "success_threshold": success_threshold,
        "object_name": object_name,
        "episodes": stats.episode_details,
    }
    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return stats


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Save arm pick dataset images from MCAP recordings"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing .mcap episode files",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_output",
        help="Directory to save filtered images (default: dataset_output)",
    )
    parser.add_argument(
        "--image-interval",
        type=int,
        default=5,
        help="Save every Nth image frame (default: 5)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.525,
        help="Min z-position for grip success (default: 0.525)",
    )
    parser.add_argument(
        "--object-name",
        default="red_cube",
        help="Name of target object in body_poses (default: red_cube)",
    )
    parser.add_argument(
        "--joint-dim",
        type=int,
        default=None,
        help="Number of joints to save (default: all)",
    )
    args = parser.parse_args()

    print(f"Processing MCAP episodes from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image interval: every {args.image_interval} frames")
    print(f"Success threshold: {args.object_name}.z >= {args.success_threshold}m")
    print()

    stats = process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_interval=args.image_interval,
        success_threshold=args.success_threshold,
        object_name=args.object_name,
        joint_dim=args.joint_dim,
    )

    print("=== Dataset Processing Results ===")
    print(f"  Total episodes:      {stats.total_episodes}")
    print(f"  Successful (kept):   {stats.successful_episodes}")
    print(f"  Failed (discarded):  {stats.failed_episodes}")
    print(f"  Skipped (no images): {stats.skipped_no_images}")
    print(f"  Total images saved:  {stats.total_images_saved}")
    print()

    for detail in stats.episode_details:
        status = detail["status"]
        ep = detail["episode"]
        if status == "success":
            print(f"  [KEPT]      {ep}: {detail['images_saved']} images saved")
        elif status == "failed_grip":
            print(f"  [DISCARDED] {ep}: {detail.get('reason', 'grip failed')}")
        else:
            print(f"  [SKIPPED]   {ep}: no images")

    print(f"\nSummary written to: {args.output_dir}/dataset_summary.json")


if __name__ == "__main__":
    main()
