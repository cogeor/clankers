"""Integration tests for the arm pick dataset pipeline.

Tests the full pipeline: MCAP creation -> filtering by grip success ->
image saving to disk at intervals.

Creates synthetic MCAP files with body_poses (some with successful grip,
some without), verifies that only successful episodes' images are saved
and failed episodes are correctly discarded.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MCAP writing helpers (create synthetic test data)
# ---------------------------------------------------------------------------

try:
    from mcap.writer import CompressionType
    from mcap.writer import Writer as McapWriter

    _MCAP_WRITE_AVAILABLE = True
except ImportError:
    _MCAP_WRITE_AVAILABLE = False

try:
    from clankers.mcap_loader import McapEpisodeLoader

    _MCAP_READ_AVAILABLE = True
except ImportError:
    _MCAP_READ_AVAILABLE = False


def _require_mcap_write() -> None:
    if not _MCAP_WRITE_AVAILABLE:
        pytest.skip("mcap writer not available")


def _require_mcap_read() -> None:
    if not _MCAP_READ_AVAILABLE:
        pytest.skip("mcap reader not available")


def write_synthetic_mcap(
    path: str,
    n_steps: int = 20,
    n_joints: int = 8,
    image_width: int = 4,
    image_height: int = 4,
    image_channels: int = 4,
    cube_final_z: float = 0.3,
    include_images: bool = True,
    include_body_poses: bool = True,
) -> None:
    """Write a synthetic MCAP file with joint states, images, and body_poses.

    Parameters
    ----------
    path : str
        Output MCAP file path.
    n_steps : int
        Number of timesteps.
    n_joints : int
        Number of joints.
    image_width, image_height, image_channels : int
        Image dimensions.
    cube_final_z : float
        Final z-position of red_cube (>= 0.525 = success).
    include_images : bool
        Whether to include camera images.
    include_body_poses : bool
        Whether to include body_poses.
    """
    _require_mcap_write()

    with open(path, "wb") as f:
        writer = McapWriter(f, compression=CompressionType.NONE)
        writer.start()

        # Register JSON schema
        json_schema_id = writer.register_schema(name="json", encoding="jsonschema", data=b"")

        # Joint states channel
        joint_ch = writer.register_channel(
            schema_id=json_schema_id,
            topic="/joint_states",
            message_encoding="application/json",
        )

        # Actions channel
        action_ch = writer.register_channel(
            schema_id=json_schema_id,
            topic="/actions",
            message_encoding="application/json",
        )

        # Body poses channel
        body_pose_ch = None
        if include_body_poses:
            body_pose_ch = writer.register_channel(
                schema_id=json_schema_id,
                topic="/body_poses",
                message_encoding="application/json",
            )

        # Image channel
        image_ch = None
        if include_images:
            binary_schema_id = writer.register_schema(
                name="binary", encoding="application/octet-stream", data=b""
            )
            image_ch = writer.register_channel(
                schema_id=binary_schema_id,
                topic="/camera/image",
                message_encoding="application/octet-stream",
                metadata={
                    "width": str(image_width),
                    "height": str(image_height),
                    "channels": str(image_channels),
                },
            )

        rng = np.random.RandomState(42)

        for step in range(n_steps):
            ts_ns = step * 20_000_000  # 20ms per step

            # Joint states
            positions = rng.uniform(-1, 1, n_joints).tolist()
            velocities = rng.uniform(-0.1, 0.1, n_joints).tolist()
            joint_frame = {
                "timestamp_ns": ts_ns,
                "names": [f"joint_{i}" for i in range(n_joints)],
                "positions": positions,
                "velocities": velocities,
                "torques": [0.0] * n_joints,
            }
            writer.add_message(
                channel_id=joint_ch,
                log_time=ts_ns,
                data=json.dumps(joint_frame).encode("utf-8"),
                publish_time=ts_ns,
            )

            # Actions
            action_frame = {
                "timestamp_ns": ts_ns,
                "data": rng.uniform(-1, 1, n_joints).tolist(),
            }
            writer.add_message(
                channel_id=action_ch,
                log_time=ts_ns,
                data=json.dumps(action_frame).encode("utf-8"),
                publish_time=ts_ns,
            )

            # Body poses
            if body_pose_ch is not None:
                # Linearly interpolate cube z from 0.425 (table) to cube_final_z
                t = step / max(n_steps - 1, 1)
                cube_z = 0.425 + t * (cube_final_z - 0.425)

                pose_frame = {
                    "timestamp_ns": ts_ns,
                    "poses": {
                        "red_cube": [0.3, 0.0, cube_z, 0.0, 0.0, 0.0, 1.0],
                        "end_effector": [0.3, 0.0, cube_z + 0.05, 0.0, 0.0, 0.0, 1.0],
                        "table": [0.35, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0],
                    },
                }
                writer.add_message(
                    channel_id=body_pose_ch,
                    log_time=ts_ns,
                    data=json.dumps(pose_frame).encode("utf-8"),
                    publish_time=ts_ns,
                )

            # Images (synthetic solid color)
            if image_ch is not None:
                # Create a simple colored image: color changes with step
                color = int((step / n_steps) * 255)
                pixel = bytes([color, 255 - color, 128, 255][:image_channels])
                image_data = pixel * (image_width * image_height)
                writer.add_message(
                    channel_id=image_ch,
                    log_time=ts_ns,
                    data=image_data,
                    publish_time=ts_ns,
                )

        writer.finish()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMcapLoaderBodyPoses:
    """Test McapEpisodeLoader body_poses support."""

    def test_load_body_poses(self, tmp_path: object) -> None:
        """Verify body_poses are loaded from MCAP."""
        _require_mcap_write()
        _require_mcap_read()

        import pathlib

        p = pathlib.Path(str(tmp_path))
        mcap_path = str(p / "test_ep.mcap")
        write_synthetic_mcap(mcap_path, n_steps=10, cube_final_z=0.6)

        loader = McapEpisodeLoader(mcap_path)
        data = loader.load()

        assert data["body_poses"] is not None
        assert len(data["body_poses"]) == 10

        # Check final frame has red_cube pose
        final = data["body_poses"][-1]
        assert "red_cube" in final["poses"]
        cube_z = final["poses"]["red_cube"][2]
        assert cube_z == pytest.approx(0.6, abs=0.01)

    def test_load_without_body_poses(self, tmp_path: object) -> None:
        """Verify graceful handling when body_poses are absent."""
        _require_mcap_write()
        _require_mcap_read()

        import pathlib

        p = pathlib.Path(str(tmp_path))
        mcap_path = str(p / "test_ep_no_poses.mcap")
        write_synthetic_mcap(mcap_path, n_steps=5, include_body_poses=False)

        loader = McapEpisodeLoader(mcap_path)
        data = loader.load()

        assert data["body_poses"] is None

    def test_load_images_and_joints(self, tmp_path: object) -> None:
        """Verify images and joint positions are loaded correctly."""
        _require_mcap_write()
        _require_mcap_read()

        import pathlib

        p = pathlib.Path(str(tmp_path))
        mcap_path = str(p / "test_ep_full.mcap")
        write_synthetic_mcap(
            mcap_path, n_steps=10, image_width=4, image_height=4, image_channels=4, n_joints=8
        )

        loader = McapEpisodeLoader(mcap_path)
        data = loader.load()

        assert data["images"] is not None
        assert data["images"].shape == (10, 4, 4, 4)
        assert data["images"].dtype == np.uint8

        assert data["joint_positions"] is not None
        assert data["joint_positions"].shape == (10, 8)


class TestGripSuccessCheck:
    """Test grip success detection from body_poses."""

    def test_successful_grip(self) -> None:
        """Cube lifted above threshold -> success."""
        from clankers.dataset_saver import check_grip_success

        body_poses = [
            {"poses": {"red_cube": [0.3, 0.0, 0.425, 0.0, 0.0, 0.0, 1.0]}},
            {"poses": {"red_cube": [0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]}},
            {"poses": {"red_cube": [0.3, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0]}},
        ]
        assert check_grip_success(body_poses) is True

    def test_failed_grip(self) -> None:
        """Cube stays on table -> failure."""
        from clankers.dataset_saver import check_grip_success

        body_poses = [
            {"poses": {"red_cube": [0.3, 0.0, 0.425, 0.0, 0.0, 0.0, 1.0]}},
            {"poses": {"red_cube": [0.3, 0.0, 0.43, 0.0, 0.0, 0.0, 1.0]}},
            {"poses": {"red_cube": [0.3, 0.0, 0.42, 0.0, 0.0, 0.0, 1.0]}},
        ]
        assert check_grip_success(body_poses) is False

    def test_empty_poses(self) -> None:
        """No body_poses -> failure."""
        from clankers.dataset_saver import check_grip_success

        assert check_grip_success([]) is False

    def test_custom_threshold(self) -> None:
        """Custom threshold works."""
        from clankers.dataset_saver import check_grip_success

        body_poses = [
            {"poses": {"red_cube": [0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]}},
        ]
        assert check_grip_success(body_poses, z_threshold=0.45) is True
        assert check_grip_success(body_poses, z_threshold=0.55) is False

    def test_missing_object(self) -> None:
        """Object not in poses -> failure."""
        from clankers.dataset_saver import check_grip_success

        body_poses = [
            {"poses": {"table": [0.35, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]}},
        ]
        assert check_grip_success(body_poses, object_name="red_cube") is False


class TestDatasetPipeline:
    """End-to-end dataset pipeline tests."""

    def test_filter_keeps_successful_discards_failed(self, tmp_path: object) -> None:
        """Verify successful episodes are kept and failed ones are discarded."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        # Write 3 episodes: 2 successful, 1 failed
        write_synthetic_mcap(
            str(input_dir / "ep_001.mcap"),
            n_steps=20,
            cube_final_z=0.6,  # SUCCESS: cube lifted
        )
        write_synthetic_mcap(
            str(input_dir / "ep_002.mcap"),
            n_steps=20,
            cube_final_z=0.42,  # FAIL: cube stayed on table
        )
        write_synthetic_mcap(
            str(input_dir / "ep_003.mcap"),
            n_steps=20,
            cube_final_z=0.55,  # SUCCESS: cube lifted
        )

        stats = process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            image_interval=1,
        )

        assert stats.total_episodes == 3
        assert stats.successful_episodes == 2
        assert stats.failed_episodes == 1

        # Check that only ep_001 and ep_003 directories exist
        assert (output_dir / "ep_001").exists()
        assert not (output_dir / "ep_002").exists()
        assert (output_dir / "ep_003").exists()

    def test_image_interval(self, tmp_path: object) -> None:
        """Verify images are saved at correct intervals."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        write_synthetic_mcap(
            str(input_dir / "ep_success.mcap"),
            n_steps=20,
            cube_final_z=0.6,
        )

        stats = process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            image_interval=5,  # save every 5th frame
        )

        assert stats.successful_episodes == 1
        # 20 frames / 5 interval = 4 images (frames 0, 5, 10, 15)
        assert stats.total_images_saved == 4

        # Check files exist
        ep_dir = output_dir / "ep_success"
        saved_files = sorted(f for f in os.listdir(str(ep_dir)) if f.startswith("frame_"))
        assert len(saved_files) == 4

    def test_no_images_skipped(self, tmp_path: object) -> None:
        """Episodes without images are skipped."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        write_synthetic_mcap(
            str(input_dir / "ep_no_imgs.mcap"),
            n_steps=10,
            cube_final_z=0.6,
            include_images=False,
        )

        stats = process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
        )

        assert stats.total_episodes == 1
        assert stats.skipped_no_images == 1
        assert stats.successful_episodes == 0
        assert stats.total_images_saved == 0

    def test_joint_data_saved(self, tmp_path: object) -> None:
        """Joint positions are saved alongside images for successful episodes."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        write_synthetic_mcap(
            str(input_dir / "ep_joints.mcap"),
            n_steps=15,
            n_joints=8,
            cube_final_z=0.6,
        )

        stats = process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            image_interval=1,
            save_joint_data=True,
        )

        assert stats.successful_episodes == 1
        joint_path = output_dir / "ep_joints" / "joint_positions.npy"
        assert joint_path.exists()

        positions = np.load(str(joint_path))
        assert positions.shape == (15, 8)

    def test_summary_json_written(self, tmp_path: object) -> None:
        """Verify dataset_summary.json is created."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        write_synthetic_mcap(
            str(input_dir / "ep_001.mcap"),
            n_steps=10,
            cube_final_z=0.6,
        )
        write_synthetic_mcap(
            str(input_dir / "ep_002.mcap"),
            n_steps=10,
            cube_final_z=0.4,
        )

        process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            image_interval=2,
        )

        summary_path = output_dir / "dataset_summary.json"
        assert summary_path.exists()

        with open(str(summary_path)) as f:
            summary = json.load(f)

        assert summary["total_episodes"] == 2
        assert summary["successful_episodes"] == 1
        assert summary["failed_episodes"] == 1
        assert summary["image_interval"] == 2
        assert len(summary["episodes"]) == 2

    def test_mixed_pipeline_report(self, tmp_path: object) -> None:
        """Comprehensive test: mix of success, fail, and no-image episodes."""
        _require_mcap_write()
        _require_mcap_read()
        import pathlib

        from clankers.dataset_saver import process_dataset

        p = pathlib.Path(str(tmp_path))
        input_dir = p / "recordings"
        output_dir = p / "dataset"
        input_dir.mkdir()

        # Episode 1: success with images
        write_synthetic_mcap(
            str(input_dir / "ep_001.mcap"),
            n_steps=30,
            cube_final_z=0.6,
        )
        # Episode 2: fail (cube not lifted)
        write_synthetic_mcap(
            str(input_dir / "ep_002.mcap"),
            n_steps=30,
            cube_final_z=0.43,
        )
        # Episode 3: success but no images
        write_synthetic_mcap(
            str(input_dir / "ep_003.mcap"),
            n_steps=30,
            cube_final_z=0.7,
            include_images=False,
        )
        # Episode 4: success with images
        write_synthetic_mcap(
            str(input_dir / "ep_004.mcap"),
            n_steps=30,
            cube_final_z=0.55,
        )

        stats = process_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            image_interval=10,
        )

        assert stats.total_episodes == 4
        assert stats.successful_episodes == 2  # ep_001, ep_004
        assert stats.failed_episodes == 1  # ep_002
        assert stats.skipped_no_images == 1  # ep_003

        # 30 frames / 10 interval = 3 images per episode, 2 episodes = 6
        assert stats.total_images_saved == 6

        # Only successful episodes with images have directories
        assert (output_dir / "ep_001").exists()
        assert not (output_dir / "ep_002").exists()
        assert not (output_dir / "ep_003").exists()
        assert (output_dir / "ep_004").exists()
