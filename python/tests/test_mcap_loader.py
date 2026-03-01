"""Integration tests for McapEpisodeLoader and EpisodeDataset.

These tests create synthetic MCAP files using the Python mcap writer, then
verify that the loader reads them back correctly into numpy arrays.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

try:
    import mcap  # noqa: F401
    from mcap.writer import CompressionType, Writer

    HAS_MCAP = True
except ImportError:
    HAS_MCAP = False

pytestmark = pytest.mark.skipif(not HAS_MCAP, reason="mcap not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_STEPS = 20
NUM_JOINTS = 2
NUM_ACTION_DIMS = 3


def _write_synthetic_mcap(path: str) -> None:
    """Write a synthetic MCAP file with known data for testing."""
    with open(path, "wb") as f:
        writer = Writer(f, compression=CompressionType.NONE)
        writer.start()

        schema_id = writer.register_schema(name="json", encoding="jsonschema", data=b"")
        joint_ch = writer.register_channel(
            topic="/joint_states",
            message_encoding="application/json",
            schema_id=schema_id,
        )
        action_ch = writer.register_channel(
            topic="/actions",
            message_encoding="application/json",
            schema_id=schema_id,
        )
        reward_ch = writer.register_channel(
            topic="/reward",
            message_encoding="application/json",
            schema_id=schema_id,
        )

        for i in range(NUM_STEPS):
            ts = (i + 1) * 1_000_000  # nanoseconds

            # Joint states: JointFrame-like dict with positions, velocities,
            # torques, and timestamp_ns.
            joint_frame = {
                "timestamp_ns": ts,
                "names": ["shoulder", "elbow"],
                "positions": [0.1 * i, -0.1 * i],
                "velocities": [0.01 * i, 0.02 * i],
                "torques": [1.0 * i, -0.5 * i],
            }
            writer.add_message(
                channel_id=joint_ch,
                log_time=ts,
                publish_time=ts,
                data=json.dumps(joint_frame).encode("utf-8"),
            )

            # Actions: bare list of floats (the loader expects
            # np.array(list_of_lists) to produce (T, A) shape).
            action = [0.5 * i, -0.5 * i, 0.0]
            writer.add_message(
                channel_id=action_ch,
                log_time=ts,
                publish_time=ts,
                data=json.dumps(action).encode("utf-8"),
            )

            # Rewards: bare scalar float.
            reward = float(i) * 0.1
            writer.add_message(
                channel_id=reward_ch,
                log_time=ts,
                publish_time=ts,
                data=json.dumps(reward).encode("utf-8"),
            )

        writer.finish()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_mcap(tmp_path):
    """Create a single synthetic MCAP file and return its path."""
    path = os.path.join(str(tmp_path), "episode_001.mcap")
    _write_synthetic_mcap(path)
    return path


@pytest.fixture()
def synthetic_mcap_dir(tmp_path):
    """Create a directory with 3 synthetic MCAP files."""
    for i in range(3):
        path = os.path.join(str(tmp_path), f"episode_{i:03d}.mcap")
        _write_synthetic_mcap(path)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoaderWithSyntheticMcap:
    """Tests for McapEpisodeLoader.load() with synthetic data."""

    def test_shapes(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        data = loader.load()

        assert data["timestamps_ns"] is not None
        assert data["timestamps_ns"].shape == (NUM_STEPS,)

        assert data["joint_positions"] is not None
        assert data["joint_positions"].shape == (NUM_STEPS, NUM_JOINTS)

        assert data["actions"] is not None
        assert data["actions"].shape == (NUM_STEPS, NUM_ACTION_DIMS)

        assert data["rewards"] is not None
        assert data["rewards"].shape == (NUM_STEPS,)

    def test_dtypes(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        data = loader.load()

        assert data["timestamps_ns"].dtype == np.int64
        assert data["joint_positions"].dtype == np.float32
        assert data["actions"].dtype == np.float32
        assert data["rewards"].dtype == np.float32

    def test_values(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        data = loader.load()

        # Check first timestamp.
        assert data["timestamps_ns"][0] == 1_000_000

        # Check positions at step 0: [0.0, 0.0].
        np.testing.assert_allclose(data["joint_positions"][0], [0.0, 0.0], atol=1e-6)

        # Check positions at step 5: [0.5, -0.5].
        np.testing.assert_allclose(data["joint_positions"][5], [0.5, -0.5], atol=1e-6)

        # Check rewards at step 10: 1.0.
        np.testing.assert_allclose(data["rewards"][10], 1.0, atol=1e-6)


class TestSb3ReplayBuffer:
    """Tests for McapEpisodeLoader.to_sb3_replay_buffer()."""

    def test_keys(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        buf = loader.to_sb3_replay_buffer()

        assert "observations" in buf
        assert "next_observations" in buf
        assert "actions" in buf
        assert "rewards" in buf
        assert "dones" in buf

    def test_shapes(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        buf = loader.to_sb3_replay_buffer()

        # T-1 transitions for T=20 steps.
        assert buf["observations"].shape == (NUM_STEPS - 1, NUM_JOINTS)
        assert buf["next_observations"].shape == (NUM_STEPS - 1, NUM_JOINTS)
        assert buf["actions"].shape == (NUM_STEPS - 1, NUM_ACTION_DIMS)
        assert buf["rewards"].shape == (NUM_STEPS - 1,)
        assert buf["dones"].shape == (NUM_STEPS - 1,)

    def test_dones_last_is_terminal(self, synthetic_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        loader = McapEpisodeLoader(synthetic_mcap)
        buf = loader.to_sb3_replay_buffer()

        # Last transition should be terminal.
        assert buf["dones"][-1] == 1.0
        # All others should be zero.
        assert np.all(buf["dones"][:-1] == 0.0)


class TestEpisodeDataset:
    """Tests for EpisodeDataset."""

    def test_len(self, synthetic_mcap_dir):
        from clankers.mcap_loader import EpisodeDataset

        dataset = EpisodeDataset(synthetic_mcap_dir)
        assert len(dataset) == 3

    def test_getitem_returns_buffer(self, synthetic_mcap_dir):
        from clankers.mcap_loader import EpisodeDataset

        dataset = EpisodeDataset(synthetic_mcap_dir)
        buf = dataset[0]

        assert "observations" in buf
        assert "actions" in buf
        assert buf["observations"].shape[0] == NUM_STEPS - 1


class TestRustFormatMcap:
    """Tests for MCAP files written by Rust's RecorderPlugin.

    Rust serializes ActionFrame as {"timestamp_ns": ..., "data": [...]}
    and RewardFrame as {"timestamp_ns": ..., "reward": 0.5}. The loader
    must unwrap these dicts to extract the bare values.
    """

    @staticmethod
    def _write_rust_format_mcap(path: str) -> None:
        """Write MCAP with Rust-style ActionFrame/RewardFrame dicts."""
        with open(path, "wb") as f:
            writer = Writer(f, compression=CompressionType.NONE)
            writer.start()

            schema_id = writer.register_schema(name="json", encoding="jsonschema", data=b"")
            joint_ch = writer.register_channel(
                topic="/joint_states",
                message_encoding="application/json",
                schema_id=schema_id,
            )
            action_ch = writer.register_channel(
                topic="/actions",
                message_encoding="application/json",
                schema_id=schema_id,
            )
            reward_ch = writer.register_channel(
                topic="/reward",
                message_encoding="application/json",
                schema_id=schema_id,
            )

            for i in range(NUM_STEPS):
                ts = (i + 1) * 1_000_000

                joint_frame = {
                    "timestamp_ns": ts,
                    "names": ["shoulder", "elbow"],
                    "positions": [0.1 * i, -0.1 * i],
                    "velocities": [0.01 * i, 0.02 * i],
                    "torques": [1.0 * i, -0.5 * i],
                }
                writer.add_message(
                    channel_id=joint_ch,
                    log_time=ts,
                    publish_time=ts,
                    data=json.dumps(joint_frame).encode("utf-8"),
                )

                # Rust-style ActionFrame dict
                action_frame = {
                    "timestamp_ns": ts,
                    "data": [0.5 * i, -0.5 * i, 0.0],
                }
                writer.add_message(
                    channel_id=action_ch,
                    log_time=ts,
                    publish_time=ts,
                    data=json.dumps(action_frame).encode("utf-8"),
                )

                # Rust-style RewardFrame dict
                reward_frame = {
                    "timestamp_ns": ts,
                    "reward": float(i) * 0.1,
                }
                writer.add_message(
                    channel_id=reward_ch,
                    log_time=ts,
                    publish_time=ts,
                    data=json.dumps(reward_frame).encode("utf-8"),
                )

            writer.finish()

    @pytest.fixture()
    def rust_mcap(self, tmp_path):
        path = os.path.join(str(tmp_path), "rust_episode.mcap")
        self._write_rust_format_mcap(path)
        return path

    def test_actions_unwrapped(self, rust_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        data = McapEpisodeLoader(rust_mcap).load()
        assert data["actions"] is not None
        assert data["actions"].shape == (NUM_STEPS, NUM_ACTION_DIMS)
        # Step 0: [0.0, 0.0, 0.0]
        np.testing.assert_allclose(data["actions"][0], [0.0, 0.0, 0.0], atol=1e-6)
        # Step 5: [2.5, -2.5, 0.0]
        np.testing.assert_allclose(data["actions"][5], [2.5, -2.5, 0.0], atol=1e-6)

    def test_rewards_unwrapped(self, rust_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        data = McapEpisodeLoader(rust_mcap).load()
        assert data["rewards"] is not None
        assert data["rewards"].shape == (NUM_STEPS,)
        np.testing.assert_allclose(data["rewards"][0], 0.0, atol=1e-6)
        np.testing.assert_allclose(data["rewards"][10], 1.0, atol=1e-6)

    def test_sb3_buffer_from_rust_format(self, rust_mcap):
        from clankers.mcap_loader import McapEpisodeLoader

        buf = McapEpisodeLoader(rust_mcap).to_sb3_replay_buffer()
        assert buf["actions"].shape == (NUM_STEPS - 1, NUM_ACTION_DIMS)
        assert buf["rewards"].shape == (NUM_STEPS - 1,)
        assert buf["dones"][-1] == 1.0


class TestLoaderErrors:
    """Tests for error handling."""

    def test_missing_file(self, tmp_path):
        from clankers.mcap_loader import McapEpisodeLoader

        nonexistent = os.path.join(str(tmp_path), "does_not_exist.mcap")
        loader = McapEpisodeLoader(nonexistent)

        with pytest.raises((FileNotFoundError, OSError)):
            loader.load()
