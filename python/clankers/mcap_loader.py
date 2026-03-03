"""MCAP episode loader for offline training data.

Reads ``.mcap`` files written by the ``clankers-record`` crate and returns
numpy arrays suitable for use with Stable-Baselines3 replay buffers or
PyTorch ``DataLoader`` instances.

Channels
--------
``/joint_states``
    JSON-encoded ``JointFrame`` messages (timestamp_ns, names, positions,
    velocities, torques).
``/actions``
    JSON-encoded action vectors (float arrays).
``/reward``
    JSON-encoded scalar reward values.
``/camera/image``
    Raw ``uint8`` bytes with ``width`` and ``height`` in the channel
    metadata. Each message is a flat ``H*W*C`` buffer.

Optional dependency
-------------------
Requires ``mcap>=1.0.0``.  Install with::

    pip install mcap

The module imports ``mcap`` lazily so that the rest of ``clankers``
remains importable even when ``mcap`` is not installed.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from mcap.reader import make_reader  # type: ignore[import-untyped, import-not-found]

    _MCAP_AVAILABLE = True
except ImportError:
    _MCAP_AVAILABLE = False


def _require_mcap() -> None:
    if not _MCAP_AVAILABLE:
        raise ImportError(
            "mcap is required for McapEpisodeLoader. Install with: pip install mcap>=1.0.0"
        )


class McapEpisodeLoader:
    """Reads a single ``.mcap`` episode file and returns numpy arrays.

    Parameters
    ----------
    path : str
        Path to the ``.mcap`` file.

    Examples
    --------
    >>> loader = McapEpisodeLoader("episode_001.mcap")
    >>> data = loader.load()
    >>> data["joint_positions"].shape  # (T, n_joints)
    >>> data["images"].shape           # (T, H, W, C) or None
    >>> buf = loader.to_sb3_replay_buffer()
    >>> buf["observations"].shape      # (T-1, ...) SB3-compatible
    """

    CHANNEL_JOINT_STATES = "/joint_states"
    CHANNEL_ACTIONS = "/actions"
    CHANNEL_REWARD = "/reward"
    CHANNEL_IMAGE = "/camera/image"
    CHANNEL_BODY_POSES = "/body_poses"

    def __init__(self, path: str) -> None:
        _require_mcap()
        self.path = path
        self._loaded: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Load the episode from disk.

        Returns a dict with the following keys:

        ``timestamps_ns`` : NDArray[np.int64], shape (T,)
            Message timestamps in nanoseconds (from ``/joint_states``).
        ``joint_positions`` : NDArray[np.float32] | None, shape (T, J)
            Joint positions in radians.  ``None`` if channel not present.
        ``joint_velocities`` : NDArray[np.float32] | None, shape (T, J)
            Joint velocities.  ``None`` if channel not present.
        ``joint_torques`` : NDArray[np.float32] | None, shape (T, J)
            Joint torques.  ``None`` if channel not present.
        ``actions`` : NDArray[np.float32] | None, shape (T, A)
            Actions taken at each step.  ``None`` if channel not present.
        ``rewards`` : NDArray[np.float32] | None, shape (T,)
            Scalar rewards.  ``None`` if channel not present.
        ``images`` : NDArray[np.uint8] | None, shape (T, H, W, C)
            Camera images.  ``None`` if channel not present.
        """
        if self._loaded is not None:
            return self._loaded

        raw: dict[str, list[Any]] = {
            "joint_states": [],
            "actions": [],
            "rewards": [],
            "images": [],
            "body_poses": [],
            "timestamps_ns": [],
        }
        # Metadata for images (width, height, channels) extracted from channel
        # schema or first message.
        image_meta: dict[str, int] = {}

        with open(self.path, "rb") as f:
            reader = make_reader(f)  # type: ignore[possibly-undefined]
            # Collect channel metadata first.
            summary = reader.get_summary()
            assert summary is not None, "MCAP file has no summary"
            for channel in summary.channels.values():
                topic = channel.topic
                if topic == self.CHANNEL_IMAGE and channel.metadata:
                    try:
                        image_meta["width"] = int(channel.metadata.get("width", 0))
                        image_meta["height"] = int(channel.metadata.get("height", 0))
                        image_meta["channels"] = int(channel.metadata.get("channels", 3))
                    except (ValueError, TypeError):
                        pass

            # Read all messages.
            for _schema, channel, message in reader.iter_messages():
                topic = channel.topic
                if topic == self.CHANNEL_JOINT_STATES:
                    frame = json.loads(message.data.decode("utf-8"))
                    raw["joint_states"].append(frame)
                    raw["timestamps_ns"].append(int(frame.get("timestamp_ns", message.log_time)))
                elif topic == self.CHANNEL_ACTIONS:
                    action = json.loads(message.data.decode("utf-8"))
                    # Handle both Rust ActionFrame dict and bare list
                    if isinstance(action, dict):
                        raw["actions"].append(action.get("data", []))
                    else:
                        raw["actions"].append(action)
                elif topic == self.CHANNEL_REWARD:
                    reward = json.loads(message.data.decode("utf-8"))
                    # Handle both Rust RewardFrame dict and bare float
                    if isinstance(reward, dict):
                        raw["rewards"].append(reward.get("reward", 0.0))
                    else:
                        raw["rewards"].append(reward)
                elif topic == self.CHANNEL_BODY_POSES:
                    frame = json.loads(message.data.decode("utf-8"))
                    raw["body_poses"].append(frame)
                elif topic == self.CHANNEL_IMAGE:
                    raw["images"].append(bytes(message.data))
                    # Try to get image dimensions from first message if metadata missing.
                    if not image_meta:
                        # Cannot infer without metadata; will require explicit metadata.
                        pass

        result: dict[str, Any] = {
            "timestamps_ns": None,
            "joint_positions": None,
            "joint_velocities": None,
            "joint_torques": None,
            "actions": None,
            "rewards": None,
            "images": None,
            "body_poses": None,
        }

        if raw["timestamps_ns"]:
            result["timestamps_ns"] = np.array(raw["timestamps_ns"], dtype=np.int64)

        if raw["joint_states"]:
            positions = [f.get("positions", []) for f in raw["joint_states"]]
            velocities = [f.get("velocities", []) for f in raw["joint_states"]]
            torques = [f.get("torques", []) for f in raw["joint_states"]]
            result["joint_positions"] = np.array(positions, dtype=np.float32)
            result["joint_velocities"] = np.array(velocities, dtype=np.float32)
            result["joint_torques"] = np.array(torques, dtype=np.float32)

        if raw["actions"]:
            result["actions"] = np.array(raw["actions"], dtype=np.float32)

        if raw["rewards"]:
            result["rewards"] = np.array(raw["rewards"], dtype=np.float32)

        if raw["images"] and image_meta:
            width = image_meta.get("width", 0)
            height = image_meta.get("height", 0)
            channels = image_meta.get("channels", 3)
            if width > 0 and height > 0:
                frames = []
                expected = height * width * channels
                for raw_bytes in raw["images"]:
                    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    if arr.size == expected:
                        frames.append(arr.reshape(height, width, channels))
                if frames:
                    result["images"] = np.stack(frames, axis=0)

        if raw["body_poses"]:
            # Each entry is a dict with "timestamp_ns" and "poses" keys.
            # "poses" maps body name -> [x, y, z, qx, qy, qz, qw].
            result["body_poses"] = raw["body_poses"]

        self._loaded = result  # type: ignore[assignment]
        return result

    def to_sb3_replay_buffer(self) -> dict[str, NDArray[Any]]:
        """Convert the episode to an SB3-compatible replay buffer dict.

        The returned dict has the following keys:

        ``observations`` : shape (T-1, ...)
            Observations at time t.  Channel-first uint8 ``(C, H, W)`` for
            Image spaces, or float32 ``(J,)`` for joint-state spaces.
        ``next_observations`` : shape (T-1, ...)
            Observations at time t+1.
        ``actions`` : shape (T-1, A)
            Actions taken.
        ``rewards`` : shape (T-1,)
            Scalar rewards.
        ``dones`` : shape (T-1,)
            Episode termination flags (all ``False`` for mid-episode steps;
            last step is ``True``).

        Returns
        -------
        dict[str, np.ndarray]

        Raises
        ------
        ValueError
            If neither joint states nor images are available in the episode,
            or if required channels are missing.
        """
        data = self.load()

        # Determine observations array.
        obs_array: NDArray[Any] | None = None
        if data.get("images") is not None:
            imgs = data["images"]  # (T, H, W, C)
            assert imgs is not None
            # Transpose to channel-first: (T, C, H, W).
            obs_array = imgs.transpose(0, 3, 1, 2)
        elif data.get("joint_positions") is not None:
            obs_array = data["joint_positions"]
        else:
            raise ValueError(
                "Episode contains neither images nor joint_positions. "
                "Cannot build SB3 replay buffer."
            )

        if data.get("actions") is None:
            raise ValueError("Episode missing /actions channel.")
        if data.get("rewards") is None:
            raise ValueError("Episode missing /reward channel.")

        actions = data["actions"]
        rewards = data["rewards"]
        assert actions is not None
        assert rewards is not None

        # Align lengths: T steps produce T-1 transitions.
        assert obs_array is not None  # guaranteed by ValueError branches above
        t = min(len(obs_array), len(actions), len(rewards))
        if t < 2:
            raise ValueError(f"Episode too short to form transitions: {t} steps.")

        obs = obs_array[: t - 1]
        next_obs = obs_array[1:t]
        act = actions[: t - 1]
        rew = rewards[: t - 1]

        # Last transition in the episode is terminal.
        dones = np.zeros(t - 1, dtype=np.float32)
        dones[-1] = 1.0

        return {
            "observations": obs,
            "next_observations": next_obs,
            "actions": act,
            "rewards": rew,
            "dones": dones,
        }


class EpisodeDataset:
    """Wraps a directory of ``.mcap`` files for use with a DataLoader.

    Each item returned by ``__getitem__`` is the SB3 replay buffer dict for
    one episode (see :meth:`McapEpisodeLoader.to_sb3_replay_buffer`).

    Parameters
    ----------
    directory : str
        Path to the directory containing ``.mcap`` files.

    Examples
    --------
    >>> dataset = EpisodeDataset("/data/episodes")
    >>> len(dataset)
    42
    >>> sample = dataset[0]
    >>> sample["observations"].shape
    """

    def __init__(self, directory: str) -> None:
        _require_mcap()
        self.directory = directory
        self._paths: list[str] = sorted(
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mcap")
        )

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int) -> dict[str, NDArray[Any]]:
        path = self._paths[idx]
        loader = McapEpisodeLoader(path)
        return loader.to_sb3_replay_buffer()
