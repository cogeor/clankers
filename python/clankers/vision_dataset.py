"""Multimodal dataset for vision-based imitation learning.

Loads MCAP episodes containing camera images and joint states, and returns
``(image, positions, velocity)`` tuples suitable for training a
:class:`~clankers.vision_model.VisionPolicyNet`.

Images are normalised to ``[0, 1]`` float32 in channel-first ``(C, H, W)``
layout.  Velocities are computed as finite differences:
``(pos_{t+1} - pos_t) / control_dt``.

Requires ``mcap>=1.0.0`` and ``torch>=2.0.0``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from clankers.mcap_loader import McapEpisodeLoader


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for VisionPositionDataset. pip install torch>=2.0.0")


class VisionPositionDataset:
    """PyTorch Dataset of ``(image, positions, velocity)`` from MCAP episodes.

    Each sample corresponds to one timestep *t* in an episode:

    - ``image``:     ``(C, H, W)`` float32 in ``[0, 1]``
    - ``positions``: ``(J,)`` float32 joint positions (radians)
    - ``velocity``:  ``(J,)`` float32 joint velocity (rad/s)

    Velocities are finite-differenced from consecutive position frames:
    ``vel_t = (pos_{t+1} - pos_t) / control_dt``.

    Parameters
    ----------
    images : NDArray[np.uint8]
        Stacked images ``(N, H, W, C)`` in HWC uint8 format.
    positions : NDArray[np.float32]
        Joint positions ``(N, J)`` in radians.
    velocities : NDArray[np.float32]
        Target velocities ``(N, J)`` in rad/s.
    """

    def __init__(
        self,
        images: NDArray[np.uint8],
        positions: NDArray[np.float32],
        velocities: NDArray[np.float32],
    ) -> None:
        _require_torch()
        assert len(images) == len(positions) == len(velocities)
        self._images = images
        self._positions = positions
        self._velocities = velocities

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        # HWC uint8 → CHW float32 normalised to [0, 1]
        img = self._images[idx]  # (H, W, C) uint8
        img_f = img.astype(np.float32) / 255.0
        img_chw = np.transpose(img_f, (2, 0, 1))  # (C, H, W)

        pos = self._positions[idx]
        vel = self._velocities[idx]

        return (
            torch.from_numpy(img_chw),
            torch.from_numpy(pos),
            torch.from_numpy(vel),
        )

    # -- Properties ----------------------------------------------------------

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` shape of a single image sample."""
        _, h, w, c = self._images.shape
        return (c, h, w)

    @property
    def joint_dim(self) -> int:
        """Number of joint dimensions."""
        return int(self._positions.shape[1])

    @property
    def velocity_dim(self) -> int:
        """Number of velocity dimensions (same as joint_dim)."""
        return int(self._velocities.shape[1])

    # -- Factories -----------------------------------------------------------

    @classmethod
    def from_mcap_file(
        cls,
        path: str,
        joint_dim: int | None = None,
        control_dt: float = 0.02,
    ) -> VisionPositionDataset:
        """Load a single MCAP episode file.

        Parameters
        ----------
        path : str
            Path to ``.mcap`` file recorded by ``clankers-record``.
        joint_dim : int | None
            Number of joints to use.  If ``None``, uses all available.
        control_dt : float
            Timestep for velocity computation (seconds).

        Raises
        ------
        ValueError
            If the MCAP file lacks images or joint positions.
        """
        data = McapEpisodeLoader(path).load()

        images, positions = _validate_and_extract(data, joint_dim)
        velocities = _compute_velocities(positions, control_dt)

        # Trim to T-1 (velocity needs next frame)
        return cls(images[:-1], positions[:-1], velocities)

    @classmethod
    def from_mcap_dir(
        cls,
        directory: str,
        joint_dim: int | None = None,
        control_dt: float = 0.02,
    ) -> VisionPositionDataset:
        """Load all MCAP episodes from a directory and concatenate.

        Parameters
        ----------
        directory : str
            Directory containing ``.mcap`` files.
        joint_dim : int | None
            Number of joints to use.  If ``None``, uses all available.
        control_dt : float
            Timestep for velocity computation (seconds).
        """
        mcap_files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mcap")
        )
        if not mcap_files:
            raise ValueError(f"No .mcap files found in {directory}")

        all_images: list[NDArray[np.uint8]] = []
        all_positions: list[NDArray[np.float32]] = []
        all_velocities: list[NDArray[np.float32]] = []

        for path in mcap_files:
            data = McapEpisodeLoader(path).load()
            try:
                images, positions = _validate_and_extract(data, joint_dim)
            except ValueError:
                continue  # Skip episodes without images or joints

            velocities = _compute_velocities(positions, control_dt)
            all_images.append(images[:-1])
            all_positions.append(positions[:-1])
            all_velocities.append(velocities)

        if not all_images:
            raise ValueError(f"No valid episodes with images found in {directory}")

        return cls(
            np.concatenate(all_images, axis=0),
            np.concatenate(all_positions, axis=0),
            np.concatenate(all_velocities, axis=0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_and_extract(
    data: dict[str, NDArray[Any] | None],
    joint_dim: int | None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """Validate loaded MCAP data and extract images + positions."""
    images = data.get("images")
    positions = data.get("joint_positions")

    if images is None:
        raise ValueError("MCAP episode has no /camera/image channel")
    if positions is None:
        raise ValueError("MCAP episode has no /joint_states channel")

    assert images is not None
    assert positions is not None

    # Align lengths (images and joint states may have different sample counts)
    t = min(len(images), len(positions))
    if t < 2:
        raise ValueError(f"Episode too short for velocity computation: {t} steps")

    images = images[:t]
    positions = positions[:t]

    # Optionally truncate to requested joint count
    if joint_dim is not None and positions.shape[1] > joint_dim:
        positions = positions[:, :joint_dim]

    return images, positions


def _compute_velocities(
    positions: NDArray[np.float32],
    control_dt: float,
) -> NDArray[np.float32]:
    """Finite-difference velocity: (pos_{t+1} - pos_t) / dt."""
    return (positions[1:] - positions[:-1]) / control_dt
