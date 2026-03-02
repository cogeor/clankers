"""Trajectory dataset for behavioral cloning from JSON trace files.

Loads execution traces (single ``dry_run_trace.json`` or packaged
``episodes/ep_*.json``) and uses :class:`JointEncoder` to produce consistent
joint-position vectors.  Returns ``(pos_t, pos_t+1)`` pairs suitable for
training a next-step predictor, ``(pos_t, action_t)`` when
``include_actions=True``, or ``(pos_t, vel_t)`` when
``include_velocities=True`` where ``vel_t = (pos_{t+1} - pos_t) / control_dt``.

Example::

    ds = TrajectoryDataset.from_trace_file(
        "output/dry_run_trace.json",
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
    )
    pos_t, pos_next = ds[0]

    # Velocity mode
    ds_vel = TrajectoryDataset.from_trace_file(
        "output/dry_run_trace.json",
        include_velocities=True,
        control_dt=0.02,
    )
    pos_t, vel_t = ds_vel[0]  # vel_t in rad/s
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from clankers.joint_encoder import JointEncoder


def _extract_positions(obs: list[float], n_joints: int) -> list[float]:
    """Extract joint positions from interleaved obs [pos0, vel0, pos1, vel1, ...]."""
    return [obs[i * 2] for i in range(n_joints)]


def _load_trace_steps(path: Path) -> list[dict]:
    """Load steps from a trace JSON file (both single-trace and episode formats)."""
    data = json.loads(path.read_text())
    if "steps" in data:
        return data["steps"]
    raise ValueError(f"Unrecognized trace format in {path}")


class TrajectoryDataset(Dataset):
    """PyTorch Dataset of consecutive joint-position pairs from trace data.

    Each sample is ``(pos_t, target_t)`` where:

    - ``pos_t`` is a float32 tensor of shape ``(dof,)`` — joint positions at step *t*
    - ``target_t`` is either ``pos_{t+1}`` (default) or ``action_t`` when
      ``include_actions=True``
    """

    def __init__(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        encoder: JointEncoder,
    ) -> None:
        """Direct constructor (prefer factory class methods).

        Parameters
        ----------
        positions : np.ndarray
            (N, dof) array of joint positions.
        targets : np.ndarray
            (N, target_dim) array of targets (next positions or actions).
        encoder : JointEncoder
            The encoder used to produce *positions*.
        """
        assert len(positions) == len(targets)
        self._positions = torch.tensor(positions, dtype=torch.float32)
        self._targets = torch.tensor(targets, dtype=torch.float32)
        self._encoder = encoder

    @property
    def encoder(self) -> JointEncoder:
        """The JointEncoder used for this dataset."""
        return self._encoder

    @property
    def input_dim(self) -> int:
        """Dimension of input (position) vectors."""
        return self._positions.shape[1]

    @property
    def target_dim(self) -> int:
        """Dimension of target vectors."""
        return self._targets.shape[1]

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._positions[idx], self._targets[idx]

    # -- Factory methods -----------------------------------------------------

    @classmethod
    def from_trace_file(
        cls,
        path: str | Path,
        joint_names: list[str] | None = None,
        scene_spec: str | Path | None = None,
        include_actions: bool = False,
        include_velocities: bool = False,
        control_dt: float = 0.02,
    ) -> TrajectoryDataset:
        """Load a single trace JSON file.

        Parameters
        ----------
        path : str or Path
            Path to the trace JSON.
        joint_names : list[str], optional
            Joint names for the encoder.  If not given, *scene_spec* must be
            provided, or names are auto-generated as ``joint_0``, ``joint_1``, etc.
        scene_spec : str or Path, optional
            Path to a SceneSpec JSON to read joint names from.
        include_actions : bool
            If True, targets are ``action_t`` instead of ``pos_{t+1}``.
        include_velocities : bool
            If True, targets are ``(pos_{t+1} - pos_t) / control_dt`` (rad/s).
            Mutually exclusive with *include_actions*.
        control_dt : float
            Control timestep for velocity computation (default 0.02s).
        """
        if include_actions and include_velocities:
            raise ValueError("include_actions and include_velocities are mutually exclusive")
        path = Path(path)
        steps = _load_trace_steps(path)
        encoder = _resolve_encoder(joint_names, scene_spec, steps)
        return _build_dataset(steps, encoder, include_actions, include_velocities, control_dt)

    @classmethod
    def from_dataset_dir(
        cls,
        directory: str | Path,
        joint_names: list[str] | None = None,
        scene_spec: str | Path | None = None,
        include_actions: bool = False,
        include_velocities: bool = False,
        control_dt: float = 0.02,
    ) -> TrajectoryDataset:
        """Load all episode files from a packaged dataset directory.

        Expects ``episodes/ep_*.json`` inside *directory*.
        """
        if include_actions and include_velocities:
            raise ValueError("include_actions and include_velocities are mutually exclusive")
        directory = Path(directory)
        ep_dir = directory / "episodes"
        if not ep_dir.exists():
            raise FileNotFoundError(f"No episodes/ folder in {directory}")

        ep_files = sorted(ep_dir.glob("ep_*.json"))
        if not ep_files:
            raise FileNotFoundError(f"No ep_*.json files in {ep_dir}")

        all_steps: list[dict] = []
        for f in ep_files:
            all_steps.extend(_load_trace_steps(f))

        encoder = _resolve_encoder(joint_names, scene_spec, all_steps)
        return _build_dataset(all_steps, encoder, include_actions, include_velocities, control_dt)


def _resolve_encoder(
    joint_names: list[str] | None,
    scene_spec: str | Path | None,
    steps: list[dict],
) -> JointEncoder:
    """Resolve a JointEncoder from explicit names, scene spec, or obs dim."""
    if joint_names is not None:
        return JointEncoder(joint_names)
    if scene_spec is not None:
        return JointEncoder.from_scene_spec(scene_spec)
    # Auto-detect from obs dimension
    if steps:
        obs_len = len(steps[0].get("obs", []))
        n_joints = obs_len // 2
        if n_joints > 0:
            names = [f"joint_{i}" for i in range(n_joints)]
            return JointEncoder(names)
    raise ValueError(
        "Cannot determine joint names: provide joint_names, scene_spec, or a trace with obs data"
    )


def _build_dataset(
    steps: list[dict],
    encoder: JointEncoder,
    include_actions: bool,
    include_velocities: bool = False,
    control_dt: float = 0.02,
) -> TrajectoryDataset:
    """Extract position/target arrays from trace steps and build dataset."""
    n_joints = encoder.dof
    positions_list: list[list[float]] = []
    targets_list: list[list[float]] = []

    if include_actions:
        # (pos_t, action_t) pairs
        for step in steps:
            obs = step.get("obs", [])
            action = step.get("action", [])
            if len(obs) >= 2 * n_joints and action:
                pos = _extract_positions(obs, n_joints)
                positions_list.append(pos)
                targets_list.append(action)
    else:
        # Extract all positions first (needed for both position and velocity modes)
        all_pos: list[list[float]] = []
        for step in steps:
            obs = step.get("obs", [])
            if len(obs) >= 2 * n_joints:
                all_pos.append(_extract_positions(obs, n_joints))

        for i in range(len(all_pos) - 1):
            positions_list.append(all_pos[i])
            if include_velocities:
                # vel_t = (pos_{t+1} - pos_t) / dt
                vel = [
                    (all_pos[i + 1][j] - all_pos[i][j]) / control_dt
                    for j in range(n_joints)
                ]
                targets_list.append(vel)
            else:
                targets_list.append(all_pos[i + 1])

    if not positions_list:
        raise ValueError("No valid samples extracted from trace steps")

    positions = np.array(positions_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    return TrajectoryDataset(positions, targets, encoder)
