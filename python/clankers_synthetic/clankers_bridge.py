"""Bridge adapter: wraps ClankerEnv to match the StepEnv protocol.

``ClankerEnv.step()`` returns a 4-tuple ``(obs, terminated, truncated, info)``
while ``StepEnv`` expects a 5-tuple ``(obs, reward, terminated, truncated, info)``.
This adapter adds ``reward=0.0`` and handles gripper action mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from clankers.env import ClankerEnv
from numpy.typing import NDArray


class ClankersBridgeEnv:
    """Adapter that wraps :class:`ClankerEnv` to satisfy the ``StepEnv`` protocol.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server TCP port.
    n_arm_joints : int
        Number of arm joints (default 6). The compiler sends actions for these.
    n_gripper_joints : int
        Number of gripper joints (default 2). These are appended by the bridge.
    gripper_open_width : float
        Gripper finger position when open (default 0.03).
    gripper_close_width : float
        Gripper finger position when closed (default 0.0).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9880,
        n_arm_joints: int = 6,
        n_gripper_joints: int = 2,
        gripper_open_width: float = 0.03,
        gripper_close_width: float = 0.0,
    ) -> None:
        self._env = ClankerEnv(host=host, port=port)
        self._n_arm = n_arm_joints
        self._n_gripper = n_gripper_joints
        self._gripper_open = gripper_open_width
        self._gripper_close = gripper_close_width
        self._gripper_width = gripper_open_width  # start open

    @property
    def gripper_width(self) -> float:
        """Current gripper target width."""
        return self._gripper_width

    @gripper_width.setter
    def gripper_width(self, width: float) -> None:
        """Set gripper target width (clamped to valid range)."""
        self._gripper_width = float(
            np.clip(width, self._gripper_close, self._gripper_open)
        )

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        self._env.connect(seed=seed)
        obs, info = self._env.reset(seed=seed)
        self._gripper_width = self._gripper_open  # reset gripper to open
        return obs, info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take one step.

        The compiler sends ``n_arm_joints``-dim actions. This adapter appends
        the current gripper width to form the full ``n_arm + n_gripper``-dim
        action expected by the sim.

        Returns
        -------
        observation : np.ndarray
        reward : float
            Always 0.0 (synthetic pipeline does not use reward).
        terminated : bool
        truncated : bool
        info : dict
        """
        action = np.asarray(action, dtype=np.float32)

        # If action is arm-only, append gripper
        if len(action) == self._n_arm:
            gripper_action = np.full(self._n_gripper, self._gripper_width, dtype=np.float32)
            full_action = np.concatenate([action, gripper_action])
        else:
            full_action = action

        obs, terminated, truncated, info = self._env.step(full_action)
        reward = 0.0
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the connection."""
        self._env.close()

    def __enter__(self) -> ClankersBridgeEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
