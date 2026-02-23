"""Single-environment Gymnasium-compatible wrapper.

``ClankerEnv`` connects to a Clankers ``GymServer`` and provides the
standard ``reset`` / ``step`` / ``close`` interface expected by training
frameworks like Stable-Baselines3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from clanker_gym.client import GymClient
from clanker_gym.spaces import Box, Discrete, space_from_dict


class ClankerEnv:
    """Gymnasium-compatible environment connected to a Clankers server.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self._client = GymClient(host=host, port=port)
        self.observation_space: Box | Discrete | None = None
        self.action_space: Box | Discrete | None = None
        self._connected = False

    def connect(self, seed: int | None = None) -> dict[str, Any]:
        """Connect and perform handshake. Returns InitResponse."""
        resp = self._client.connect(seed=seed)
        env_info = resp.get("env_info", {})
        obs_data = env_info.get("observation_space", {})
        act_data = env_info.get("action_space", {})
        if obs_data:
            self.observation_space = space_from_dict(obs_data)
        if act_data:
            self.action_space = space_from_dict(act_data)
        self._connected = True
        return resp

    def reset(
        self, seed: int | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray
            Initial observation.
        info : dict
            Reset metadata.
        """
        req: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            req["seed"] = seed
        resp = self._client.send(req)
        obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        info = resp.get("info", {})
        return obs, info

    def step(
        self, action: NDArray[np.float32] | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take one step.

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        if isinstance(action, (int, np.integer)):
            action_payload: dict[str, Any] = {"Discrete": int(action)}
        else:
            action_payload = {"Continuous": np.asarray(action, dtype=np.float32).tolist()}

        resp = self._client.send({"type": "step", "action": action_payload})
        obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        reward = float(resp["reward"])
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"])
        info = resp.get("info", {})
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the connection."""
        self._client.close()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __enter__(self) -> ClankerEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
