"""Vectorized environment client for batched training.

``ClankerVecEnv`` connects to a ``VecGymServer`` and provides batched
``step`` / ``reset`` methods using the ``batch_step`` / ``batch_reset``
protocol messages.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from clanker_gym.client import GymClient
from clanker_gym.spaces import Box, Discrete, space_from_dict


class ClankerVecEnv:
    """Vectorized environment connected to a Clankers VecGymServer.

    Observations are returned as batched numpy arrays of shape
    ``(num_envs, obs_dim)``.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9876) -> None:
        self._client = GymClient(
            host=host, port=port, capabilities={"batch_step": True}
        )
        self.num_envs: int = 0
        self.observation_space: Box | Discrete | None = None
        self.action_space: Box | Discrete | None = None

    def connect(self, seed: int | None = None) -> dict[str, Any]:
        """Connect and perform handshake. Returns InitResponse."""
        resp = self._client.connect(seed=seed)
        env_info = resp.get("env_info", {})
        self.num_envs = env_info.get("n_agents", 0)
        obs_data = env_info.get("observation_space", {})
        act_data = env_info.get("action_space", {})
        if obs_data:
            self.observation_space = space_from_dict(obs_data)
        if act_data:
            self.action_space = space_from_dict(act_data)
        return resp

    def reset(
        self,
        env_ids: list[int] | None = None,
        seeds: list[int | None] | None = None,
    ) -> tuple[NDArray[np.float32], list[dict[str, Any]]]:
        """Reset environments.

        Parameters
        ----------
        env_ids : list[int] | None
            Which environments to reset. If None, resets all.
        seeds : list[int | None] | None
            Per-env seeds. Must match length of env_ids if provided.

        Returns
        -------
        observations : np.ndarray
            Shape ``(len(env_ids), obs_dim)``.
        infos : list[dict]
            Per-env reset info.
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        req: dict[str, Any] = {
            "type": "batch_reset",
            "env_ids": env_ids,
        }
        if seeds is not None:
            req["seeds"] = seeds

        resp = self._client.send(req)
        observations = np.array(
            [obs["data"] for obs in resp["observations"]], dtype=np.float32
        )
        infos = resp.get("infos", [{} for _ in env_ids])
        return observations, infos

    def step(
        self, actions: NDArray[np.float32] | list[Any]
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.bool_],
        NDArray[np.bool_],
        list[dict[str, Any]],
    ]:
        """Step all environments.

        Rewards are not included in the server response — compute them
        Python-side using :mod:`clanker_gym.rewards`.

        Parameters
        ----------
        actions : np.ndarray or list
            Actions for each environment. For continuous spaces,
            shape ``(num_envs, action_dim)``. For discrete, list of ints.

        Returns
        -------
        observations : np.ndarray
            Shape ``(num_envs, obs_dim)``.
        terminated : np.ndarray of bool
            Shape ``(num_envs,)``.
        truncated : np.ndarray of bool
            Shape ``(num_envs,)``.
        infos : list[dict]
            Per-env step info.
        """
        action_list: list[dict[str, Any]] = []
        for a in actions:
            if isinstance(a, (int, np.integer)):
                action_list.append({"Discrete": int(a)})
            else:
                action_list.append({"Continuous": np.asarray(a, dtype=np.float32).tolist()})

        resp = self._client.send({"type": "batch_step", "actions": action_list})
        observations = np.array(
            [obs["data"] for obs in resp["observations"]], dtype=np.float32
        )
        terminated = np.array(resp["terminated"], dtype=np.bool_)
        truncated = np.array(resp["truncated"], dtype=np.bool_)
        infos = resp.get("infos", [{} for _ in range(self.num_envs)])
        return observations, terminated, truncated, infos

    def close(self) -> None:
        """Close the connection."""
        self._client.close()

    def __enter__(self) -> ClankerVecEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
