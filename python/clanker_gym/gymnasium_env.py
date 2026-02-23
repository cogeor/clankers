"""Gymnasium-compatible wrapper for Stable-Baselines3 integration.

Provides ``ClankerGymnasiumEnv``, a ``gymnasium.Env`` subclass that
wraps :class:`~clanker_gym.env.ClankerEnv` and computes rewards via a
pluggable :class:`~clanker_gym.rewards.RewardFunction`.

Requires the ``sb3`` extra: ``pip install clanker-gym[sb3]``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for ClankerGymnasiumEnv. Install with: pip install clanker-gym[sb3]"
    ) from exc

from clanker_gym.client import GymClient
from clanker_gym.rewards import RewardFunction
from clanker_gym.spaces import Box, Discrete, space_from_dict


def _to_gymnasium_space(
    space: Box | Discrete,
) -> gym_spaces.Box | gym_spaces.Discrete:
    """Convert a clanker_gym space to a gymnasium space."""
    if isinstance(space, Box):
        return gym_spaces.Box(low=space.low, high=space.high, dtype=np.float32)
    if isinstance(space, Discrete):
        return gym_spaces.Discrete(n=space.n)
    msg = f"Unsupported space type: {type(space)}"
    raise TypeError(msg)


class ClankerGymnasiumEnv(gymnasium.Env):  # type: ignore[misc]
    """Gymnasium-compatible environment for Stable-Baselines3.

    Wraps the Clankers TCP client and computes rewards Python-side
    using a pluggable :class:`RewardFunction`.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    reward_fn : RewardFunction
        Reward function computing scalar reward each step.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        reward_fn: RewardFunction | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._reward_fn = reward_fn
        self._client: GymClient | None = None
        self._step_count = 0
        self._last_obs: NDArray[np.float32] | None = None

        # Spaces are set after connect; gymnasium requires them on the instance.
        self.observation_space: gym_spaces.Box | gym_spaces.Discrete = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space: gym_spaces.Box | gym_spaces.Discrete = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def _ensure_connected(self, seed: int | None = None) -> None:
        """Connect to the server if not already connected."""
        if self._client is not None:
            return
        self._client = GymClient(host=self._host, port=self._port)
        resp = self._client.connect(seed=seed)
        env_info = resp.get("env_info", {})

        obs_data = env_info.get("observation_space", {})
        act_data = env_info.get("action_space", {})
        if obs_data:
            self.observation_space = _to_gymnasium_space(space_from_dict(obs_data))
        if act_data:
            self.action_space = _to_gymnasium_space(space_from_dict(act_data))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        super().reset(seed=seed, options=options)
        self._ensure_connected(seed=seed)
        assert self._client is not None

        req: dict[str, Any] = {"type": "reset"}
        if seed is not None:
            req["seed"] = seed
        resp = self._client.send(req)

        obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        info: dict[str, Any] = resp.get("info", {})
        self._step_count = 0
        self._last_obs = obs
        return obs, info

    def step(
        self,
        action: NDArray[np.float32] | int,
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
        assert self._client is not None, "Call reset() before step()"

        if isinstance(action, (int, np.integer)):
            action_payload: dict[str, Any] = {"Discrete": int(action)}
        else:
            action_payload = {"Continuous": np.asarray(action, dtype=np.float32).tolist()}

        resp = self._client.send({"type": "step", "action": action_payload})
        obs = np.asarray(resp["observation"]["data"], dtype=np.float32)
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"])
        info: dict[str, Any] = resp.get("info", {})

        self._step_count += 1

        # Compute reward via pluggable function.
        if self._reward_fn is not None and self._last_obs is not None:
            reward = self._reward_fn.compute(
                obs=self._last_obs,
                action=action,
                next_obs=obs,
                info=info,
            )
        else:
            reward = 0.0

        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def render(self) -> None:
        """No-op; Clankers rendering is handled server-side."""
