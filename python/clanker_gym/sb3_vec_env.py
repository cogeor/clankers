"""Stable-Baselines3 VecEnv adapter for batched Clankers training.

``ClankerSB3VecEnv`` wraps :class:`~clanker_gym.vec_env.ClankerVecEnv` as a
``stable_baselines3.common.vec_env.VecEnv`` subclass.  It handles auto-reset
(the SB3 convention), pluggable reward/termination functions, and the
async step interface required by SB3.

Requires the ``sb3`` extra: ``pip install clanker-gym[sb3]``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

try:
    import gymnasium
    from gymnasium import spaces as gym_spaces
except ImportError as exc:
    raise ImportError(
        "gymnasium is required for ClankerSB3VecEnv. "
        "Install with: pip install clanker-gym[sb3]"
    ) from exc

try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
except ImportError as exc:
    raise ImportError(
        "stable_baselines3 is required for ClankerSB3VecEnv. "
        "Install with: pip install clanker-gym[sb3]"
    ) from exc

from clanker_gym.rewards import ConstantReward, RewardFunction
from clanker_gym.spaces import Box, Discrete
from clanker_gym.terminations import TerminationFn, cartpole_termination
from clanker_gym.vec_env import ClankerVecEnv


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


# Sentinel used when no VecEnvIndices are provided.
VecEnvIndices = Union[None, int, "Iterable[int]"]


class ClankerSB3VecEnv(VecEnv):
    """SB3-compatible vectorized environment backed by a Clankers VecGymServer.

    Wraps :class:`ClankerVecEnv` and adds:
    - Pluggable :class:`RewardFunction` (rewards computed Python-side)
    - Pluggable :class:`TerminationFn` (termination override Python-side)
    - Auto-reset: when an env terminates or truncates, it is immediately
      reset. The terminal observation is stored in
      ``info["terminal_observation"]`` and the returned obs is the initial
      observation from the new episode (SB3 convention).

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Server port.
    reward_fn : RewardFunction | None
        Reward function. If ``None``, rewards are 0.0.
    termination_fn : TerminationFn | None
        Optional Python-side termination override.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        reward_fn: RewardFunction | None = None,
        termination_fn: TerminationFn | None = None,
    ) -> None:
        self._vec_env = ClankerVecEnv(host=host, port=port)
        self._vec_env.connect()

        # Convert clanker_gym spaces to gymnasium spaces.
        assert self._vec_env.observation_space is not None
        assert self._vec_env.action_space is not None
        obs_space = _to_gymnasium_space(self._vec_env.observation_space)
        act_space = _to_gymnasium_space(self._vec_env.action_space)

        # Bypass the VecEnv.__init__ render_mode check by providing
        # get_attr before calling super().__init__.
        self._render_mode: str | None = None

        super().__init__(
            num_envs=self._vec_env.num_envs,
            observation_space=obs_space,
            action_space=act_space,
        )

        self._reward_fn = reward_fn
        self._termination_fn = termination_fn
        self._actions: NDArray[np.float32] | None = None
        self._last_obs: NDArray[np.float32] | None = None
        # Per-env step counters for termination_fn.
        self._step_counts = np.zeros(self.num_envs, dtype=np.int64)

    # ------------------------------------------------------------------
    # VecEnv abstract methods
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvObs:
        """Reset all environments and return initial observations."""
        obs, _infos = self._vec_env.reset()
        self._last_obs = obs
        self._step_counts[:] = 0
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions; actual step happens in :meth:`step_wait`."""
        self._actions = np.asarray(actions)

    def step_wait(self) -> VecEnvStepReturn:
        """Execute the stored actions, compute rewards, and handle auto-reset.

        Returns ``(obs, rewards, dones, infos)`` following SB3 convention:
        - ``dones[i]`` is ``terminated[i] or truncated[i]``
        - When ``dones[i]`` is True, ``infos[i]["terminal_observation"]``
          contains the final obs and ``obs[i]`` is the initial obs from
          the auto-reset.
        """
        assert self._actions is not None, "Call step_async() before step_wait()"
        actions = self._actions
        self._actions = None

        obs, terminated, truncated, infos = self._vec_env.step(actions)
        self._step_counts += 1

        # -- Compute rewards -------------------------------------------------
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        if self._reward_fn is not None and self._last_obs is not None:
            for i in range(self.num_envs):
                action_i: Any
                if isinstance(self.action_space, gym_spaces.Discrete):
                    action_i = int(actions[i])
                else:
                    action_i = actions[i]
                rewards[i] = self._reward_fn.compute(
                    obs=self._last_obs[i],
                    action=action_i,
                    next_obs=obs[i],
                    info=infos[i],
                )

        # -- Python-side termination override --------------------------------
        if self._termination_fn is not None:
            for i in range(self.num_envs):
                if not terminated[i]:
                    if self._termination_fn.is_terminated(
                        obs[i], int(self._step_counts[i])
                    ):
                        terminated[i] = True

        # -- SB3 dones = terminated | truncated ------------------------------
        dones = np.logical_or(terminated, truncated)

        # -- Auto-reset for done envs (SB3 convention) ----------------------
        done_indices = [int(i) for i in np.where(dones)[0]]
        if done_indices:
            # Store terminal observations in infos.
            for i in done_indices:
                infos[i]["terminal_observation"] = obs[i].copy()

            # Reset the done environments.
            reset_obs, _reset_infos = self._vec_env.reset(env_ids=done_indices)

            # Replace observations with initial obs from reset.
            for idx, env_id in enumerate(done_indices):
                obs[env_id] = reset_obs[idx]

            # Reset step counters for done envs.
            self._step_counts[done_indices] = 0

        self._last_obs = obs
        return obs, rewards, dones, infos

    def close(self) -> None:
        """Close the underlying vectorized environment."""
        self._vec_env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Return attribute from vectorized environment.

        Only ``render_mode`` is supported; other attributes raise
        ``AttributeError``.
        """
        if attr_name == "render_mode":
            indices_list = self._get_indices(indices)
            return [self._render_mode for _ in indices_list]
        raise AttributeError(
            f"ClankerSB3VecEnv does not expose attribute '{attr_name}'"
        )

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute (not supported, raises ``AttributeError``)."""
        raise AttributeError(
            f"ClankerSB3VecEnv does not support setting attribute '{attr_name}'"
        )

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call environment method (not supported, raises ``AttributeError``)."""
        raise AttributeError(
            f"ClankerSB3VecEnv does not support calling method '{method_name}'"
        )

    def env_is_wrapped(
        self,
        wrapper_class: type[gymnasium.Wrapper],
        indices: VecEnvIndices = None,
    ) -> list[bool]:
        """Check if environments are wrapped (always returns False)."""
        indices_list = self._get_indices(indices)
        return [False for _ in indices_list]

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        """No-op: seeds are handled server-side during reset."""
        return [None for _ in range(self.num_envs)]


def make_cartpole_sb3_vec_env(
    host: str = "127.0.0.1",
    port: int = 9878,
) -> ClankerSB3VecEnv:
    """Create SB3 VecEnv connected to a cartpole_vec_gym server.

    Uses ``ConstantReward(+1)`` and standard CartPole-v1 termination
    conditions (pole angle > 12 deg or cart position > 2.4 m).

    Parameters
    ----------
    host : str
        Server address (default: 127.0.0.1).
    port : int
        Server port (default: 9878 for cartpole_vec_gym).
    """
    return ClankerSB3VecEnv(
        host=host,
        port=port,
        reward_fn=ConstantReward(value=1.0),
        termination_fn=cartpole_termination(),
    )
