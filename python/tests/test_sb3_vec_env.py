"""Tests for ClankerSB3VecEnv (unit tests with mocked ClankerVecEnv).

Stable-Baselines3 requires PyTorch at import time.  When torch is not
available we inject a lightweight shim into ``sys.modules`` so that the
adapter module (and these tests) can be exercised without a full SB3 /
torch installation.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure gymnasium is available (skip entire file otherwise).
# ---------------------------------------------------------------------------
gymnasium = pytest.importorskip("gymnasium")

# ---------------------------------------------------------------------------
# Provide a lightweight SB3 VecEnv shim when torch is unavailable.
# ---------------------------------------------------------------------------
_NEED_SB3_SHIM = False
try:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv as _SB3VecEnv
except (ImportError, ModuleNotFoundError):
    _NEED_SB3_SHIM = True

if _NEED_SB3_SHIM:
    # Build a minimal shim that mirrors the real VecEnv ABC just enough for
    # our adapter to subclass and for tests to exercise.
    from collections.abc import Iterable
    from copy import deepcopy

    from gymnasium import spaces

    VecEnvIndices = Union[None, int, Iterable[int]]
    VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]
    VecEnvStepReturn = tuple[VecEnvObs, np.ndarray, np.ndarray, list[dict]]

    class _ShimVecEnv(ABC):
        """Minimal stand-in for ``stable_baselines3.common.vec_env.VecEnv``."""

        def __init__(
            self,
            num_envs: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
        ) -> None:
            self.num_envs = num_envs
            self.observation_space = observation_space
            self.action_space = action_space
            self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
            self._seeds: list[Optional[int]] = [None for _ in range(num_envs)]
            self._options: list[dict[str, Any]] = [{} for _ in range(num_envs)]

            try:
                render_modes = self.get_attr("render_mode")
            except AttributeError:
                render_modes = [None for _ in range(num_envs)]

            self.render_mode = render_modes[0] if render_modes else None
            self.metadata = {"render_modes": []}

        @abstractmethod
        def reset(self) -> VecEnvObs:
            raise NotImplementedError

        @abstractmethod
        def step_async(self, actions: np.ndarray) -> None:
            raise NotImplementedError

        @abstractmethod
        def step_wait(self) -> VecEnvStepReturn:
            raise NotImplementedError

        @abstractmethod
        def close(self) -> None:
            raise NotImplementedError

        @abstractmethod
        def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
            raise NotImplementedError

        @abstractmethod
        def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
            raise NotImplementedError

        @abstractmethod
        def env_method(self, method_name: str, *method_args: Any, indices: VecEnvIndices = None, **method_kwargs: Any) -> list[Any]:
            raise NotImplementedError

        @abstractmethod
        def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
            raise NotImplementedError

        def step(self, actions: np.ndarray) -> VecEnvStepReturn:
            self.step_async(actions)
            return self.step_wait()

        def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
            if seed is None:
                seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            self._seeds = [seed + idx for idx in range(self.num_envs)]
            return self._seeds

        def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
            if indices is None:
                indices = range(self.num_envs)
            elif isinstance(indices, int):
                indices = [indices]
            return indices

    # Inject the shim into sys.modules so that our adapter module can
    # ``from stable_baselines3.common.vec_env.base_vec_env import VecEnv``.
    _sb3_base = type(sys)("stable_baselines3.common.vec_env.base_vec_env")
    _sb3_base.VecEnv = _ShimVecEnv  # type: ignore[attr-defined]
    _sb3_base.VecEnvObs = VecEnvObs  # type: ignore[attr-defined]
    _sb3_base.VecEnvStepReturn = VecEnvStepReturn  # type: ignore[attr-defined]

    # Ensure all parent packages exist as empty modules.
    for mod_name in [
        "stable_baselines3",
        "stable_baselines3.common",
        "stable_baselines3.common.vec_env",
        "stable_baselines3.common.vec_env.base_vec_env",
    ]:
        if mod_name not in sys.modules:
            if mod_name == "stable_baselines3.common.vec_env.base_vec_env":
                sys.modules[mod_name] = _sb3_base
            else:
                sys.modules[mod_name] = type(sys)(mod_name)

# ---------------------------------------------------------------------------
# Now we can safely import the adapter.
# ---------------------------------------------------------------------------
from clanker_gym.rewards import ConstantReward, DistanceReward  # noqa: E402
from clanker_gym.sb3_vec_env import (  # noqa: E402
    ClankerSB3VecEnv,
    make_cartpole_sb3_vec_env,
)
from clanker_gym.spaces import Box  # noqa: E402
from clanker_gym.terminations import BoundsTermination  # noqa: E402

NUM_ENVS = 3
OBS_DIM = 4


def _make_env(
    reward_fn=None,
    termination_fn=None,
    num_envs: int = NUM_ENVS,
    obs_dim: int = OBS_DIM,
) -> ClankerSB3VecEnv:
    """Create a ClankerSB3VecEnv with a mocked ClankerVecEnv underneath.

    Bypasses __init__ to avoid needing a real server connection.
    """
    mock_vec_env = MagicMock()
    mock_vec_env.num_envs = num_envs
    mock_vec_env.observation_space = Box(
        low=[-np.inf] * obs_dim, high=[np.inf] * obs_dim
    )
    mock_vec_env.action_space = Box(low=[-1.0], high=[1.0])

    obs_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    act_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Build instance without calling __init__ (avoids server connect).
    env = ClankerSB3VecEnv.__new__(ClankerSB3VecEnv)
    env._vec_env = mock_vec_env
    env._reward_fn = reward_fn
    env._termination_fn = termination_fn
    env._actions = None
    env._last_obs = None
    env._step_counts = np.zeros(num_envs, dtype=np.int64)
    env._render_mode = None

    # Fields set by VecEnv.__init__ that we replicate manually.
    env.num_envs = num_envs
    env.observation_space = obs_space
    env.action_space = act_space
    env.reset_infos = [{} for _ in range(num_envs)]
    env._seeds = [None for _ in range(num_envs)]
    env._options = [{} for _ in range(num_envs)]
    env.render_mode = None
    env.metadata = {"render_modes": []}

    return env


class TestReset:
    def test_reset_returns_correct_shape(self):
        env = _make_env()
        reset_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        env._vec_env.reset.return_value = (reset_obs, [{} for _ in range(NUM_ENVS)])

        obs = env.reset()

        assert obs.shape == (NUM_ENVS, OBS_DIM)
        assert obs.dtype == np.float32
        env._vec_env.reset.assert_called_once()

    def test_reset_clears_step_counts(self):
        env = _make_env()
        env._step_counts[:] = 50
        reset_obs = np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32)
        env._vec_env.reset.return_value = (reset_obs, [{} for _ in range(NUM_ENVS)])

        env.reset()

        np.testing.assert_array_equal(env._step_counts, [0, 0, 0])

    def test_reset_stores_last_obs(self):
        env = _make_env()
        reset_obs = np.arange(NUM_ENVS * OBS_DIM, dtype=np.float32).reshape(
            NUM_ENVS, OBS_DIM
        )
        env._vec_env.reset.return_value = (reset_obs, [{} for _ in range(NUM_ENVS)])

        obs = env.reset()

        np.testing.assert_array_equal(env._last_obs, obs)


class TestStep:
    def _setup_step(self, env, obs_data=None, terminated=None, truncated=None):
        """Configure mock for a step call."""
        if obs_data is None:
            obs_data = np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32)
        if terminated is None:
            terminated = np.array([False] * NUM_ENVS, dtype=np.bool_)
        if truncated is None:
            truncated = np.array([False] * NUM_ENVS, dtype=np.bool_)

        env._vec_env.step.return_value = (
            obs_data,
            terminated,
            truncated,
            [{} for _ in range(NUM_ENVS)],
        )
        # Need last_obs set for reward computation.
        if env._last_obs is None:
            env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

    def test_step_returns_correct_shapes(self):
        env = _make_env()
        self._setup_step(env)

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)

        assert obs.shape == (NUM_ENVS, OBS_DIM)
        assert rewards.shape == (NUM_ENVS,)
        assert dones.shape == (NUM_ENVS,)
        assert len(infos) == NUM_ENVS

    def test_step_async_and_wait(self):
        env = _make_env()
        self._setup_step(env)

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()

        assert obs.shape == (NUM_ENVS, OBS_DIM)
        env._vec_env.step.assert_called_once()

    def test_dones_is_terminated_or_truncated(self):
        env = _make_env()
        terminated = np.array([True, False, False], dtype=np.bool_)
        truncated = np.array([False, True, False], dtype=np.bool_)
        self._setup_step(env, terminated=terminated, truncated=truncated)

        # Need auto-reset mock for done envs.
        env._vec_env.reset.return_value = (
            np.zeros((2, OBS_DIM), dtype=np.float32),
            [{}, {}],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        _, _, dones, _ = env.step(actions)

        np.testing.assert_array_equal(dones, [True, True, False])


class TestAutoReset:
    def test_auto_reset_stores_terminal_observation(self):
        env = _make_env()
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        terminal_obs = np.array(
            [
                [9.0, 9.0, 9.0, 9.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        env._vec_env.step.return_value = (
            terminal_obs.copy(),
            np.array([True, False, False], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        # Reset for env 0 returns fresh obs.
        reset_obs = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        env._vec_env.reset.return_value = (reset_obs, [{}])

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)

        # Env 0 was done: obs should be the reset obs, not terminal.
        np.testing.assert_array_equal(obs[0], [0.0, 0.0, 0.0, 0.0])
        # Terminal obs should be stored in info.
        np.testing.assert_array_equal(
            infos[0]["terminal_observation"], [9.0, 9.0, 9.0, 9.0]
        )
        # Env 1 and 2 are unchanged.
        np.testing.assert_array_equal(obs[1], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(obs[2], [2.0, 2.0, 2.0, 2.0])

    def test_auto_reset_on_truncated(self):
        env = _make_env()
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        terminal_obs = np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32)
        env._vec_env.step.return_value = (
            terminal_obs.copy(),
            np.array([False, False, False], dtype=np.bool_),
            np.array([False, True, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        reset_obs = np.array([[5.0, 5.0, 5.0, 5.0]], dtype=np.float32)
        env._vec_env.reset.return_value = (reset_obs, [{}])

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        obs, _, dones, infos = env.step(actions)

        assert dones[1]
        np.testing.assert_array_equal(obs[1], [5.0, 5.0, 5.0, 5.0])
        np.testing.assert_array_equal(
            infos[1]["terminal_observation"], [1.0, 1.0, 1.0, 1.0]
        )

    def test_auto_reset_calls_batch_reset_with_done_ids(self):
        env = _make_env()
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        env._vec_env.step.return_value = (
            np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32),
            np.array([True, False, True], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        reset_obs = np.zeros((2, OBS_DIM), dtype=np.float32)
        env._vec_env.reset.return_value = (reset_obs, [{}, {}])

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        env.step(actions)

        # Should reset env_ids [0, 2].
        env._vec_env.reset.assert_called_once_with(env_ids=[0, 2])

    def test_auto_reset_resets_step_counts(self):
        env = _make_env()
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        env._step_counts[:] = [10, 20, 30]

        env._vec_env.step.return_value = (
            np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32),
            np.array([True, False, False], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )
        env._vec_env.reset.return_value = (
            np.zeros((1, OBS_DIM), dtype=np.float32),
            [{}],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        env.step(actions)

        # Env 0 reset: step_count goes to 0. Others incremented by 1.
        assert env._step_counts[0] == 0
        assert env._step_counts[1] == 21
        assert env._step_counts[2] == 31

    def test_no_auto_reset_when_no_done(self):
        env = _make_env()
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        env._vec_env.step.return_value = (
            np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32),
            np.array([False, False, False], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        env.step(actions)

        # reset should NOT be called.
        env._vec_env.reset.assert_not_called()


class TestRewardFn:
    def test_reward_fn_called_with_correct_args(self):
        mock_reward = MagicMock()
        mock_reward.compute.return_value = 42.0
        env = _make_env(reward_fn=mock_reward)

        prev_obs = np.arange(NUM_ENVS * OBS_DIM, dtype=np.float32).reshape(
            NUM_ENVS, OBS_DIM
        )
        env._last_obs = prev_obs.copy()

        next_obs = np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32)
        env._vec_env.step.return_value = (
            next_obs,
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            [{} for _ in range(NUM_ENVS)],
        )

        actions = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        _, rewards, _, _ = env.step(actions)

        assert mock_reward.compute.call_count == NUM_ENVS
        np.testing.assert_array_equal(rewards, [42.0, 42.0, 42.0])

        # Verify first call args.
        call_kwargs = mock_reward.compute.call_args_list[0][1]
        np.testing.assert_array_equal(call_kwargs["obs"], prev_obs[0])
        np.testing.assert_array_equal(call_kwargs["next_obs"], next_obs[0])

    def test_constant_reward(self):
        env = _make_env(reward_fn=ConstantReward(value=1.0))
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        env._vec_env.step.return_value = (
            np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32),
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            [{} for _ in range(NUM_ENVS)],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        _, rewards, _, _ = env.step(actions)

        np.testing.assert_array_equal(rewards, [1.0, 1.0, 1.0])

    def test_no_reward_fn_returns_zeros(self):
        env = _make_env(reward_fn=None)
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        env._vec_env.step.return_value = (
            np.ones((NUM_ENVS, OBS_DIM), dtype=np.float32),
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            np.array([False] * NUM_ENVS, dtype=np.bool_),
            [{} for _ in range(NUM_ENVS)],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        _, rewards, _, _ = env.step(actions)

        np.testing.assert_array_equal(rewards, [0.0, 0.0, 0.0])


class TestTerminationFn:
    def test_termination_fn_overrides_server(self):
        # Terminate when obs[2] > 0.2 (like cartpole angle).
        term_fn = BoundsTermination(obs_index=2, threshold=0.2, label="AngleLimit")
        env = _make_env(termination_fn=term_fn)
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        # Server says not terminated, but obs[2] = 0.3 > 0.2 for env 0.
        obs_data = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        obs_data[0, 2] = 0.3  # exceeds threshold
        env._vec_env.step.return_value = (
            obs_data.copy(),
            np.array([False, False, False], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        # Auto-reset will be triggered for env 0.
        env._vec_env.reset.return_value = (
            np.zeros((1, OBS_DIM), dtype=np.float32),
            [{}],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        _, _, dones, infos = env.step(actions)

        assert dones[0]  # Python-side termination fired.
        assert not dones[1]
        assert not dones[2]
        # Terminal observation stored.
        assert "terminal_observation" in infos[0]

    def test_termination_fn_does_not_un_terminate(self):
        term_fn = BoundsTermination(obs_index=2, threshold=0.2, label="AngleLimit")
        env = _make_env(termination_fn=term_fn)
        env._last_obs = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)

        # Server says terminated for env 0, obs within threshold.
        obs_data = np.zeros((NUM_ENVS, OBS_DIM), dtype=np.float32)
        obs_data[0, 2] = 0.01  # within threshold
        env._vec_env.step.return_value = (
            obs_data.copy(),
            np.array([True, False, False], dtype=np.bool_),
            np.array([False, False, False], dtype=np.bool_),
            [{}, {}, {}],
        )

        env._vec_env.reset.return_value = (
            np.zeros((1, OBS_DIM), dtype=np.float32),
            [{}],
        )

        actions = np.zeros((NUM_ENVS, 1), dtype=np.float32)
        _, _, dones, _ = env.step(actions)

        # Server terminated remains True.
        assert dones[0]


class TestClose:
    def test_close_delegates(self):
        env = _make_env()
        env.close()
        env._vec_env.close.assert_called_once()


class TestStubs:
    def test_seed_returns_nones(self):
        env = _make_env()
        result = env.seed(42)
        assert result == [None, None, None]

    def test_get_attr_render_mode(self):
        env = _make_env()
        result = env.get_attr("render_mode")
        assert result == [None, None, None]

    def test_get_attr_unknown_raises(self):
        env = _make_env()
        with pytest.raises(AttributeError):
            env.get_attr("nonexistent")

    def test_set_attr_raises(self):
        env = _make_env()
        with pytest.raises(AttributeError):
            env.set_attr("foo", 42)

    def test_env_method_raises(self):
        env = _make_env()
        with pytest.raises(AttributeError):
            env.env_method("some_method")

    def test_env_is_wrapped_returns_false(self):
        env = _make_env()
        result = env.env_is_wrapped(gymnasium.Wrapper)
        assert result == [False, False, False]


class TestMakeCartpoleSB3VecEnv:
    @patch("clanker_gym.sb3_vec_env.ClankerVecEnv")
    def test_factory_creates_configured_env(self, mock_cls):
        """Verify the factory passes correct args; skip actual connect."""
        mock_instance = MagicMock()
        mock_instance.num_envs = 4
        mock_instance.observation_space = Box(
            low=[-4.8, -10, -0.42, -10], high=[4.8, 10, 0.42, 10]
        )
        mock_instance.action_space = Box(low=[-1.0], high=[1.0])
        mock_cls.return_value = mock_instance

        env = make_cartpole_sb3_vec_env(host="localhost", port=1234)

        assert isinstance(env, ClankerSB3VecEnv)
        assert isinstance(env._reward_fn, ConstantReward)
        assert env._termination_fn is not None
        mock_cls.assert_called_once_with(host="localhost", port=1234)
        mock_instance.connect.assert_called_once()
