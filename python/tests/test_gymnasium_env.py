"""Tests for ClankerGymnasiumEnv (unit tests with mock client)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip entire module if gymnasium is not installed.
gymnasium = pytest.importorskip("gymnasium")

from clanker_gym.gymnasium_env import (  # noqa: E402
    ClankerGymnasiumEnv,
    _to_gymnasium_space,
    make_cartpole_gymnasium_env,
)
from clanker_gym.rewards import ConstantReward, DistanceReward  # noqa: E402
from clanker_gym.spaces import Box, Discrete  # noqa: E402
from clanker_gym.terminations import BoundsTermination, cartpole_termination  # noqa: E402


class TestToGymnasiumSpace:
    def test_box_conversion(self):
        box = Box(low=[-1.0, -2.0], high=[1.0, 2.0])
        gym_box = _to_gymnasium_space(box)
        assert isinstance(gym_box, gymnasium.spaces.Box)
        np.testing.assert_allclose(gym_box.low, [-1.0, -2.0])
        np.testing.assert_allclose(gym_box.high, [1.0, 2.0])

    def test_discrete_conversion(self):
        discrete = Discrete(n=5)
        gym_discrete = _to_gymnasium_space(discrete)
        assert isinstance(gym_discrete, gymnasium.spaces.Discrete)
        assert gym_discrete.n == 5


class TestClankerGymnasiumEnv:
    def _make_env(self, reward_fn=None, termination_fn=None):
        """Create env with mocked client."""
        env = ClankerGymnasiumEnv.__new__(ClankerGymnasiumEnv)
        env._host = "127.0.0.1"
        env._port = 9876
        env._reward_fn = reward_fn
        env._termination_fn = termination_fn
        env._client = MagicMock()
        env._step_count = 0
        env._last_obs = None
        env.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        env.action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        env.np_random = np.random.default_rng(0)
        return env

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        env._client.send.return_value = {
            "type": "reset",
            "observation": {"data": [1.0, 2.0, 3.0, 4.0]},
            "info": {"seed": 42},
        }

        obs, info = env.reset()
        assert obs.shape == (4,)
        assert obs.dtype == np.float32
        assert info["seed"] == 42
        assert env._step_count == 0

    def test_step_returns_five_tuple(self):
        env = self._make_env()
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [1.0, 2.0, 3.0, 4.0]},
            "terminated": False,
            "truncated": True,
            "info": {},
        }

        obs, reward, terminated, truncated, info = env.step(np.array([0.5, -0.3]))
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert not terminated
        assert truncated
        assert env._step_count == 1

    def test_step_uses_reward_fn(self):
        reward_fn = DistanceReward(pos_a_indices=[0, 1], pos_b_indices=[2, 3])
        env = self._make_env(reward_fn=reward_fn)
        env._last_obs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0, 0.0, 0.5, 0.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        obs, reward, terminated, truncated, info = env.step(np.array([0.1, 0.2]))
        # DistanceReward returns -distance between positions in _last_obs
        # pos_a=[0,0], pos_b=[1,0] -> dist=1.0, reward=-1.0
        assert reward == pytest.approx(-1.0)

    def test_step_no_reward_fn_returns_zero(self):
        env = self._make_env(reward_fn=None)
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [1.0, 2.0, 3.0, 4.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward == 0.0

    def test_close(self):
        env = self._make_env()
        env.close()
        env._client.close.assert_called_once()
        assert env._client is None

    def test_step_discrete_action(self):
        env = self._make_env()
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0, 0.0, 0.0, 0.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        env.step(3)
        sent = env._client.send.call_args[0][0]
        assert sent["action"] == {"Discrete": 3}

    def test_termination_fn_fires(self):
        term_fn = BoundsTermination(obs_index=2, threshold=0.2, label="AngleLimit")
        env = self._make_env(termination_fn=term_fn)
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Server says not terminated, but angle 0.3 > 0.2 threshold
        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0, 0.0, 0.3, 0.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        assert terminated  # Python-side termination overrides

    def test_termination_fn_does_not_override_server(self):
        term_fn = BoundsTermination(obs_index=2, threshold=0.2, label="AngleLimit")
        env = self._make_env(termination_fn=term_fn)
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Server says terminated, Python should not un-terminate
        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0, 0.0, 0.01, 0.0]},
            "terminated": True,
            "truncated": False,
            "info": {},
        }

        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        assert terminated  # Server says terminated, stays terminated

    def test_no_termination_fn_no_override(self):
        env = self._make_env(termination_fn=None)
        env._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0, 0.0, 0.3, 0.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        assert not terminated  # No termination fn, server says false


class TestMakeCartpoleGymnasiumEnv:
    def test_returns_configured_env(self):
        env = make_cartpole_gymnasium_env(port=9999)
        assert isinstance(env, ClankerGymnasiumEnv)
        assert isinstance(env._reward_fn, ConstantReward)
        assert env._termination_fn is not None
        assert env._port == 9999
