"""Tests for ClankerEnv (unit tests with mock client)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from clanker_gym.env import ClankerEnv


class TestClankerEnv:
    def test_connect_sets_spaces(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = False
        env.observation_space = None
        env.action_space = None

        env._client.connect.return_value = {
            "type": "init_response",
            "env_info": {
                "observation_space": {"low": [-1.0, -1.0], "high": [1.0, 1.0]},
                "action_space": {"n": 4},
            },
        }

        env.connect()
        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.is_connected

    def test_reset_returns_obs_and_info(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = True

        env._client.send.return_value = {
            "type": "reset",
            "observation": {"data": [0.0, 0.0, 0.0]},
            "info": {"seed": 42},
        }

        obs, info = env.reset(seed=42)
        assert obs.shape == (3,)
        assert obs.dtype == np.float32
        assert info["seed"] == 42

    def test_step_continuous(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = True

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [1.0, 2.0]},
            "terminated": False,
            "truncated": True,
            "info": {"episode_length": 10},
        }

        obs, terminated, truncated, info = env.step(np.array([0.5, -0.3]))
        assert obs.shape == (2,)
        assert not terminated
        assert truncated

        # Verify the action was sent as Continuous
        sent = env._client.send.call_args[0][0]
        np.testing.assert_allclose(sent["action"]["Continuous"], [0.5, -0.3], atol=1e-6)

    def test_step_discrete(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = True

        env._client.send.return_value = {
            "type": "step",
            "observation": {"data": [0.0]},
            "terminated": False,
            "truncated": False,
            "info": {},
        }

        env.step(3)
        sent = env._client.send.call_args[0][0]
        assert sent["action"] == {"Discrete": 3}

    def test_close(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = True

        env.close()
        env._client.close.assert_called_once()
        assert not env.is_connected

    def test_context_manager(self):
        env = ClankerEnv.__new__(ClankerEnv)
        env._client = MagicMock()
        env._connected = True

        with env:
            pass

        env._client.close.assert_called_once()
