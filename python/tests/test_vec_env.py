"""Tests for ClankerVecEnv (unit tests with mock client)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from clanker_gym.vec_env import ClankerVecEnv


class TestClankerVecEnv:
    def test_connect_sets_num_envs(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()
        env.num_envs = 0
        env.observation_space = None
        env.action_space = None

        env._client.connect.return_value = {
            "type": "init_response",
            "env_info": {
                "n_agents": 4,
                "observation_space": {"low": [-1.0, -1.0], "high": [1.0, 1.0]},
                "action_space": {"low": [-1.0], "high": [1.0]},
            },
        }

        env.connect()
        assert env.num_envs == 4
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_reset_all(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()
        env.num_envs = 3

        env._client.send.return_value = {
            "type": "batch_reset",
            "observations": [
                {"data": [0.0, 0.0]},
                {"data": [0.0, 0.0]},
                {"data": [0.0, 0.0]},
            ],
            "infos": [{}, {}, {}],
        }

        obs, infos = env.reset()
        assert obs.shape == (3, 2)
        assert obs.dtype == np.float32
        assert len(infos) == 3

        sent = env._client.send.call_args[0][0]
        assert sent["type"] == "batch_reset"
        assert sent["env_ids"] == [0, 1, 2]

    def test_reset_selective(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()
        env.num_envs = 4

        env._client.send.return_value = {
            "type": "batch_reset",
            "observations": [{"data": [0.0, 0.0]}],
            "infos": [{}],
        }

        obs, infos = env.reset(env_ids=[2], seeds=[42])
        assert obs.shape == (1, 2)

        sent = env._client.send.call_args[0][0]
        assert sent["env_ids"] == [2]
        assert sent["seeds"] == [42]

    def test_step_continuous(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()
        env.num_envs = 2

        env._client.send.return_value = {
            "type": "batch_step",
            "observations": [{"data": [1.0, 2.0]}, {"data": [3.0, 4.0]}],
            "rewards": [1.0, 0.5],
            "terminated": [False, True],
            "truncated": [False, False],
            "infos": [{}, {}],
        }

        actions = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        obs, rewards, terminated, truncated, infos = env.step(actions)

        assert obs.shape == (2, 2)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert not terminated[0]
        assert terminated[1]

        sent = env._client.send.call_args[0][0]
        assert sent["type"] == "batch_step"
        assert len(sent["actions"]) == 2
        assert "Continuous" in sent["actions"][0]

    def test_step_discrete(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()
        env.num_envs = 2

        env._client.send.return_value = {
            "type": "batch_step",
            "observations": [{"data": [0.0]}, {"data": [0.0]}],
            "rewards": [0.0, 0.0],
            "terminated": [False, False],
            "truncated": [False, False],
            "infos": [{}, {}],
        }

        obs, rewards, terminated, truncated, infos = env.step([0, 3])
        sent = env._client.send.call_args[0][0]
        assert sent["actions"][0] == {"Discrete": 0}
        assert sent["actions"][1] == {"Discrete": 3}

    def test_close(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()

        env.close()
        env._client.close.assert_called_once()

    def test_context_manager(self):
        env = ClankerVecEnv.__new__(ClankerVecEnv)
        env._client = MagicMock()

        with env:
            pass
        env._client.close.assert_called_once()
