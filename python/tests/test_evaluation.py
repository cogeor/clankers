"""Tests for the evaluation module."""
import numpy as np
import pytest

import gymnasium
from gymnasium import spaces as gym_spaces

from clanker_gym.evaluation import EvalResult, evaluate_policy
from clanker_gym.rewards import CompositeReward, ConstantReward, ActionPenaltyReward


class _FixedEnv(gymnasium.Env):
    """Deterministic env that always returns fixed reward and terminates after N steps."""

    def __init__(self, reward: float = 1.0, max_steps: int = 5, is_success: bool = True):
        self.observation_space = gym_spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self._reward = reward
        self._max_steps = max_steps
        self._is_success = is_success
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(4, dtype=np.float32)
        terminated = self._step_count >= self._max_steps
        info = {}
        if terminated:
            info["is_success"] = self._is_success
        return obs, self._reward, terminated, False, info


def test_eval_result_defaults():
    r = EvalResult()
    assert r.mean_reward == 0.0
    assert r.episode_rewards == []
    assert np.isnan(r.success_rate)


def test_evaluate_basic():
    env = _FixedEnv(reward=2.0, max_steps=3)
    policy = lambda obs: np.zeros(2, dtype=np.float32)
    result = evaluate_policy(env, policy, n_episodes=5)
    assert result.mean_reward == pytest.approx(6.0)  # 2.0 * 3 steps
    assert result.mean_length == pytest.approx(3.0)
    assert len(result.episode_rewards) == 5
    assert result.success_rate == pytest.approx(1.0)


def test_evaluate_failure():
    env = _FixedEnv(reward=1.0, max_steps=2, is_success=False)
    policy = lambda obs: np.zeros(2, dtype=np.float32)
    result = evaluate_policy(env, policy, n_episodes=3)
    assert result.success_rate == pytest.approx(0.0)


def test_evaluate_with_breakdown():
    env = _FixedEnv(reward=1.0, max_steps=2)
    policy = lambda obs: np.zeros(2, dtype=np.float32)
    reward_fn = CompositeReward([
        (ConstantReward(1.0), 1.0),
        (ActionPenaltyReward(scale=0.0), 1.0),
    ])
    result = evaluate_policy(env, policy, n_episodes=2, reward_fn=reward_fn)
    assert "ConstantReward" in result.reward_breakdown
    # 2 episodes * 2 steps = 4 total steps, constant 1.0 per step
    assert result.reward_breakdown["ConstantReward"] == pytest.approx(1.0)


def test_evaluate_no_success_info():
    """When env doesn't provide is_success, success_rate is NaN."""
    class _NoSuccessEnv(gymnasium.Env):
        def __init__(self):
            self.observation_space = gym_spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.action_space = gym_spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self._step = 0
        def reset(self, **kwargs):
            self._step = 0
            return np.zeros(4, dtype=np.float32), {}
        def step(self, action):
            self._step += 1
            return np.zeros(4, dtype=np.float32), 0.0, self._step >= 2, False, {}

    env = _NoSuccessEnv()
    policy = lambda obs: np.zeros(2, dtype=np.float32)
    result = evaluate_policy(env, policy, n_episodes=2)
    assert np.isnan(result.success_rate)


def test_evaluate_std_reward():
    """Std reward should be 0 for constant reward env."""
    env = _FixedEnv(reward=1.0, max_steps=3)
    policy = lambda obs: np.zeros(2, dtype=np.float32)
    result = evaluate_policy(env, policy, n_episodes=4)
    assert result.std_reward == pytest.approx(0.0)
