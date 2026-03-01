"""Tests for the goal-conditioned environment wrapper (ClankerGoalEnv).

All tests are self-contained and mock the base ClankerGymnasiumEnv
so no server connection is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip entire module if gymnasium is not installed.
gymnasium = pytest.importorskip("gymnasium")

from clankers.goal_env import (  # noqa: E402
    ClankerGoalEnv,
    DenseGoalReward,
    GoalRewardFn,
    SparseGoalReward,
)
from clankers.gymnasium_env import ClankerGymnasiumEnv  # noqa: E402


# ---------------------------------------------------------------------------
# GoalRewardFn tests
# ---------------------------------------------------------------------------


class TestSparseGoalReward:
    def test_within_threshold_returns_zero(self):
        reward_fn = SparseGoalReward(threshold=0.1)
        achieved = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        desired = np.array([0.01, 0.01, 0.0], dtype=np.float32)
        assert reward_fn.compute_reward(achieved, desired, {}) == 0.0

    def test_outside_threshold_returns_negative_one(self):
        reward_fn = SparseGoalReward(threshold=0.05)
        achieved = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        desired = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert reward_fn.compute_reward(achieved, desired, {}) == -1.0

    def test_exact_threshold_returns_negative_one(self):
        """At exactly the threshold distance, reward should be -1.0 (strict <)."""
        reward_fn = SparseGoalReward(threshold=1.0)
        achieved = np.array([0.0], dtype=np.float32)
        desired = np.array([1.0], dtype=np.float32)
        assert reward_fn.compute_reward(achieved, desired, {}) == -1.0

    def test_identical_goals_returns_zero(self):
        reward_fn = SparseGoalReward(threshold=0.01)
        goal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert reward_fn.compute_reward(goal, goal.copy(), {}) == 0.0

    def test_default_threshold(self):
        reward_fn = SparseGoalReward()
        achieved = np.array([0.0], dtype=np.float32)
        desired = np.array([0.04], dtype=np.float32)
        assert reward_fn.compute_reward(achieved, desired, {}) == 0.0

    def test_name(self):
        assert SparseGoalReward().name == "SparseGoalReward"

    def test_is_goal_reward_fn(self):
        assert isinstance(SparseGoalReward(), GoalRewardFn)


class TestDenseGoalReward:
    def test_zero_distance(self):
        reward_fn = DenseGoalReward()
        goal = np.array([1.0, 2.0], dtype=np.float32)
        assert reward_fn.compute_reward(goal, goal.copy(), {}) == pytest.approx(0.0)

    def test_negative_distance(self):
        reward_fn = DenseGoalReward()
        achieved = np.array([0.0, 0.0], dtype=np.float32)
        desired = np.array([3.0, 4.0], dtype=np.float32)
        # L2 = 5.0, reward = -5.0
        assert reward_fn.compute_reward(achieved, desired, {}) == pytest.approx(-5.0)

    def test_unit_distance(self):
        reward_fn = DenseGoalReward()
        achieved = np.array([0.0], dtype=np.float32)
        desired = np.array([1.0], dtype=np.float32)
        assert reward_fn.compute_reward(achieved, desired, {}) == pytest.approx(-1.0)

    def test_name(self):
        assert DenseGoalReward().name == "DenseGoalReward"

    def test_is_goal_reward_fn(self):
        assert isinstance(DenseGoalReward(), GoalRewardFn)

    def test_symmetry(self):
        """Distance is symmetric: reward(a, b) == reward(b, a)."""
        reward_fn = DenseGoalReward()
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        assert reward_fn.compute_reward(a, b, {}) == pytest.approx(
            reward_fn.compute_reward(b, a, {})
        )


# ---------------------------------------------------------------------------
# ClankerGoalEnv tests
# ---------------------------------------------------------------------------


def _make_mock_base_env(obs_dim: int = 6, action_dim: int = 2) -> ClankerGymnasiumEnv:
    """Create a mock ClankerGymnasiumEnv with proper spaces."""
    env = MagicMock(spec=ClankerGymnasiumEnv)
    env.observation_space = gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    env.action_space = gymnasium.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
    )
    return env


class TestClankerGoalEnvInit:
    def test_observation_space_is_dict(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        assert isinstance(env.observation_space, gymnasium.spaces.Dict)
        assert set(env.observation_space.spaces.keys()) == {
            "observation",
            "achieved_goal",
            "desired_goal",
        }

    def test_observation_sub_space_shapes(self):
        base = _make_mock_base_env(obs_dim=10)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2, 3, 4, 5, 6],
            achieved_goal_indices=[7, 8, 9],
            goal_dim=3,
        )
        assert env.observation_space["observation"].shape == (7,)
        assert env.observation_space["achieved_goal"].shape == (3,)
        assert env.observation_space["desired_goal"].shape == (3,)

    def test_action_space_preserved(self):
        base = _make_mock_base_env(obs_dim=6, action_dim=4)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        assert env.action_space == base.action_space

    def test_default_reward_fn_is_sparse(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        assert isinstance(env._reward_fn, SparseGoalReward)

    def test_custom_reward_fn(self):
        base = _make_mock_base_env(obs_dim=6)
        reward_fn = DenseGoalReward()
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
            goal_reward_fn=reward_fn,
        )
        assert env._reward_fn is reward_fn

    def test_slice_indices(self):
        """Supports slice objects for indices."""
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=slice(0, 3),
            achieved_goal_indices=slice(3, 6),
            goal_dim=3,
        )
        assert env.observation_space["observation"].shape == (3,)
        assert env.observation_space["achieved_goal"].shape == (3,)


class TestClankerGoalEnvReset:
    def test_reset_returns_dict_and_info(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        base.reset.return_value = (flat_obs, {"seed": 42})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )

        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
        np.testing.assert_array_equal(obs["observation"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(obs["achieved_goal"], [4.0, 5.0, 6.0])
        assert info["seed"] == 42

    def test_reset_desired_goal_is_current(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs = np.zeros(6, dtype=np.float32)
        base.reset.return_value = (flat_obs, {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        goal = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        env.set_goal(goal)

        obs, _ = env.reset()
        np.testing.assert_array_equal(obs["desired_goal"], goal)

    def test_reset_passes_seed(self):
        base = _make_mock_base_env(obs_dim=6)
        base.reset.return_value = (np.zeros(6, dtype=np.float32), {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        env.reset(seed=123)
        base.reset.assert_called_once_with(seed=123, options=None)


class TestClankerGoalEnvStep:
    def _make_goal_env(
        self, obs_dim=6, goal_reward_fn=None, success_threshold=0.05
    ) -> tuple[ClankerGoalEnv, MagicMock]:
        base = _make_mock_base_env(obs_dim=obs_dim)
        base.reset.return_value = (np.zeros(obs_dim, dtype=np.float32), {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=list(range(obs_dim - 3)),
            achieved_goal_indices=list(range(obs_dim - 3, obs_dim)),
            goal_dim=3,
            goal_reward_fn=goal_reward_fn,
            success_threshold=success_threshold,
        )
        return env, base

    def test_step_returns_five_tuple_with_dict_obs(self):
        env, base = self._make_goal_env()
        flat_obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"observation", "achieved_goal", "desired_goal"}
        np.testing.assert_array_almost_equal(obs["observation"], [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(obs["achieved_goal"], [0.4, 0.5, 0.6])
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_is_success_in_info_when_close(self):
        env, base = self._make_goal_env(success_threshold=1.0)
        # Achieved goal = [0.1, 0.1, 0.1], desired goal = [0, 0, 0] (default)
        # dist = sqrt(0.03) ~ 0.173, which is < 1.0
        flat_obs = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.1], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        assert info["is_success"] is True

    def test_step_is_success_in_info_when_far(self):
        env, base = self._make_goal_env(success_threshold=0.01)
        flat_obs = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        assert info["is_success"] is False

    def test_step_sparse_reward_at_goal(self):
        env, base = self._make_goal_env(success_threshold=0.1)
        # Achieved = desired = [0, 0, 0] -> dist = 0 -> reward = 0.0
        flat_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward == 0.0

    def test_step_sparse_reward_far_from_goal(self):
        env, base = self._make_goal_env(success_threshold=0.05)
        flat_obs = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        assert reward == -1.0

    def test_step_dense_reward(self):
        dense_fn = DenseGoalReward()
        env, base = self._make_goal_env(goal_reward_fn=dense_fn)
        flat_obs = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0], dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, False, False, {})

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        # dist = sqrt(9+16) = 5.0, reward = -5.0
        assert reward == pytest.approx(-5.0)

    def test_step_preserves_terminated_truncated(self):
        env, base = self._make_goal_env()
        flat_obs = np.zeros(6, dtype=np.float32)
        base.step.return_value = (flat_obs, 0.0, True, True, {})

        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        assert terminated is True
        assert truncated is True

    def test_step_ignores_base_reward(self):
        """Goal env computes its own reward; base reward is discarded."""
        env, base = self._make_goal_env()
        flat_obs = np.zeros(6, dtype=np.float32)
        base.step.return_value = (flat_obs, 999.0, False, False, {})

        _, reward, _, _, _ = env.step(np.array([0.0, 0.0]))
        # At goal (all zeros), sparse reward = 0.0, not 999.0
        assert reward == 0.0


class TestClankerGoalEnvSetGoal:
    def test_set_goal_changes_desired(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs = np.zeros(6, dtype=np.float32)
        base.reset.return_value = (flat_obs, {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )

        new_goal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        env.set_goal(new_goal)

        obs, _ = env.reset()
        np.testing.assert_array_equal(obs["desired_goal"], new_goal)

    def test_set_goal_affects_reward(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs_step = np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        base.step.return_value = (flat_obs_step, 0.0, False, False, {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
            success_threshold=0.01,
        )

        # Set goal to exactly match achieved goal
        env.set_goal(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        _, reward, _, _, info = env.step(np.array([0.0, 0.0]))
        assert reward == 0.0
        assert info["is_success"] is True


class TestComputeRewardExternal:
    """Test that compute_reward can be called externally for HER relabeling."""

    def test_external_compute_reward_sparse(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
            success_threshold=0.1,
        )

        # Close to goal
        achieved = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        desired = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        assert env.compute_reward(achieved, desired, {}) == 0.0

        # Far from goal
        desired_far = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        assert env.compute_reward(achieved, desired_far, {}) == -1.0

    def test_external_compute_reward_dense(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
            goal_reward_fn=DenseGoalReward(),
        )

        achieved = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        desired = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        assert env.compute_reward(achieved, desired, {}) == pytest.approx(-5.0)

    def test_external_compute_reward_with_arbitrary_goals(self):
        """HER pattern: compute reward with relabeled goals."""
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
            goal_reward_fn=DenseGoalReward(),
        )

        # Simulate HER: use a past achieved_goal as the new desired_goal
        past_achieved = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        # Relabel desired to be the same as achieved -> reward = 0
        reward = env.compute_reward(past_achieved, past_achieved.copy(), {})
        assert reward == pytest.approx(0.0)


class TestClankerGoalEnvClose:
    def test_close_delegates(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        env.close()
        base.close.assert_called_once()


class TestClankerGoalEnvRender:
    def test_render_delegates(self):
        base = _make_mock_base_env(obs_dim=6)
        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        env.render()
        base.render.assert_called_once()


class TestObsDictCopies:
    """Verify that obs dict arrays are copies, not views into the flat obs."""

    def test_obs_dict_values_are_copies(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        base.reset.return_value = (flat_obs, {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )

        obs, _ = env.reset()
        # Mutate the returned arrays; original flat_obs should not change
        obs["observation"][0] = 999.0
        obs["achieved_goal"][0] = 888.0
        assert flat_obs[0] == 1.0
        assert flat_obs[3] == 4.0

    def test_desired_goal_is_copy(self):
        base = _make_mock_base_env(obs_dim=6)
        flat_obs = np.zeros(6, dtype=np.float32)
        base.reset.return_value = (flat_obs, {})

        env = ClankerGoalEnv(
            env=base,
            obs_indices=[0, 1, 2],
            achieved_goal_indices=[3, 4, 5],
            goal_dim=3,
        )
        goal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        env.set_goal(goal)

        obs, _ = env.reset()
        obs["desired_goal"][0] = 999.0
        # Internal desired goal should not change
        np.testing.assert_array_equal(env._desired_goal, [1.0, 2.0, 3.0])
