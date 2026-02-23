"""Comprehensive tests for the Python reward module."""

from __future__ import annotations

import numpy as np
import pytest

from clanker_gym.rewards import (
    ActionPenaltyReward,
    CompositeReward,
    DistanceReward,
    RewardFunction,
    SparseReward,
)


# ---------------------------------------------------------------------------
# DistanceReward
# ---------------------------------------------------------------------------


class TestDistanceReward:
    def test_zero_distance(self):
        obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        reward = DistanceReward(pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5])
        assert abs(reward.compute(obs)) < 1e-6

    def test_negative_when_apart(self):
        # distance = sqrt(3^2 + 4^2) = 5
        obs = np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0], dtype=np.float32)
        reward = DistanceReward(pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5])
        assert abs(reward.compute(obs) - (-5.0)) < 1e-5

    def test_unit_distance(self):
        obs = np.array([0.0, 1.0], dtype=np.float32)
        reward = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        assert abs(reward.compute(obs) - (-1.0)) < 1e-6

    def test_name(self):
        reward = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        assert reward.name == "DistanceReward"

    def test_is_reward_function(self):
        reward = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        assert isinstance(reward, RewardFunction)

    def test_mismatched_indices_raises(self):
        with pytest.raises(ValueError, match="must match"):
            DistanceReward(pos_a_indices=[0, 1], pos_b_indices=[2])

    def test_2d_positions(self):
        obs = np.array([1.0, 0.0, 4.0, 0.0], dtype=np.float32)
        reward = DistanceReward(pos_a_indices=[0, 1], pos_b_indices=[2, 3])
        assert abs(reward.compute(obs) - (-3.0)) < 1e-5

    def test_ignores_action(self):
        obs = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        reward = DistanceReward(pos_a_indices=[0, 1], pos_b_indices=[2, 3])
        r1 = reward.compute(obs, action=None)
        r2 = reward.compute(obs, action=np.array([99.0]))
        assert abs(r1 - r2) < 1e-6


# ---------------------------------------------------------------------------
# SparseReward
# ---------------------------------------------------------------------------


class TestSparseReward:
    def test_success_within_threshold(self):
        obs = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        reward = SparseReward(
            pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5], threshold=0.1
        )
        assert reward.compute(obs) == 1.0

    def test_failure_outside_threshold(self):
        obs = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        reward = SparseReward(
            pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5], threshold=0.1
        )
        assert reward.compute(obs) == 0.0

    def test_exact_boundary(self):
        # At exactly the threshold, it should NOT fire (strict <)
        obs = np.array([0.0, 0.1], dtype=np.float32)
        reward = SparseReward(pos_a_indices=[0], pos_b_indices=[1], threshold=0.1)
        assert reward.compute(obs) == 0.0

    def test_name(self):
        reward = SparseReward(pos_a_indices=[0], pos_b_indices=[1], threshold=0.5)
        assert reward.name == "SparseReward"

    def test_mismatched_indices_raises(self):
        with pytest.raises(ValueError, match="must match"):
            SparseReward(pos_a_indices=[0, 1], pos_b_indices=[2], threshold=0.1)


# ---------------------------------------------------------------------------
# ActionPenaltyReward
# ---------------------------------------------------------------------------


class TestActionPenaltyReward:
    def test_no_action(self):
        obs = np.zeros(2, dtype=np.float32)
        reward = ActionPenaltyReward(scale=0.01)
        assert abs(reward.compute(obs, action=None)) < 1e-6

    def test_discrete_action_returns_zero(self):
        obs = np.zeros(2, dtype=np.float32)
        reward = ActionPenaltyReward(scale=0.01)
        assert abs(reward.compute(obs, action=3)) < 1e-6

    def test_computes_negative_norm_sq(self):
        obs = np.zeros(2, dtype=np.float32)
        # ||[3, 4]||^2 = 9 + 16 = 25
        action = np.array([3.0, 4.0], dtype=np.float32)
        reward = ActionPenaltyReward(scale=0.01)
        assert abs(reward.compute(obs, action=action) - (-0.25)) < 1e-6

    def test_zero_action(self):
        obs = np.zeros(2, dtype=np.float32)
        action = np.array([0.0, 0.0], dtype=np.float32)
        reward = ActionPenaltyReward(scale=1.0)
        assert abs(reward.compute(obs, action=action)) < 1e-6

    def test_scale_factor(self):
        obs = np.zeros(2, dtype=np.float32)
        action = np.array([1.0], dtype=np.float32)
        # scale=0.5, norm_sq=1.0 -> -0.5
        reward = ActionPenaltyReward(scale=0.5)
        assert abs(reward.compute(obs, action=action) - (-0.5)) < 1e-6

    def test_name(self):
        reward = ActionPenaltyReward(scale=0.01)
        assert reward.name == "ActionPenaltyReward"

    def test_default_scale(self):
        reward = ActionPenaltyReward()
        obs = np.zeros(1, dtype=np.float32)
        action = np.array([10.0], dtype=np.float32)
        # scale=0.01, norm_sq=100 -> -1.0
        assert abs(reward.compute(obs, action=action) - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# CompositeReward
# ---------------------------------------------------------------------------


class TestCompositeReward:
    def test_empty_returns_zero(self):
        reward = CompositeReward()
        obs = np.zeros(2, dtype=np.float32)
        assert abs(reward.compute(obs)) < 1e-6

    def test_single_reward(self):
        distance = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        reward = CompositeReward().add(distance, weight=1.0)
        obs = np.array([0.0, 5.0], dtype=np.float32)
        assert abs(reward.compute(obs) - (-5.0)) < 1e-5

    def test_weighted_sum(self):
        r1 = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        r2 = ActionPenaltyReward(scale=1.0)
        reward = CompositeReward().add(r1, weight=0.5).add(r2, weight=1.0)
        obs = np.array([0.0, 2.0], dtype=np.float32)
        action = np.array([1.0], dtype=np.float32)
        # distance: -2.0 * 0.5 = -1.0; penalty: -1.0 * 1.0 = -1.0; total = -2.0
        assert abs(reward.compute(obs, action=action) - (-2.0)) < 1e-5

    def test_breakdown(self):
        r1 = DistanceReward(pos_a_indices=[0], pos_b_indices=[1])
        r2 = ActionPenaltyReward(scale=1.0)
        reward = CompositeReward().add(r1, weight=0.5).add(r2, weight=-1.0)
        obs = np.array([0.0, 10.0], dtype=np.float32)
        action = np.array([2.0], dtype=np.float32)
        bd = reward.breakdown(obs, action=action)
        assert len(bd) == 2
        assert bd[0][0] == "DistanceReward"
        assert abs(bd[0][1] - (-5.0)) < 1e-5  # -10 * 0.5
        assert bd[1][0] == "ActionPenaltyReward"
        assert abs(bd[1][1] - 4.0) < 1e-5  # -4 * -1.0

    def test_name(self):
        reward = CompositeReward()
        assert reward.name == "CompositeReward"

    def test_chaining(self):
        reward = (
            CompositeReward()
            .add(ActionPenaltyReward(scale=0.01), weight=1.0)
            .add(ActionPenaltyReward(scale=0.02), weight=2.0)
        )
        assert isinstance(reward, CompositeReward)

    def test_initial_rewards(self):
        r1 = ActionPenaltyReward(scale=1.0)
        reward = CompositeReward(rewards=[(r1, 1.0)])
        obs = np.zeros(1, dtype=np.float32)
        action = np.array([3.0], dtype=np.float32)
        assert abs(reward.compute(obs, action=action) - (-9.0)) < 1e-6
