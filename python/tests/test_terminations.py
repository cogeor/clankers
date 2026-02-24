"""Comprehensive tests for the Python termination module."""

from __future__ import annotations

import numpy as np
import pytest

from clanker_gym.terminations import (
    BoundsTermination,
    CompositeTermination,
    FailureTermination,
    SuccessTermination,
    TerminationFn,
    TimeoutTermination,
    cartpole_termination,
)

# ---------------------------------------------------------------------------
# SuccessTermination
# ---------------------------------------------------------------------------


class TestSuccessTermination:
    def test_fires_when_close(self):
        obs = np.array([0.0, 0.0, 0.01, 0.0], dtype=np.float32)
        term = SuccessTermination(pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1)
        assert term.is_terminated(obs)

    def test_does_not_fire_when_far(self):
        obs = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)
        term = SuccessTermination(pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1)
        assert not term.is_terminated(obs)

    def test_exact_boundary_does_not_fire(self):
        obs = np.array([0.0, 0.1], dtype=np.float32)
        term = SuccessTermination(pos_a_indices=[0], pos_b_indices=[1], threshold=0.1)
        assert not term.is_terminated(obs)

    def test_name(self):
        term = SuccessTermination(pos_a_indices=[0], pos_b_indices=[1], threshold=0.5)
        assert term.name == "SuccessTermination"

    def test_is_termination_fn(self):
        term = SuccessTermination(pos_a_indices=[0], pos_b_indices=[1], threshold=0.1)
        assert isinstance(term, TerminationFn)

    def test_mismatched_indices_raises(self):
        with pytest.raises(ValueError, match="must match"):
            SuccessTermination(pos_a_indices=[0, 1], pos_b_indices=[2], threshold=0.1)

    def test_3d_positions(self):
        obs = np.array([0.0, 0.0, 0.0, 0.005, 0.005, 0.005], dtype=np.float32)
        term = SuccessTermination(pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5], threshold=0.1)
        # distance = sqrt(0.005^2 * 3) = ~0.0087 < 0.1
        assert term.is_terminated(obs)


# ---------------------------------------------------------------------------
# TimeoutTermination
# ---------------------------------------------------------------------------


class TestTimeoutTermination:
    def test_fires_at_limit(self):
        obs = np.zeros(2, dtype=np.float32)
        term = TimeoutTermination(max_steps=100)
        assert term.is_terminated(obs, step_count=100)

    def test_fires_above_limit(self):
        obs = np.zeros(2, dtype=np.float32)
        term = TimeoutTermination(max_steps=100)
        assert term.is_terminated(obs, step_count=200)

    def test_does_not_fire_before_limit(self):
        obs = np.zeros(2, dtype=np.float32)
        term = TimeoutTermination(max_steps=100)
        assert not term.is_terminated(obs, step_count=50)

    def test_fires_at_zero_steps(self):
        obs = np.zeros(2, dtype=np.float32)
        term = TimeoutTermination(max_steps=0)
        assert term.is_terminated(obs, step_count=0)

    def test_name(self):
        term = TimeoutTermination(max_steps=100)
        assert term.name == "TimeoutTermination"


# ---------------------------------------------------------------------------
# FailureTermination
# ---------------------------------------------------------------------------


class TestFailureTermination:
    def test_fires_below_threshold(self):
        obs = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        term = FailureTermination(obs_index=1, min_value=0.0)
        assert term.is_terminated(obs)

    def test_does_not_fire_above_threshold(self):
        obs = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        term = FailureTermination(obs_index=1, min_value=0.0)
        assert not term.is_terminated(obs)

    def test_exact_boundary_does_not_fire(self):
        obs = np.array([0.0], dtype=np.float32)
        term = FailureTermination(obs_index=0, min_value=0.0)
        assert not term.is_terminated(obs)

    def test_name(self):
        term = FailureTermination(obs_index=0, min_value=0.0)
        assert term.name == "FailureTermination"

    def test_negative_threshold(self):
        obs = np.array([-2.0], dtype=np.float32)
        term = FailureTermination(obs_index=0, min_value=-1.5)
        assert term.is_terminated(obs)


# ---------------------------------------------------------------------------
# CompositeTermination
# ---------------------------------------------------------------------------


class TestCompositeTermination:
    def test_empty_returns_false(self):
        term = CompositeTermination()
        obs = np.zeros(2, dtype=np.float32)
        assert not term.is_terminated(obs)

    def test_single_true(self):
        term = CompositeTermination().add(TimeoutTermination(max_steps=0))
        obs = np.zeros(2, dtype=np.float32)
        assert term.is_terminated(obs, step_count=0)

    def test_single_false(self):
        term = CompositeTermination().add(TimeoutTermination(max_steps=100))
        obs = np.zeros(2, dtype=np.float32)
        assert not term.is_terminated(obs, step_count=0)

    def test_or_logic_any_true(self):
        term = (
            CompositeTermination()
            .add(TimeoutTermination(max_steps=100))  # false at step 0
            .add(TimeoutTermination(max_steps=0))  # true at step 0
        )
        obs = np.zeros(2, dtype=np.float32)
        assert term.is_terminated(obs, step_count=0)

    def test_or_logic_all_false(self):
        term = (
            CompositeTermination()
            .add(TimeoutTermination(max_steps=100))
            .add(TimeoutTermination(max_steps=200))
        )
        obs = np.zeros(2, dtype=np.float32)
        assert not term.is_terminated(obs, step_count=50)

    def test_name(self):
        term = CompositeTermination()
        assert term.name == "CompositeTermination"

    def test_chaining(self):
        term = (
            CompositeTermination()
            .add(TimeoutTermination(max_steps=100))
            .add(FailureTermination(obs_index=0, min_value=0.0))
        )
        assert isinstance(term, CompositeTermination)

    def test_initial_conditions(self):
        conditions = [
            TimeoutTermination(max_steps=0),
        ]
        term = CompositeTermination(conditions=conditions)
        obs = np.zeros(2, dtype=np.float32)
        assert term.is_terminated(obs, step_count=0)

    def test_mixed_conditions(self):
        obs = np.array([0.0, -1.0, 5.0, 0.0], dtype=np.float32)
        term = (
            CompositeTermination()
            .add(
                SuccessTermination(pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1)
            )  # false: distance > 0.1
            .add(FailureTermination(obs_index=1, min_value=0.0))  # true: -1.0 < 0.0
        )
        assert term.is_terminated(obs)


# ---------------------------------------------------------------------------
# BoundsTermination
# ---------------------------------------------------------------------------


class TestBoundsTermination:
    def test_fires_above_threshold(self):
        obs = np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32)
        term = BoundsTermination(obs_index=2, threshold=0.2)
        assert term.is_terminated(obs)

    def test_fires_below_negative_threshold(self):
        obs = np.array([0.0, 0.0, -0.3, 0.0], dtype=np.float32)
        term = BoundsTermination(obs_index=2, threshold=0.2)
        assert term.is_terminated(obs)

    def test_does_not_fire_within_bounds(self):
        obs = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32)
        term = BoundsTermination(obs_index=2, threshold=0.2)
        assert not term.is_terminated(obs)

    def test_exact_boundary_does_not_fire(self):
        # Use 0.25 which is exactly representable in float32
        obs = np.array([0.25], dtype=np.float32)
        term = BoundsTermination(obs_index=0, threshold=0.25)
        assert not term.is_terminated(obs)

    def test_custom_label(self):
        term = BoundsTermination(obs_index=0, threshold=1.0, label="MyBounds")
        assert term.name == "MyBounds"

    def test_default_label(self):
        term = BoundsTermination(obs_index=0, threshold=1.0)
        assert term.name == "BoundsTermination"

    def test_is_termination_fn(self):
        term = BoundsTermination(obs_index=0, threshold=1.0)
        assert isinstance(term, TerminationFn)


# ---------------------------------------------------------------------------
# cartpole_termination factory
# ---------------------------------------------------------------------------


class TestCartpoleTermination:
    def test_returns_composite(self):
        term = cartpole_termination()
        assert isinstance(term, CompositeTermination)

    def test_angle_triggers(self):
        # pole_angle at index 2 exceeds 12 degrees (0.2094 rad)
        obs = np.array([0.0, 0.0, 0.25, 0.0], dtype=np.float32)
        term = cartpole_termination()
        assert term.is_terminated(obs)

    def test_position_triggers(self):
        # cart_pos at index 0 exceeds 2.4m
        obs = np.array([2.5, 0.0, 0.0, 0.0], dtype=np.float32)
        term = cartpole_termination()
        assert term.is_terminated(obs)

    def test_negative_position_triggers(self):
        obs = np.array([-2.5, 0.0, 0.0, 0.0], dtype=np.float32)
        term = cartpole_termination()
        assert term.is_terminated(obs)

    def test_within_bounds_no_trigger(self):
        obs = np.array([1.0, 0.0, 0.1, 0.0], dtype=np.float32)
        term = cartpole_termination()
        assert not term.is_terminated(obs)

    def test_custom_thresholds(self):
        obs = np.array([0.5, 0.0, 0.05, 0.0], dtype=np.float32)
        term = cartpole_termination(angle_threshold=0.04, position_threshold=0.4)
        assert term.is_terminated(obs)  # both exceeded
