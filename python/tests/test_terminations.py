"""Comprehensive tests for the Python termination module."""

from __future__ import annotations

import numpy as np
import pytest

from clanker_gym.terminations import (
    CompositeTermination,
    FailureTermination,
    SuccessTermination,
    TerminationFn,
    TimeoutTermination,
)


# ---------------------------------------------------------------------------
# SuccessTermination
# ---------------------------------------------------------------------------


class TestSuccessTermination:
    def test_fires_when_close(self):
        obs = np.array([0.0, 0.0, 0.01, 0.0], dtype=np.float32)
        term = SuccessTermination(
            pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1
        )
        assert term.is_terminated(obs)

    def test_does_not_fire_when_far(self):
        obs = np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32)
        term = SuccessTermination(
            pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1
        )
        assert not term.is_terminated(obs)

    def test_exact_boundary_does_not_fire(self):
        obs = np.array([0.0, 0.1], dtype=np.float32)
        term = SuccessTermination(
            pos_a_indices=[0], pos_b_indices=[1], threshold=0.1
        )
        assert not term.is_terminated(obs)

    def test_name(self):
        term = SuccessTermination(
            pos_a_indices=[0], pos_b_indices=[1], threshold=0.5
        )
        assert term.name == "SuccessTermination"

    def test_is_termination_fn(self):
        term = SuccessTermination(
            pos_a_indices=[0], pos_b_indices=[1], threshold=0.1
        )
        assert isinstance(term, TerminationFn)

    def test_mismatched_indices_raises(self):
        with pytest.raises(ValueError, match="must match"):
            SuccessTermination(
                pos_a_indices=[0, 1], pos_b_indices=[2], threshold=0.1
            )

    def test_3d_positions(self):
        obs = np.array([0.0, 0.0, 0.0, 0.005, 0.005, 0.005], dtype=np.float32)
        term = SuccessTermination(
            pos_a_indices=[0, 1, 2], pos_b_indices=[3, 4, 5], threshold=0.1
        )
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
                SuccessTermination(
                    pos_a_indices=[0, 1], pos_b_indices=[2, 3], threshold=0.1
                )
            )  # false: distance > 0.1
            .add(FailureTermination(obs_index=1, min_value=0.0))  # true: -1.0 < 0.0
        )
        assert term.is_terminated(obs)
