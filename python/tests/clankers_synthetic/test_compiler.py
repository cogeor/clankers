"""Tests for clankers_synthetic.compiler -- SkillCompiler with mocked env."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from clankers_synthetic.compiler import SkillCompiler
from clankers_synthetic.specs import (
    CanonicalPlan,
    GuardCondition,
    ResolvedSkill,
)

# ---------------------------------------------------------------------------
# MockEnv
# ---------------------------------------------------------------------------


class MockEnv:
    """Mock gym env for testing compiler."""

    def __init__(self, n_joints=6, n_steps_before_success=5):
        self.n_joints = n_joints
        self.step_count = 0
        self.n_steps_before_success = n_steps_before_success
        self.last_action = None
        self.actions_history: list[np.ndarray] = []

    def reset(self):
        self.step_count = 0
        self.actions_history = []
        obs = np.zeros(self.n_joints, dtype=np.float32)
        info = {
            "body_poses": {
                "end_effector": [0.0, 0.0, 0.9, 0, 0, 0, 1],
            },
            "contact_events": [],
            "is_success": False,
        }
        return obs, info

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.actions_history.append(action.copy())
        obs = np.zeros(self.n_joints, dtype=np.float32)
        is_success = self.step_count >= self.n_steps_before_success
        info = {
            "body_poses": {
                "end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1],
            },
            "contact_events": [],
            "is_success": is_success,
        }
        return obs, -0.1, is_success, False, info


class TerminatingMockEnv(MockEnv):
    """Mock env that terminates after a fixed number of steps."""

    def __init__(self, n_joints=6, terminate_at=3):
        super().__init__(n_joints=n_joints, n_steps_before_success=999)
        self.terminate_at = terminate_at

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.actions_history.append(action.copy())
        obs = np.zeros(self.n_joints, dtype=np.float32)
        terminated = self.step_count >= self.terminate_at
        info = {
            "body_poses": {
                "end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1],
            },
            "contact_events": [],
            "is_success": False,
        }
        return obs, -0.1, terminated, False, info


class ContactMockEnv(MockEnv):
    """Mock env that returns contact events after a fixed number of steps."""

    def __init__(self, n_joints=6, contact_at=3, contact_force=50.0):
        super().__init__(n_joints=n_joints, n_steps_before_success=999)
        self.contact_at = contact_at
        self.contact_force = contact_force

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.actions_history.append(action.copy())
        obs = np.zeros(self.n_joints, dtype=np.float32)
        contacts = []
        if self.step_count >= self.contact_at:
            contacts = [
                {
                    "body_a": "end_effector",
                    "body_b": "table",
                    "force_magnitude": self.contact_force,
                }
            ]
        info = {
            "body_poses": {
                "end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1],
            },
            "contact_events": contacts,
            "is_success": False,
        }
        return obs, -0.1, False, False, info


class DistanceMockEnv(MockEnv):
    """Mock env that returns decreasing distance between EE and target body."""

    def __init__(self, n_joints=6, close_at=3):
        super().__init__(n_joints=n_joints, n_steps_before_success=999)
        self.close_at = close_at

    def step(self, action):
        self.step_count += 1
        self.last_action = action
        self.actions_history.append(action.copy())
        obs = np.zeros(self.n_joints, dtype=np.float32)
        # Before close_at: EE and target are far apart
        # At close_at and after: EE and target are close
        if self.step_count >= self.close_at:
            ee_pos = [0.2, 0.0, 0.9, 0, 0, 0, 1]
            target_pos = [0.2, 0.0, 0.9, 0, 0, 0, 1]  # same position
        else:
            ee_pos = [0.0, 0.0, 0.9, 0, 0, 0, 1]
            target_pos = [0.5, 0.0, 0.9, 0, 0, 0, 1]  # far apart
        info = {
            "body_poses": {
                "end_effector": ee_pos,
                "target_object": target_pos,
            },
            "contact_events": [],
            "is_success": False,
        }
        return obs, -0.1, False, False, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
JOINT_LIMITS = {
    "j1": [-3.14, 3.14],
    "j2": [-3.14, 3.14],
    "j3": [-3.14, 3.14],
    "j4": [-3.14, 3.14],
    "j5": [-3.14, 3.14],
    "j6": [-3.14, 3.14],
}


def _make_compiler(**kwargs):
    defaults = dict(
        joint_names=JOINT_NAMES,
        joint_limits=JOINT_LIMITS,
    )
    defaults.update(kwargs)
    return SkillCompiler(**defaults)


def _make_plan(skills, plan_id="test_plan"):
    return CanonicalPlan(plan_id=plan_id, skills=skills)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkillCompiler:
    """Tests for the SkillCompiler."""

    def test_wait_skill_steps(self):
        """wait(steps=5) calls env.step exactly 5 times."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(name="wait", params={"steps": 5}),
        ])

        compiler.execute(plan, env)

        assert env.step_count == 5

    def test_wait_records_trace(self):
        """Trace has correct number of steps for a wait skill."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(name="wait", params={"steps": 3}),
        ])

        trace = compiler.execute(plan, env)

        assert len(trace.steps) == 3
        for step in trace.steps:
            assert len(step.obs) == 6
            assert len(step.action) == 6
            assert len(step.next_obs) == 6
            assert isinstance(step.reward, float)

    def test_set_gripper_steps(self):
        """set_gripper with wait_settle_steps=3 steps 3 times."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(
                name="set_gripper",
                params={"width": 0.04, "wait_settle_steps": 3},
            ),
        ])

        trace = compiler.execute(plan, env)

        assert env.step_count == 3
        assert len(trace.steps) == 3

    def test_move_to_with_ik(self):
        """move_to calls IK solver, interpolates, steps env."""
        # Create a mock IK solver
        mock_ik_solver = MagicMock()
        mock_result = MagicMock()
        mock_result.joint_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        mock_ik_solver.solve.return_value = mock_result

        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler(ik_solver=mock_ik_solver)
        plan = _make_plan([
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.3, 0.0, 0.8],
                params={"speed_fraction": 0.5},
            ),
        ])

        trace = compiler.execute(plan, env)

        # IK solver should have been called
        mock_ik_solver.solve.assert_called_once()
        # Should have interpolated steps
        assert len(trace.steps) >= 1
        assert env.step_count >= 1

    def test_move_to_without_ik(self):
        """No IK solver -> still steps env (holds position)."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler(ik_solver=None)
        plan = _make_plan([
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.3, 0.0, 0.8],
                params={"speed_fraction": 0.5},
            ),
        ])

        trace = compiler.execute(plan, env)

        # Should still produce at least 1 step even without IK
        assert len(trace.steps) >= 1
        assert env.step_count >= 1

    def test_move_linear_steps(self):
        """move_linear takes multiple steps."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(
                name="move_linear",
                params={
                    "direction": [0.0, 0.0, -1.0],
                    "distance": 0.1,
                    "speed_fraction": 0.5,
                },
            ),
        ])

        trace = compiler.execute(plan, env)

        # Should take multiple steps based on distance and speed
        assert len(trace.steps) >= 1
        assert env.step_count >= 1

    def test_guard_contact_stops_early(self):
        """Contact guard triggers early termination of skill."""
        env = ContactMockEnv(n_joints=6, contact_at=3, contact_force=50.0)
        compiler = _make_compiler()
        # Use a wait skill with many steps but a contact guard
        plan = _make_plan([
            ResolvedSkill(
                name="move_linear",
                params={
                    "direction": [0.0, 0.0, -1.0],
                    "distance": 1.0,
                    "speed_fraction": 0.1,
                },
                guard=GuardCondition(
                    type="contact",
                    body="end_effector",
                    min_force=10.0,
                ),
            ),
        ])

        trace = compiler.execute(plan, env)

        # Should stop at step 3 (when contact occurs), not run all steps
        # The exact step count is 3 because contact happens at step 3
        # and the guard fires after that step.
        assert env.step_count <= 5
        assert env.step_count >= 3

    def test_guard_distance_stops_early(self):
        """Distance guard triggers early stop when EE is near target body."""
        env = DistanceMockEnv(n_joints=6, close_at=3)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(
                name="move_linear",
                params={
                    "direction": [1.0, 0.0, 0.0],
                    "distance": 1.0,
                    "speed_fraction": 0.1,
                },
                guard=GuardCondition(
                    type="distance",
                    from_body="target_object",
                    threshold=0.05,
                ),
            ),
        ])

        trace = compiler.execute(plan, env)

        # Should stop around step 3 when distance guard triggers
        assert env.step_count >= 3
        assert env.step_count <= 5

    def test_action_normalization(self):
        """Verify actions are in [-1, 1] range."""
        # Joint limits: [-3.14, 3.14] for all joints
        # Center = 0, half_range = 3.14
        # A target of [0, 0, 0, 0, 0, 0] should normalize to [0, 0, 0, 0, 0, 0]
        compiler = _make_compiler()
        target = np.zeros(6)
        action = compiler.normalize_action(target)
        np.testing.assert_allclose(action, np.zeros(6), atol=1e-10)

        # Target at upper limit should normalize to +1
        upper = np.array([3.14] * 6)
        action_upper = compiler.normalize_action(upper)
        np.testing.assert_allclose(action_upper, np.ones(6), atol=1e-10)

        # Target at lower limit should normalize to -1
        lower = np.array([-3.14] * 6)
        action_lower = compiler.normalize_action(lower)
        np.testing.assert_allclose(action_lower, -np.ones(6), atol=1e-10)

    def test_full_plan_execution(self):
        """Multi-skill plan executed in sequence."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(name="wait", params={"steps": 2}),
            ResolvedSkill(
                name="set_gripper",
                params={"width": 0.08, "wait_settle_steps": 3},
            ),
            ResolvedSkill(name="wait", params={"steps": 4}),
        ])

        trace = compiler.execute(plan, env)

        # Total steps = 2 + 3 + 4 = 9
        assert env.step_count == 9
        assert len(trace.steps) == 9

    def test_terminated_stops_execution(self):
        """Env returns terminated -> remaining skills skipped."""
        env = TerminatingMockEnv(n_joints=6, terminate_at=3)
        compiler = _make_compiler()
        plan = _make_plan([
            ResolvedSkill(name="wait", params={"steps": 5}),
            ResolvedSkill(name="wait", params={"steps": 5}),
        ])

        trace = compiler.execute(plan, env)

        # Should stop at step 3 (when terminated), not run the full 10 steps
        assert env.step_count == 3
        assert len(trace.steps) == 3
        assert trace.terminated is True

    def test_execution_trace_fields(self):
        """Verify plan_id, total_reward, steps populated."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan(
            [ResolvedSkill(name="wait", params={"steps": 3})],
            plan_id="my_plan_42",
        )

        trace = compiler.execute(plan, env)

        assert trace.plan_id == "my_plan_42"
        assert len(trace.steps) == 3
        # Each step has reward -0.1, so total = -0.3
        assert trace.total_reward == pytest.approx(-0.3, abs=1e-6)
        assert trace.terminated is False
        assert trace.truncated is False
        assert isinstance(trace.final_info, dict)

    def test_extract_joint_positions_from_info(self):
        """Extract joint positions from info dict when available."""
        compiler = _make_compiler()
        obs = np.zeros(6)
        info = {
            "joint_positions": {
                "j1": 0.1,
                "j2": 0.2,
                "j3": 0.3,
                "j4": 0.4,
                "j5": 0.5,
                "j6": 0.6,
            }
        }
        jp = compiler._extract_joint_positions(obs, info)
        expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(jp, expected)

    def test_extract_joint_positions_from_obs(self):
        """Fallback to obs array when info has no joint_positions."""
        compiler = _make_compiler()
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        info = {}
        jp = compiler._extract_joint_positions(obs, info)
        np.testing.assert_allclose(jp, obs)
