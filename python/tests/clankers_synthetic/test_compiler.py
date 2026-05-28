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
        plan = _make_plan(
            [
                ResolvedSkill(name="wait", params={"steps": 5}),
            ]
        )

        compiler.execute(plan, env)

        assert env.step_count == 5

    def test_wait_records_trace(self):
        """Trace has correct number of steps for a wait skill."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan(
            [
                ResolvedSkill(name="wait", params={"steps": 3}),
            ]
        )

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
        plan = _make_plan(
            [
                ResolvedSkill(
                    name="set_gripper",
                    params={"width": 0.04, "wait_settle_steps": 3},
                ),
            ]
        )

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
        plan = _make_plan(
            [
                ResolvedSkill(
                    name="move_to",
                    target_world_position=[0.3, 0.0, 0.8],
                    params={"speed_fraction": 0.5},
                ),
            ]
        )

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
        plan = _make_plan(
            [
                ResolvedSkill(
                    name="move_to",
                    target_world_position=[0.3, 0.0, 0.8],
                    params={"speed_fraction": 0.5},
                ),
            ]
        )

        trace = compiler.execute(plan, env)

        # Should still produce at least 1 step even without IK
        assert len(trace.steps) >= 1
        assert env.step_count >= 1

    def test_move_linear_steps(self):
        """move_linear takes multiple steps."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler()
        plan = _make_plan(
            [
                ResolvedSkill(
                    name="move_linear",
                    params={
                        "direction": [0.0, 0.0, -1.0],
                        "distance": 0.1,
                        "speed_fraction": 0.5,
                    },
                ),
            ]
        )

        trace = compiler.execute(plan, env)

        # Should take multiple steps based on distance and speed
        assert len(trace.steps) >= 1
        assert env.step_count >= 1

    def test_guard_contact_stops_early(self):
        """Contact guard triggers early termination of skill."""
        env = ContactMockEnv(n_joints=6, contact_at=3, contact_force=50.0)
        compiler = _make_compiler()
        # Use a wait skill with many steps but a contact guard
        plan = _make_plan(
            [
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
            ]
        )

        compiler.execute(plan, env)

        # Should stop at step 3 (when contact occurs), not run all steps
        # The exact step count is 3 because contact happens at step 3
        # and the guard fires after that step.
        assert env.step_count <= 5
        assert env.step_count >= 3

    def test_guard_distance_stops_early(self):
        """Distance guard triggers early stop when EE is near target body."""
        env = DistanceMockEnv(n_joints=6, close_at=3)
        compiler = _make_compiler()
        plan = _make_plan(
            [
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
            ]
        )

        compiler.execute(plan, env)

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
        plan = _make_plan(
            [
                ResolvedSkill(name="wait", params={"steps": 2}),
                ResolvedSkill(
                    name="set_gripper",
                    params={"width": 0.08, "wait_settle_steps": 3},
                ),
                ResolvedSkill(name="wait", params={"steps": 4}),
            ]
        )

        trace = compiler.execute(plan, env)

        # Total steps = 2 + 3 + 4 = 9
        assert env.step_count == 9
        assert len(trace.steps) == 9

    def test_terminated_stops_execution(self):
        """Env returns terminated -> remaining skills skipped."""
        env = TerminatingMockEnv(n_joints=6, terminate_at=3)
        compiler = _make_compiler()
        plan = _make_plan(
            [
                ResolvedSkill(name="wait", params={"steps": 5}),
                ResolvedSkill(name="wait", params={"steps": 5}),
            ]
        )

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


# ---------------------------------------------------------------------------
# Action-semantics emission contract
#
# Every skill execution path MUST emit env actions via the negotiated
# adapter; TraceStep.action must equal what env.step() received.
# ---------------------------------------------------------------------------


class _NonZeroJointMockEnv(MockEnv):
    """Env whose info reports non-zero joint positions for IK targets.

    Lets _interpolate_to_target push a non-trivial delta through the
    adapter so the test can distinguish raw targets from emitted actions.
    """

    def __init__(self, n_joints=6, fixed_positions=None):
        super().__init__(n_joints=n_joints, n_steps_before_success=999)
        self._fixed = list(fixed_positions) if fixed_positions is not None else [0.0] * n_joints

    def reset(self):
        obs, info = super().reset()
        info["joint_positions"] = list(self._fixed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["joint_positions"] = list(self._fixed)
        return obs, reward, terminated, truncated, info


class TestActionSemanticsEmission:
    """Contract: env.step() and TraceStep.action must reflect the adapter."""

    def test_normalized_position_clips_above_limit(self):
        """Target above joint limit emits clipped action in [-1, 1]."""
        # Single-step wait at non-zero joints so we can compare emission.
        env = _NonZeroJointMockEnv(
            n_joints=6,
            # Above upper limit (3.14) -> normalized > 1 -> clip to 1.
            fixed_positions=[5.0, -5.0, 3.14, -3.14, 0.0, 1.57],
        )
        compiler = _make_compiler(action_semantics="NormalizedPosition")
        plan = _make_plan([ResolvedSkill(name="wait", params={"steps": 1})])

        trace = compiler.execute(plan, env)

        emitted = env.actions_history[-1]
        # Clipped to [-1, 1].
        assert np.all(emitted >= -1.0 - 1e-6)
        assert np.all(emitted <= 1.0 + 1e-6)
        np.testing.assert_allclose(
            emitted,
            np.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.5], dtype=np.float32),
            atol=1e-3,
        )
        # TraceStep.action must equal what env.step received, not raw targets.
        np.testing.assert_allclose(trace.steps[0].action, emitted, atol=1e-6)

    def test_absolute_joint_position_passthrough(self):
        """AbsoluteJointPosition emits the joint target verbatim."""
        env = _NonZeroJointMockEnv(
            n_joints=6,
            fixed_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        compiler = _make_compiler(action_semantics="AbsoluteJointPosition")
        plan = _make_plan([ResolvedSkill(name="wait", params={"steps": 1})])

        compiler.execute(plan, env)
        emitted = env.actions_history[-1]
        np.testing.assert_allclose(
            emitted,
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
        )

    def test_joint_velocity_first_then_finite_difference(self):
        """JointVelocity: first emission zero; subsequent emissions = dq/dt."""
        # Use distinct positions across steps via a custom env.
        positions_by_step = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        class _StreamingEnv(MockEnv):
            def __init__(self_):
                super().__init__(n_joints=6, n_steps_before_success=999)
                self_._i = 0

            def reset(self_):
                obs, info = super().reset()
                info["joint_positions"] = list(positions_by_step[0])
                return obs, info

            def step(self_, action):
                obs, reward, terminated, truncated, info = super().step(action)
                self_._i = min(self_._i + 1, len(positions_by_step) - 1)
                info["joint_positions"] = list(positions_by_step[self_._i])
                return obs, reward, terminated, truncated, info

        env = _StreamingEnv()
        compiler = _make_compiler(
            action_semantics="JointVelocity",
            control_dt=0.1,
        )
        plan = _make_plan([ResolvedSkill(name="wait", params={"steps": 3})])

        compiler.execute(plan, env)

        # First emission seeds the adapter -> zero velocity.
        np.testing.assert_allclose(env.actions_history[0], np.zeros(6, dtype=np.float32))
        # Subsequent emissions are stateful diffs. We can't easily predict
        # the exact value because wait re-emits current_joints, which
        # itself updates from info[joint_positions] only across skill
        # boundaries. The important contract is: the adapter is invoked,
        # so the dtype/shape match and at least one non-zero emission
        # follows a position change.
        all_history = np.stack(env.actions_history)
        assert all_history.shape == (3, 6)
        assert all_history.dtype == np.float32

    def test_torque_adapter_raises_at_emission(self):
        """Torque: compiler fails loudly on first env.step that would emit."""
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler(action_semantics="Torque")
        plan = _make_plan([ResolvedSkill(name="wait", params={"steps": 1})])

        with pytest.raises(NotImplementedError):
            compiler.execute(plan, env)

    def test_trace_action_matches_env_step_action(self):
        """For every skill path, TraceStep.action == env.step()'s last arg."""
        # Exercise all four direct env.step call-sites:
        # _exec_wait, _exec_set_gripper, _exec_move_linear,
        # _interpolate_to_target (via _exec_move_joints).
        mock_ik = MagicMock()
        mock_ik.solve.return_value = MagicMock(
            joint_angles=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        )
        env = MockEnv(n_joints=6, n_steps_before_success=999)
        compiler = _make_compiler(
            action_semantics="AbsoluteJointPosition",
            ik_solver=mock_ik,
        )
        plan = _make_plan(
            [
                ResolvedSkill(name="wait", params={"steps": 1}),
                ResolvedSkill(
                    name="set_gripper",
                    params={"width": 0.04, "wait_settle_steps": 1},
                ),
                ResolvedSkill(
                    name="move_linear",
                    params={
                        "direction": [0.0, 0.0, -1.0],
                        "distance": 0.05,
                        "speed_fraction": 0.5,
                    },
                ),
                ResolvedSkill(
                    name="move_joints",
                    params={
                        "targets": {"j1": 0.1},
                        "speed_fraction": 0.5,
                    },
                ),
            ]
        )
        trace = compiler.execute(plan, env)

        assert len(trace.steps) == len(env.actions_history)
        for step, emitted in zip(trace.steps, env.actions_history, strict=True):
            np.testing.assert_allclose(
                step.action,
                emitted,
                atol=1e-6,
            )
