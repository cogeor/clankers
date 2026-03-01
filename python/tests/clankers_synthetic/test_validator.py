"""Tests for clankers_synthetic.validator — SimValidator with hard/soft gates."""
from __future__ import annotations

import pytest

from clankers_synthetic.specs import (
    ConstraintSpec,
    ExecutionTrace,
    SuccessCriterion,
    TaskSpec,
    TraceStep,
)
from clankers_synthetic.validator import SimValidator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constraint_spec(**overrides) -> ConstraintSpec:
    defaults = {
        "workspace_bounds_min": [-0.5, -0.5, 0.0],
        "workspace_bounds_max": [0.5, 0.5, 1.2],
        "max_contact_force": 100.0,
        "max_ee_speed": 1.0,
    }
    defaults.update(overrides)
    return ConstraintSpec(**defaults)


def _make_task_spec(**overrides) -> TaskSpec:
    defaults = {
        "task_id": "reach_001",
        "task_text": "Move end-effector to target.",
        "success_criteria": [
            SuccessCriterion(
                type="ee_near_target",
                params={"target": [0.2, 0.0, 0.8], "threshold": 0.05},
            )
        ],
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_step(
    ee_pos=(0.2, 0.0, 0.9),
    contacts=None,
    is_success=False,
    joint_positions=None,
    body_poses_extra=None,
):
    """Build a TraceStep with info containing body_poses and contacts."""
    info = {
        "body_poses": {"end_effector": list(ee_pos) + [0, 0, 0, 1]},
        "contact_events": contacts or [],
        "is_success": is_success,
    }
    if joint_positions is not None:
        info["joint_positions"] = joint_positions
    if body_poses_extra is not None:
        info["body_poses"].update(body_poses_extra)
    return TraceStep(
        obs=[0.0] * 3,
        action=[0.0] * 2,
        next_obs=[0.0] * 3,
        reward=0.0,
        terminated=False,
        truncated=False,
        info=info,
    )


def _make_trace(steps, plan_id="test_plan"):
    """Build an ExecutionTrace from a list of TraceSteps."""
    return ExecutionTrace(
        plan_id=plan_id,
        steps=steps,
        total_reward=sum(s.reward for s in steps),
        terminated=False,
        truncated=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimValidator:
    """Tests for the SimValidator hard and soft gate logic."""

    def test_valid_trace_passes(self):
        """Trace where final step has is_success=True, all within bounds -> passed=True."""
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.1, 0.0, 0.6)),
            _make_step(ee_pos=(0.2, 0.0, 0.8), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is True
        assert report.task_success is True
        assert report.failure_reason is None
        # No hard violations
        hard = [v for v in report.constraint_violations if not v.type.startswith("soft_")]
        assert len(hard) == 0

    def test_task_failure_at_end(self):
        """Final step is_success=False -> passed=False, task_failure violation."""
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.1, 0.0, 0.6)),
            _make_step(ee_pos=(0.2, 0.0, 0.8), is_success=False),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        assert report.task_success is False
        violation_types = [v.type for v in report.constraint_violations]
        assert "task_failure" in violation_types

    def test_workspace_bounds_violation(self):
        """EE pos [0.6, 0.0, 0.9] exceeds max X=0.5."""
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.6, 0.0, 0.9)),  # X=0.6 > max 0.5
            _make_step(ee_pos=(0.2, 0.0, 0.8), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        ws_violations = [v for v in report.constraint_violations if v.type == "workspace_bounds"]
        assert len(ws_violations) >= 1
        assert ws_violations[0].step == 1
        assert "X" in ws_violations[0].details

    def test_workspace_bounds_min_violation(self):
        """EE below min bounds."""
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(-0.6, 0.0, 0.5)),  # X=-0.6 < min -0.5
            _make_step(ee_pos=(0.0, 0.0, 0.5), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        ws_violations = [v for v in report.constraint_violations if v.type == "workspace_bounds"]
        assert len(ws_violations) >= 1
        assert "below" in ws_violations[0].details

    def test_contact_force_violation(self):
        """Contact force 150 > limit 100."""
        contacts = [
            {"body_a": "end_effector", "body_b": "table", "force_magnitude": 150.0}
        ]
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.1, 0.0, 0.6), contacts=contacts),
            _make_step(ee_pos=(0.2, 0.0, 0.8), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec(max_contact_force=100.0)
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        force_violations = [v for v in report.constraint_violations if v.type == "max_force"]
        assert len(force_violations) == 1
        assert force_violations[0].step == 1
        assert "150.0" in force_violations[0].details
        assert report.metrics.max_contact_force == 150.0

    def test_soft_ee_speed_warning(self):
        """EE moves fast but doesn't cause hard failure (soft gate only)."""
        # Two consecutive steps with large EE displacement.
        # dt = 0.02, so displacement of 0.04 => speed = 2.0 m/s > max_ee_speed 1.0
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.04, 0.0, 0.5)),  # speed = 0.04/0.02 = 2.0 m/s
            _make_step(ee_pos=(0.04, 0.0, 0.5), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec(max_ee_speed=1.0)
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        # Should still pass -- soft gate does not reject
        assert report.passed is True
        soft_violations = [v for v in report.constraint_violations if v.type == "soft_ee_speed"]
        assert len(soft_violations) >= 1
        assert report.metrics.max_ee_speed > 1.0

    def test_joint_limit_violation(self):
        """Joint position exceeds URDF limits."""
        joint_limits = {"j1": [-2.89, 2.89], "j2": [-1.57, 1.57]}
        steps = [
            _make_step(
                ee_pos=(0.0, 0.0, 0.5),
                joint_positions={"j1": 0.0, "j2": 0.0},
            ),
            _make_step(
                ee_pos=(0.1, 0.0, 0.6),
                joint_positions={"j1": 3.0, "j2": 0.5},  # j1=3.0 > 2.89
            ),
            _make_step(
                ee_pos=(0.2, 0.0, 0.8),
                joint_positions={"j1": 0.0, "j2": 0.0},
                is_success=True,
            ),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator(joint_limits=joint_limits)

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        jl_violations = [v for v in report.constraint_violations if v.type == "joint_limit"]
        assert len(jl_violations) == 1
        assert "j1" in jl_violations[0].details
        assert jl_violations[0].step == 1

    def test_multiple_violations_collected(self):
        """Trace with force + bounds violations: both should be collected."""
        contacts = [
            {"body_a": "end_effector", "body_b": "table", "force_magnitude": 200.0}
        ]
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(ee_pos=(0.6, 0.0, 0.9), contacts=contacts),  # OOB + force
            _make_step(ee_pos=(0.2, 0.0, 0.8), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec(max_contact_force=100.0)
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        violation_types = set(v.type for v in report.constraint_violations)
        assert "max_force" in violation_types
        assert "workspace_bounds" in violation_types

    def test_empty_trace(self):
        """0 steps -> task_failure (no success at end)."""
        trace = _make_trace([])
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is False
        assert report.task_success is False
        violation_types = [v.type for v in report.constraint_violations]
        assert "task_failure" in violation_types
        assert report.metrics.total_steps == 0

    def test_metrics_computed_correctly(self):
        """Verify max_contact_force, max_ee_speed, total_steps, success_at_step."""
        contacts_step1 = [
            {"body_a": "ee", "body_b": "obj", "force_magnitude": 30.0}
        ]
        contacts_step2 = [
            {"body_a": "ee", "body_b": "obj", "force_magnitude": 50.0}
        ]
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5), contacts=contacts_step1),
            _make_step(ee_pos=(0.01, 0.0, 0.5), contacts=contacts_step2),
            _make_step(ee_pos=(0.02, 0.0, 0.5), is_success=True),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is True
        assert report.metrics.total_steps == 3
        assert report.metrics.max_contact_force == 50.0
        assert report.metrics.success_at_step == 2
        # EE speed: displacement 0.01 m in 0.02 s = 0.5 m/s
        assert report.metrics.max_ee_speed == pytest.approx(0.5, abs=0.01)

    def test_success_at_step_tracked(self):
        """is_success first true at step 80 (out of 100)."""
        steps = []
        for i in range(100):
            steps.append(
                _make_step(
                    ee_pos=(0.0, 0.0, 0.5),
                    is_success=(i >= 80),
                )
            )
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is True
        assert report.metrics.success_at_step == 80

    def test_final_object_poses_recorded(self):
        """body_poses from last step appear in metrics."""
        extra_poses = {
            "cube": [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
            "bin": [-0.2, 0.1, 0.75, 0.0, 0.0, 0.0, 1.0],
        }
        steps = [
            _make_step(ee_pos=(0.0, 0.0, 0.5)),
            _make_step(
                ee_pos=(0.2, 0.0, 0.8),
                is_success=True,
                body_poses_extra=extra_poses,
            ),
        ]
        trace = _make_trace(steps)
        constraints = _make_constraint_spec()
        task = _make_task_spec()
        validator = SimValidator()

        report = validator.validate(trace, task, constraints)

        assert report.passed is True
        poses = report.metrics.final_object_poses
        assert "cube" in poses
        assert "bin" in poses
        assert "end_effector" in poses
        assert poses["cube"] == [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
        assert poses["bin"] == [-0.2, 0.1, 0.75, 0.0, 0.0, 0.0, 1.0]
