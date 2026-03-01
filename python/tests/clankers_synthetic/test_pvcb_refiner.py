"""Tests for PVCBRefiner -- deterministic rewrite rules + LLM fallback."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clankers_synthetic.pvcb_refiner import PVCBRefiner
from clankers_synthetic.specs import (
    CanonicalPlan,
    ConstraintSpec,
    ConstraintViolation,
    ObservationSpec,
    PlanRejection,
    ResolvedSkill,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
    TaskSpec,
    ValidationMetrics,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(speed_fraction: float = 0.8) -> CanonicalPlan:
    """Create a minimal CanonicalPlan with one move_to and one set_gripper.

    The params match what PlanParser produces after resolving targets:
    move_to params contain only speed_fraction (target is in target_world_position).
    """
    return CanonicalPlan(
        plan_id="test_plan_001",
        skills=[
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.3, 0.0, 0.2],
                target_orientation=[0.0, 0.0, 0.0, 1.0],
                params={"speed_fraction": speed_fraction},
                guard=None,
            ),
            ResolvedSkill(
                name="set_gripper",
                target_world_position=None,
                target_orientation=None,
                params={"width": 0.04, "wait_settle_steps": 5},
                guard=None,
            ),
        ],
        assumptions=["object reachable"],
        metadata={"plan_type": "skill_plan"},
    )


def _make_report(
    violations: list[ConstraintViolation],
    failure_reason: str = "constraint_violation",
) -> ValidationReport:
    """Create a ValidationReport with given violations, passed=False."""
    return ValidationReport(
        passed=False,
        task_success=False,
        constraint_violations=violations,
        metrics=ValidationMetrics(
            total_steps=100,
            max_contact_force=150.0,
            max_joint_velocity=1.5,
            max_ee_speed=0.8,
        ),
        failure_reason=failure_reason,
    )


def _make_scene() -> SceneSpec:
    """Create a minimal SceneSpec for testing."""
    return SceneSpec(
        scene_id="test_scene",
        simulation=SimulationSpec(),
        robot=RobotSpec(
            name="test_robot",
            urdf_path="test.urdf",
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
        ),
        objects=[],
        constraints=ConstraintSpec(
            workspace_bounds_min=[-1.0, -1.0, 0.0],
            workspace_bounds_max=[1.0, 1.0, 1.0],
        ),
        observations=ObservationSpec(),
    )


def _make_task() -> TaskSpec:
    """Create a minimal TaskSpec for testing."""
    return TaskSpec(
        task_id="test_task",
        task_text="Pick up the cube.",
        success_criteria=[],
    )


def _mock_planner(return_plan: dict | None = None) -> MagicMock:
    """Create a mock LLMPlanner that returns a valid plan dict."""
    planner = MagicMock()
    if return_plan is None:
        return_plan = {
            "plan_id": "llm_refined_001",
            "plan_type": "skill_plan",
            "rationale": "revised approach",
            "assumptions": [],
            "skills": [
                {
                    "name": "move_to",
                    "params": {
                        "target": {
                            "frame": "world",
                            "position": [0.35, 0.0, 0.25],
                        },
                        "speed_fraction": 0.3,
                    },
                },
                {
                    "name": "set_gripper",
                    "params": {"width": 0.04},
                },
            ],
        }
    planner.refine_candidate.return_value = {
        "plan": return_plan,
        "provenance": {"model": "mock"},
    }
    return planner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeterministicFixes:
    """Tests for deterministic rewrite rules."""

    def test_deterministic_speed_reduction_on_force(self) -> None:
        """max_force violation reduces speed_fraction by 30%."""
        plan = _make_plan(speed_fraction=0.8)
        report = _make_report([
            ConstraintViolation(type="max_force", step=50, details="force exceeded"),
        ])
        refiner = PVCBRefiner(max_iterations=1)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is not None
        move_skill = result.skills[0]
        expected_sf = 0.8 * 0.7  # 0.56
        assert abs(move_skill.params["speed_fraction"] - expected_sf) < 1e-9

    def test_deterministic_speed_reduction_on_ee_speed(self) -> None:
        """soft_ee_speed violation reduces speed_fraction by 20%."""
        plan = _make_plan(speed_fraction=0.8)
        report = _make_report([
            ConstraintViolation(
                type="soft_ee_speed", step=30, details="ee too fast"
            ),
        ])
        refiner = PVCBRefiner(max_iterations=1)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is not None
        move_skill = result.skills[0]
        expected_sf = 0.8 * 0.8  # 0.64
        assert abs(move_skill.params["speed_fraction"] - expected_sf) < 1e-9

    def test_deterministic_settle_increase_on_force(self) -> None:
        """max_force violation increases set_gripper wait_settle_steps."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(type="max_force", step=50, details="force exceeded"),
        ])
        refiner = PVCBRefiner(max_iterations=1)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is not None
        gripper_skill = result.skills[1]
        # Original was 5, should increase by 5 to 10
        assert gripper_skill.params["wait_settle_steps"] == 10

    def test_unfixable_workspace_returns_none(self) -> None:
        """workspace_bounds violation is not fixable deterministically."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="workspace_bounds", step=10, details="out of bounds"
            ),
        ])
        refiner = PVCBRefiner(max_iterations=1)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is None

    def test_unfixable_task_failure_returns_none(self) -> None:
        """task_failure violation is not fixable deterministically."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="task_failure", step=100, details="task not completed"
            ),
        ])
        refiner = PVCBRefiner(max_iterations=1)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is None


class TestLLMFallback:
    """Tests for LLM re-proposal fallback."""

    def test_llm_fallback_on_unfixable(self) -> None:
        """When deterministic fails, calls planner.refine_candidate."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="workspace_bounds", step=10, details="out of bounds"
            ),
        ])
        mock_planner = _mock_planner()
        refiner = PVCBRefiner(planner=mock_planner, max_iterations=1)

        refiner.refine(plan, report, _make_scene(), _make_task())

        mock_planner.refine_candidate.assert_called_once()

    def test_llm_fallback_returns_parsed_plan(self) -> None:
        """LLM refinement returns a valid CanonicalPlan after parsing."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="workspace_bounds", step=10, details="out of bounds"
            ),
        ])
        mock_planner = _mock_planner()
        refiner = PVCBRefiner(planner=mock_planner, max_iterations=3)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is not None
        assert isinstance(result, CanonicalPlan)
        assert result.plan_id == "llm_refined_001"

    def test_no_planner_returns_none_on_unfixable(self) -> None:
        """No LLM planner configured returns None for unfixable violations."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="task_failure", step=100, details="task failed"
            ),
        ])
        refiner = PVCBRefiner(planner=None, max_iterations=3)

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is None


class TestLoopDetection:
    """Tests for loop detection and iteration limits."""

    def test_loop_detection_terminates(self) -> None:
        """Same plan hash detected terminates refinement."""
        plan = _make_plan(speed_fraction=0.05)
        # At 0.05, reduction * 0.7 = 0.035, then clamped to 0.05 minimum
        # This should converge quickly and loop detect.
        # Actually, 0.05 * 0.7 = 0.035 which is below 0.05 minimum, so
        # the fixed plan will have speed_fraction=0.05 (same as input).
        # The hash will match on the second attempt.
        report = _make_report([
            ConstraintViolation(type="max_force", step=50, details="force exceeded"),
        ])
        refiner = PVCBRefiner(planner=None, max_iterations=3)

        # The first deterministic fix returns plan with sf=max(0.05*0.7, 0.05)=0.05
        # but the plan_id changes ("_refined" suffix), so it won't hash the same.
        # Actually, _plan_hash ignores plan_id -- it only hashes skills.
        # The skills would have speed_fraction=max(0.05*0.7, 0.05)=0.05 which
        # is the same params but the hash also includes "target" field.
        # Wait, 0.05 * 0.7 = 0.035, max(0.035, 0.05) = 0.05. Same speed_fraction.
        # The gripper gets wait_settle_steps: min(5+5, 50) = 10 on first pass.
        # So first fix: move_to sf=0.05, gripper wait=10 -- different from original.
        # First fix is returned (different hash). Let me adjust the test.

        # Actually the refine() method returns on the first successful deterministic
        # fix. Loop detection only kicks in on subsequent iterations.
        # To test loop detection, we need the first fix to differ from original
        # (which it will, because of gripper settle increase), then call refine
        # again with the refined plan.

        # Use a planner mock that returns a plan whose parsed result has
        # the same skills (and therefore the same hash) as the input plan.
        # The parser resolves move_to target into target_world_position and
        # strips the "target" key from params, so the parsed result will
        # match _make_plan's structure.
        mock_planner = MagicMock()
        mock_planner.refine_candidate.return_value = {
            "plan": {
                "plan_id": "test_plan_001",
                "plan_type": "skill_plan",
                "rationale": "",
                "assumptions": [],
                "skills": [
                    {
                        "name": "move_to",
                        "params": {
                            "target": {
                                "frame": "world",
                                "position": [0.3, 0.0, 0.2],
                            },
                            "speed_fraction": 0.8,
                        },
                    },
                    {
                        "name": "set_gripper",
                        "params": {
                            "width": 0.04,
                            "wait_settle_steps": 5,
                        },
                    },
                ],
            },
            "provenance": {"model": "mock"},
        }

        # Build a plan whose hash matches what the parser will produce
        # from the mock LLM plan above.  _make_plan already matches
        # the parser's output format (speed_fraction only in params,
        # target resolved to target_world_position).
        plan_for_loop = _make_plan(speed_fraction=0.8)
        report_loop = _make_report([
            ConstraintViolation(
                type="workspace_bounds", step=10, details="bounds"
            ),
        ])
        refiner_loop = PVCBRefiner(planner=mock_planner, max_iterations=5)

        # The LLM returns a plan that parses to same skills as the original.
        # However, the parser output has target_world_position=[0.3, 0.0, 0.2]
        # and params={"speed_fraction": 0.8} -- we need to verify the hashes
        # actually match.  Build the expected parsed plan to check.
        from clankers_synthetic.parser import PlanParser
        test_parser = PlanParser()
        raw_plan = mock_planner.refine_candidate.return_value["plan"]
        parsed = test_parser.parse(raw_plan, _make_scene())
        # Confirm the parsed plan has the same hash as plan_for_loop
        assert refiner_loop._plan_hash(parsed) == refiner_loop._plan_hash(plan_for_loop), (
            "Test setup error: mock LLM plan does not hash-match the input plan"
        )

        result = refiner_loop.refine(
            plan_for_loop, report_loop, _make_scene(), _make_task()
        )

        assert result is None
        # Planner should have been called (up to max_iterations times)
        assert mock_planner.refine_candidate.call_count > 0

    def test_max_iterations_respected(self) -> None:
        """Exceeding max_iterations returns None."""
        plan = _make_plan()
        report = _make_report([
            ConstraintViolation(
                type="workspace_bounds", step=10, details="bounds"
            ),
        ])
        # LLM always returns a rejected plan
        mock_planner = MagicMock()
        mock_planner.refine_candidate.side_effect = RuntimeError("API error")

        refiner = PVCBRefiner(
            planner=mock_planner, max_iterations=2
        )

        result = refiner.refine(plan, report, _make_scene(), _make_task())

        assert result is None
        assert mock_planner.refine_candidate.call_count == 2


class TestPlanHash:
    """Tests for plan hash computation."""

    def test_plan_hash_consistency(self) -> None:
        """Same plan produces same hash."""
        plan = _make_plan()
        refiner = PVCBRefiner()

        h1 = refiner._plan_hash(plan)
        h2 = refiner._plan_hash(plan)

        assert h1 == h2
        assert len(h1) == 16  # hex chars from sha256[:16]

    def test_plan_hash_changes_on_modification(self) -> None:
        """Different params produce different hash."""
        plan1 = _make_plan(speed_fraction=0.8)
        plan2 = _make_plan(speed_fraction=0.5)
        refiner = PVCBRefiner()

        h1 = refiner._plan_hash(plan1)
        h2 = refiner._plan_hash(plan2)

        assert h1 != h2
