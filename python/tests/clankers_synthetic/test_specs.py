"""Tests for clankers_synthetic.specs Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clankers_synthetic.specs import (
    CanonicalPlan,
    ConstraintSpec,
    ConstraintViolation,
    DatasetManifest,
    ExecutionTrace,
    GuardCondition,
    LLMProposedPlan,
    LLMRequest,
    ObjectSpec,
    ObservationSpec,
    PlanRejection,
    ProposedSkill,
    ResolvedSkill,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
    SkillParams,
    SuccessCriterion,
    TaskSpec,
    TraceStep,
    ValidationMetrics,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable model instances
# ---------------------------------------------------------------------------


def _make_simulation_spec(**overrides) -> SimulationSpec:
    return SimulationSpec(**overrides)


def _make_robot_spec(**overrides) -> RobotSpec:
    defaults = {
        "name": "panda",
        "urdf_path": "/robots/panda.urdf",
        "base_position": [0.0, 0.0, 0.0],
        "base_orientation": [0.0, 0.0, 0.0, 1.0],
    }
    defaults.update(overrides)
    return RobotSpec(**defaults)


def _make_object_spec(**overrides) -> ObjectSpec:
    defaults = {
        "name": "cube",
        "shape": "box",
        "shape_params": {"half_extents": [0.025, 0.025, 0.025]},
        "position": [0.2, 0.0, 0.8],
    }
    defaults.update(overrides)
    return ObjectSpec(**defaults)


def _make_constraint_spec(**overrides) -> ConstraintSpec:
    defaults = {
        "workspace_bounds_min": [-0.5, -0.5, 0.0],
        "workspace_bounds_max": [0.5, 0.5, 1.2],
    }
    defaults.update(overrides)
    return ConstraintSpec(**defaults)


def _make_observation_spec(**overrides) -> ObservationSpec:
    return ObservationSpec(**overrides)


def _make_scene_spec(**overrides) -> SceneSpec:
    defaults = {
        "scene_id": "test_scene_001",
        "simulation": _make_simulation_spec(),
        "robot": _make_robot_spec(),
        "objects": [_make_object_spec()],
        "constraints": _make_constraint_spec(),
        "observations": _make_observation_spec(),
    }
    defaults.update(overrides)
    return SceneSpec(**defaults)


def _make_success_criterion(**overrides) -> SuccessCriterion:
    defaults = {
        "type": "ee_near_target",
        "params": {"target": [0.2, 0.0, 0.8], "threshold": 0.05},
    }
    defaults.update(overrides)
    return SuccessCriterion(**defaults)


def _make_task_spec(**overrides) -> TaskSpec:
    defaults = {
        "task_id": "reach_001",
        "task_text": "Move the end-effector to the cube position.",
        "success_criteria": [_make_success_criterion()],
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_trace_step(**overrides) -> TraceStep:
    defaults = {
        "obs": [0.0, 0.1, 0.2],
        "action": [0.5, -0.3],
        "next_obs": [0.01, 0.11, 0.21],
        "reward": -0.1,
        "terminated": False,
        "truncated": False,
    }
    defaults.update(overrides)
    return TraceStep(**defaults)


def _make_validation_metrics(**overrides) -> ValidationMetrics:
    defaults = {
        "total_steps": 100,
        "max_contact_force": 25.0,
        "max_joint_velocity": 1.5,
        "max_ee_speed": 0.8,
    }
    defaults.update(overrides)
    return ValidationMetrics(**defaults)


# ---------------------------------------------------------------------------
# Scene model tests
# ---------------------------------------------------------------------------


class TestSimulationSpec:
    def test_defaults(self):
        spec = SimulationSpec()
        assert spec.gravity == [0.0, 0.0, -9.81]
        assert spec.physics_dt == 0.001
        assert spec.control_dt == 0.02
        assert spec.max_episode_steps == 500
        assert spec.seed == 42

    def test_custom_values(self):
        spec = SimulationSpec(gravity=[0, 0, -10], physics_dt=0.002, seed=123)
        assert spec.gravity == [0, 0, -10]
        assert spec.physics_dt == 0.002
        assert spec.seed == 123

    def test_roundtrip(self):
        spec = SimulationSpec(seed=99)
        data = spec.dict()
        restored = SimulationSpec(**data)
        assert restored == spec


class TestRobotSpec:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            RobotSpec()  # type: ignore[call-arg]

    def test_construction(self):
        robot = _make_robot_spec()
        assert robot.name == "panda"
        assert robot.fixed_base is True
        assert robot.control_mode == "position_pd"
        assert robot.ee_link_name == "end_effector"

    def test_defaults(self):
        robot = _make_robot_spec()
        assert robot.joint_names == []
        assert robot.joint_limits == {}
        assert robot.joint_types == {}
        assert robot.pd_gains == {}

    def test_roundtrip(self):
        robot = _make_robot_spec(
            joint_names=["j1", "j2"],
            joint_limits={"j1": [-3.14, 3.14], "j2": [-1.57, 1.57]},
        )
        data = robot.dict()
        restored = RobotSpec(**data)
        assert restored == robot


class TestObjectSpec:
    def test_construction(self):
        obj = _make_object_spec()
        assert obj.name == "cube"
        assert obj.shape == "box"
        assert obj.mass == 0.1

    def test_defaults(self):
        obj = _make_object_spec()
        assert obj.orientation == [0, 0, 0, 1]
        assert obj.friction == 0.8
        assert obj.restitution == 0.1
        assert obj.is_static is False
        assert obj.is_graspable is False
        assert obj.is_container is False

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ObjectSpec()  # type: ignore[call-arg]

    def test_roundtrip(self):
        obj = _make_object_spec(is_graspable=True, mass=0.5)
        data = obj.dict()
        restored = ObjectSpec(**data)
        assert restored == obj


class TestConstraintSpec:
    def test_construction(self):
        cs = _make_constraint_spec()
        assert cs.workspace_bounds_min == [-0.5, -0.5, 0.0]
        assert cs.workspace_bounds_max == [0.5, 0.5, 1.2]

    def test_defaults(self):
        cs = _make_constraint_spec()
        assert cs.max_ee_speed == 1.0
        assert cs.max_joint_velocity == 2.0
        assert cs.max_contact_force == 100.0
        assert cs.keep_upright == []
        assert cs.avoid_regions == []
        assert cs.no_self_collision is True

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ConstraintSpec()  # type: ignore[call-arg]


class TestObservationSpec:
    def test_defaults(self):
        obs = ObservationSpec()
        assert obs.modalities == ["proprio"]
        assert obs.joint_state_dim == 0
        assert obs.ee_pose_dim == 7
        assert obs.object_pose_dim == 0
        assert obs.cameras == []


class TestSceneSpec:
    def test_construction(self):
        scene = _make_scene_spec()
        assert scene.scene_id == "test_scene_001"
        assert len(scene.objects) == 1
        assert scene.objects[0].name == "cube"

    def test_default_units(self):
        scene = _make_scene_spec()
        assert scene.units == {"length": "m", "angle": "rad", "time": "s"}

    def test_nested_objects(self):
        objects = [
            _make_object_spec(name="cube", position=[0.2, 0.0, 0.8]),
            _make_object_spec(name="bin", shape="cylinder", position=[-0.2, 0.1, 0.75],
                              is_container=True, is_static=True),
            _make_object_spec(name="sphere", shape="sphere",
                              shape_params={"radius": 0.03},
                              position=[0.1, -0.1, 0.8], is_graspable=True),
        ]
        scene = _make_scene_spec(objects=objects)
        assert len(scene.objects) == 3
        assert scene.objects[1].is_container is True
        assert scene.objects[2].is_graspable is True

    def test_roundtrip(self):
        scene = _make_scene_spec()
        data = scene.dict()
        restored = SceneSpec(**data)
        assert restored == scene

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            SceneSpec()  # type: ignore[call-arg]

    def test_roundtrip_with_nested(self):
        """Ensure deeply nested scene survives dump -> restore."""
        scene = _make_scene_spec(
            objects=[
                _make_object_spec(name="a"),
                _make_object_spec(name="b"),
            ]
        )
        data = scene.dict()
        restored = SceneSpec(**data)
        assert restored.objects[0].name == "a"
        assert restored.objects[1].name == "b"
        assert restored == scene


# ---------------------------------------------------------------------------
# Task model tests
# ---------------------------------------------------------------------------


class TestSuccessCriterion:
    def test_construction(self):
        sc = _make_success_criterion()
        assert sc.type == "ee_near_target"
        assert "threshold" in sc.params

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            SuccessCriterion()  # type: ignore[call-arg]


class TestTaskSpec:
    def test_construction(self):
        task = _make_task_spec()
        assert task.task_id == "reach_001"
        assert len(task.success_criteria) == 1

    def test_defaults(self):
        task = _make_task_spec()
        assert task.reward_hint == ""
        assert task.preferences == {}

    def test_multiple_criteria(self):
        criteria = [
            _make_success_criterion(type="ee_near_target"),
            _make_success_criterion(
                type="object_in_container",
                params={"object": "cube", "container": "bin"},
            ),
        ]
        task = _make_task_spec(success_criteria=criteria)
        assert len(task.success_criteria) == 2

    def test_roundtrip(self):
        task = _make_task_spec(preferences={"minimize_time": True})
        data = task.dict()
        restored = TaskSpec(**data)
        assert restored == task


# ---------------------------------------------------------------------------
# Plan model tests
# ---------------------------------------------------------------------------


class TestGuardCondition:
    def test_contact_guard(self):
        g = GuardCondition(type="contact", body="cube", min_force=0.5)
        assert g.type == "contact"
        assert g.body == "cube"
        assert g.min_force == 0.5
        assert g.threshold is None

    def test_distance_guard(self):
        g = GuardCondition(type="distance", from_body="cube", threshold=0.05)
        assert g.from_body == "cube"
        assert g.threshold == 0.05

    def test_timeout_guard(self):
        g = GuardCondition(type="timeout", steps=50)
        assert g.steps == 50


class TestSkillParams:
    def test_all_optional(self):
        sp = SkillParams()
        assert sp.target is None
        assert sp.speed_fraction is None

    def test_partial(self):
        sp = SkillParams(speed_fraction=0.5, width=0.04)
        assert sp.speed_fraction == 0.5
        assert sp.width == 0.04
        assert sp.direction is None


class TestProposedSkill:
    def test_construction(self):
        skill = ProposedSkill(
            name="move_to",
            params={"target": {"frame": "world", "position": [0.2, 0.0, 0.9]}},
            comment="Move above cube",
        )
        assert skill.name == "move_to"
        assert skill.comment == "Move above cube"

    def test_no_comment(self):
        skill = ProposedSkill(name="wait", params={"steps": 10})
        assert skill.comment is None


class TestLLMProposedPlan:
    def test_construction(self):
        plan = LLMProposedPlan(
            plan_id="pick_001",
            plan_type="skill_plan",
            rationale="Top-down grasp approach.",
            skills=[
                ProposedSkill(name="move_to", params={"speed_fraction": 0.8}),
                ProposedSkill(name="set_gripper", params={"width": 0.0}),
            ],
        )
        assert plan.plan_id == "pick_001"
        assert len(plan.skills) == 2

    def test_defaults(self):
        plan = LLMProposedPlan(
            plan_id="x",
            plan_type="skill_plan",
            rationale="test",
            skills=[],
        )
        assert plan.assumptions == []
        assert plan.uncertainty_flags == []

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            LLMProposedPlan()  # type: ignore[call-arg]


class TestResolvedSkill:
    def test_construction(self):
        skill = ResolvedSkill(
            name="move_to",
            target_world_position=[0.2, 0.0, 0.95],
            target_orientation=[0.0, 0.707, 0.0, 0.707],
            params={"speed_fraction": 0.8, "tolerance": 0.005},
        )
        assert skill.target_world_position == [0.2, 0.0, 0.95]
        assert skill.target_orientation is not None

    def test_with_guard(self):
        guard = GuardCondition(type="contact", body="cube", min_force=0.5)
        skill = ResolvedSkill(
            name="move_linear",
            params={"direction": [0, 0, -1], "distance": 0.1},
            guard=guard,
        )
        assert skill.guard is not None
        assert skill.guard.type == "contact"

    def test_defaults(self):
        skill = ResolvedSkill(name="wait", params={"steps": 10})
        assert skill.target_world_position is None
        assert skill.target_orientation is None
        assert skill.guard is None


class TestCanonicalPlan:
    def test_construction_with_multiple_skills(self):
        skills = [
            ResolvedSkill(
                name="move_to",
                target_world_position=[0.2, 0.0, 0.95],
                params={"speed_fraction": 0.8},
            ),
            ResolvedSkill(
                name="move_linear",
                params={"direction": [0, 0, -1], "distance": 0.1},
                guard=GuardCondition(type="contact", body="cube", min_force=0.5),
            ),
            ResolvedSkill(
                name="set_gripper",
                params={"width": 0.0, "force": 40.0},
            ),
            ResolvedSkill(
                name="move_linear",
                target_world_position=[0.2, 0.0, 1.1],
                params={"direction": [0, 0, 1], "distance": 0.15},
            ),
        ]
        plan = CanonicalPlan(
            plan_id="pick_cube_001",
            skills=skills,
            assumptions=["Cube is reachable from above"],
            metadata={"llm_model": "gpt-5", "prompt_version": "1.0.0"},
        )
        assert plan.plan_id == "pick_cube_001"
        assert len(plan.skills) == 4
        assert plan.skills[1].guard is not None
        assert plan.metadata["llm_model"] == "gpt-5"

    def test_defaults(self):
        plan = CanonicalPlan(plan_id="p1", skills=[])
        assert plan.assumptions == []
        assert plan.metadata == {}

    def test_roundtrip(self):
        plan = CanonicalPlan(
            plan_id="rt_plan",
            skills=[
                ResolvedSkill(name="wait", params={"steps": 5}),
                ResolvedSkill(
                    name="move_to",
                    target_world_position=[0.1, 0.2, 0.3],
                    params={},
                ),
            ],
            assumptions=["test assumption"],
            metadata={"key": "value"},
        )
        data = plan.dict()
        restored = CanonicalPlan(**data)
        assert restored == plan


class TestPlanRejection:
    def test_construction(self):
        rejection = PlanRejection(
            reasons=["Unknown skill: fly_to", "Object 'sphere' not in scene"],
            raw_plan={"plan_id": "bad_plan", "skills": []},
            error_codes=["UNKNOWN_SKILL", "UNKNOWN_OBJECT"],
        )
        assert len(rejection.reasons) == 2
        assert "UNKNOWN_SKILL" in rejection.error_codes

    def test_defaults(self):
        rejection = PlanRejection(reasons=["test"])
        assert rejection.raw_plan == {}
        assert rejection.error_codes == []


# ---------------------------------------------------------------------------
# Execution model tests
# ---------------------------------------------------------------------------


class TestTraceStep:
    def test_construction(self):
        step = _make_trace_step()
        assert step.reward == -0.1
        assert step.terminated is False
        assert step.truncated is False

    def test_default_info(self):
        step = _make_trace_step()
        assert step.info == {}

    def test_with_info(self):
        step = _make_trace_step(info={"ee_pos": [0.1, 0.2, 0.3]})
        assert step.info["ee_pos"] == [0.1, 0.2, 0.3]

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            TraceStep()  # type: ignore[call-arg]


class TestExecutionTrace:
    def test_construction_with_steps(self):
        steps = [
            _make_trace_step(reward=-0.5),
            _make_trace_step(reward=-0.3),
            _make_trace_step(reward=-0.1, terminated=True),
        ]
        trace = ExecutionTrace(
            plan_id="exec_001",
            steps=steps,
            total_reward=-0.9,
            terminated=True,
            truncated=False,
        )
        assert trace.plan_id == "exec_001"
        assert len(trace.steps) == 3
        assert trace.total_reward == pytest.approx(-0.9)
        assert trace.terminated is True

    def test_empty_steps(self):
        trace = ExecutionTrace(
            plan_id="empty",
            steps=[],
            total_reward=0.0,
            terminated=False,
            truncated=True,
        )
        assert len(trace.steps) == 0
        assert trace.truncated is True

    def test_default_final_info(self):
        trace = ExecutionTrace(
            plan_id="t",
            steps=[],
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
        assert trace.final_info == {}

    def test_roundtrip(self):
        trace = ExecutionTrace(
            plan_id="rt",
            steps=[_make_trace_step(), _make_trace_step(terminated=True)],
            total_reward=-0.2,
            terminated=True,
            truncated=False,
            final_info={"is_success": True},
        )
        data = trace.dict()
        restored = ExecutionTrace(**data)
        assert restored == trace

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ExecutionTrace()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Validation model tests
# ---------------------------------------------------------------------------


class TestConstraintViolation:
    def test_construction(self):
        v = ConstraintViolation(
            type="workspace_bounds",
            step=42,
            details="EE position [0.6, 0.0, 0.9] exceeds max X bound 0.5",
        )
        assert v.type == "workspace_bounds"
        assert v.step == 42

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ConstraintViolation()  # type: ignore[call-arg]


class TestValidationMetrics:
    def test_construction(self):
        m = _make_validation_metrics()
        assert m.total_steps == 100
        assert m.max_contact_force == 25.0

    def test_defaults(self):
        m = _make_validation_metrics()
        assert m.final_object_poses == {}
        assert m.success_at_step is None

    def test_with_success_step(self):
        m = _make_validation_metrics(success_at_step=87)
        assert m.success_at_step == 87


class TestValidationReport:
    def test_passing_report(self):
        report = ValidationReport(
            passed=True,
            task_success=True,
            metrics=_make_validation_metrics(success_at_step=95),
        )
        assert report.passed is True
        assert report.task_success is True
        assert report.constraint_violations == []
        assert report.failure_reason is None

    def test_failing_report_with_violations(self):
        violations = [
            ConstraintViolation(
                type="workspace_bounds",
                step=42,
                details="EE X exceeded max bound",
            ),
            ConstraintViolation(
                type="max_force",
                step=67,
                details="Contact force 150N exceeds limit 100N",
            ),
        ]
        report = ValidationReport(
            passed=False,
            task_success=False,
            constraint_violations=violations,
            metrics=_make_validation_metrics(max_contact_force=150.0),
            failure_reason="Workspace and force violations detected.",
        )
        assert report.passed is False
        assert len(report.constraint_violations) == 2
        assert report.failure_reason is not None

    def test_roundtrip(self):
        report = ValidationReport(
            passed=True,
            task_success=True,
            constraint_violations=[
                ConstraintViolation(type="soft_speed", step=10, details="EE speed warning"),
            ],
            metrics=_make_validation_metrics(),
        )
        data = report.dict()
        restored = ValidationReport(**data)
        assert restored == report

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ValidationReport()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# LLM request model tests
# ---------------------------------------------------------------------------


class TestLLMRequest:
    def test_construction(self):
        req = LLMRequest(
            system_message="You are a robotic manipulation planner.",
            user_message="Plan a reach task.",
            model="gpt-5",
            temperature=0.3,
            max_tokens=4096,
        )
        assert req.model == "gpt-5"
        assert req.temperature == 0.3

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            LLMRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Dataset model tests
# ---------------------------------------------------------------------------


class TestDatasetManifest:
    def test_construction(self):
        manifest = DatasetManifest(
            output_dir="/data/dataset_001",
            n_episodes=1000,
            n_original=100,
            n_augmented=900,
            schema_version="1.0.0",
            scene_spec_hash="sha256:abc123",
            task_spec_hash="sha256:def456",
            prompt_template_version="1.0.0",
            llm_model="gpt-5",
            stats={"mean_reward": -0.23, "std_reward": 0.15},
            split_sizes={"train": 800, "val": 100, "test": 100},
        )
        assert manifest.n_episodes == 1000
        assert manifest.stats["mean_reward"] == -0.23
        assert manifest.split_sizes["train"] == 800

    def test_defaults(self):
        manifest = DatasetManifest(
            output_dir="/tmp/test",
            n_episodes=10,
            n_original=10,
            n_augmented=0,
            schema_version="1.0.0",
            scene_spec_hash="sha256:000",
            task_spec_hash="sha256:111",
            prompt_template_version="1.0.0",
            llm_model="gpt-5",
        )
        assert manifest.stats == {}
        assert manifest.split_sizes == {}

    def test_roundtrip(self):
        manifest = DatasetManifest(
            output_dir="/data/ds",
            n_episodes=50,
            n_original=50,
            n_augmented=0,
            schema_version="1.0.0",
            scene_spec_hash="sha256:aaa",
            task_spec_hash="sha256:bbb",
            prompt_template_version="1.0.0",
            llm_model="gpt-5",
            stats={"mean_reward": 0.5},
            split_sizes={"train": 40, "val": 5, "test": 5},
        )
        data = manifest.dict()
        restored = DatasetManifest(**data)
        assert restored == manifest

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            DatasetManifest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cross-model integration tests
# ---------------------------------------------------------------------------


class TestSceneSpecNested:
    """Test SceneSpec with deeply nested and varied objects."""

    def test_full_scene_with_multiple_objects(self):
        scene = _make_scene_spec(
            scene_id="integration_scene",
            objects=[
                _make_object_spec(name="table", shape="box",
                                  shape_params={"half_extents": [0.4, 0.3, 0.02]},
                                  position=[0.0, 0.0, 0.74], is_static=True),
                _make_object_spec(name="red_cube", shape="box",
                                  shape_params={"half_extents": [0.025, 0.025, 0.025]},
                                  position=[0.2, 0.0, 0.79], is_graspable=True, mass=0.05),
                _make_object_spec(name="bin", shape="cylinder",
                                  shape_params={"radius": 0.1, "height": 0.12},
                                  position=[-0.2, 0.1, 0.8], is_container=True,
                                  is_static=True),
            ],
        )
        assert len(scene.objects) == 3
        names = [o.name for o in scene.objects]
        assert "table" in names
        assert "red_cube" in names
        assert "bin" in names

        # Verify roundtrip preserves all nested data
        data = scene.dict()
        restored = SceneSpec(**data)
        assert restored == scene

    def test_scene_with_robot_joints(self):
        robot = _make_robot_spec(
            joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
            joint_limits={
                "j1": [-2.89, 2.89],
                "j2": [-1.76, 1.76],
                "j3": [-2.89, 2.89],
                "j4": [-3.07, -0.07],
                "j5": [-2.89, 2.89],
                "j6": [-0.02, 3.75],
            },
            joint_types={f"j{i}": "revolute" for i in range(1, 7)},
            pd_gains={f"j{i}": [100.0, 10.0] for i in range(1, 7)},
        )
        scene = _make_scene_spec(robot=robot)
        assert len(scene.robot.joint_names) == 6
        assert scene.robot.joint_limits["j1"] == [-2.89, 2.89]


class TestCanonicalPlanIntegration:
    """Test CanonicalPlan with realistic multi-skill plans."""

    def test_pick_and_place_plan(self):
        plan = CanonicalPlan(
            plan_id="pick_cube_001",
            skills=[
                ResolvedSkill(
                    name="move_to",
                    target_world_position=[0.2, 0.0, 0.95],
                    target_orientation=[0.0, 0.707, 0.0, 0.707],
                    params={"speed_fraction": 0.8, "tolerance": 0.005},
                ),
                ResolvedSkill(
                    name="move_linear",
                    params={"direction": [0, 0, -1], "distance": 0.1, "speed_fraction": 0.3},
                    guard=GuardCondition(type="contact", body="cube", min_force=0.5),
                ),
                ResolvedSkill(
                    name="set_gripper",
                    params={"width": 0.0, "force": 40.0, "wait_settle_steps": 10},
                ),
                ResolvedSkill(
                    name="move_linear",
                    params={"direction": [0, 0, 1], "distance": 0.15, "speed_fraction": 0.5},
                ),
                ResolvedSkill(
                    name="move_to",
                    target_world_position=[-0.2, 0.1, 1.05],
                    params={"speed_fraction": 0.6},
                ),
                ResolvedSkill(
                    name="move_linear",
                    params={"direction": [0, 0, -1], "distance": 0.08, "speed_fraction": 0.2},
                ),
                ResolvedSkill(
                    name="set_gripper",
                    params={"width": 0.08, "wait_settle_steps": 5},
                ),
                ResolvedSkill(
                    name="move_linear",
                    params={"direction": [0, 0, 1], "distance": 0.15, "speed_fraction": 0.5},
                ),
            ],
            assumptions=[
                "Cube is reachable from above",
                "Top-down grasp is feasible",
                "Bin opening is wide enough",
            ],
            metadata={"llm_model": "gpt-5", "prompt_version": "1.0.0"},
        )
        assert len(plan.skills) == 8
        assert plan.skills[0].name == "move_to"
        assert plan.skills[1].guard is not None
        assert plan.skills[2].name == "set_gripper"

        # Roundtrip
        data = plan.dict()
        restored = CanonicalPlan(**data)
        assert restored == plan


class TestValidationReportIntegration:
    """Test ValidationReport with realistic validation outcomes."""

    def test_full_report_with_mixed_violations(self):
        violations = [
            ConstraintViolation(
                type="workspace_bounds",
                step=42,
                details="EE position [0.55, 0.0, 0.9] exceeds workspace max X=0.5",
            ),
            ConstraintViolation(
                type="max_force",
                step=67,
                details="Contact force 150.3N exceeds limit 100.0N",
            ),
            ConstraintViolation(
                type="joint_limit",
                step=89,
                details="Joint j3 position 3.0 exceeds upper limit 2.89",
            ),
        ]
        metrics = ValidationMetrics(
            total_steps=100,
            max_contact_force=150.3,
            max_joint_velocity=1.8,
            max_ee_speed=0.95,
            final_object_poses={
                "cube": [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                "bin": [-0.2, 0.1, 0.75, 0.0, 0.0, 0.0, 1.0],
            },
        )
        report = ValidationReport(
            passed=False,
            task_success=False,
            constraint_violations=violations,
            metrics=metrics,
            failure_reason="Multiple hard-gate violations: workspace bounds, contact force, "
            "joint limit.",
        )
        assert not report.passed
        assert len(report.constraint_violations) == 3
        assert report.metrics.max_contact_force == 150.3
        assert "cube" in report.metrics.final_object_poses

        # Roundtrip
        data = report.dict()
        restored = ValidationReport(**data)
        assert restored == report


class TestExecutionTraceIntegration:
    """Test ExecutionTrace with realistic step sequences."""

    def test_multi_step_trace(self):
        steps = []
        for i in range(50):
            steps.append(TraceStep(
                obs=[float(i), float(i) * 0.1, 0.5],
                action=[0.1, -0.05],
                next_obs=[float(i) + 0.02, float(i) * 0.1 + 0.01, 0.5],
                reward=-0.5 + i * 0.01,
                terminated=(i == 49),
                truncated=False,
                info={"step": i},
            ))
        trace = ExecutionTrace(
            plan_id="trace_integration",
            steps=steps,
            total_reward=sum(s.reward for s in steps),
            terminated=True,
            truncated=False,
            final_info={"is_success": True, "episode_length": 50},
        )
        assert len(trace.steps) == 50
        assert trace.steps[-1].terminated is True
        assert trace.final_info["is_success"] is True

        # Roundtrip
        data = trace.dict()
        restored = ExecutionTrace(**data)
        assert restored == trace
        assert len(restored.steps) == 50
