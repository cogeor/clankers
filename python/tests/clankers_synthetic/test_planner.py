"""Tests for clankers_synthetic.planner module."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from clankers_synthetic.specs import (
    ConstraintSpec,
    LLMRequest,
    ObjectSpec,
    ObservationSpec,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
    SuccessCriterion,
    TaskSpec,
)
from clankers_synthetic.planner import LLMPlanner, PromptAssembler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_scene_spec(**overrides) -> SceneSpec:
    defaults = {
        "scene_id": "test_scene_001",
        "simulation": SimulationSpec(),
        "robot": RobotSpec(
            name="panda",
            urdf_path="/robots/panda.urdf",
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
        ),
        "objects": [
            ObjectSpec(
                name="cube",
                shape="box",
                shape_params={"half_extents": [0.025, 0.025, 0.025]},
                position=[0.2, 0.0, 0.8],
            )
        ],
        "constraints": ConstraintSpec(
            workspace_bounds_min=[-0.5, -0.5, 0.0],
            workspace_bounds_max=[0.5, 0.5, 1.2],
        ),
        "observations": ObservationSpec(),
    }
    defaults.update(overrides)
    return SceneSpec(**defaults)


def _make_task_spec(**overrides) -> TaskSpec:
    defaults = {
        "task_id": "reach_001",
        "task_text": "Move the end-effector to the cube position.",
        "success_criteria": [
            SuccessCriterion(
                type="ee_near_target",
                params={"target": [0.2, 0.0, 0.8], "threshold": 0.05},
            )
        ],
    }
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_mock_client(n_results: int = 1) -> MagicMock:
    """Create a mock OpenAIClient with request_json returning n_results."""
    client = MagicMock()
    results = []
    for i in range(n_results):
        results.append(
            {
                "content": {
                    "plan_id": f"plan_{i}",
                    "plan_type": "skill_plan",
                    "rationale": f"Test plan {i}",
                    "assumptions": [],
                    "uncertainty_flags": [],
                    "skills": [
                        {"name": "move_to", "params": {"speed_fraction": 0.5}}
                    ],
                },
                "provenance": {
                    "model": "gpt-5",
                    "request_id": f"req-{i}",
                    "prompt_hash": f"hash-{i}",
                    "response_hash": f"rhash-{i}",
                    "timestamp": 1700000000.0 + i,
                },
            }
        )
    client.request_json.return_value = results
    return client


# ---------------------------------------------------------------------------
# PromptAssembler tests
# ---------------------------------------------------------------------------


class TestPromptAssemblerSystemMessageContainsSkills:
    """test_prompt_assembler_system_message_contains_skills -- verify skill table present."""

    def test_system_message_contains_skill_vocabulary(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "move_to" in request.system_message
        assert "move_linear" in request.system_message
        assert "move_relative" in request.system_message
        assert "set_gripper" in request.system_message
        assert "wait" in request.system_message
        assert "move_joints" in request.system_message
        assert "Available Skills" in request.system_message


class TestPromptAssemblerSystemMessageContainsRules:
    """test_prompt_assembler_system_message_contains_rules -- verify rules present."""

    def test_system_message_contains_rules(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "meters" in request.system_message
        assert "Z-up" in request.system_message
        assert "quaternions" in request.system_message
        assert "speed_fraction" in request.system_message
        assert "workspace bounds" in request.system_message
        assert "valid JSON" in request.system_message
        assert "Rules" in request.system_message


class TestPromptAssemblerUserMessageContainsScene:
    """test_prompt_assembler_user_message_contains_scene -- scene JSON in user msg."""

    def test_user_message_contains_scene_json(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "## Scene" in request.user_message
        assert "test_scene_001" in request.user_message
        assert "panda" in request.user_message
        assert "cube" in request.user_message
        # Verify it's valid JSON embedded in the message
        scene_dict = scene.dict()
        scene_json = json.dumps(scene_dict, indent=2, default=str)
        assert scene_json in request.user_message


class TestPromptAssemblerUserMessageContainsTask:
    """test_prompt_assembler_user_message_contains_task -- task JSON in user msg."""

    def test_user_message_contains_task_json(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "## Task" in request.user_message
        assert "reach_001" in request.user_message
        assert "Move the end-effector" in request.user_message
        task_dict = task.dict()
        task_json = json.dumps(task_dict, indent=2, default=str)
        assert task_json in request.user_message


class TestPromptAssemblerWithFewShot:
    """test_prompt_assembler_with_few_shot -- examples included in user message."""

    def test_few_shot_examples_in_user_message(self) -> None:
        example_scene = _make_scene_spec(scene_id="example_scene")
        example_task = _make_task_spec(task_id="example_task")
        example_plan = {
            "plan_id": "example_plan",
            "plan_type": "skill_plan",
            "rationale": "Example approach",
            "skills": [{"name": "move_to", "params": {"speed_fraction": 0.5}}],
        }

        assembler = PromptAssembler(
            few_shot_examples=[(example_scene, example_task, example_plan)]
        )
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "## Example 1" in request.user_message
        assert "example_scene" in request.user_message
        assert "example_task" in request.user_message
        assert "example_plan" in request.user_message

    def test_multiple_few_shot_examples(self) -> None:
        examples = []
        for i in range(3):
            ex_scene = _make_scene_spec(scene_id=f"ex_scene_{i}")
            ex_task = _make_task_spec(task_id=f"ex_task_{i}")
            ex_plan = {"plan_id": f"ex_plan_{i}", "skills": []}
            examples.append((ex_scene, ex_task, ex_plan))

        assembler = PromptAssembler(few_shot_examples=examples)
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "## Example 1" in request.user_message
        assert "## Example 2" in request.user_message
        assert "## Example 3" in request.user_message

    def test_no_few_shot_by_default(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "## Example" not in request.user_message


class TestPromptAssemblerTemplateVersion:
    """test_prompt_assembler_template_version -- version string in system msg."""

    def test_template_version_in_system_message(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()
        request = assembler.assemble(scene, task)

        assert "1.0.0" in request.system_message
        assert "Prompt template version:" in request.system_message

    def test_template_version_class_attribute(self) -> None:
        assert PromptAssembler.TEMPLATE_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# LLMPlanner tests
# ---------------------------------------------------------------------------


class TestLLMPlannerProposeReturnsCandidates:
    """test_llm_planner_propose_returns_candidates -- mock client, verify n results."""

    def test_propose_returns_correct_number_of_candidates(self) -> None:
        mock_client = _make_mock_client(n_results=3)
        planner = LLMPlanner(client=mock_client, n_candidates=3)
        scene = _make_scene_spec()
        task = _make_task_spec()

        results = planner.propose(scene, task)

        assert len(results) == 3
        for i, r in enumerate(results):
            assert "plan" in r
            assert "provenance" in r
            assert r["plan"]["plan_id"] == f"plan_{i}"

    def test_propose_with_override_n_candidates(self) -> None:
        mock_client = _make_mock_client(n_results=2)
        planner = LLMPlanner(client=mock_client, n_candidates=3)
        scene = _make_scene_spec()
        task = _make_task_spec()

        results = planner.propose(scene, task, n_candidates=2)

        # The client was called; verify n=2 was passed
        call_kwargs = mock_client.request_json.call_args
        assert call_kwargs.kwargs.get("n") == 2 or call_kwargs[1].get("n") == 2

    def test_propose_with_seed(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client, n_candidates=1)
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.propose(scene, task, seed=42)

        call_kwargs = mock_client.request_json.call_args
        assert call_kwargs.kwargs.get("seed") == 42 or call_kwargs[1].get("seed") == 42


class TestLLMPlannerProposePassesModelAndTemp:
    """test_llm_planner_propose_passes_model_and_temp -- verify API params."""

    def test_model_and_temperature_in_assembled_request(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(
            client=mock_client, model="gpt-5-turbo", temperature=0.7
        )
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.propose(scene, task)

        # The request passed to client.request_json should have model/temp set
        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]  # first positional arg
        assert isinstance(request_arg, LLMRequest)
        assert request_arg.model == "gpt-5-turbo"
        assert request_arg.temperature == 0.7

    def test_max_tokens_in_assembled_request(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client, max_tokens=2048)
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.propose(scene, task)

        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert request_arg.max_tokens == 2048


class TestLLMPlannerRefineIncludesFailure:
    """test_llm_planner_refine_includes_failure -- refinement prompt contains failure report."""

    def test_refine_includes_failure_reason(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client)
        scene = _make_scene_spec()
        task = _make_task_spec()

        failed_plan = {
            "plan_id": "failed_001",
            "skills": [{"name": "move_to", "params": {"speed_fraction": 0.9}}],
        }
        failure_report = {
            "failure_reason": "Workspace bounds exceeded at step 42",
            "constraint_violations": [
                {"type": "workspace_bounds", "step": 42, "details": "EE out of bounds"}
            ],
        }

        result = planner.refine_candidate(failed_plan, failure_report, scene, task)

        # Verify the refinement context was passed to the client
        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert "Workspace bounds exceeded" in request_arg.user_message
        assert "Previous Plan Failed" in request_arg.user_message
        assert "Validation Report" in request_arg.user_message
        assert "Constraints Violated" in request_arg.user_message
        assert "workspace_bounds" in request_arg.user_message

        # Verify result structure
        assert "plan" in result
        assert "provenance" in result

    def test_refine_increases_temperature(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client, temperature=0.3)
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.refine_candidate(
            {"plan_id": "x", "skills": []},
            {"failure_reason": "test"},
            scene,
            task,
        )

        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert request_arg.temperature == pytest.approx(0.4)

    def test_refine_temperature_capped_at_1(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client, temperature=0.95)
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.refine_candidate(
            {"plan_id": "x", "skills": []},
            {"failure_reason": "test"},
            scene,
            task,
        )

        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert request_arg.temperature == 1.0


class TestLLMPlannerRefineIncludesHistory:
    """test_llm_planner_refine_includes_history -- attempt history in prompt."""

    def test_refine_with_attempt_history(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client)
        scene = _make_scene_spec()
        task = _make_task_spec()

        attempt_history = [
            {"attempt": 1, "failure": "speed too high"},
            {"attempt": 2, "failure": "workspace bounds"},
        ]

        planner.refine_candidate(
            {"plan_id": "x", "skills": []},
            {"failure_reason": "test failure"},
            scene,
            task,
            attempt_history=attempt_history,
        )

        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert "Attempt History" in request_arg.user_message
        assert "speed too high" in request_arg.user_message
        assert "workspace bounds" in request_arg.user_message

    def test_refine_without_attempt_history(self) -> None:
        mock_client = _make_mock_client(n_results=1)
        planner = LLMPlanner(client=mock_client)
        scene = _make_scene_spec()
        task = _make_task_spec()

        planner.refine_candidate(
            {"plan_id": "x", "skills": []},
            {"failure_reason": "test failure"},
            scene,
            task,
        )

        call_args = mock_client.request_json.call_args
        request_arg = call_args[0][0]
        assert "Attempt History" not in request_arg.user_message


class TestLLMRequestStructure:
    """test_llm_request_structure -- assembled request has correct fields."""

    def test_assembled_request_fields(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()

        request = assembler.assemble(
            scene, task, model="gpt-5", temperature=0.3, max_tokens=4096
        )

        assert isinstance(request, LLMRequest)
        assert request.model == "gpt-5"
        assert request.temperature == 0.3
        assert request.max_tokens == 4096
        assert isinstance(request.system_message, str)
        assert isinstance(request.user_message, str)
        assert len(request.system_message) > 0
        assert len(request.user_message) > 0

    def test_assembled_request_custom_params(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()

        request = assembler.assemble(
            scene, task, model="gpt-5-turbo", temperature=0.9, max_tokens=8192
        )

        assert request.model == "gpt-5-turbo"
        assert request.temperature == 0.9
        assert request.max_tokens == 8192

    def test_user_message_ends_with_plan_instruction(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()

        request = assembler.assemble(scene, task)

        assert request.user_message.endswith("## Plan the trajectory.")

    def test_system_message_contains_robotic_planner_role(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()

        request = assembler.assemble(scene, task)

        assert "robotic manipulation planner" in request.system_message

    def test_system_message_contains_output_schema(self) -> None:
        assembler = PromptAssembler()
        scene = _make_scene_spec()
        task = _make_task_spec()

        request = assembler.assemble(scene, task)

        assert "plan_id" in request.system_message
        assert "plan_type" in request.system_message
        assert "rationale" in request.system_message
        assert "skills" in request.system_message
        assert "Output Schema" in request.system_message
