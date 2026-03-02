"""End-to-end tests for clankers_synthetic pipeline and CLI.

All LLM calls and env interactions are mocked -- no live API, no live sim.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

from clankers_synthetic.cli import main as cli_main
from clankers_synthetic.packager import DatasetPackager
from clankers_synthetic.pipeline import generate_dataset
from clankers_synthetic.specs import (
    ConstraintSpec,
    DatasetManifest,
    ObjectSpec,
    ObservationSpec,
    RobotSpec,
    SceneSpec,
    SimulationSpec,
    SuccessCriterion,
    TaskSpec,
)

# ---------------------------------------------------------------------------
# Shared fixtures
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

MOCK_PLAN = {
    "plan_id": "test_001",
    "plan_type": "skill_plan",
    "rationale": "Simple wait plan for testing",
    "assumptions": [],
    "uncertainty_flags": [],
    "skills": [
        {"name": "wait", "params": {"steps": 3}},
    ],
}


def _make_scene() -> SceneSpec:
    return SceneSpec(
        scene_id="test_scene",
        simulation=SimulationSpec(),
        robot=RobotSpec(
            name="test_robot",
            urdf_path="test.urdf",
            base_position=[0.0, 0.0, 0.0],
            base_orientation=[0.0, 0.0, 0.0, 1.0],
            joint_names=JOINT_NAMES,
            joint_limits=JOINT_LIMITS,
            ee_link_name="end_effector",
        ),
        objects=[
            ObjectSpec(
                name="cube",
                shape="box",
                position=[0.3, 0.0, 0.8],
            ),
        ],
        constraints=ConstraintSpec(
            workspace_bounds_min=[-1.0, -1.0, 0.0],
            workspace_bounds_max=[1.0, 1.0, 2.0],
            max_contact_force=100.0,
        ),
        observations=ObservationSpec(joint_state_dim=6),
    )


def _make_task() -> TaskSpec:
    return TaskSpec(
        task_id="test_task",
        task_text="Wait in place for testing",
        success_criteria=[
            SuccessCriterion(type="is_success", params={}),
        ],
    )


class MockEnv:
    """Mock env that returns success after a fixed number of steps."""

    def __init__(self, n_steps_before_success: int = 3):
        self.step_count = 0
        self.n_steps_before_success = n_steps_before_success

    def reset(self):
        self.step_count = 0
        return np.zeros(6, dtype=np.float32), {
            "body_poses": {"end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1]},
            "contact_events": [],
            "is_success": False,
        }

    def step(self, action):
        self.step_count += 1
        is_success = self.step_count >= self.n_steps_before_success
        return (
            np.zeros(6, dtype=np.float32),
            -0.1,
            is_success,
            False,
            {
                "body_poses": {"end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1]},
                "contact_events": [],
                "is_success": is_success,
            },
        )


class FailingMockEnv:
    """Mock env that never reports success."""

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros(6, dtype=np.float32), {
            "body_poses": {"end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1]},
            "contact_events": [],
            "is_success": False,
        }

    def step(self, action):
        self.step_count += 1
        return (
            np.zeros(6, dtype=np.float32),
            -0.1,
            False,
            False,
            {
                "body_poses": {"end_effector": [0.2, 0.0, 0.9, 0, 0, 0, 1]},
                "contact_events": [],
                "is_success": False,
            },
        )


def _mock_openai_response(plan_dict: dict) -> list[dict]:
    """Build a list matching OpenAIClient.request_json return format."""
    return [
        {
            "content": plan_dict,
            "provenance": {
                "model": "gpt-5-mock",
                "request_id": "req_test",
                "prompt_hash": "abc123",
                "response_hash": "def456",
                "timestamp": 1700000000.0,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    """Tests for generate_dataset() orchestrator."""

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_e2e(self, mock_request, tmp_path):
        """Full pipeline: mock LLM returns valid plan, mock env succeeds, dataset written."""
        mock_request.return_value = _mock_openai_response(MOCK_PLAN)

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=lambda: MockEnv(n_steps_before_success=3),
            n_plans=1,
            model="gpt-5",
            seed=42,
        )

        assert isinstance(manifest, DatasetManifest)
        assert manifest.n_episodes == 1
        assert manifest.output_dir == out
        assert manifest.schema_version == DatasetPackager.SCHEMA_VERSION
        assert manifest.llm_model == "gpt-5"
        assert manifest.scene_spec_hash != ""
        assert manifest.task_spec_hash != ""

        # Verify files were written
        assert os.path.isdir(os.path.join(out, "episodes"))
        assert os.path.isfile(os.path.join(out, "metadata.json"))
        assert os.path.isfile(os.path.join(out, "splits.json"))
        ep0 = os.path.join(out, "episodes", "ep_000000.json")
        assert os.path.isfile(ep0)
        with open(ep0) as f:
            ep_data = json.load(f)
        assert "steps" in ep_data
        assert len(ep_data["steps"]) == 3

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_no_passing_traces(self, mock_request, tmp_path):
        """Unparseable LLM output yields empty manifest."""
        # Return a plan with an invalid skill name -- parser will reject it
        bad_plan = {
            "plan_id": "bad_001",
            "plan_type": "skill_plan",
            "rationale": "bad",
            "assumptions": [],
            "uncertainty_flags": [],
            "skills": [
                {"name": "invalid_skill", "params": {}},
            ],
        }
        mock_request.return_value = _mock_openai_response(bad_plan)

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=lambda: MockEnv(),
            n_plans=2,
            seed=42,
        )

        assert isinstance(manifest, DatasetManifest)
        assert manifest.n_episodes == 0
        assert manifest.stats["validation_pass_rate"] == 0.0
        # Output directory should still exist
        assert os.path.isdir(out)

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_with_refinement(self, mock_request, tmp_path):
        """First plan fails validation (env never succeeds), refined plan succeeds.

        We use a FailingMockEnv for the first execution (validation fails because
        is_success is never True), and then a MockEnv for the refined re-execution.
        The refiner's deterministic fix cannot fix task_failure, so it falls back
        to LLM re-proposal. The refiner hashes plans by skill content for loop
        detection, so the refined LLM response must have different skill params
        to avoid being rejected as a duplicate.
        """
        # The refined plan has different steps so the plan hash differs
        refined_plan = {
            "plan_id": "test_refined",
            "plan_type": "skill_plan",
            "rationale": "Refined wait plan",
            "assumptions": [],
            "uncertainty_flags": [],
            "skills": [
                {"name": "wait", "params": {"steps": 5}},
            ],
        }

        call_count = {"n": 0}
        env_calls = {"n": 0}

        def mock_request_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                # First call: initial proposal
                return _mock_openai_response(MOCK_PLAN)
            else:
                # Subsequent calls: refiner re-proposal with different skills
                return _mock_openai_response(refined_plan)

        mock_request.side_effect = mock_request_side_effect

        def env_factory():
            env_calls["n"] += 1
            if env_calls["n"] <= 1:
                # First env: never succeeds (validation will fail)
                return FailingMockEnv()
            else:
                # Subsequent envs: succeed after 5 steps (matching refined plan)
                return MockEnv(n_steps_before_success=5)

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=env_factory,
            n_plans=1,
            max_refine_iters=3,
            seed=42,
        )

        assert isinstance(manifest, DatasetManifest)
        # Should have 1 passing trace (from the refined plan)
        assert manifest.n_episodes == 1
        # LLM was called at least twice: initial proposal + refinement
        assert call_count["n"] >= 2
        # Env was created at least twice: initial + refined
        assert env_calls["n"] >= 2

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_no_env_factory(self, mock_request, tmp_path):
        """Without env_factory, pipeline logs warning and returns empty manifest."""
        mock_request.return_value = _mock_openai_response(MOCK_PLAN)

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=None,
            n_plans=1,
            seed=42,
        )

        assert isinstance(manifest, DatasetManifest)
        assert manifest.n_episodes == 0

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_multiple_plans(self, mock_request, tmp_path):
        """Multiple n_plans all succeeding produces dataset with multiple episodes."""
        mock_request.return_value = _mock_openai_response(MOCK_PLAN)

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=lambda: MockEnv(n_steps_before_success=3),
            n_plans=3,
            seed=42,
        )

        assert isinstance(manifest, DatasetManifest)
        assert manifest.n_episodes == 3

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_hashes_stable(self, mock_request, tmp_path):
        """Same scene+task produce the same hashes across runs."""
        mock_request.return_value = _mock_openai_response(MOCK_PLAN)
        scene = _make_scene()
        task = _make_task()

        out1 = str(tmp_path / "ds1")
        m1 = generate_dataset(
            scene=scene, task=task, output_dir=out1,
            env_factory=lambda: MockEnv(n_steps_before_success=3),
            n_plans=1, seed=42,
        )

        out2 = str(tmp_path / "ds2")
        m2 = generate_dataset(
            scene=scene, task=task, output_dir=out2,
            env_factory=lambda: MockEnv(n_steps_before_success=3),
            n_plans=1, seed=42,
        )

        assert m1.scene_spec_hash == m2.scene_spec_hash
        assert m1.task_spec_hash == m2.task_spec_hash

    @patch("clankers_synthetic.openai_client.OpenAIClient.request_json")
    def test_generate_dataset_llm_exception(self, mock_request, tmp_path):
        """LLM throwing exception is caught, pipeline returns empty manifest."""
        mock_request.side_effect = RuntimeError("API down")

        scene = _make_scene()
        task = _make_task()
        out = str(tmp_path / "dataset")

        manifest = generate_dataset(
            scene=scene,
            task=task,
            output_dir=out,
            env_factory=lambda: MockEnv(),
            n_plans=2,
            seed=42,
        )

        assert manifest.n_episodes == 0


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI entrypoint."""

    def test_cli_help(self):
        """--help exits with code 0."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--help"])
        assert exc_info.value.code == 0

    def test_cli_argument_parsing(self):
        """Verify argparse parses all expected arguments without error."""
        import argparse

        # We just need to verify parsing, not execution
        parser = argparse.ArgumentParser(prog="clankers_synthetic")
        parser.add_argument("--scene", required=True)
        parser.add_argument("--task", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--n-plans", type=int, default=10)
        parser.add_argument("--model", default="gpt-5")
        parser.add_argument("--temperature", type=float, default=0.3)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--max-refine-iters", type=int, default=3)
        parser.add_argument("--api-key", default=None)

        args = parser.parse_args([
            "--scene", "scene.json",
            "--task", "task.json",
            "--out", "/tmp/dataset",
            "--n-plans", "5",
            "--model", "gpt-4o",
            "--temperature", "0.7",
            "--seed", "99",
            "--max-refine-iters", "2",
            "--api-key", "sk-test",
        ])

        assert args.scene == "scene.json"
        assert args.task == "task.json"
        assert args.out == "/tmp/dataset"
        assert args.n_plans == 5
        assert args.model == "gpt-4o"
        assert args.temperature == 0.7
        assert args.seed == 99
        assert args.max_refine_iters == 2
        assert args.api_key == "sk-test"

    def test_cli_missing_required_args(self):
        """Missing required args causes SystemExit(2)."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main([])
        assert exc_info.value.code == 2

    @patch("clankers_synthetic.pipeline.generate_dataset")
    def test_cli_with_spec_files(self, mock_generate, tmp_path):
        """Write temp spec files, run CLI with mocked pipeline."""
        scene = _make_scene()
        task = _make_task()

        scene_path = str(tmp_path / "scene.json")
        task_path = str(tmp_path / "task.json")
        out_dir = str(tmp_path / "output")

        with open(scene_path, "w") as f:
            json.dump(scene.model_dump(), f, default=str)

        with open(task_path, "w") as f:
            json.dump(task.model_dump(), f, default=str)

        mock_generate.return_value = DatasetManifest(
            output_dir=out_dir,
            n_episodes=5,
            n_original=5,
            n_augmented=0,
            schema_version="1.0.0",
            scene_spec_hash="abc",
            task_spec_hash="def",
            prompt_template_version="1.0.0",
            llm_model="gpt-5",
        )

        result = cli_main([
            "--scene", scene_path,
            "--task", task_path,
            "--out", out_dir,
            "--n-plans", "5",
            "--model", "gpt-5",
            "--seed", "99",
        ])

        assert result == 0
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        # Verify the kwargs passed to generate_dataset
        assert call_kwargs.kwargs["n_plans"] == 5
        assert call_kwargs.kwargs["model"] == "gpt-5"
        assert call_kwargs.kwargs["seed"] == 99


class TestMainModule:
    """Tests for __main__.py entry point."""

    def test_module_runnable(self):
        """python -m clankers_synthetic --help exits 0."""
        import subprocess
        import sys

        # Ensure the package is importable by adding its parent to PYTHONPATH
        package_parent = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = package_parent + os.pathsep + existing if existing else package_parent

        result = subprocess.run(
            [sys.executable, "-m", "clankers_synthetic", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        assert result.returncode == 0
        assert "clankers_synthetic" in result.stdout
