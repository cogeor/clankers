"""Tests for clankers_synthetic.packager -- DatasetPackager."""
from __future__ import annotations

import json

import pytest

from clankers_synthetic.packager import DatasetPackager
from clankers_synthetic.specs import (
    DatasetManifest,
    ExecutionTrace,
    TraceStep,
    ValidationMetrics,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace_pair(n_steps=10, total_reward=-1.0, max_force=25.0):
    """Create an (ExecutionTrace, ValidationReport) tuple."""
    steps = [
        TraceStep(
            obs=[0.0] * 3,
            action=[0.0] * 2,
            next_obs=[0.0] * 3,
            reward=total_reward / n_steps,
            terminated=False,
            truncated=False,
        )
        for _ in range(n_steps)
    ]
    trace = ExecutionTrace(
        plan_id="test",
        steps=steps,
        total_reward=total_reward,
        terminated=False,
        truncated=False,
    )
    metrics = ValidationMetrics(
        total_steps=n_steps,
        max_contact_force=max_force,
        max_joint_velocity=1.0,
        max_ee_speed=0.5,
    )
    report = ValidationReport(passed=True, task_success=True, metrics=metrics)
    return trace, report


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetPackager:
    """Tests for DatasetPackager.package()."""

    def test_package_creates_directory_structure(self, tmp_path):
        """package() creates episodes/, provenance/plans/, provenance/validation/, provenance/prompts/."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair()]
        packager.package(traces, out)

        assert (tmp_path / "dataset" / "episodes").is_dir()
        assert (tmp_path / "dataset" / "provenance" / "plans").is_dir()
        assert (tmp_path / "dataset" / "provenance" / "validation").is_dir()
        assert (tmp_path / "dataset" / "provenance" / "prompts").is_dir()

    def test_package_writes_metadata_json(self, tmp_path):
        """metadata.json contains schema_version, n_episodes, and stats."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair(), _make_trace_pair()]
        packager.package(traces, out)

        meta_path = tmp_path / "dataset" / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == "1.0.0"
        assert meta["n_episodes"] == 2
        assert "stats" in meta
        assert "mean_reward" in meta["stats"]
        assert "std_reward" in meta["stats"]
        assert "mean_episode_length" in meta["stats"]

    def test_package_writes_splits_json(self, tmp_path):
        """splits.json has train/val/test keys with integer indices."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair() for _ in range(20)]
        packager.package(traces, out, split_ratios=(0.8, 0.1, 0.1))

        splits_path = tmp_path / "dataset" / "splits.json"
        assert splits_path.exists()
        splits = json.loads(splits_path.read_text())
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        # All indices should be covered
        all_indices = sorted(splits["train"] + splits["val"] + splits["test"])
        assert all_indices == list(range(20))

    def test_splits_deterministic_from_seed(self, tmp_path):
        """Same seed produces identical splits."""
        packager = DatasetPackager()
        traces = [_make_trace_pair() for _ in range(20)]

        out1 = str(tmp_path / "ds1")
        packager.package(traces, out1, seed=123)
        splits1 = json.loads((tmp_path / "ds1" / "splits.json").read_text())

        out2 = str(tmp_path / "ds2")
        packager.package(traces, out2, seed=123)
        splits2 = json.loads((tmp_path / "ds2" / "splits.json").read_text())

        assert splits1 == splits2

    def test_splits_different_seed(self, tmp_path):
        """Different seeds produce different splits."""
        packager = DatasetPackager()
        traces = [_make_trace_pair() for _ in range(20)]

        out1 = str(tmp_path / "ds1")
        packager.package(traces, out1, seed=1)
        splits1 = json.loads((tmp_path / "ds1" / "splits.json").read_text())

        out2 = str(tmp_path / "ds2")
        packager.package(traces, out2, seed=2)
        splits2 = json.loads((tmp_path / "ds2" / "splits.json").read_text())

        # With 20 episodes and different seeds, splits should differ
        assert splits1 != splits2

    def test_package_writes_episode_files(self, tmp_path):
        """ep_000000.json and ep_000001.json exist with correct structure."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair(), _make_trace_pair()]
        packager.package(traces, out)

        ep0 = tmp_path / "dataset" / "episodes" / "ep_000000.json"
        ep1 = tmp_path / "dataset" / "episodes" / "ep_000001.json"
        assert ep0.exists()
        assert ep1.exists()

        data = json.loads(ep0.read_text())
        assert "plan_id" in data
        assert "steps" in data
        assert "total_reward" in data
        assert "terminated" in data
        assert "truncated" in data

    def test_package_writes_validation_provenance(self, tmp_path):
        """validation/*.json files exist for each episode."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair(), _make_trace_pair()]
        packager.package(traces, out)

        val0 = tmp_path / "dataset" / "provenance" / "validation" / "ep_000000.json"
        val1 = tmp_path / "dataset" / "provenance" / "validation" / "ep_000001.json"
        assert val0.exists()
        assert val1.exists()

        data = json.loads(val0.read_text())
        assert "passed" in data
        assert "metrics" in data

    def test_package_writes_plan_provenance(self, tmp_path):
        """plans/*.json files exist when plans are provided."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair(), _make_trace_pair()]
        plans = [
            {"plan_id": "p0", "skills": []},
            {"plan_id": "p1", "skills": []},
        ]
        packager.package(traces, out, plans=plans)

        plan0 = tmp_path / "dataset" / "provenance" / "plans" / "ep_000000.json"
        plan1 = tmp_path / "dataset" / "provenance" / "plans" / "ep_000001.json"
        assert plan0.exists()
        assert plan1.exists()

        data = json.loads(plan0.read_text())
        assert data["plan_id"] == "p0"

    def test_package_writes_prompt_provenance(self, tmp_path):
        """prompts/*.json files exist when prompts are provided."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair()]
        prompt_data = [{"system": "You are a robot.", "response": "OK"}]
        packager.package(traces, out, prompts=prompt_data)

        prompt0 = tmp_path / "dataset" / "provenance" / "prompts" / "ep_000000.json"
        assert prompt0.exists()

        data = json.loads(prompt0.read_text())
        assert data["system"] == "You are a robot."

    def test_package_returns_manifest(self, tmp_path):
        """package() returns a DatasetManifest with correct fields."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair() for _ in range(10)]
        manifest = packager.package(
            traces,
            out,
            scene_spec_hash="abc123",
            task_spec_hash="def456",
            llm_model="gpt-4o",
        )

        assert isinstance(manifest, DatasetManifest)
        assert manifest.output_dir == out
        assert manifest.n_episodes == 10
        assert manifest.n_original == 10
        assert manifest.n_augmented == 0
        assert manifest.schema_version == "1.0.0"
        assert manifest.scene_spec_hash == "abc123"
        assert manifest.task_spec_hash == "def456"
        assert manifest.llm_model == "gpt-4o"
        assert "train" in manifest.split_sizes
        assert "val" in manifest.split_sizes
        assert "test" in manifest.split_sizes

    def test_package_stats_computed(self, tmp_path):
        """Stats include correct mean_reward, std_reward, mean_episode_length."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        # Two traces with different rewards and lengths
        t1 = _make_trace_pair(n_steps=10, total_reward=-2.0, max_force=20.0)
        t2 = _make_trace_pair(n_steps=20, total_reward=-4.0, max_force=30.0)
        manifest = packager.package([t1, t2], out)

        stats = manifest.stats
        # mean_reward = (-2.0 + -4.0) / 2 = -3.0
        assert stats["mean_reward"] == -3.0
        # std_reward = sqrt(((1)^2 + (1)^2) / 2) = 1.0
        assert stats["std_reward"] == 1.0
        # mean_episode_length = (10 + 20) / 2 = 15.0
        assert stats["mean_episode_length"] == 15.0
        # mean_contact_force = (20 + 30) / 2 = 25.0
        assert stats["mean_contact_force"] == 25.0
        # max_contact_force = 30.0
        assert stats["max_contact_force"] == 30.0

    def test_empty_traces_raises(self, tmp_path):
        """package() raises ValueError on empty trace list."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        with pytest.raises(ValueError, match="No traces"):
            packager.package([], out)

    def test_episode_data_loadable(self, tmp_path):
        """Write then read episode JSON; verify obs/action shapes are preserved."""
        packager = DatasetPackager()
        out = str(tmp_path / "dataset")
        traces = [_make_trace_pair(n_steps=5)]
        packager.package(traces, out)

        ep_path = tmp_path / "dataset" / "episodes" / "ep_000000.json"
        data = json.loads(ep_path.read_text())

        assert len(data["steps"]) == 5
        step = data["steps"][0]
        assert len(step["obs"]) == 3
        assert len(step["action"]) == 2
        assert len(step["next_obs"]) == 3
        assert isinstance(step["reward"], float)
        assert isinstance(step["terminated"], bool)
        assert isinstance(step["truncated"], bool)
