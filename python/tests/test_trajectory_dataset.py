"""Tests for clankers.trajectory_dataset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from clankers.trajectory_dataset import TrajectoryDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trace(n_steps: int = 5, n_joints: int = 3) -> dict:
    """Create a synthetic trace with interleaved obs [pos0, vel0, pos1, vel1, ...]."""
    steps = []
    for t in range(n_steps):
        obs = []
        for j in range(n_joints):
            obs.append(float(t * n_joints + j))       # position
            obs.append(float((t * n_joints + j) * 0.1))  # velocity
        action = [float(j + 0.5) for j in range(n_joints)]
        steps.append({
            "obs": obs,
            "action": action,
            "next_obs": obs,  # simplified
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        })
    return {"plan_id": "test", "steps": steps}


# ---------------------------------------------------------------------------
# from_trace_file
# ---------------------------------------------------------------------------


def test_from_trace_file_consecutive_pairs(tmp_path):
    trace = _make_trace(n_steps=5, n_joints=3)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    names = ["a", "b", "c"]
    ds = TrajectoryDataset.from_trace_file(p, joint_names=names)

    # 5 steps → 4 consecutive pairs
    assert len(ds) == 4
    pos, target = ds[0]
    assert pos.shape == (3,)
    assert target.shape == (3,)

    # Positions are alphabetically sorted (a=0, b=1, c=2 at step 0)
    np.testing.assert_array_almost_equal(pos.numpy(), [0.0, 1.0, 2.0])
    # Next step positions (a=3, b=4, c=5)
    np.testing.assert_array_almost_equal(target.numpy(), [3.0, 4.0, 5.0])


def test_from_trace_file_with_actions(tmp_path):
    trace = _make_trace(n_steps=3, n_joints=2)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_trace_file(
        p, joint_names=["x", "y"], include_actions=True
    )

    # With actions: one sample per step (3 steps)
    assert len(ds) == 3
    pos, action = ds[0]
    assert pos.shape == (2,)
    assert action.shape == (2,)


def test_from_trace_file_auto_detect_joints(tmp_path):
    trace = _make_trace(n_steps=4, n_joints=6)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_trace_file(p)

    assert ds.encoder.dof == 6
    assert ds.input_dim == 6
    assert len(ds) == 3  # 4 steps → 3 consecutive pairs


def test_from_trace_file_with_scene_spec(tmp_path):
    trace = _make_trace(n_steps=3, n_joints=2)
    (tmp_path / "trace.json").write_text(json.dumps(trace))

    spec = {"robot": {"joint_names": ["z_joint", "a_joint"]}}
    (tmp_path / "scene.json").write_text(json.dumps(spec))

    ds = TrajectoryDataset.from_trace_file(
        tmp_path / "trace.json",
        scene_spec=tmp_path / "scene.json",
    )
    # Alphabetic: a_joint, z_joint
    assert ds.encoder.names == ("a_joint", "z_joint")


# ---------------------------------------------------------------------------
# from_dataset_dir
# ---------------------------------------------------------------------------


def test_from_dataset_dir(tmp_path):
    ep_dir = tmp_path / "episodes"
    ep_dir.mkdir()

    # Write 2 episode files with 3 steps each
    for i in range(2):
        trace = _make_trace(n_steps=3, n_joints=2)
        trace["plan_id"] = f"ep_{i}"
        (ep_dir / f"ep_{i:06d}.json").write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_dataset_dir(
        tmp_path, joint_names=["a", "b"]
    )

    # 2 episodes × 3 steps = 6 total steps → 5 consecutive pairs
    # (pairs cross episode boundaries — this is by design for simple BC)
    assert len(ds) == 5


def test_from_dataset_dir_missing_episodes(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrajectoryDataset.from_dataset_dir(tmp_path)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_input_target_dim(tmp_path):
    trace = _make_trace(n_steps=3, n_joints=4)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_trace_file(
        p, joint_names=["a", "b", "c", "d"]
    )
    assert ds.input_dim == 4
    assert ds.target_dim == 4


def test_target_dim_with_actions(tmp_path):
    # Actions can have different dim than positions
    trace = _make_trace(n_steps=3, n_joints=3)
    # Override actions to have 5 dims
    for step in trace["steps"]:
        step["action"] = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_trace_file(
        p, joint_names=["a", "b", "c"], include_actions=True
    )
    assert ds.input_dim == 3
    assert ds.target_dim == 5


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


def test_dataloader_batch(tmp_path):
    trace = _make_trace(n_steps=10, n_joints=3)
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    ds = TrajectoryDataset.from_trace_file(p, joint_names=["a", "b", "c"])
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    batch_pos, batch_target = next(iter(loader))
    assert batch_pos.shape == (4, 3)
    assert batch_target.shape == (4, 3)


# ---------------------------------------------------------------------------
# Real trace file (integration, skipped if not available)
# ---------------------------------------------------------------------------


def test_load_real_trace():
    from pathlib import Path

    trace_path = Path("output/arm_pick_dataset/dry_run_trace.json")
    if not trace_path.exists():
        pytest.skip("Real trace not available")

    ds = TrajectoryDataset.from_trace_file(trace_path)
    assert len(ds) > 0
    pos, target = ds[0]
    assert pos.shape[0] == ds.encoder.dof
