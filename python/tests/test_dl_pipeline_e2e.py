"""End-to-end integration tests for the DL pipeline.

Tests the data flow: trace JSON → TrajectoryDataset → train MLP → export ONNX → load & infer.
No gym server required — uses synthetic trace data.
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "python/examples")

from clankers.joint_encoder import JointEncoder
from clankers.trajectory_dataset import TrajectoryDataset


def _make_trace(n_steps: int = 50, n_joints: int = 6, dt: float = 0.02) -> dict:
    """Generate a synthetic trace with smooth sinusoidal motion."""
    steps = []
    for t in range(n_steps):
        phase = t * dt
        pos = [0.1 * np.sin(phase * (j + 1)) for j in range(n_joints)]
        vel = [0.1 * (j + 1) * np.cos(phase * (j + 1)) for j in range(n_joints)]
        # Interleaved obs: [pos0, vel0, pos1, vel1, ...]
        obs = []
        for p, v in zip(pos, vel):
            obs.extend([p, v])
        # Action = target position (same as next step's position in smooth motion)
        action = [0.1 * np.sin((phase + dt) * (j + 1)) for j in range(n_joints)]
        steps.append({
            "obs": obs,
            "action": action,
            "next_obs": obs,  # simplified
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        })
    return {"plan_id": "test_trace", "steps": steps}


JOINT_NAMES = [f"joint_{i}" for i in range(6)]


@pytest.fixture
def trace_file(tmp_path: Path) -> Path:
    """Write a synthetic trace to a temp file."""
    trace = _make_trace()
    path = tmp_path / "test_trace.json"
    path.write_text(json.dumps(trace))
    return path


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a multi-episode dataset directory."""
    ep_dir = tmp_path / "episodes"
    ep_dir.mkdir()
    for i in range(3):
        trace = _make_trace(n_steps=30)
        trace["plan_id"] = f"ep_{i}"
        (ep_dir / f"ep_{i:06d}.json").write_text(json.dumps(trace))
    return tmp_path


class TestTraceToDataset:
    """Test: trace JSON → TrajectoryDataset loading."""

    def test_position_mode_shape(self, trace_file: Path):
        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES
        )
        assert len(ds) == 49  # 50 steps → 49 consecutive pairs
        assert ds.input_dim == 6
        assert ds.target_dim == 6

    def test_velocity_mode_shape(self, trace_file: Path):
        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
            include_velocities=True, control_dt=0.02,
        )
        assert len(ds) == 49
        pos, vel = ds[0]
        assert pos.shape == (6,)
        assert vel.shape == (6,)

    def test_velocity_values_correct(self, trace_file: Path):
        ds_pos = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
        )
        ds_vel = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
            include_velocities=True, control_dt=0.02,
        )
        # vel = (pos_{t+1} - pos_t) / dt
        pos_t, pos_next = ds_pos[0]
        _, vel_t = ds_vel[0]
        expected_vel = (pos_next - pos_t) / 0.02
        assert torch.allclose(vel_t, expected_vel, atol=1e-5)

    def test_action_mode_shape(self, trace_file: Path):
        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
            include_actions=True,
        )
        assert len(ds) == 50  # action mode: one per step
        assert ds.target_dim == 6

    def test_dataset_dir_loading(self, dataset_dir: Path):
        ds = TrajectoryDataset.from_dataset_dir(
            dataset_dir, joint_names=JOINT_NAMES,
        )
        # 3 episodes × 30 steps concatenated = 90 total steps → 89 pairs
        assert len(ds) == 89

    def test_mutual_exclusivity(self, trace_file: Path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TrajectoryDataset.from_trace_file(
                trace_file, joint_names=JOINT_NAMES,
                include_actions=True, include_velocities=True,
            )


class TestTrainAndExport:
    """Test: dataset → train MLP → export ONNX → reload."""

    def test_train_position_mode(self, trace_file: Path):
        from train_joint_bc import JointMLP, train

        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES
        )
        model = JointMLP(input_dim=ds.input_dim, output_dim=ds.target_dim, hidden=32)
        trained = train(ds, model, epochs=5, batch_size=16)

        # Verify model produces output of correct shape
        with torch.no_grad():
            dummy = torch.randn(1, ds.input_dim)
            out = trained(dummy)
            assert out.shape == (1, ds.target_dim)

    def test_train_velocity_mode(self, trace_file: Path):
        from train_joint_bc import JointMLP, train

        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
            include_velocities=True, control_dt=0.02,
        )
        model = JointMLP(input_dim=ds.input_dim, output_dim=ds.target_dim, hidden=32)
        trained = train(ds, model, epochs=5, batch_size=16)

        with torch.no_grad():
            dummy = torch.randn(1, ds.input_dim)
            out = trained(dummy)
            assert out.shape == (1, ds.target_dim)

    def test_onnx_export_and_reload(self, trace_file: Path, tmp_path: Path):
        from train_joint_bc import JointMLP, export_onnx, train

        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES
        )
        model = JointMLP(input_dim=ds.input_dim, output_dim=ds.target_dim, hidden=32)
        trained = train(ds, model, epochs=5, batch_size=16)

        onnx_path = str(tmp_path / "test_model.onnx")
        export_onnx(trained, ds, onnx_path, mode="position", control_dt=0.02)

        # Reload and verify
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Check metadata
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
        assert meta.get("clanker_prediction_mode") == "position"
        assert meta.get("clanker_control_dt") == "0.02"
        assert "clanker_joint_encoder" in meta

        # Check inference
        dummy = np.random.randn(1, ds.input_dim).astype(np.float32)
        result = session.run(None, {input_name: dummy})[0]
        assert result.shape == (1, ds.target_dim)

    def test_onnx_velocity_metadata(self, trace_file: Path, tmp_path: Path):
        from train_joint_bc import JointMLP, export_onnx, train

        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES,
            include_velocities=True, control_dt=0.02,
        )
        model = JointMLP(input_dim=ds.input_dim, output_dim=ds.target_dim, hidden=32)
        trained = train(ds, model, epochs=3, batch_size=16)

        onnx_path = str(tmp_path / "vel_model.onnx")
        export_onnx(trained, ds, onnx_path, mode="velocity", control_dt=0.02)

        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
        assert meta.get("clanker_prediction_mode") == "velocity"


class TestEncoderRoundTrip:
    """Test: encoder save/load through ONNX metadata."""

    def test_encoder_in_onnx(self, trace_file: Path, tmp_path: Path):
        from train_joint_bc import JointMLP, export_onnx, train

        ds = TrajectoryDataset.from_trace_file(
            trace_file, joint_names=JOINT_NAMES
        )
        model = JointMLP(input_dim=ds.input_dim, output_dim=ds.target_dim, hidden=32)
        trained = train(ds, model, epochs=3, batch_size=16)

        onnx_path = str(tmp_path / "enc_model.onnx")
        export_onnx(trained, ds, onnx_path)

        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        meta = dict(session.get_modelmeta().custom_metadata_map or {})
        recovered_encoder = JointEncoder.from_json(meta["clanker_joint_encoder"])
        assert recovered_encoder == ds.encoder

    def test_encoder_save_load_file(self, tmp_path: Path):
        encoder = JointEncoder(JOINT_NAMES)
        path = str(tmp_path / "encoder.json")
        encoder.save(path)
        loaded = JointEncoder.load(path)
        assert loaded == encoder


class TestOpenLoopReplay:
    """Test: trained model open-loop rollout matches expectations."""

    def test_position_rollout_shape(self, trace_file: Path):
        from replay_policy import extract_positions, open_loop_rollout

        trace_data = json.loads(trace_file.read_text())
        encoder = JointEncoder(JOINT_NAMES)
        gt_positions = extract_positions(trace_data, encoder)

        assert len(gt_positions) == 50

        # Identity model (returns input as output)
        def identity_fn(x):
            return x

        preds = open_loop_rollout(identity_fn, gt_positions[0], 10)
        assert len(preds) == 10
        # With identity model, all predictions should be the same as initial
        for p in preds:
            assert np.allclose(p, gt_positions[0])

    def test_velocity_rollout_integration(self, trace_file: Path):
        from replay_policy import open_loop_rollout

        initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        dt = 0.02

        # Constant velocity model: always predict 1.0 rad/s per joint
        def const_vel_fn(x):
            return np.ones_like(x)

        preds = open_loop_rollout(const_vel_fn, initial, 5, mode="velocity", control_dt=dt)
        assert len(preds) == 5

        # After 4 integration steps at 1.0 rad/s with dt=0.02:
        # pos = 4 * 1.0 * 0.02 = 0.08
        expected_final = np.full(6, 4 * dt)
        assert np.allclose(preds[4], expected_final, atol=1e-6)
