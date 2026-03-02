"""Replay a trained policy against a ground-truth trace for comparison.

Runs the model in open-loop rollout mode: starting from the initial joint
positions, feed predictions back as the next input.  Compares predicted
trajectory against ground truth at the joint level.

Outputs:
1. A comparison JSON with per-step GT and predicted joint positions.
2. A trace JSON compatible with ``arm_pick_replay`` (Rust binary).
3. Optional matplotlib plots showing per-joint GT vs predicted.

Usage::

    # Using ONNX model
    py -3.12 python/examples/replay_policy.py \\
        --model joint_bc.onnx \\
        --trace output/arm_pick_dataset/dry_run_trace.json

    # Using PyTorch checkpoint (needs encoder JSON)
    py -3.12 python/examples/replay_policy.py \\
        --model joint_bc.pt \\
        --encoder joint_bc_encoder.json \\
        --trace output/arm_pick_dataset/dry_run_trace.json

    # With scene spec and matplotlib plot
    py -3.12 python/examples/replay_policy.py \\
        --model joint_bc.onnx \\
        --trace output/arm_pick_dataset/dry_run_trace.json \\
        --scene python/clankers_synthetic/scenes/arm_pick_cube.json \\
        --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "python")

from clankers.joint_encoder import JointEncoder

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_onnx_model(path: str) -> tuple:
    """Load ONNX model, return (inference_fn, encoder_or_None, metadata_dict)."""
    import onnxruntime as ort

    session = ort.InferenceSession(path)
    input_name = session.get_inputs()[0].name

    def infer(x: np.ndarray) -> np.ndarray:
        return session.run(None, {input_name: x.astype(np.float32)})[0]

    # Try to extract encoder and metadata from ONNX
    encoder = None
    model_meta: dict[str, str] = {}
    meta = session.get_modelmeta()
    if meta.custom_metadata_map:
        model_meta = dict(meta.custom_metadata_map)
        enc_json = model_meta.get("clanker_joint_encoder")
        if enc_json:
            encoder = JointEncoder.from_json(enc_json)

    input_dim = session.get_inputs()[0].shape[1]
    output_dim = session.get_outputs()[0].shape[1]
    mode = model_meta.get("clanker_prediction_mode", "position")
    print(f"ONNX model: input_dim={input_dim}, output_dim={output_dim}, mode={mode}")

    return infer, encoder, model_meta


def load_pt_model(path: str, encoder: JointEncoder) -> tuple:
    """Load PyTorch checkpoint, return (inference_fn, encoder, metadata_dict)."""
    import torch

    # Import the model class from the trainer
    sys.path.insert(0, "python/examples")
    from train_joint_bc import JointMLP

    state_dict = torch.load(path, weights_only=True)
    # Infer dims from first/last layer
    weight_keys = [k for k in state_dict if "weight" in k]
    first_key = weight_keys[0]
    last_key = weight_keys[-1]
    input_dim = state_dict[first_key].shape[1]
    output_dim = state_dict[last_key].shape[0]

    model = JointMLP(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(state_dict)
    model.eval()

    def infer(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return model(t).numpy()

    print(f"PyTorch model: input_dim={input_dim}, output_dim={output_dim}")
    return infer, encoder, {}


# ---------------------------------------------------------------------------
# Trace extraction
# ---------------------------------------------------------------------------


def extract_positions(trace_data: dict, encoder: JointEncoder) -> list[np.ndarray]:
    """Extract joint position vectors from trace steps."""
    n_joints = encoder.dof
    positions = []
    for step in trace_data["steps"]:
        obs = step.get("obs", [])
        if len(obs) >= 2 * n_joints:
            pos = [obs[i * 2] for i in range(n_joints)]
            positions.append(np.array(pos, dtype=np.float32))
    return positions


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def open_loop_rollout(
    infer_fn,
    initial_pos: np.ndarray,
    n_steps: int,
    mode: str = "position",
    control_dt: float = 0.02,
) -> list[np.ndarray]:
    """Run model in open-loop: pos → model → next_pos → model → ...

    In velocity mode, the model predicts joint velocities (rad/s) and
    positions are integrated: pos_{t+1} = pos_t + vel_predicted * dt.
    """
    predictions = [initial_pos.copy()]
    current = initial_pos.copy().reshape(1, -1)

    for _ in range(n_steps - 1):
        pred = infer_fn(current)
        if mode == "velocity":
            # Integrate: pos_{t+1} = pos_t + vel * dt
            next_pos = current[0] + pred[0] * control_dt
            predictions.append(next_pos.copy())
            current = next_pos.reshape(1, -1)
        else:
            # Position mode: prediction IS the next position
            predictions.append(pred[0].copy())
            current = pred

    return predictions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_comparison_json(
    gt_positions: list[np.ndarray],
    pred_positions: list[np.ndarray],
    encoder: JointEncoder,
    path: str,
) -> None:
    """Write GT vs predicted comparison to JSON."""
    n = min(len(gt_positions), len(pred_positions))
    data = {
        "joint_names": list(encoder.names),
        "n_steps": n,
        "gt_positions": [pos.tolist() for pos in gt_positions[:n]],
        "pred_positions": [pos.tolist() for pos in pred_positions[:n]],
        "per_step_mse": [
            float(np.mean((gt_positions[i] - pred_positions[i]) ** 2)) for i in range(n)
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Comparison JSON written to {path}")


def write_replay_trace(
    trace_data: dict,
    pred_positions: list[np.ndarray],
    encoder: JointEncoder,
    path: str,
) -> None:
    """Write a trace JSON compatible with arm_pick_replay.

    Copies body_poses from ground truth (since we don't have FK), but marks
    the trace as a prediction and adds predicted joint positions to info.
    """
    steps = trace_data["steps"]
    n = min(len(steps), len(pred_positions))

    output_steps = []
    for i in range(n):
        step = steps[i]
        info = dict(step.get("info", {}))
        info["predicted_joint_positions"] = dict(
            zip(encoder.names, pred_positions[i].tolist(), strict=True)
        )
        # Copy body_poses from GT for visualization
        output_steps.append(
            {
                "obs": pred_positions[i].tolist(),
                "action": step.get("action", []),
                "next_obs": (
                    pred_positions[i + 1].tolist()
                    if i + 1 < len(pred_positions)
                    else pred_positions[i].tolist()
                ),
                "reward": step.get("reward", 0.0),
                "terminated": step.get("terminated", False),
                "truncated": step.get("truncated", False),
                "info": info,
            }
        )

    output = {
        "plan_id": f"{trace_data.get('plan_id', 'unknown')}_predicted",
        "steps": output_steps,
    }
    Path(path).write_text(json.dumps(output, indent=2))
    print(f"Replay trace written to {path}")


def plot_comparison(
    gt_positions: list[np.ndarray],
    pred_positions: list[np.ndarray],
    encoder: JointEncoder,
) -> None:
    """Matplotlib plot: per-joint GT vs predicted overlaid."""
    import matplotlib.pyplot as plt

    n = min(len(gt_positions), len(pred_positions))
    gt = np.array(gt_positions[:n])
    pred = np.array(pred_positions[:n])
    t = np.arange(n)

    n_joints = encoder.dof
    cols = 2
    rows = (n_joints + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=True)
    axes = axes.flatten()

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t, gt[:, j], label="GT", linewidth=1.5)
        ax.plot(t, pred[:, j], label="Pred", linewidth=1.5, linestyle="--")
        ax.set_title(encoder.names[j])
        ax.set_ylabel("rad")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_joints, len(axes)):
        axes[j].set_visible(False)

    axes[-2].set_xlabel("Step")
    if n_joints % 2 == 0:
        axes[-1].set_xlabel("Step")

    fig.suptitle("Joint Position: Ground Truth vs Policy Prediction", fontsize=14)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a trained policy against ground-truth trace data"
    )
    parser.add_argument("--model", required=True, help="Path to .onnx or .pt model")
    parser.add_argument("--trace", required=True, help="Path to ground-truth trace JSON")
    parser.add_argument(
        "--output", default=None, help="Output replay trace JSON (default: <trace>_predicted.json)"
    )
    parser.add_argument("--scene", default=None, help="Scene spec JSON for joint names")
    parser.add_argument(
        "--encoder", default=None, help="Encoder JSON path (required for .pt models without scene)"
    )
    parser.add_argument("--plot", action="store_true", help="Show matplotlib comparison plot")
    args = parser.parse_args()

    # Load model
    model_path = args.model
    model_meta: dict[str, str] = {}
    if model_path.endswith(".onnx"):
        infer_fn, model_encoder, model_meta = load_onnx_model(model_path)
    elif model_path.endswith(".pt"):
        if args.encoder:
            enc = JointEncoder.load(args.encoder)
        elif args.scene:
            enc = JointEncoder.from_scene_spec(args.scene)
        else:
            parser.error(".pt model requires --encoder or --scene for joint names")
            return
        infer_fn, model_encoder, model_meta = load_pt_model(model_path, enc)
    else:
        parser.error("Model must be .onnx or .pt")
        return

    prediction_mode = model_meta.get("clanker_prediction_mode", "position")
    control_dt = float(model_meta.get("clanker_control_dt", "0.02"))
    print(f"Prediction mode: {prediction_mode}, control_dt: {control_dt}")

    # Load trace
    trace_path = Path(args.trace)
    trace_data = json.loads(trace_path.read_text())
    print(f"Trace: {len(trace_data['steps'])} steps")

    # Resolve encoder
    encoder = model_encoder
    if encoder is None:
        if args.scene:
            encoder = JointEncoder.from_scene_spec(args.scene)
        elif args.encoder:
            encoder = JointEncoder.load(args.encoder)
        else:
            # Auto-detect from obs dim
            obs_len = len(trace_data["steps"][0].get("obs", []))
            n_joints = obs_len // 2
            encoder = JointEncoder([f"joint_{i}" for i in range(n_joints)])

    print(f"Encoder: {encoder}")

    # Extract GT positions
    gt_positions = extract_positions(trace_data, encoder)
    print(f"Extracted {len(gt_positions)} GT position vectors")

    # Open-loop rollout
    pred_positions = open_loop_rollout(
        infer_fn, gt_positions[0], len(gt_positions),
        mode=prediction_mode, control_dt=control_dt,
    )

    # Compute overall MSE
    n = min(len(gt_positions), len(pred_positions))
    mse_per_step = [float(np.mean((gt_positions[i] - pred_positions[i]) ** 2)) for i in range(n)]
    print(f"Open-loop rollout: {n} steps")
    print(f"  Final step MSE: {mse_per_step[-1]:.6f}")
    print(f"  Mean MSE: {np.mean(mse_per_step):.6f}")
    print(f"  Max MSE: {np.max(mse_per_step):.6f}")

    # Write comparison JSON
    comparison_path = str(trace_path).replace(".json", "_comparison.json")
    write_comparison_json(gt_positions, pred_positions, encoder, comparison_path)

    # Write replay trace
    output_path = args.output or str(trace_path).replace(".json", "_predicted.json")
    write_replay_trace(trace_data, pred_positions, encoder, output_path)

    # Optional plot
    if args.plot:
        plot_comparison(gt_positions, pred_positions, encoder)


if __name__ == "__main__":
    main()
