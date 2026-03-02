#!/usr/bin/env python3
"""Closed-loop evaluation of a trained pick policy in arm_pick_gym.

Loads a trained ONNX (or PyTorch) model and runs it closed-loop against
the arm_pick_gym server. Reports success rate, cube heights, and episode
statistics.

The model predicts arm joint targets (6-dim). Gripper control uses a
simple time-based strategy derived from the training demonstrations:
open at start, close at a configurable step, stay closed.

Usage:
    1. Start gym server:
       cargo run -j 24 -p clankers-examples --bin arm_pick_gym

    2. Evaluate:
       py -3.12 python/examples/eval_pick_policy.py \\
           --model joint_bc.onnx \\
           --n-episodes 10

    Options:
       --model PATH       Path to .onnx or .pt model
       --n-episodes N     Number of evaluation episodes (default: 10)
       --port PORT        Gym server port (default: 9880)
       --gripper-close-step N  Step at which to close gripper (default: 120)
       --max-steps N      Max steps per episode (default: 500)
       --control-dt DT    Override control dt (auto from ONNX metadata)
       --verbose          Print per-step info
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "python")

from clankers.joint_encoder import JointEncoder


def load_model(path: str, encoder_path: str | None = None, scene_path: str | None = None):
    """Load model, return (infer_fn, encoder, metadata)."""
    if path.endswith(".onnx"):
        import onnxruntime as ort

        session = ort.InferenceSession(path)
        input_name = session.get_inputs()[0].name

        def infer(x: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: x.astype(np.float32)})[0]

        meta = dict(session.get_modelmeta().custom_metadata_map or {})
        encoder = None
        enc_json = meta.get("clanker_joint_encoder")
        if enc_json:
            encoder = JointEncoder.from_json(enc_json)

        return infer, encoder, meta

    elif path.endswith(".pt"):
        import torch

        sys.path.insert(0, "python/examples")
        from train_joint_bc import JointMLP

        state_dict = torch.load(path, weights_only=True)
        weight_keys = [k for k in state_dict if "weight" in k]
        input_dim = state_dict[weight_keys[0]].shape[1]
        output_dim = state_dict[weight_keys[-1]].shape[0]

        model = JointMLP(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(state_dict)
        model.eval()

        def infer(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(x, dtype=torch.float32)
                return model(t).numpy()

        encoder = None
        if encoder_path:
            encoder = JointEncoder.load(encoder_path)
        elif scene_path:
            encoder = JointEncoder.from_scene_spec(scene_path)

        return infer, encoder, {}

    else:
        raise ValueError(f"Unsupported model format: {path}")


def extract_arm_positions(obs: np.ndarray, n_arm: int = 6) -> np.ndarray:
    """Extract arm joint positions from interleaved obs [pos0, vel0, ...]."""
    return np.array([obs[i * 2] for i in range(n_arm)], dtype=np.float32)


def make_action(
    arm_targets: np.ndarray,
    gripper_width: float,
    n_gripper: int = 2,
) -> np.ndarray:
    """Build full 8-dim action from arm targets + gripper width."""
    gripper = np.full(n_gripper, gripper_width, dtype=np.float32)
    return np.concatenate([arm_targets, gripper])


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation of a trained pick policy"
    )
    parser.add_argument("--model", required=True, help="Path to .onnx or .pt model")
    parser.add_argument("--encoder", default=None, help="Encoder JSON (for .pt models)")
    parser.add_argument("--scene", default=None, help="Scene spec JSON for joint names")
    parser.add_argument("--host", default="127.0.0.1", help="Gym server host")
    parser.add_argument("--port", type=int, default=9880, help="Gym server port")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument(
        "--gripper-close-step",
        type=int,
        default=120,
        help="Step at which to close gripper (default: 120)",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--control-dt", type=float, default=None, help="Override control dt")
    parser.add_argument("--gripper-open", type=float, default=0.03, help="Gripper open width")
    parser.add_argument("--gripper-closed", type=float, default=0.0, help="Gripper closed width")
    parser.add_argument("--verbose", action="store_true", help="Print per-step info")
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    from clankers.env import ClankerEnv

    print("=" * 60)
    print("  Closed-Loop Pick Policy Evaluation")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {args.model}")
    infer_fn, encoder, model_meta = load_model(args.model, args.encoder, args.scene)

    prediction_mode = model_meta.get("clanker_prediction_mode", "position")
    control_dt = args.control_dt or float(model_meta.get("clanker_control_dt", "0.02"))
    output_dim = int(model_meta.get("clanker_target_dim", "6"))
    print(f"  Mode: {prediction_mode}, control_dt: {control_dt}, output_dim: {output_dim}")
    if encoder:
        print(f"  Encoder: {encoder}")

    # Determine arm DOF from model
    n_arm = output_dim
    n_gripper = 2
    success_threshold = 0.525

    # Run episodes
    print(f"\nRunning {args.n_episodes} episodes (max {args.max_steps} steps each)")
    print(f"  Gripper closes at step {args.gripper_close_step}")
    print(f"  Server: {args.host}:{args.port}")
    print()

    results = []
    t_start = time.time()

    for ep in range(args.n_episodes):
        env = ClankerEnv(host=args.host, port=args.port)
        env.connect()
        obs, info = env.reset()

        current_pos = extract_arm_positions(obs, n_arm)
        success = False
        cube_z = 0.0
        terminated = False
        truncated = False
        step = 0

        for step in range(args.max_steps):
            if terminated or truncated:
                break

            # Gripper control: open until gripper_close_step, then closed
            if step < args.gripper_close_step:
                gripper_w = args.gripper_open
            else:
                gripper_w = args.gripper_closed

            # Policy prediction
            model_input = current_pos.reshape(1, -1)
            prediction = infer_fn(model_input)[0]

            if prediction_mode == "velocity":
                # Integrate: next_pos = current_pos + vel * dt
                arm_targets = current_pos + prediction * control_dt
            else:
                # Position or action mode: use prediction directly
                arm_targets = prediction

            arm_targets = arm_targets.astype(np.float32)
            action = make_action(arm_targets, gripper_w, n_gripper)

            obs, terminated, truncated, info = env.step(action)
            current_pos = extract_arm_positions(obs, n_arm)

            # Check cube height
            body_poses = info.get("body_poses", {})
            if "red_cube" in body_poses:
                cube_z = body_poses["red_cube"][2]
                if cube_z >= success_threshold:
                    success = True

            if args.verbose and step % 50 == 0:
                print(
                    f"  ep={ep} step={step}: cube_z={cube_z:.4f} "
                    f"arm_targets={arm_targets[:3].tolist()}"
                )

        env.close()

        status = "OK" if success else "FAIL"
        ep_result = {
            "episode": ep,
            "success": success,
            "cube_z": cube_z,
            "steps": step + 1,
        }
        results.append(ep_result)
        print(f"  Episode {ep:3d}: {status}  cube_z={cube_z:.4f}  steps={step + 1}")

    elapsed = time.time() - t_start

    # Summary
    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)
    success_rate = n_success / max(n_total, 1)
    cube_heights = [r["cube_z"] for r in results]
    step_counts = [r["steps"] for r in results]

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Prediction mode:    {prediction_mode}")
    print(f"  Total episodes:     {n_total}")
    print(f"  Successful:         {n_success}")
    print(f"  Success rate:       {success_rate:.1%}")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    if cube_heights:
        print(f"  Cube z (mean):      {sum(cube_heights) / len(cube_heights):.4f}")
        print(f"  Cube z (max):       {max(cube_heights):.4f}")
    if step_counts:
        print(f"  Steps (mean):       {sum(step_counts) / len(step_counts):.0f}")
    print("=" * 60)

    # Save results
    if args.output:
        report = {
            "model": args.model,
            "prediction_mode": prediction_mode,
            "n_total": n_total,
            "n_success": n_success,
            "success_rate": success_rate,
            "elapsed_seconds": elapsed,
            "gripper_close_step": args.gripper_close_step,
            "episodes": results,
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
