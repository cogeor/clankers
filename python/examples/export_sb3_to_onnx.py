"""Export a Stable-Baselines3 PPO model to ONNX with clanker metadata.

Converts the SB3 actor (policy) network to ONNX format suitable for
Rust-side inference via the clankers-policy OnnxPolicy.

Usage:
    py -3.12 python/examples/export_sb3_to_onnx.py
    py -3.12 python/examples/export_sb3_to_onnx.py --model path/to/model.zip --output path/to/output.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, "python")

from stable_baselines3 import PPO


class PolicyWrapper(torch.nn.Module):
    """Wraps an SB3 policy to export only the deterministic actor output.

    Replicates model.predict(obs, deterministic=True) exactly:
      1. Extract the Gaussian mean from the action distribution
      2. Clip to the action space bounds (same as SB3's post-predict clip)

    With clipping baked in, action_transform="none" is correct because
    the ONNX model output is already in the valid action range.
    """

    def __init__(self, policy, act_low: torch.Tensor, act_high: torch.Tensor):
        super().__init__()
        self.policy = policy
        self.register_buffer("act_low", act_low)
        self.register_buffer("act_high", act_high)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # get_distribution runs: features_extractor -> mlp_extractor -> action_net
        # .distribution.mean gives the raw Gaussian mean (pre-clip).
        # SB3's predict(deterministic=True) returns this mean clipped to the
        # action space bounds, so we replicate that clipping here.
        raw_action = self.policy.get_distribution(obs).distribution.mean
        return torch.clamp(raw_action, self.act_low, self.act_high)


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
) -> None:
    # Load SB3 model (no env needed for export)
    print(f"Loading SB3 model from {model_path}...")
    model = PPO.load(model_path, device="cpu")

    obs_dim = model.observation_space.shape[0]
    act_dim = model.action_space.shape[0]
    act_low = model.action_space.low.tolist()
    act_high = model.action_space.high.tolist()

    print(f"  obs_dim:  {obs_dim}")
    print(f"  act_dim:  {act_dim}")
    print(f"  act_low:  {act_low}")
    print(f"  act_high: {act_high}")

    # Wrap for deterministic forward pass (includes action clipping)
    act_low_t = torch.tensor(act_low, dtype=torch.float32)
    act_high_t = torch.tensor(act_high, dtype=torch.float32)
    wrapper = PolicyWrapper(model.policy, act_low_t, act_high_t)
    wrapper.eval()

    # Dummy input for tracing
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        wrapper,
        (dummy_obs,),
        output_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=opset_version,
    )
    print(f"  Exported to {output_path}")

    # Add clanker metadata
    add_clanker_metadata(output_path, obs_dim, act_dim, act_low, act_high)

    # Validate
    validate_onnx(output_path, obs_dim, act_dim)

    # Cross-check against SB3 predict
    cross_check(model, output_path, obs_dim)


def add_clanker_metadata(
    onnx_path: str,
    obs_dim: int,
    act_dim: int,
    act_low: list[float],
    act_high: list[float],
) -> None:
    print("Adding clanker metadata...")
    onnx_model = onnx.load(onnx_path)

    metadata = {
        "clanker_policy_version": "1.0.0",
        "action_space": json.dumps({
            "type": "Box",
            "shape": [act_dim],
            "dtype": "float32",
            "low": act_low,
            "high": act_high,
        }),
        "action_transform": "none",
        "action_scale": json.dumps([1.0] * act_dim),
        "action_offset": json.dumps([0.0] * act_dim),
        "training_framework": "stable-baselines3",
        "recurrent": "false",
        "deterministic_mode": "false",
        "batch_inference": "true",
    }

    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(onnx_model, onnx_path)
    print(f"  Added {len(metadata)} metadata entries")


def validate_onnx(onnx_path: str, obs_dim: int, act_dim: int) -> None:
    print("Validating ONNX model...")

    # Structural validation
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  onnx.checker: OK")

    # Check tensor names
    input_names = [inp.name for inp in onnx_model.graph.input]
    output_names = [out.name for out in onnx_model.graph.output]
    assert "obs" in input_names, f"Missing 'obs' input, found: {input_names}"
    assert "action" in output_names, f"Missing 'action' output, found: {output_names}"
    print(f"  inputs:  {input_names}")
    print(f"  outputs: {output_names}")

    # Check metadata
    metadata = {p.key: p.value for p in onnx_model.metadata_props}
    assert "clanker_policy_version" in metadata, "Missing clanker_policy_version"
    assert "action_space" in metadata, "Missing action_space"
    assert "action_transform" in metadata, "Missing action_transform"
    print(f"  metadata keys: {list(metadata.keys())}")

    # Runtime inference check
    session = ort.InferenceSession(onnx_path)
    dummy = np.zeros((1, obs_dim), dtype=np.float32)
    outputs = session.run(["action"], {"obs": dummy})
    action = outputs[0]
    assert action.shape == (1, act_dim), f"Expected shape (1, {act_dim}), got {action.shape}"
    print(f"  inference check: OK (output shape {action.shape})")

    # Batch inference check
    batch = np.zeros((4, obs_dim), dtype=np.float32)
    outputs = session.run(["action"], {"obs": batch})
    assert outputs[0].shape == (4, act_dim), f"Batch shape mismatch: {outputs[0].shape}"
    print(f"  batch inference: OK (batch=4, output shape {outputs[0].shape})")


def cross_check(model, onnx_path: str, obs_dim: int, n_samples: int = 10) -> None:
    print("Cross-checking SB3 vs ONNX outputs...")
    session = ort.InferenceSession(onnx_path)
    np.random.seed(42)

    max_diff = 0.0
    for i in range(n_samples):
        obs = np.random.uniform(-1, 1, size=(obs_dim,)).astype(np.float32)

        # SB3 deterministic prediction
        sb3_action, _ = model.predict(obs, deterministic=True)

        # ONNX prediction
        onnx_action = session.run(
            ["action"], {"obs": obs.reshape(1, -1)}
        )[0].flatten()

        diff = np.max(np.abs(sb3_action - onnx_action))
        max_diff = max(max_diff, diff)

        if i < 3:  # Print first few
            print(f"  sample {i}: sb3={sb3_action}, onnx={onnx_action}, diff={diff:.2e}")

    print(f"  max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Cross-check failed: max diff {max_diff} >= 1e-5"
    print("  Cross-check: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SB3 PPO model to ONNX with clanker metadata.",
    )
    parser.add_argument(
        "--model",
        default="python/examples/cartpole_ppo_model.zip",
        help="Path to SB3 model .zip file",
    )
    parser.add_argument(
        "--output",
        default="python/examples/cartpole_ppo.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    print("=== SB3 to ONNX Export ===\n")
    export_to_onnx(args.model, args.output, opset_version=args.opset)
    print("\nDone.")


if __name__ == "__main__":
    main()
