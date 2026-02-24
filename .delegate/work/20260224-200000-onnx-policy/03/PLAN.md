# Loop 03: Create Python export script to convert SB3 models to ONNX with clanker metadata

## Overview

This loop creates `python/examples/export_sb3_to_onnx.py`, a standalone script that loads a Stable-Baselines3 PPO model (`.zip`), extracts the deterministic actor network, exports it to ONNX format via `torch.onnx.export`, embeds clanker metadata per `POLICY_ONNX_SPEC.md`, and validates the result with `onnxruntime`. The script is then run to produce `python/examples/cartpole_ppo.onnx` from the existing `cartpole_ppo_model.zip`.

The exported ONNX model will be compatible with the `OnnxPolicy` struct implemented in Loop 02 and will be consumed by the viz binary in Loop 04.

## Tasks

### Task 1: Create `python/examples/export_sb3_to_onnx.py`

**Goal:** Write a self-contained Python script that converts any SB3 PPO continuous-action model to an ONNX file with clanker metadata. The script follows the existing code style seen in `cartpole_train_ppo.py` and `cartpole_play_ppo.py` (docstring, `sys.path` insert, `main()` entry point, progress printing).

**Files:**
| Action | Path |
|--------|------|
| CREATE | `python/examples/export_sb3_to_onnx.py` |

**Steps:**

1. Add module docstring and imports. Follow the pattern from existing example scripts:
   ```python
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
   from stable_baselines3 import PPO
   ```

2. Define `PolicyWrapper` class that extracts the deterministic (mean) action from the SB3 policy. SB3's `MlpPolicy` for continuous actions internally applies tanh squashing through its `get_distribution` method. We want the post-squash output (the deterministic mean action), which is what `model.predict(obs, deterministic=True)` returns:

   ```python
   class PolicyWrapper(torch.nn.Module):
       """Wraps an SB3 policy to export only the deterministic actor output."""

       def __init__(self, policy):
           super().__init__()
           self.policy = policy

       def forward(self, obs: torch.Tensor) -> torch.Tensor:
           # get_distribution returns an SB3 Distribution object.
           # For continuous actions with squashing, .distribution.mean
           # gives the post-tanh deterministic action in [-1, 1].
           return self.policy.get_distribution(obs).distribution.mean
   ```

   **Key detail:** `get_distribution(obs)` runs the MLP features extractor and action net, then wraps the result in a `SquashedDiagGaussianDistribution`. The `.distribution.mean` attribute gives the tanh-squashed mean -- exactly the value `model.predict(obs, deterministic=True)` returns. This is the correct output to export because:
   - SB3 applies tanh internally for `Box` action spaces
   - The output is already in `[-1, 1]`
   - The Rust side should use `action_transform: "none"` since no further transform is needed

3. Define `export_to_onnx()` function that performs the core export:

   ```python
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

       # Wrap for deterministic forward pass
       wrapper = PolicyWrapper(model.policy)
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
   ```

4. Define `add_clanker_metadata()` function that loads the ONNX model, adds the required metadata per `POLICY_ONNX_SPEC.md`, and saves it back:

   ```python
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
   ```

   **Rationale for `action_transform: "none"`:** The exported model already outputs post-tanh actions in `[-1, 1]`. For the cartpole env with `Box([-1], [1])`, the action space bounds match the tanh output range, so no further denormalization is needed. The Rust `OnnxPolicy` with `ActionTransform::None` will use the raw output directly.

5. Define `validate_onnx()` function that checks the model loads in onnxruntime and produces correct-shape output:

   ```python
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
   ```

6. Define `cross_check()` function that compares SB3 predictions with ONNX inference on several random observations:

   ```python
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
   ```

7. Define `main()` with argparse for CLI flexibility:

   ```python
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
   ```

**Verify:** `py -3.12 -c "import ast; ast.parse(open('python/examples/export_sb3_to_onnx.py').read())"` succeeds (syntax check without running).

---

### Task 2: Run the export script to generate `cartpole_ppo.onnx`

**Goal:** Execute the export script to produce the ONNX model file from the existing trained SB3 model. This validates the full pipeline end-to-end.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `python/examples/cartpole_ppo.onnx` (generated) |

**Steps:**

1. Run the export script:
   ```bash
   cd /c/Users/costa/src/clankers
   py -3.12 python/examples/export_sb3_to_onnx.py
   ```

2. Expected console output:
   ```
   === SB3 to ONNX Export ===

   Loading SB3 model from python/examples/cartpole_ppo_model.zip...
     obs_dim:  4
     act_dim:  1
     act_low:  [-1.0]
     act_high: [1.0]
   Exporting to ONNX (opset 17)...
     Exported to python/examples/cartpole_ppo.onnx
   Adding clanker metadata...
     Added 8 metadata entries
   Validating ONNX model...
     onnx.checker: OK
     inputs:  ['obs']
     outputs: ['action']
     metadata keys: ['clanker_policy_version', 'action_space', ...]
     inference check: OK (output shape (1, 1))
     batch inference: OK (batch=4, output shape (4, 1))
   Cross-checking SB3 vs ONNX outputs...
     sample 0: sb3=[...], onnx=[...], diff=...
     ...
     max absolute difference: ...
     Cross-check: PASS

   Done.
   ```

3. Verify the output file exists and is reasonable size:
   ```bash
   ls -la python/examples/cartpole_ppo.onnx
   ```
   Expected: A few tens of KB (the MLP is 64x64 with 4 inputs and 1 output, so the model should be small).

**Verify:** The script exits with code 0, `python/examples/cartpole_ppo.onnx` exists, and all assertions in the script pass (structural check, tensor names, metadata, inference shape, cross-check).

---

### Task 3: Validate the exported model metadata and add to `.gitignore` considerations

**Goal:** Manually inspect the exported ONNX model to confirm all clanker metadata is correct and the file is ready for consumption by `OnnxPolicy` (Loop 02) and the viz binary (Loop 04). Decide on whether to commit the `.onnx` binary.

**Files:**
| Action | Path |
|--------|------|
| READ | `python/examples/cartpole_ppo.onnx` (inspect via Python) |

**Steps:**

1. Run a quick inspection script to dump metadata:
   ```bash
   py -3.12 -c "
   import onnx
   m = onnx.load('python/examples/cartpole_ppo.onnx')
   print('Inputs:', [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
   print('Outputs:', [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
   print('Metadata:')
   for p in m.metadata_props:
       print(f'  {p.key}: {p.value}')
   "
   ```

2. Confirm the following metadata matches the ONNX spec requirements:
   - `clanker_policy_version`: `"1.0.0"`
   - `action_space`: `{"type":"Box","shape":[1],"dtype":"float32","low":[-1.0],"high":[1.0]}`
   - `action_transform`: `"none"`
   - `action_scale`: `[1.0]`
   - `action_offset`: `[0.0]`
   - `training_framework`: `"stable-baselines3"`
   - `recurrent`: `"false"`

3. Confirm input tensor is named `"obs"` with shape `[batch, 4]` and output tensor is named `"action"` with shape `[batch, 1]`.

4. The `.onnx` file should be committed to the repo so that Loop 04 (viz binary) can use it without requiring the Python export step. It is small enough (tens of KB) to track in git.

**Verify:** All metadata keys and tensor shapes match the expected values listed above. The file can be loaded by `onnxruntime` in both Python and (once Loop 02 is complete) by `OnnxPolicy::from_file()` in Rust.

---

## Acceptance Criteria

- [ ] `python/examples/export_sb3_to_onnx.py` exists and follows project code style (docstring, argparse, `main()` entry point)
- [ ] Script loads SB3 PPO model from `--model` path (default: `python/examples/cartpole_ppo_model.zip`)
- [ ] Script exports only the deterministic actor network (not the critic) to ONNX
- [ ] Exported ONNX model has input named `"obs"` with shape `[batch, 4]` (float32)
- [ ] Exported ONNX model has output named `"action"` with shape `[batch, 1]` (float32)
- [ ] Dynamic batch axes are set for both input and output tensors
- [ ] All required clanker metadata is embedded: `clanker_policy_version`, `action_space`, `action_transform`, `action_scale`, `action_offset`
- [ ] `action_transform` is `"none"` (SB3 already applies tanh internally)
- [ ] `onnx.checker.check_model()` passes on the exported model
- [ ] `onnxruntime` can load and run inference on the exported model
- [ ] Cross-check between SB3 `model.predict()` and ONNX inference produces max absolute difference < 1e-5
- [ ] `python/examples/cartpole_ppo.onnx` is generated and committed
