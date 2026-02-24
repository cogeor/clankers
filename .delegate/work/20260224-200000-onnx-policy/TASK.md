# Task: ONNX Policy Loading for Rust-Side Inference

## Context

Trained policies (SB3 PPO) are currently `.zip` files containing PyTorch weights. Inference
runs in Python (`cartpole_play_ppo.py`). To visualize trained policies in the Rust sim
(like `pendulum_viz`) without Python, we need ONNX-format model loading in Rust.

The `Policy` trait exists in `clankers-core` (`fn get_action(&self, obs: &Observation) -> Action`).
`clankers-policy` crate has implementations (Zero, Constant, Random, Scripted) and `PolicyRunner`
resource that drives the decide phase via `policy_decide_system` in `ClankersSet::Decide`.

A detailed ONNX model contract spec exists at `.delegate/doc/spec/POLICY_ONNX_SPEC.md`.

## Requirements

1. **`OnnxPolicy` in `clankers-policy`** behind `onnx` feature flag:
   - New module `src/onnx.rs`
   - `OnnxPolicy` struct holding `ort::Session` + action_dim
   - Implements `Policy` trait: loads `.onnx` file, runs inference on observation
   - Start with basic MLP vector policy (no recurrent, no image obs)
   - Handle continuous and discrete action spaces
   - Add `ort` as optional workspace dependency

2. **Python export script** `python/examples/export_sb3_to_onnx.py`:
   - Load SB3 `.zip` model
   - Export to `.onnx` via `torch.onnx.export`
   - Embed clanker metadata (action_space, action_transform, version) per spec
   - Export the existing `cartpole_ppo_model.zip` as validation

3. **Viz example binary** `examples/src/bin/cartpole_policy_viz.rs`:
   - Accept `--model path/to/model.onnx` CLI arg
   - Load `OnnxPolicy::from_file(path)`
   - Insert `PolicyRunner` + `ClankersPolicyPlugin`
   - Reuse visual setup from `pendulum_viz.rs`
   - Policy drives joints automatically each frame

4. **Tests**: Unit tests for OnnxPolicy with a small test ONNX model

## Acceptance Criteria

- `OnnxPolicy` loads an ONNX model and produces correct actions
- Python export script converts SB3 model to valid ONNX
- Viz binary shows trained policy balancing cart-pole without Python
- Feature flag: `cargo build` without `onnx` feature still works
- All existing tests pass
