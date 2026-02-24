# Implementation Log

## Task 1: Create `python/examples/export_sb3_to_onnx.py`

Completed: 2026-02-24T15:55:00

### Changes

- `python/examples/export_sb3_to_onnx.py`: Created export script with the following structure:
  - `PolicyWrapper(nn.Module)`: Wraps SB3 policy to extract the deterministic actor output (Gaussian mean) and clips to action space bounds. Uses `policy.get_distribution(obs).distribution.mean` followed by `torch.clamp()` to replicate `model.predict(obs, deterministic=True)` exactly.
  - `export_to_onnx()`: Loads SB3 PPO model, wraps actor, exports via `torch.onnx.export` with dynamic batch axes and opset 17.
  - `add_clanker_metadata()`: Embeds 9 metadata entries per POLICY_ONNX_SPEC.md (clanker_policy_version, action_space, action_transform, action_scale, action_offset, training_framework, recurrent, deterministic_mode, batch_inference).
  - `validate_onnx()`: Runs `onnx.checker.check_model`, verifies tensor names, metadata presence, single inference shape, and batch inference shape.
  - `cross_check()`: Compares 10 random observations between SB3 `predict()` and ONNX Runtime, asserts max diff < 1e-5.
  - `main()`: argparse CLI with `--model`, `--output`, `--opset` flags.

### Verification

- [x] Syntax check passes: `py -3.12 -c "import ast; ast.parse(...)"`
- [x] Follows project code style: docstring, `sys.path.insert(0, "python")`, `main()` entry point, progress printing

### Notes

- The PLAN.md initially suggested using `get_distribution(obs).distribution.mean` directly without clipping, expecting it would match SB3's `predict()`. However, SB3's `predict(deterministic=True)` clips the Gaussian mean to action space bounds. The initial run showed max diff of 1.89 because raw means can exceed `[-1, 1]`. The fix was to add `torch.clamp(raw_action, act_low, act_high)` in the wrapper, making the ONNX output match `predict()` exactly (max diff 5.96e-08).
- The model has `squash_output=False` (no tanh squashing), so `action_transform="none"` is correct since clipping is baked into the ONNX graph.
- The trained model has obs_dim=4 and act_dim=2 (not 1 as some plan notes assumed).
- Installed `onnx` and `onnxruntime` pip packages (were not previously installed in the Python 3.12 environment).

---

## Task 2: Run the export script to generate `cartpole_ppo.onnx`

Completed: 2026-02-24T15:55:30

### Changes

- `python/examples/cartpole_ppo.onnx`: Generated ONNX model file (21,204 bytes)

### Verification

- [x] Script exits with code 0
- [x] `onnx.checker.check_model()` passes
- [x] Input tensor named `obs` with shape `[batch, 4]` (float32)
- [x] Output tensor named `action` with shape `[batch, 2]` (float32)
- [x] Dynamic batch axes set for both input and output
- [x] Single inference: output shape (1, 2) -- OK
- [x] Batch inference: output shape (4, 2) -- OK
- [x] Cross-check: max absolute difference 5.96e-08 < 1e-5 -- PASS

### Notes

- Full console output confirmed all validation steps passed on first run (after the clipping fix in Task 1).

---

## Task 3: Validate the exported model metadata

Completed: 2026-02-24T15:56:00

### Changes

- No file changes (inspection only)

### Verification

- [x] `clanker_policy_version`: `"1.0.0"` -- correct
- [x] `action_space`: `{"type": "Box", "shape": [2], "dtype": "float32", "low": [-1.0, -1.0], "high": [1.0, 1.0]}` -- correct
- [x] `action_transform`: `"none"` -- correct (clipping baked into ONNX graph)
- [x] `action_scale`: `[1.0, 1.0]` -- correct
- [x] `action_offset`: `[0.0, 0.0]` -- correct
- [x] `training_framework`: `"stable-baselines3"` -- correct
- [x] `recurrent`: `"false"` -- correct
- [x] `deterministic_mode`: `"false"` -- correct
- [x] `batch_inference`: `"true"` -- correct
- [x] Input tensor: `obs` shape `['batch', 4]` -- correct
- [x] Output tensor: `action` shape `['batch', 2]` -- correct
- [x] File size: 21,204 bytes (reasonable for 64x64 MLP with 4 inputs and 2 outputs)

### Notes

- The model's action dimension is 2, not 1 as some plan notes assumed. The cartpole environment uses a 2D continuous action space with bounds `[-1, 1]` per dimension.
- The `.onnx` file is small enough (21 KB) to commit to the repository for use by Loop 04 (viz binary).

---
