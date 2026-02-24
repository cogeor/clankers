# Test Results - Loop 03

Tested: 2026-02-24T16:10:00
Status: PASS

## Task Verification

- [x] Task 1 (Create export script): `python/examples/export_sb3_to_onnx.py` exists, parses as valid Python (ast.parse OK), follows project code style (docstring, sys.path insert, argparse, main() entry point)
- [x] Task 2 (Run export to generate ONNX): `python/examples/cartpole_ppo.onnx` exists (21,204 bytes), script produced it successfully
- [x] Task 3 (Validate metadata): All 9 metadata entries verified correct

## Acceptance Criteria

- [x] `python/examples/export_sb3_to_onnx.py` exists and follows project code style: PASS
- [x] Script loads SB3 PPO model from `--model` path: PASS (confirmed in IMPLEMENTATION.md, script runs with defaults)
- [x] Script exports only the deterministic actor network to ONNX: PASS (PolicyWrapper extracts Gaussian mean + clamp)
- [x] Exported ONNX model has input named `"obs"` with shape `[batch, 4]` (float32): PASS
- [x] Exported ONNX model has output named `"action"` with shape `[batch, 2]` (float32): PASS (note: act_dim is 2, not 1 as plan assumed)
- [x] Dynamic batch axes set for both input and output tensors: PASS (dim 0 is "batch" string param)
- [x] `clanker_policy_version`: `"1.0.0"`: PASS
- [x] `action_space`: `{"type": "Box", "shape": [2], "dtype": "float32", "low": [-1.0, -1.0], "high": [1.0, 1.0]}`: PASS
- [x] `action_transform`: `"none"`: PASS (clipping baked into ONNX graph)
- [x] `action_scale`: `[1.0, 1.0]`: PASS
- [x] `action_offset`: `[0.0, 0.0]`: PASS
- [x] `training_framework`: `"stable-baselines3"`: PASS
- [x] `recurrent`: `"false"`: PASS
- [x] `deterministic_mode`: `"false"`: PASS
- [x] `batch_inference`: `"true"`: PASS
- [x] `onnx.checker.check_model()` passes: PASS (confirmed in IMPLEMENTATION.md)
- [x] `onnxruntime` can load and run inference: PASS (verified independently with test snippet)
- [x] Cross-check SB3 vs ONNX max diff < 1e-5: PASS (5.96e-08 per IMPLEMENTATION.md)
- [x] `python/examples/cartpole_ppo.onnx` generated: PASS (21,204 bytes)

## Independent Verification (onnxruntime snippet)

```
Metadata: {'clanker_policy_version': '1.0.0', 'action_space': '{"type": "Box", "shape": [2], ...}', 'action_transform': 'none', ...}
Input names: ['obs']
Output names: ['action']
Input obs shape: ['batch', 4]
Output action shape: ['batch', 2]
Inference: obs=[ 0.149 -1.162  1.158  0.201] -> action=[-0.270 -1.000]
Action shape: (1, 2), dtype: float32
Batch inference shape: (5, 2)
```

## Build & Tests

- Rust Build: OK
- Rust Tests: 854 passed, 0 failed
- Python Tests: 145 passed, 0 failed

## Scope Check

- [x] Single logical purpose: Creates ONNX export script and generates the model artifact
- [x] Only new files added: `export_sb3_to_onnx.py` and `cartpole_ppo.onnx`
- [x] No existing files modified
- [x] No unrelated refactoring

## Notes

- The trained model has act_dim=2 (2D continuous action space), not act_dim=1 as the PLAN.md initially assumed. The implementation correctly adapted to this.
- The PLAN.md suggested using `get_distribution(obs).distribution.mean` directly (expecting tanh squashing), but the model has `squash_output=False`. The implementation correctly added `torch.clamp()` to match SB3's `predict(deterministic=True)` behavior, bringing cross-check diff from 1.89 down to 5.96e-08.

---

Ready for Commit: yes
Commit Message: feat(python): add SB3-to-ONNX export script with clanker metadata
