# Loop 02: Implementation Report

## Task 1: Add `ndarray` and `thiserror` dependencies to `clankers-policy`

Completed: 2026-02-24

### Changes

- `Cargo.toml` (workspace root): Added `ndarray = "0.16"` to `[workspace.dependencies]`. Updated `ort` entry to include features `["std", "download-binaries", "copy-dylibs", "load-dynamic", "tls-native"]` (was previously using a minimal set).
- `crates/clankers-policy/Cargo.toml`: Added `ndarray = { workspace = true, optional = true }`, `ort = { workspace = true, optional = true }`, `serde_json.workspace = true`, and `thiserror.workspace = true` to `[dependencies]`. Added `[features]` section with `onnx = ["dep:ort", "dep:ndarray"]`.

### Verification

- [x] `cargo check -p clankers-policy --features onnx`: PASS

### Notes

- The plan originally specified only `dep:ort` and `dep:ndarray` for the onnx feature; `serde_json` and `thiserror` are non-optional dependencies added for metadata JSON parsing and the error enum respectively.
- The ort workspace entry required additional features beyond what was in the plan: `std` (for `commit_from_file`), `download-binaries` + `tls-native` (for automatic ONNX Runtime binary download), `copy-dylibs` (for DLL deployment), and `load-dynamic` (to avoid static linking issues on Windows with MSVC version mismatches).

---

## Task 2: Create `crates/clankers-policy/src/onnx.rs`

Completed: 2026-02-24

### Changes

- `crates/clankers-policy/src/onnx.rs`: Created (519 lines). Contains:
  - `OnnxPolicyError` enum with 8 variants (LoadFailed, MissingObsInput, MissingActionOutput, UnknownObsDim, UnknownActionDim, ObsDimMismatch, MetadataParse, InferenceFailed).
  - `ActionTransform` enum with 3 variants (None, Tanh, Clip).
  - `OnnxPolicy` struct wrapping `Mutex<Session>` with fields for obs_dim, action_dim, action_transform, input_name, output_name.
  - `OnnxPolicy::from_file()` constructor that loads model, resolves tensor names, extracts dimensions, and parses metadata.
  - `impl Policy for OnnxPolicy` with `get_action()`, `name()`, `is_deterministic()`.
  - Helper functions: `find_tensor_name`, `extract_dim_from_input`, `extract_dim_from_output`, `read_metadata`, `parse_action_transform`, `parse_f32_array`, `parse_clip_bounds`.
  - 10 unit tests in a `#[cfg(test)] mod tests` block.

### Verification

- [x] `cargo check -p clankers-policy --features onnx`: PASS
- [x] All 10 unit tests pass

### Notes

- **Deviation: `Mutex<Session>` instead of bare `Session`**: The ort v2 API requires `&mut self` for `Session::run()`, but the `Policy` trait defines `get_action(&self, ...)`. Wrapping in `Mutex<Session>` solves this without unsafe code. The plan stated `Session` is `Send + Sync` directly, but the mutable borrow requirement made the Mutex necessary.
- **Deviation: Raw slice API instead of ndarray**: The plan specified using `ndarray::Array2` for tensor I/O. However, ort v2 RC11 bundles ndarray 0.17 internally while the workspace uses ndarray 0.16. To avoid version conflicts, the implementation uses ort's raw slice-based API: `TensorRef::from_array_view((shape_tuple, &[f32]))` for input and `try_extract_tensor::<f32>()` returning `(&Shape, &[f32])` for output. The `ndarray` crate is still listed as an optional dependency but is not actively used in the current implementation.
- **Deviation: Metadata API**: The plan assumed `session.metadata()?.custom()` returns `HashMap<String, String>`. The actual ort v2 RC11 API uses `session.metadata()` returning `Result<ModelMetadata>`, then `meta.custom_keys()` returning `Result<Vec<String>>`, and `meta.custom(&key)` returning `Option<String>`. The `read_metadata` helper iterates keys individually.
- **Deviation: `inputs!` macro**: The plan assumed `ort::inputs!` returns a `Result`. The named form `ort::inputs!["name" => tensor]` actually returns `Vec<(Cow<str>, SessionInputValue)>` directly (no Result wrapper).

---

## Task 3: Wire up the `onnx` module in `lib.rs` and update the prelude

Completed: 2026-02-24

### Changes

- `crates/clankers-policy/src/lib.rs`: Added `#[cfg(feature = "onnx")] pub mod onnx;` after `pub mod runner;` (line 24-25). Added `#[cfg(feature = "onnx")] pub use crate::onnx::{OnnxPolicy, OnnxPolicyError};` to the `prelude` module (line 66-67).

### Verification

- [x] `cargo check -p clankers-policy` (without onnx feature): PASS
- [x] `cargo check -p clankers-policy --features onnx`: PASS
- [x] `cargo check` (full workspace): PASS

### Notes

None. This task was implemented exactly as planned.

---

## Task 4: Create test fixture ONNX model via Python script

Completed: 2026-02-24

### Changes

- `crates/clankers-policy/tests/fixtures/create_test_model.py`: Created. Generates two minimal ONNX models using the `onnx` and `numpy` Python packages. Uses identity-like weight matrices and zero bias for deterministic test outputs.
- `crates/clankers-policy/tests/fixtures/test_policy_none.onnx`: Generated (313 bytes). 4 obs -> 1 action, `action_transform=none`.
- `crates/clankers-policy/tests/fixtures/test_policy_tanh.onnx`: Generated (397 bytes). 4 obs -> 2 actions, `action_transform=tanh`, `action_scale=[2.0, 2.0]`, `action_offset=[0.0, 0.0]`.

### Verification

- [x] Python script runs successfully via `uv run --with onnx --with numpy --python 3.11`
- [x] Both `.onnx` files generated and valid
- [x] Models load successfully in unit tests

### Notes

- **Deviation: Model naming**: The plan specified `test_policy.onnx` for the no-transform model. The implementation uses `test_policy_none.onnx` to be more descriptive and consistent with `test_policy_tanh.onnx`.
- **Deviation: Model dimensions**: The plan specified both models as 4 obs -> 2 actions. The `test_policy_none.onnx` model uses 4 obs -> 1 action to test a different output dimension and better exercise the dimension extraction code.
- **Deviation: Python execution**: `pip install onnx` failed due to MINGW Python build issues. Used `uv run --with onnx --with numpy --python 3.11 -- python create_test_model.py` instead.

---

## Task 5: Add unit tests in `onnx.rs`

Completed: 2026-02-24

### Changes

- `crates/clankers-policy/src/onnx.rs`: Added 10 unit tests in a `#[cfg(test)] mod tests` block (lines 395-519):
  1. `load_valid_model` -- loads `test_policy_none.onnx`, verifies obs_dim=4, action_dim=1
  2. `load_tanh_model_has_tanh_transform` -- loads tanh model, verifies ActionTransform::Tanh with correct scale/offset
  3. `error_on_missing_file` -- verifies OnnxPolicyError::LoadFailed for nonexistent path
  4. `get_action_returns_correct_dim` -- verifies action vector length matches model output
  5. `get_action_zero_obs_returns_zero_action` -- zero input yields near-zero output (identity weights)
  6. `get_action_deterministic_across_calls` -- same input produces identical output twice
  7. `tanh_transform_scales_output` -- verifies tanh transform applies scale correctly (raw * 2.0 + 0.0)
  8. `policy_name` -- verifies `name()` returns "OnnxPolicy"
  9. `policy_is_deterministic` -- verifies `is_deterministic()` returns true
  10. `onnx_policy_is_send_sync` -- compile-time check that OnnxPolicy: Send + Sync

### Verification

- [x] `cargo test -p clankers-policy --features onnx`: 38/38 tests pass (10 onnx + 28 existing)
- [x] `cargo test -p clankers-policy` (without onnx feature): 28/28 tests pass (no regressions)

### Notes

- Tests require `ORT_DYLIB_PATH` environment variable to be set pointing to an onnxruntime.dll v1.23.0 or later. The system had an old v1.17.1 DLL in `C:\Windows\System32\` which is incompatible with ort v2.0.0-rc.11. The correct DLL was downloaded from the official GitHub releases and placed at `target/debug/deps/onnxruntime.dll`.
- The `tanh_transform_scales_output` test was simplified from the plan. Instead of comparing against a separate no-transform model (which has different dimensions), it directly asserts expected values based on the known identity weights and scale factors.

---

## Overall Verification Summary

| Check | Result |
|-------|--------|
| `cargo check -p clankers-policy --features onnx` | PASS |
| `cargo test -p clankers-policy --features onnx` | 38/38 PASS |
| `cargo test -p clankers-policy` (without onnx) | 28/28 PASS |
| `cargo check` (full workspace) | PASS |

## Deviations from Plan

1. **ort features**: Added `std`, `download-binaries`, `copy-dylibs`, `load-dynamic`, `tls-native` features to the workspace ort dependency. These were needed for file loading, binary download, and dynamic linking on Windows.
2. **Mutex wrapping**: `Session::run()` requires `&mut self` in ort v2, so the session is wrapped in `Mutex<Session>` to satisfy the `&self` signature of `Policy::get_action`.
3. **Raw slice API**: Used `TensorRef::from_array_view((shape, &[f32]))` and `try_extract_tensor::<f32>()` returning `(&Shape, &[f32])` instead of ndarray arrays, to avoid ndarray version conflicts (ort bundles 0.17, workspace has 0.16).
4. **Metadata API**: Used `custom_keys()` + `custom(&key)` iteration pattern instead of a single `custom()` call returning a HashMap.
5. **Test fixture naming**: Used `test_policy_none.onnx` (4->1) instead of `test_policy.onnx` (4->2).
6. **ORT_DYLIB_PATH requirement**: Tests on this Windows system require the environment variable to point to a compatible onnxruntime.dll (v1.23.0+), since the system-level DLL is outdated.
