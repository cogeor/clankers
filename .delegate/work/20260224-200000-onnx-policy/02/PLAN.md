# Loop 02: Implement OnnxPolicy struct with Policy trait, metadata parsing, and unit tests

## Overview

This loop creates the `OnnxPolicy` struct in `crates/clankers-policy/src/onnx.rs`, gated behind `#[cfg(feature = "onnx")]`. The struct wraps an `ort::Session`, parses ONNX model metadata (action transform, scale, offset), and implements the `Policy` trait from `clankers-core`. It also adds a Python helper script to generate a tiny test fixture ONNX model, and comprehensive unit tests covering loading, inference, action transforms, and error paths.

The implementation targets the "Basic" + "Standard" conformance levels from `POLICY_ONNX_SPEC.md` -- vector observations, continuous actions, action transform metadata, and tensor name resolution. Recurrent state and image observations are out of scope for this loop.

## Tasks

### Task 1: Add `ndarray` and `thiserror` dependencies to `clankers-policy`

**Goal:** The `ort` v2 API requires `ndarray` arrays for input/output tensor creation. Add `ndarray` as an optional dependency (behind the `onnx` feature) and `thiserror` for the error enum.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `Cargo.toml` (workspace root) |
| MODIFY | `crates/clankers-policy/Cargo.toml` |

**Steps:**
1. In `C:\Users\costa\src\clankers\Cargo.toml`, add to the `[workspace.dependencies]` section (after the `ort` line):
   ```toml
   ndarray = "0.16"
   ```
2. In `C:\Users\costa\src\clankers\crates\clankers-policy\Cargo.toml`, add `ndarray` and `thiserror` as dependencies:
   ```toml
   [dependencies]
   # ... existing deps ...
   ndarray = { workspace = true, optional = true }
   thiserror.workspace = true
   ```
3. Update the `[features]` section to include `ndarray` in the `onnx` feature:
   ```toml
   [features]
   onnx = ["dep:ort", "dep:ndarray"]
   ```

**Verify:** `cargo check -p clankers-policy --features onnx` succeeds.

---

### Task 2: Create `crates/clankers-policy/src/onnx.rs`

**Goal:** Implement the `OnnxPolicy` struct, error types, action transform logic, and `Policy` trait implementation. The module is fully gated behind `#[cfg(feature = "onnx")]`.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `crates/clankers-policy/src/onnx.rs` |

**Steps:**

1. Define `OnnxPolicyError` enum using `thiserror`:
   ```rust
   use std::path::PathBuf;
   use thiserror::Error;

   #[derive(Debug, Error)]
   pub enum OnnxPolicyError {
       #[error("Failed to load ONNX model from {path}: {source}")]
       LoadFailed {
           path: PathBuf,
           source: ort::Error,
       },

       #[error("No observation input tensor found (expected 'obs' or 'observation')")]
       MissingObsInput,

       #[error("No action output tensor found (expected 'action' or 'actions')")]
       MissingActionOutput,

       #[error("Failed to parse metadata key '{key}': {message}")]
       MetadataParse {
           key: String,
           message: String,
       },

       #[error("Inference failed: {0}")]
       InferenceFailed(#[from] ort::Error),

       #[error("Observation dimension mismatch: model expects {expected}, got {got}")]
       ObsDimMismatch { expected: usize, got: usize },

       #[error("Could not determine observation dimension from model input shape")]
       UnknownObsDim,

       #[error("Could not determine action dimension from model output shape")]
       UnknownActionDim,
   }
   ```

2. Define `ActionTransform` enum:
   ```rust
   #[derive(Debug, Clone)]
   pub enum ActionTransform {
       /// No transformation applied. Raw model output used directly.
       None,
       /// Tanh squashing: `action = raw * scale + offset`
       Tanh {
           scale: Vec<f32>,
           offset: Vec<f32>,
       },
       /// Clip to bounds. Raw output is already in environment space.
       Clip {
           low: Vec<f32>,
           high: Vec<f32>,
       },
   }
   ```

3. Define `OnnxPolicy` struct:
   ```rust
   pub struct OnnxPolicy {
       session: ort::session::Session,
       obs_dim: usize,
       action_dim: usize,
       action_transform: ActionTransform,
       input_name: String,
       output_name: String,
   }
   ```
   Note: `ort::session::Session` is `Send + Sync` in ort v2, satisfying the `Policy` trait bounds.

4. Implement `OnnxPolicy::from_file`:
   ```rust
   impl OnnxPolicy {
       pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, OnnxPolicyError> {
           let path = path.as_ref();

           // Load ONNX session via ort v2 API
           let session = ort::session::Session::builder()
               .and_then(|b| b.commit_from_file(path))
               .map_err(|e| OnnxPolicyError::LoadFailed {
                   path: path.to_path_buf(),
                   source: e,
               })?;

           // Find observation input tensor name
           let input_name = find_tensor_name(
               session.inputs.iter().map(|i| i.name.as_str()),
               &["obs", "observation"],
           )
           .ok_or(OnnxPolicyError::MissingObsInput)?;

           // Find action output tensor name
           let output_name = find_tensor_name(
               session.outputs.iter().map(|o| o.name.as_str()),
               &["action", "actions"],
           )
           .ok_or(OnnxPolicyError::MissingActionOutput)?;

           // Extract obs_dim from input shape [batch, obs_dim]
           let obs_dim = extract_dim_from_input(&session, &input_name)?;

           // Extract action_dim from output shape [batch, action_dim]
           let action_dim = extract_dim_from_output(&session, &output_name)?;

           // Parse metadata for action transform
           let metadata = read_metadata(&session);
           let action_transform = parse_action_transform(&metadata, action_dim);

           Ok(Self {
               session,
               obs_dim,
               action_dim,
               action_transform,
               input_name,
               output_name,
           })
       }

       /// Returns the observation dimension expected by the model.
       pub fn obs_dim(&self) -> usize { self.obs_dim }

       /// Returns the action dimension produced by the model.
       pub fn action_dim(&self) -> usize { self.action_dim }

       /// Returns the action transform configuration.
       pub fn action_transform(&self) -> &ActionTransform { &self.action_transform }
   }
   ```

5. Implement helper functions:

   `find_tensor_name` -- iterate session inputs/outputs, return the first matching name from a priority list:
   ```rust
   fn find_tensor_name<'a>(
       names: impl Iterator<Item = &'a str>,
       candidates: &[&str],
   ) -> Option<String> {
       let name_vec: Vec<&str> = names.collect();
       for candidate in candidates {
           if name_vec.contains(candidate) {
               return Some((*candidate).to_string());
           }
       }
       None
   }
   ```

   `extract_dim_from_input` -- read the second dimension from input shape (index 1 of `[batch, obs_dim]`). The ort v2 API exposes shape via `session.inputs[idx].input_type` -> `TensorType` -> `dimensions`. Use pattern:
   ```rust
   fn extract_dim_from_input(
       session: &ort::session::Session,
       name: &str,
   ) -> Result<usize, OnnxPolicyError> {
       for input in &session.inputs {
           if input.name == name {
               if let ort::value::ValueType::Tensor { dimensions, .. } = &input.input_type {
                   // dimensions is Vec<i64>, where -1 means dynamic
                   if dimensions.len() >= 2 {
                       let dim = dimensions[1];
                       if dim > 0 {
                           return Ok(dim as usize);
                       }
                   }
               }
           }
       }
       Err(OnnxPolicyError::UnknownObsDim)
   }
   ```

   `extract_dim_from_output` -- similar for output tensor.

   `read_metadata` -- read `metadata_props` from the session's model metadata:
   ```rust
   fn read_metadata(session: &ort::session::Session) -> std::collections::HashMap<String, String> {
       session.metadata()
           .and_then(|m| m.custom())
           .unwrap_or_default()
   }
   ```
   Note: The exact ort v2 metadata API needs verification. The `Session` may expose metadata via `session.metadata()` returning a `ModelMetadata` with `custom() -> HashMap<String, String>`. If the API differs, adapt accordingly (this is a known ort v2 API area that may vary between RC releases).

   `parse_action_transform` -- read `action_transform`, `action_scale`, `action_offset` from metadata map:
   ```rust
   fn parse_action_transform(
       metadata: &std::collections::HashMap<String, String>,
       action_dim: usize,
   ) -> ActionTransform {
       let transform_str = metadata
           .get("action_transform")
           .map(String::as_str)
           .unwrap_or("none");

       match transform_str {
           "tanh" => {
               let scale = parse_f32_array(metadata.get("action_scale"))
                   .unwrap_or_else(|| vec![1.0; action_dim]);
               let offset = parse_f32_array(metadata.get("action_offset"))
                   .unwrap_or_else(|| vec![0.0; action_dim]);
               ActionTransform::Tanh { scale, offset }
           }
           "clip" => {
               // Read from action_space metadata JSON
               let (low, high) = parse_clip_bounds(metadata, action_dim);
               ActionTransform::Clip { low, high }
           }
           _ => ActionTransform::None,
       }
   }

   fn parse_f32_array(val: Option<&String>) -> Option<Vec<f32>> {
       val.and_then(|s| serde_json::from_str::<Vec<f32>>(s).ok())
   }

   fn parse_clip_bounds(
       metadata: &std::collections::HashMap<String, String>,
       action_dim: usize,
   ) -> (Vec<f32>, Vec<f32>) {
       if let Some(space_json) = metadata.get("action_space") {
           if let Ok(v) = serde_json::from_str::<serde_json::Value>(space_json) {
               let low = v.get("low")
                   .and_then(|a| serde_json::from_value::<Vec<f32>>(a.clone()).ok())
                   .unwrap_or_else(|| vec![-1.0; action_dim]);
               let high = v.get("high")
                   .and_then(|a| serde_json::from_value::<Vec<f32>>(a.clone()).ok())
                   .unwrap_or_else(|| vec![1.0; action_dim]);
               return (low, high);
           }
       }
       (vec![-1.0; action_dim], vec![1.0; action_dim])
   }
   ```

6. Implement `apply_transform` method:
   ```rust
   impl OnnxPolicy {
       fn apply_transform(&self, raw: &mut Vec<f32>) {
           match &self.action_transform {
               ActionTransform::None => {}
               ActionTransform::Tanh { scale, offset } => {
                   for (i, val) in raw.iter_mut().enumerate() {
                       if i < scale.len() {
                           *val = *val * scale[i] + offset[i];
                       }
                   }
               }
               ActionTransform::Clip { low, high } => {
                   for (i, val) in raw.iter_mut().enumerate() {
                       if i < low.len() {
                           *val = val.clamp(low[i], high[i]);
                       }
                   }
               }
           }
       }
   }
   ```

7. Implement `Policy` trait for `OnnxPolicy`:
   ```rust
   use clankers_core::traits::Policy;
   use clankers_core::types::{Action, Observation};

   impl Policy for OnnxPolicy {
       fn get_action(&self, obs: &Observation) -> Action {
           // Build input: [1, obs_dim] f32 array
           let obs_data = obs.as_slice();
           let input_array = ndarray::Array2::<f32>::from_shape_vec(
               (1, self.obs_dim),
               obs_data.to_vec(),
           )
           .expect("obs dimension mismatch in get_action");

           // Create ort input value
           let input_value = ort::value::Value::from_array(input_array)
               .expect("failed to create input tensor");

           // Run inference
           let outputs = self.session
               .run(ort::inputs![&self.input_name => input_value].unwrap())
               .expect("ONNX inference failed");

           // Extract output tensor
           let output = &outputs[&*self.output_name];
           let output_array = output
               .try_extract_tensor::<f32>()
               .expect("failed to extract output tensor");

           // Convert to Vec<f32> -- output shape is [1, action_dim]
           let mut action_vec: Vec<f32> = output_array.iter().copied().collect();

           // Apply action transform (denormalization)
           self.apply_transform(&mut action_vec);

           Action::from(action_vec)
       }

       fn name(&self) -> &str {
           "OnnxPolicy"
       }

       fn is_deterministic(&self) -> bool {
           true
       }
   }
   ```

   **Important ort v2 API notes:**
   - `ort::inputs!` macro creates a `SessionInputs` from name-value pairs.
   - `session.run()` returns a `SessionOutputs` which can be indexed by tensor name.
   - `Value::from_array()` takes an `ndarray::ArrayBase` and returns `Result<Value, Error>`.
   - `try_extract_tensor::<f32>()` returns a `Result<ArrayViewD<f32>, Error>`.
   - If any of these APIs differ in the exact RC version, adapt the calls. The general pattern is stable across ort v2 RCs.

8. Add `Send + Sync` safety comment. The `ort::Session` type in v2 is `Send + Sync`, so `OnnxPolicy` automatically satisfies the `Policy` trait bounds. No unsafe code needed.

**Verify:** `cargo check -p clankers-policy --features onnx` succeeds with no errors.

---

### Task 3: Wire up the `onnx` module in `lib.rs` and update the prelude

**Goal:** Conditionally compile and export the `onnx` module, and add `OnnxPolicy` to the prelude behind the feature gate.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-policy/src/lib.rs` |

**Steps:**

1. After `pub mod runner;` (line 22), add:
   ```rust
   #[cfg(feature = "onnx")]
   pub mod onnx;
   ```

2. Update the `prelude` module to conditionally re-export `OnnxPolicy` and `OnnxPolicyError`:
   ```rust
   pub mod prelude {
       pub use crate::{
           ClankersPolicyPlugin,
           policies::{ConstantPolicy, RandomPolicy, ScriptedPolicy, ZeroPolicy},
           runner::{PolicyRunner, policy_decide_system},
       };

       #[cfg(feature = "onnx")]
       pub use crate::onnx::{OnnxPolicy, OnnxPolicyError};
   }
   ```

**Verify:** `cargo check -p clankers-policy` (without feature) and `cargo check -p clankers-policy --features onnx` (with feature) both succeed.

---

### Task 4: Create test fixture ONNX model via Python script

**Goal:** Create a small Python script that generates a minimal ONNX model file for testing. The model is a simple linear layer: 4 float inputs -> 2 float outputs, with clanker metadata embedded. Commit the generated `.onnx` file as a test fixture.

**Files:**
| Action | Path |
|--------|------|
| CREATE | `crates/clankers-policy/tests/fixtures/create_test_model.py` |
| CREATE | `crates/clankers-policy/tests/fixtures/test_policy.onnx` |
| CREATE | `crates/clankers-policy/tests/fixtures/test_policy_tanh.onnx` |

**Steps:**

1. Create `C:\Users\costa\src\clankers\crates\clankers-policy\tests\fixtures\create_test_model.py`:
   ```python
   """Generate tiny ONNX test models for clankers-policy unit tests.

   Requirements: pip install torch onnx numpy

   Generates:
     - test_policy.onnx: 4-input, 2-output linear model, action_transform=none
     - test_policy_tanh.onnx: 4-input, 2-output linear model, action_transform=tanh
   """
   import json
   import numpy as np
   import onnx
   from onnx import helper, TensorProto, numpy_helper

   def create_linear_model(
       obs_dim: int,
       action_dim: int,
       input_name: str = "obs",
       output_name: str = "action",
   ) -> onnx.ModelProto:
       """Create a simple linear model: output = input @ W + b."""
       # Fixed weights for deterministic test results
       np.random.seed(42)
       W = np.eye(action_dim, obs_dim, dtype=np.float32)  # identity-like
       b = np.zeros(action_dim, dtype=np.float32)

       # Define graph
       W_init = numpy_helper.from_array(W, name="W")
       b_init = numpy_helper.from_array(b, name="b")

       obs_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, obs_dim])
       action_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, action_dim])

       matmul_node = helper.make_node("MatMul", [input_name, "W_T"], ["matmul_out"])
       add_node = helper.make_node("Add", ["matmul_out", "b"], [output_name])

       # Transpose W for MatMul: [obs_dim, action_dim]
       W_T = W.T
       W_T_init = numpy_helper.from_array(W_T, name="W_T")

       graph = helper.make_graph(
           [matmul_node, add_node],
           "test_policy",
           [obs_input],
           [action_output],
           initializer=[W_T_init, b_init],
       )

       model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
       model.ir_version = 7
       return model


   def add_metadata(model, metadata: dict) -> onnx.ModelProto:
       for key, value in metadata.items():
           entry = model.metadata_props.add()
           entry.key = key
           entry.value = value
       return model


   def main():
       import os
       out_dir = os.path.dirname(os.path.abspath(__file__))

       # Model 1: no transform
       model_none = create_linear_model(4, 2)
       model_none = add_metadata(model_none, {
           "clanker_policy_version": "1.0.0",
           "action_space": json.dumps({"type": "Box", "shape": [2], "low": [-1, -1], "high": [1, 1]}),
           "action_transform": "none",
       })
       onnx.checker.check_model(model_none)
       path_none = os.path.join(out_dir, "test_policy.onnx")
       onnx.save(model_none, path_none)
       print(f"Saved {path_none}")

       # Model 2: tanh transform with scale/offset
       model_tanh = create_linear_model(4, 2)
       model_tanh = add_metadata(model_tanh, {
           "clanker_policy_version": "1.0.0",
           "action_space": json.dumps({"type": "Box", "shape": [2], "low": [-2, -2], "high": [2, 2]}),
           "action_transform": "tanh",
           "action_scale": "[2.0, 2.0]",
           "action_offset": "[0.0, 0.0]",
       })
       onnx.checker.check_model(model_tanh)
       path_tanh = os.path.join(out_dir, "test_policy_tanh.onnx")
       onnx.save(model_tanh, path_tanh)
       print(f"Saved {path_tanh}")


   if __name__ == "__main__":
       main()
   ```

2. Run the script to generate the `.onnx` files:
   ```bash
   cd crates/clankers-policy/tests/fixtures
   python create_test_model.py
   ```

3. Verify both `.onnx` files exist and are valid (small, a few KB each).

4. Both fixture files will be committed alongside the code.

**Verify:** `python crates/clankers-policy/tests/fixtures/create_test_model.py` runs without error and produces both `.onnx` files.

---

### Task 5: Add unit tests in `onnx.rs`

**Goal:** Add comprehensive unit tests gated behind `#[cfg(all(test, feature = "onnx"))]` to validate model loading, inference, action transforms, and error handling.

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `crates/clankers-policy/src/onnx.rs` |

**Steps:**

1. Add a `#[cfg(all(test, feature = "onnx"))]` test module at the bottom of `onnx.rs`:

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use clankers_core::types::Observation;

       /// Path to test fixtures relative to the crate root.
       fn fixture_path(name: &str) -> std::path::PathBuf {
           let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
           manifest_dir.join("tests").join("fixtures").join(name)
       }

       // -- Loading tests --

       #[test]
       fn load_valid_model() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx"));
           assert!(policy.is_ok(), "Failed to load: {:?}", policy.err());
           let policy = policy.unwrap();
           assert_eq!(policy.obs_dim(), 4);
           assert_eq!(policy.action_dim(), 2);
       }

       #[test]
       fn load_tanh_model_has_tanh_transform() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy_tanh.onnx")).unwrap();
           assert!(matches!(policy.action_transform(), ActionTransform::Tanh { .. }));
           if let ActionTransform::Tanh { scale, offset } = policy.action_transform() {
               assert_eq!(scale, &[2.0, 2.0]);
               assert_eq!(offset, &[0.0, 0.0]);
           }
       }

       #[test]
       fn error_on_missing_file() {
           let result = OnnxPolicy::from_file("/nonexistent/model.onnx");
           assert!(result.is_err());
           assert!(matches!(result.unwrap_err(), OnnxPolicyError::LoadFailed { .. }));
       }

       // -- Inference tests --

       #[test]
       fn get_action_returns_correct_dim() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           let obs = Observation::new(vec![1.0, 0.0, 0.0, 0.0]);
           let action = policy.get_action(&obs);
           assert_eq!(action.len(), 2);
       }

       #[test]
       fn get_action_zero_obs_returns_zero_action() {
           // With identity weights and zero bias, zero input -> zero output
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           let obs = Observation::new(vec![0.0, 0.0, 0.0, 0.0]);
           let action = policy.get_action(&obs);
           for &v in action.as_slice() {
               assert!(
                   v.abs() < 1e-6,
                   "Expected near-zero action, got {}",
                   v
               );
           }
       }

       #[test]
       fn get_action_deterministic_across_calls() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           let obs = Observation::new(vec![1.0, 2.0, 3.0, 4.0]);
           let a1 = policy.get_action(&obs);
           let a2 = policy.get_action(&obs);
           assert_eq!(a1.as_slice(), a2.as_slice());
       }

       // -- Action transform tests --

       #[test]
       fn tanh_transform_scales_output() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy_tanh.onnx")).unwrap();
           let obs = Observation::new(vec![1.0, 0.0, 0.0, 0.0]);
           let action_tanh = policy.get_action(&obs);

           // Compare with no-transform model
           let policy_none = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           let action_none = policy_none.get_action(&obs);

           // tanh model applies: result = raw * 2.0 + 0.0
           // So tanh output should be 2x the raw output for same weights
           for (t, n) in action_tanh.as_slice().iter().zip(action_none.as_slice().iter()) {
               assert!(
                   (t - n * 2.0).abs() < 1e-5,
                   "Expected tanh={}, got none*2={}",
                   t,
                   n * 2.0
               );
           }
       }

       // -- Trait compliance tests --

       #[test]
       fn policy_name() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           assert_eq!(policy.name(), "OnnxPolicy");
       }

       #[test]
       fn policy_is_deterministic() {
           let policy = OnnxPolicy::from_file(fixture_path("test_policy.onnx")).unwrap();
           assert!(policy.is_deterministic());
       }

       #[test]
       fn onnx_policy_is_send_sync() {
           fn assert_send_sync<T: Send + Sync>() {}
           assert_send_sync::<OnnxPolicy>();
       }
   }
   ```

2. The tests rely on the fixture files from Task 4. They will run with:
   ```bash
   cargo test -p clankers-policy --features onnx
   ```

**Verify:** `cargo test -p clankers-policy --features onnx` runs all tests and they pass. Also verify `cargo test -p clankers-policy` (without feature) still passes the existing tests.

---

## Acceptance Criteria

- [ ] `crates/clankers-policy/src/onnx.rs` exists and compiles behind `#[cfg(feature = "onnx")]`
- [ ] `OnnxPolicyError` enum covers load, inference, metadata parse, and dimension mismatch errors
- [ ] `ActionTransform` enum supports `None`, `Tanh { scale, offset }`, and `Clip { low, high }`
- [ ] `OnnxPolicy::from_file()` loads an ONNX model, resolves tensor names (`obs`/`observation`, `action`/`actions`), parses metadata, and validates dimensions
- [ ] `impl Policy for OnnxPolicy` produces correct-dimension actions via `get_action()`
- [ ] Action transform (tanh denormalization) is applied correctly to raw model output
- [ ] `lib.rs` exports `onnx` module and prelude re-exports behind feature gate
- [ ] Test fixture `.onnx` files exist at `crates/clankers-policy/tests/fixtures/`
- [ ] Python script at `crates/clankers-policy/tests/fixtures/create_test_model.py` can regenerate fixtures
- [ ] `cargo test -p clankers-policy --features onnx` passes all unit tests
- [ ] `cargo test -p clankers-policy` (without onnx feature) passes all existing tests with no regressions
- [ ] `cargo check` (full workspace, no onnx feature) succeeds
