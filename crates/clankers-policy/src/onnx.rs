//! ONNX policy inference via the `ort` crate.
//!
//! Requires the `onnx` feature flag.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use thiserror::Error;

use clankers_core::traits::Policy;
use clankers_core::types::{Action, Observation};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur when loading or running an ONNX policy.
#[derive(Debug, Error)]
pub enum OnnxPolicyError {
    /// Failed to load an ONNX model from disk.
    #[error("failed to load ONNX model from {path}: {source}")]
    LoadFailed {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying ort error.
        source: ort::Error,
    },
    /// No observation input tensor found in the model.
    #[error("no observation input tensor found (expected 'obs' or 'observation')")]
    MissingObsInput,
    /// No action output tensor found in the model.
    #[error("no action output tensor found (expected 'action' or 'actions')")]
    MissingActionOutput,
    /// Could not determine observation dimension from the model input shape.
    #[error("could not determine observation dimension from model input shape")]
    UnknownObsDim,
    /// Could not determine action dimension from the model output shape.
    #[error("could not determine action dimension from model output shape")]
    UnknownActionDim,
    /// Observation dimension does not match what the model expects.
    #[error("observation dimension mismatch: model expects {expected}, got {got}")]
    ObsDimMismatch {
        /// Dimension expected by the model.
        expected: usize,
        /// Dimension actually provided.
        got: usize,
    },
    /// Metadata parsing failed for a given key.
    #[error("failed to parse metadata key '{key}': {message}")]
    MetadataParse {
        /// The metadata key that failed to parse.
        key: String,
        /// Description of the parse error.
        message: String,
    },
    /// Inference failed at runtime.
    #[error("inference failed: {0}")]
    InferenceFailed(#[from] ort::Error),
}

// ---------------------------------------------------------------------------
// ActionTransform
// ---------------------------------------------------------------------------

/// Post-processing transform applied to raw model outputs before returning
/// them as actions.
#[derive(Debug, Clone)]
pub enum ActionTransform {
    /// No transformation applied. Raw model output used directly.
    None,
    /// Tanh-style denormalization: `action[i] = raw[i] * scale[i] + offset[i]`.
    Tanh {
        /// Per-dimension scale factors.
        scale: Vec<f32>,
        /// Per-dimension offsets.
        offset: Vec<f32>,
    },
    /// Clip raw output to `[low, high]` bounds.
    Clip {
        /// Per-dimension lower bounds.
        low: Vec<f32>,
        /// Per-dimension upper bounds.
        high: Vec<f32>,
    },
}

// ---------------------------------------------------------------------------
// OnnxPolicy
// ---------------------------------------------------------------------------

/// A policy backed by an ONNX Runtime session.
///
/// Wraps an [`ort::session::Session`] and implements the
/// [`Policy`](clankers_core::traits::Policy) trait. The session is protected
/// by a [`Mutex`] because `Session::run` requires `&mut self`.
pub struct OnnxPolicy {
    session: Mutex<Session>,
    obs_dim: usize,
    action_dim: usize,
    action_transform: ActionTransform,
    input_name: String,
    output_name: String,
}

impl std::fmt::Debug for OnnxPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxPolicy")
            .field("obs_dim", &self.obs_dim)
            .field("action_dim", &self.action_dim)
            .field("action_transform", &self.action_transform)
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .finish_non_exhaustive()
    }
}

impl OnnxPolicy {
    /// Load an ONNX model from a file and create a new `OnnxPolicy`.
    ///
    /// The model must have:
    /// - An input tensor named `"obs"` or `"observation"` with shape `[1, obs_dim]`.
    /// - An output tensor named `"action"` or `"actions"` with shape `[1, action_dim]`.
    ///
    /// Metadata keys `action_transform`, `action_scale`, `action_offset`,
    /// `action_low`, `action_high`, and `action_space` are optionally parsed
    /// to configure post-processing.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, OnnxPolicyError> {
        let path = path.as_ref();

        let session: Session = Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| OnnxPolicyError::LoadFailed {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Find observation input tensor name.
        let input_name = find_tensor_name(
            session.inputs().iter().map(|i| i.name()),
            &["obs", "observation"],
        )
        .ok_or(OnnxPolicyError::MissingObsInput)?;

        // Find action output tensor name.
        let output_name = find_tensor_name(
            session.outputs().iter().map(|o| o.name()),
            &["action", "actions"],
        )
        .ok_or(OnnxPolicyError::MissingActionOutput)?;

        // Extract obs_dim from input shape [batch, obs_dim].
        let obs_dim = extract_dim_from_input(&session, &input_name)?;

        // Extract action_dim from output shape [batch, action_dim].
        let action_dim = extract_dim_from_output(&session, &output_name)?;

        // Parse metadata for action transform.
        let metadata = read_metadata(&session);
        let action_transform = parse_action_transform(&metadata, action_dim);

        Ok(Self {
            session: Mutex::new(session),
            obs_dim,
            action_dim,
            action_transform,
            input_name,
            output_name,
        })
    }

    /// Returns the observation dimension expected by the model.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Returns the action dimension produced by the model.
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Returns the action transform configuration.
    pub fn action_transform(&self) -> &ActionTransform {
        &self.action_transform
    }

    /// Apply the configured action transform in-place.
    fn apply_transform(&self, raw: &mut [f32]) {
        match &self.action_transform {
            ActionTransform::None => {}
            ActionTransform::Tanh { scale, offset } => {
                for (i, val) in raw.iter_mut().enumerate() {
                    let s = scale.get(i).copied().unwrap_or(1.0);
                    let o = offset.get(i).copied().unwrap_or(0.0);
                    *val = *val * s + o;
                }
            }
            ActionTransform::Clip { low, high } => {
                for (i, val) in raw.iter_mut().enumerate() {
                    let lo = low.get(i).copied().unwrap_or(f32::NEG_INFINITY);
                    let hi = high.get(i).copied().unwrap_or(f32::INFINITY);
                    *val = val.clamp(lo, hi);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Policy trait implementation
// ---------------------------------------------------------------------------

impl Policy for OnnxPolicy {
    fn get_action(&self, obs: &Observation) -> Action {
        let obs_slice = obs.as_slice();
        let obs_len = obs_slice.len().min(self.obs_dim);

        // Build a zero-padded [1, obs_dim] input buffer.
        let mut input_data = vec![0.0f32; self.obs_dim];
        input_data[..obs_len].copy_from_slice(&obs_slice[..obs_len]);

        // Create a TensorRef from the raw slice -- shape (1, obs_dim).
        let input_tensor =
            TensorRef::<f32>::from_array_view(([1_usize, self.obs_dim], &*input_data))
                .expect("failed to create input tensor");

        let mut session = self.session.lock().expect("session lock poisoned");
        let outputs = session
            .run(ort::inputs![&self.input_name => input_tensor])
            .expect("ONNX inference failed");

        // Extract raw output as (shape, &[f32]).
        let (_shape, output_data) = outputs[&*self.output_name]
            .try_extract_tensor::<f32>()
            .expect("failed to extract action tensor");

        let mut action_data: Vec<f32> = output_data.iter().copied().collect();
        action_data.truncate(self.action_dim);

        // Apply post-processing transform.
        self.apply_transform(&mut action_data);

        Action::from(action_data)
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "OnnxPolicy"
    }

    fn is_deterministic(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Search for a tensor name from a list of candidates, returning the first
/// match found among the session's inputs or outputs.
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

/// Extract the observation dimension from the named input tensor.
///
/// Expects shape `[batch, obs_dim]` and returns `obs_dim`.
fn extract_dim_from_input(
    session: &Session,
    name: &str,
) -> Result<usize, OnnxPolicyError> {
    for input in session.inputs() {
        if input.name() == name {
            if let ValueType::Tensor { shape, .. } = input.dtype() {
                // shape is a SmallVec<i64>; for [batch, obs_dim] take index 1
                if shape.len() >= 2 {
                    let dim = shape[1];
                    if dim > 0 {
                        return Ok(dim as usize);
                    }
                }
            }
        }
    }
    Err(OnnxPolicyError::UnknownObsDim)
}

/// Extract the action dimension from the named output tensor.
///
/// Expects shape `[batch, action_dim]` and returns `action_dim`.
fn extract_dim_from_output(
    session: &Session,
    name: &str,
) -> Result<usize, OnnxPolicyError> {
    for output in session.outputs() {
        if output.name() == name {
            if let ValueType::Tensor { shape, .. } = output.dtype() {
                if shape.len() >= 2 {
                    let dim = shape[1];
                    if dim > 0 {
                        return Ok(dim as usize);
                    }
                }
            }
        }
    }
    Err(OnnxPolicyError::UnknownActionDim)
}

/// Read all custom metadata from the session as a key-value map.
fn read_metadata(session: &Session) -> HashMap<String, String> {
    let Ok(meta) = session.metadata() else {
        return HashMap::new();
    };
    let Ok(keys) = meta.custom_keys() else {
        return HashMap::new();
    };
    let mut map = HashMap::new();
    for key in keys {
        if let Some(value) = meta.custom(&key) {
            map.insert(key, value);
        }
    }
    map
}

/// Parse the action transform from model metadata.
fn parse_action_transform(
    metadata: &HashMap<String, String>,
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
            let (low, high) = parse_clip_bounds(metadata, action_dim);
            ActionTransform::Clip { low, high }
        }
        _ => ActionTransform::None,
    }
}

/// Parse a JSON array of f32 values from a metadata string.
fn parse_f32_array(val: Option<&String>) -> Option<Vec<f32>> {
    val.and_then(|s| serde_json::from_str::<Vec<f32>>(s).ok())
}

/// Parse clip bounds from `action_space` metadata or fall back to defaults.
fn parse_clip_bounds(
    metadata: &HashMap<String, String>,
    action_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    if let Some(space_json) = metadata.get("action_space") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(space_json) {
            let low = v
                .get("low")
                .and_then(|a| serde_json::from_value::<Vec<f32>>(a.clone()).ok())
                .unwrap_or_else(|| vec![-1.0; action_dim]);
            let high = v
                .get("high")
                .and_then(|a| serde_json::from_value::<Vec<f32>>(a.clone()).ok())
                .unwrap_or_else(|| vec![1.0; action_dim]);
            return (low, high);
        }
    }
    (vec![-1.0; action_dim], vec![1.0; action_dim])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let policy = OnnxPolicy::from_file(fixture_path("test_policy_none.onnx"));
        assert!(policy.is_ok(), "Failed to load: {:?}", policy.err());
        let policy = policy.unwrap();
        assert_eq!(policy.obs_dim(), 4);
        assert_eq!(policy.action_dim(), 1);
    }

    #[test]
    fn load_tanh_model_has_tanh_transform() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_tanh.onnx")).unwrap();
        assert!(matches!(
            policy.action_transform(),
            ActionTransform::Tanh { .. }
        ));
        if let ActionTransform::Tanh { scale, offset } = policy.action_transform() {
            assert_eq!(scale, &[2.0, 2.0]);
            assert_eq!(offset, &[0.0, 0.0]);
        }
    }

    #[test]
    fn error_on_missing_file() {
        let result = OnnxPolicy::from_file("/nonexistent/model.onnx");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxPolicyError::LoadFailed { .. }
        ));
    }

    // -- Inference tests --

    #[test]
    fn get_action_returns_correct_dim() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_none.onnx")).unwrap();
        let obs = Observation::new(vec![1.0, 0.0, 0.0, 0.0]);
        let action = policy.get_action(&obs);
        assert_eq!(action.len(), 1);
    }

    #[test]
    fn get_action_zero_obs_returns_zero_action() {
        // With identity weights and zero bias, zero input yields zero output.
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_none.onnx")).unwrap();
        let obs = Observation::new(vec![0.0, 0.0, 0.0, 0.0]);
        let action = policy.get_action(&obs);
        for &v in action.as_slice() {
            assert!(
                v.abs() < 1e-6,
                "Expected near-zero action, got {v}",
            );
        }
    }

    #[test]
    fn get_action_deterministic_across_calls() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_none.onnx")).unwrap();
        let obs = Observation::new(vec![1.0, 2.0, 3.0, 4.0]);
        let a1 = policy.get_action(&obs);
        let a2 = policy.get_action(&obs);
        assert_eq!(a1.as_slice(), a2.as_slice());
    }

    // -- Action transform tests --

    #[test]
    fn tanh_transform_scales_output() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_tanh.onnx")).unwrap();
        let obs = Observation::new(vec![1.0, 0.0, 0.0, 0.0]);
        let action_tanh = policy.get_action(&obs);

        // The tanh model has scale=[2.0, 2.0] and offset=[0.0, 0.0].
        // The raw model uses identity weights, so raw output for [1,0,0,0]
        // is [1.0, 0.0]. After transform: [1.0*2.0+0.0, 0.0*2.0+0.0] = [2.0, 0.0].
        let expected = [2.0f32, 0.0];
        for (t, e) in action_tanh.as_slice().iter().zip(expected.iter()) {
            assert!(
                (t - e).abs() < 1e-5,
                "Expected {e}, got {t}",
            );
        }
    }

    // -- Trait compliance tests --

    #[test]
    fn policy_name() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_none.onnx")).unwrap();
        assert_eq!(policy.name(), "OnnxPolicy");
    }

    #[test]
    fn policy_is_deterministic() {
        let policy =
            OnnxPolicy::from_file(fixture_path("test_policy_none.onnx")).unwrap();
        assert!(policy.is_deterministic());
    }

    #[test]
    fn onnx_policy_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OnnxPolicy>();
    }
}
