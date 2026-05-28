//! Boundary contract validators (G2).
//!
//! CODE_QUALITY_REVIEW § "Gap 2: Contracts Are Not Enforced
//! Consistently". Promotes documented invariants to runtime validation
//! at process boundaries. Each validator returns a typed error that
//! callers (protocol decoders, recorders, evaluators) can match on.
//!
//! ## What's here
//!
//! - [`validate_action_against_space`] — every produced action fits the
//!   declared `ActionSpace`. Catches dimensionality mismatches, bounds
//!   violations, and NaN / Inf at the wire.
//! - [`validate_observation_against_space`] — every emitted observation
//!   fits the declared `ObservationSpace`. Used by recorder + Python
//!   client roundtrip validation.
//! - [`validate_trace_step`] — invariants on a single
//!   `(action, observation, reward, terminated, truncated)` tuple
//!   recorded into a trace.
//!
//! ## What's NOT here
//!
//! - Manifest schema validators land with G3.
//! - Protocol framing validators (binary header parity) live in
//!   `clankers-gym::binary_frame` / `tensor_frame`; this module is the
//!   semantic layer above the byte layer.

use thiserror::Error;

use crate::types::{ActionSpace, ObservationSpace};

// ---------------------------------------------------------------------------
// Action validator
// ---------------------------------------------------------------------------

/// Failure modes for [`validate_action_against_space`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ActionValidationError {
    /// Action vector length disagreed with the action space.
    #[error("action length {got} does not match action space dim {expected}")]
    LengthMismatch { got: usize, expected: usize },
    /// Component `i` violated the lower or upper bound declared by the
    /// action space.
    #[error("action[{index}] = {value} out of bounds [{low}, {high}] declared by action space")]
    OutOfBounds {
        index: usize,
        value: f32,
        low: f32,
        high: f32,
    },
    /// Component `i` was NaN or Inf.
    #[error("action[{index}] is not finite ({value})")]
    NonFinite { index: usize, value: f32 },
    /// Discrete action was non-integer / negative / out of range.
    #[error(
        "action[{index}] = {value} not a valid discrete index in [0, {n}); \
         must be a finite non-negative integer"
    )]
    InvalidDiscrete { index: usize, value: f32, n: usize },
}

/// Validate that `action` is a legal sample from `space`.
///
/// Boundary check intended for the producer side (action applicator)
/// and the recorder; the protocol decoder uses the same predicate via
/// [`crate::types::ActionSpace::contains`] today, which this function
/// elevates to a typed-error contract.
pub fn validate_action_against_space(
    action: &[f32],
    space: &ActionSpace,
) -> Result<(), ActionValidationError> {
    match space {
        ActionSpace::Box { low, high } => {
            if action.len() != low.len() {
                return Err(ActionValidationError::LengthMismatch {
                    got: action.len(),
                    expected: low.len(),
                });
            }
            for (i, &v) in action.iter().enumerate() {
                if !v.is_finite() {
                    return Err(ActionValidationError::NonFinite { index: i, value: v });
                }
                if v < low[i] || v > high[i] {
                    return Err(ActionValidationError::OutOfBounds {
                        index: i,
                        value: v,
                        low: low[i],
                        high: high[i],
                    });
                }
            }
            Ok(())
        }
        ActionSpace::Discrete { n } => {
            if action.len() != 1 {
                return Err(ActionValidationError::LengthMismatch {
                    got: action.len(),
                    expected: 1,
                });
            }
            let v = action[0];
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || (v as usize) >= *n {
                return Err(ActionValidationError::InvalidDiscrete {
                    index: 0,
                    value: v,
                    n: *n,
                });
            }
            Ok(())
        }
        ActionSpace::MultiDiscrete { nvec } => {
            if action.len() != nvec.len() {
                return Err(ActionValidationError::LengthMismatch {
                    got: action.len(),
                    expected: nvec.len(),
                });
            }
            for (i, (&v, &n)) in action.iter().zip(nvec.iter()).enumerate() {
                if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || (v as usize) >= n {
                    return Err(ActionValidationError::InvalidDiscrete {
                        index: i,
                        value: v,
                        n,
                    });
                }
            }
            Ok(())
        }
        ActionSpace::MultiBinary { n } => {
            if action.len() != *n {
                return Err(ActionValidationError::LengthMismatch {
                    got: action.len(),
                    expected: *n,
                });
            }
            for (i, &v) in action.iter().enumerate() {
                if !v.is_finite() || (v != 0.0 && v != 1.0) {
                    return Err(ActionValidationError::InvalidDiscrete {
                        index: i,
                        value: v,
                        n: 2,
                    });
                }
            }
            Ok(())
        }
        ActionSpace::Dict { .. } => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Observation validator
// ---------------------------------------------------------------------------

/// Failure modes for [`validate_observation_against_space`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObservationValidationError {
    /// Observation length disagreed with the observation space.
    #[error("observation length {got} does not match observation space dim {expected}")]
    LengthMismatch { got: usize, expected: usize },
    /// Observation component `i` was NaN or Inf.
    #[error("observation[{index}] is not finite ({value})")]
    NonFinite { index: usize, value: f32 },
    /// Discrete observation was non-integer / negative / out of range.
    #[error(
        "observation[{index}] = {value} not a valid discrete index in [0, {n}); \
         must be a finite non-negative integer"
    )]
    InvalidDiscrete { index: usize, value: f32, n: usize },
}

/// Validate that `obs` matches `space` shape and finiteness.
///
/// Bounds for `Box` observation spaces are not enforced — many envs
/// declare wide synthetic bounds that are not tight on the observation
/// distribution itself (e.g. position observations declared as
/// `[-1e9, 1e9]`). The contract is: shape, finiteness, and discrete
/// integer validity.
pub fn validate_observation_against_space(
    obs: &[f32],
    space: &ObservationSpace,
) -> Result<(), ObservationValidationError> {
    match space {
        ObservationSpace::Box { low, high } => {
            if obs.len() != low.len() || obs.len() != high.len() {
                return Err(ObservationValidationError::LengthMismatch {
                    got: obs.len(),
                    expected: low.len(),
                });
            }
            for (i, &v) in obs.iter().enumerate() {
                if !v.is_finite() {
                    return Err(ObservationValidationError::NonFinite { index: i, value: v });
                }
            }
            Ok(())
        }
        ObservationSpace::Discrete { n } => {
            if obs.len() != 1 {
                return Err(ObservationValidationError::LengthMismatch {
                    got: obs.len(),
                    expected: 1,
                });
            }
            let v = obs[0];
            if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || (v as usize) >= *n {
                return Err(ObservationValidationError::InvalidDiscrete {
                    index: 0,
                    value: v,
                    n: *n,
                });
            }
            Ok(())
        }
        ObservationSpace::MultiDiscrete { nvec } => {
            if obs.len() != nvec.len() {
                return Err(ObservationValidationError::LengthMismatch {
                    got: obs.len(),
                    expected: nvec.len(),
                });
            }
            for (i, (&v, &n)) in obs.iter().zip(nvec.iter()).enumerate() {
                if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || (v as usize) >= n {
                    return Err(ObservationValidationError::InvalidDiscrete {
                        index: i,
                        value: v,
                        n,
                    });
                }
            }
            Ok(())
        }
        ObservationSpace::MultiBinary { n } => {
            if obs.len() != *n {
                return Err(ObservationValidationError::LengthMismatch {
                    got: obs.len(),
                    expected: *n,
                });
            }
            for (i, &v) in obs.iter().enumerate() {
                if !v.is_finite() || (v != 0.0 && v != 1.0) {
                    return Err(ObservationValidationError::InvalidDiscrete {
                        index: i,
                        value: v,
                        n: 2,
                    });
                }
            }
            Ok(())
        }
        ObservationSpace::Image {
            width,
            height,
            channels,
        } => {
            let expected = (*width as usize) * (*height as usize) * (*channels as usize);
            if obs.len() != expected {
                return Err(ObservationValidationError::LengthMismatch {
                    got: obs.len(),
                    expected,
                });
            }
            for (i, &v) in obs.iter().enumerate() {
                if !v.is_finite() {
                    return Err(ObservationValidationError::NonFinite { index: i, value: v });
                }
            }
            Ok(())
        }
        ObservationSpace::Dict { .. } => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Trace-step validator
// ---------------------------------------------------------------------------

/// Failure modes for [`validate_trace_step`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TraceValidationError {
    /// The contained action failed validation.
    #[error("action validation failed: {0}")]
    Action(ActionValidationError),
    /// The contained observation failed validation.
    #[error("observation validation failed: {0}")]
    Observation(ObservationValidationError),
    /// Reward was NaN.
    #[error("reward is NaN")]
    RewardNan,
    /// Both `terminated` and `truncated` flags were set on the same step.
    #[error("trace step claims both terminated AND truncated; pick one")]
    TerminatedAndTruncated,
}

/// Validate a single `(action, observation, reward, terminated,
/// truncated)` trace step against the env's declared spaces.
///
/// Used by the recorder and by Python evaluators reading back an MCAP
/// trace — catches the most common silent-drift failure modes the
/// review called out (NaN reward, action/obs shape mismatch, both
/// termination flags set).
pub fn validate_trace_step(
    action: &[f32],
    action_space: &ActionSpace,
    obs: &[f32],
    obs_space: &ObservationSpace,
    reward: f32,
    terminated: bool,
    truncated: bool,
) -> Result<(), TraceValidationError> {
    validate_action_against_space(action, action_space).map_err(TraceValidationError::Action)?;
    validate_observation_against_space(obs, obs_space)
        .map_err(TraceValidationError::Observation)?;
    if reward.is_nan() {
        return Err(TraceValidationError::RewardNan);
    }
    if terminated && truncated {
        return Err(TraceValidationError::TerminatedAndTruncated);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn box_2() -> ActionSpace {
        ActionSpace::Box {
            low: vec![-1.0, -2.0],
            high: vec![1.0, 2.0],
        }
    }

    #[test]
    fn validate_action_accepts_in_bounds() {
        validate_action_against_space(&[0.5, -1.0], &box_2()).unwrap();
    }

    #[test]
    fn validate_action_rejects_length_mismatch() {
        let err = validate_action_against_space(&[0.0], &box_2()).unwrap_err();
        assert_eq!(
            err,
            ActionValidationError::LengthMismatch {
                got: 1,
                expected: 2
            }
        );
    }

    #[test]
    fn validate_action_rejects_out_of_bounds() {
        let err = validate_action_against_space(&[0.0, 3.0], &box_2()).unwrap_err();
        assert!(matches!(err, ActionValidationError::OutOfBounds { .. }));
    }

    #[test]
    fn validate_action_rejects_nan() {
        let err = validate_action_against_space(&[f32::NAN, 0.0], &box_2()).unwrap_err();
        assert!(matches!(err, ActionValidationError::NonFinite { .. }));
    }

    #[test]
    fn validate_discrete_action_rejects_negative() {
        let err =
            validate_action_against_space(&[-1.0], &ActionSpace::Discrete { n: 4 }).unwrap_err();
        assert!(matches!(err, ActionValidationError::InvalidDiscrete { .. }));
    }

    #[test]
    fn validate_obs_accepts_finite_in_box() {
        let obs_space = ObservationSpace::Box {
            low: vec![-1.0, -1.0],
            high: vec![1.0, 1.0],
        };
        validate_observation_against_space(&[10.0, -50.0], &obs_space).unwrap();
    }

    #[test]
    fn validate_obs_rejects_inf() {
        let obs_space = ObservationSpace::Box {
            low: vec![-1.0],
            high: vec![1.0],
        };
        let err = validate_observation_against_space(&[f32::INFINITY], &obs_space).unwrap_err();
        assert!(matches!(err, ObservationValidationError::NonFinite { .. }));
    }

    #[test]
    fn validate_trace_rejects_terminated_and_truncated() {
        let err = validate_trace_step(
            &[0.0],
            &ActionSpace::Box {
                low: vec![-1.0],
                high: vec![1.0],
            },
            &[0.0],
            &ObservationSpace::Box {
                low: vec![-1.0],
                high: vec![1.0],
            },
            0.5,
            true,
            true,
        )
        .unwrap_err();
        assert_eq!(err, TraceValidationError::TerminatedAndTruncated);
    }

    #[test]
    fn validate_trace_rejects_nan_reward() {
        let err = validate_trace_step(
            &[0.0],
            &ActionSpace::Box {
                low: vec![-1.0],
                high: vec![1.0],
            },
            &[0.0],
            &ObservationSpace::Box {
                low: vec![-1.0],
                high: vec![1.0],
            },
            f32::NAN,
            false,
            false,
        )
        .unwrap_err();
        assert_eq!(err, TraceValidationError::RewardNan);
    }

    #[test]
    fn validate_image_obs_checks_pixel_count() {
        let space = ObservationSpace::Image {
            width: 4,
            height: 3,
            channels: 3,
        };
        let buf: Vec<f32> = vec![0.5; 4 * 3 * 3];
        validate_observation_against_space(&buf, &space).unwrap();
        // Wrong size.
        let err = validate_observation_against_space(&[0.5; 10], &space).unwrap_err();
        assert!(matches!(
            err,
            ObservationValidationError::LengthMismatch { .. }
        ));
    }
}
