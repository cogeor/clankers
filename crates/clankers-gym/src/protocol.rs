//! Gymnasium-compatible message protocol.
//!
//! Defines the JSON-serialisable request/response types used to communicate
//! between a training client (e.g. Python) and the Clankers simulation server.
//!
//! The protocol follows a simple command-response pattern:
//!
//! 1. Client sends a [`Request`]
//! 2. Server processes it and replies with a [`Response`]
//!
//! All messages are newline-delimited JSON.

use serde::{Deserialize, Serialize};

use clankers_core::types::{
    Action, ActionSpace, Observation, ObservationSpace, ResetInfo, ResetResult, StepInfo,
    StepResult,
};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// A request from the training client to the simulation server.
///
/// # Example
///
/// ```
/// use clankers_gym::protocol::Request;
///
/// let json = r#"{"type":"reset","seed":42}"#;
/// let req: Request = serde_json::from_str(json).unwrap();
/// assert!(matches!(req, Request::Reset { seed: Some(42) }));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Query the observation and action spaces.
    Spaces,
    /// Reset the environment, optionally with a seed.
    Reset {
        /// Optional RNG seed for reproducibility.
        #[serde(default)]
        seed: Option<u64>,
    },
    /// Take a step with the given action.
    Step {
        /// The action to apply.
        action: Action,
    },
    /// Close the environment and disconnect.
    Close,
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// A response from the simulation server to the training client.
///
/// # Example
///
/// ```
/// use clankers_gym::protocol::Response;
///
/// let resp = Response::Close;
/// let json = serde_json::to_string(&resp).unwrap();
/// assert!(json.contains("close"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// Observation and action space descriptions.
    Spaces {
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    },
    /// Result of a reset operation.
    Reset {
        observation: Observation,
        info: ResetInfo,
    },
    /// Result of a step operation.
    Step {
        observation: Observation,
        reward: f32,
        terminated: bool,
        truncated: bool,
        info: StepInfo,
    },
    /// Acknowledgement of close.
    Close,
    /// Error response.
    Error { message: String },
}

impl Response {
    /// Create a reset response from a [`ResetResult`].
    #[must_use]
    pub fn from_reset(result: ResetResult) -> Self {
        Self::Reset {
            observation: result.observation,
            info: result.info,
        }
    }

    /// Create a step response from a [`StepResult`].
    #[must_use]
    pub fn from_step(result: StepResult) -> Self {
        Self::Step {
            observation: result.observation,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        }
    }

    /// Create an error response.
    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    // ---- Request serialisation ----

    #[test]
    fn request_spaces_roundtrip() {
        let req = Request::Spaces;
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(req2, Request::Spaces));
    }

    #[test]
    fn request_reset_with_seed() {
        let req = Request::Reset { seed: Some(42) };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::Reset { seed } = req2 {
            assert_eq!(seed, Some(42));
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn request_reset_without_seed() {
        let json = r#"{"type":"reset"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        if let Request::Reset { seed } = req {
            assert_eq!(seed, None);
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn request_step_continuous() {
        let req = Request::Step {
            action: Action::Continuous(vec![0.5, -0.3]),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::Step { action } = req2 {
            assert_eq!(action, Action::Continuous(vec![0.5, -0.3]));
        } else {
            panic!("expected Step");
        }
    }

    #[test]
    fn request_step_discrete() {
        let req = Request::Step {
            action: Action::Discrete(3),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::Step { action } = req2 {
            assert_eq!(action, Action::Discrete(3));
        } else {
            panic!("expected Step");
        }
    }

    #[test]
    fn request_close_roundtrip() {
        let req = Request::Close;
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        assert!(matches!(req2, Request::Close));
    }

    // ---- Response serialisation ----

    #[test]
    fn response_spaces_roundtrip() {
        let resp = Response::Spaces {
            observation_space: ObservationSpace::Box {
                low: vec![-1.0; 3],
                high: vec![1.0; 3],
            },
            action_space: ActionSpace::Discrete { n: 4 },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Spaces {
            observation_space,
            action_space,
        } = resp2
        {
            assert_eq!(observation_space.shape(), vec![3]);
            assert_eq!(action_space.size(), 1);
        } else {
            panic!("expected Spaces");
        }
    }

    #[test]
    fn response_reset_roundtrip() {
        let resp = Response::Reset {
            observation: Observation::new(vec![1.0, 2.0]),
            info: ResetInfo {
                seed: Some(7),
                custom: HashMap::new(),
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Reset { observation, info } = resp2 {
            assert_eq!(observation.len(), 2);
            assert_eq!(info.seed, Some(7));
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn response_step_roundtrip() {
        let result = StepResult {
            observation: Observation::new(vec![0.5]),
            reward: 1.5,
            terminated: false,
            truncated: true,
            info: StepInfo {
                episode_length: 10,
                episode_reward: 5.0,
                custom: HashMap::new(),
            },
        };
        let resp = Response::from_step(result);
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Step {
            reward, truncated, ..
        } = resp2
        {
            assert!((reward - 1.5).abs() < f32::EPSILON);
            assert!(truncated);
        } else {
            panic!("expected Step");
        }
    }

    #[test]
    fn response_from_reset() {
        let result = ResetResult {
            observation: Observation::zeros(3),
            info: ResetInfo::default(),
        };
        let resp = Response::from_reset(result);
        assert!(matches!(resp, Response::Reset { .. }));
    }

    #[test]
    fn response_close_roundtrip() {
        let resp = Response::Close;
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        assert!(matches!(resp2, Response::Close));
    }

    #[test]
    fn response_error() {
        let resp = Response::error("something went wrong");
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Error { message } = resp2 {
            assert_eq!(message, "something went wrong");
        } else {
            panic!("expected Error");
        }
    }

    #[test]
    fn request_from_raw_json_step() {
        let json = r#"{"type":"step","action":{"Continuous":[0.1,0.2]}}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        if let Request::Step { action } = req {
            assert_eq!(action, Action::Continuous(vec![0.1, 0.2]));
        } else {
            panic!("expected Step");
        }
    }
}
