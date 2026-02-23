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
use thiserror::Error;

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
// Protocol Version & Constants
// ---------------------------------------------------------------------------

/// Protocol version string (semantic versioning) per `PROTOCOL_SPEC.md`.
pub const PROTOCOL_VERSION: &str = "1.0.0";

/// Maximum JSON payload size in bytes (16 MiB).
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Maximum observation data size in bytes (8 MiB).
pub const MAX_OBSERVATION_SIZE: usize = 8 * 1024 * 1024;

/// Maximum action data size in bytes (1 MiB).
pub const MAX_ACTION_SIZE: usize = 1024 * 1024;

/// Protocol timeout durations per `PROTOCOL_SPEC.md`.
pub mod timeouts {
    use std::time::Duration;

    /// Timeout for initial TCP connection.
    pub const CONNECT: Duration = Duration::from_secs(30);
    /// Timeout for handshake completion.
    pub const HANDSHAKE: Duration = Duration::from_secs(10);
    /// General request timeout.
    pub const REQUEST: Duration = Duration::from_secs(60);
    /// Timeout for step responses.
    pub const STEP: Duration = Duration::from_secs(5);
    /// Timeout for reset responses.
    pub const RESET: Duration = Duration::from_secs(30);
    /// Timeout for close acknowledgment.
    pub const CLOSE: Duration = Duration::from_secs(5);
    /// Interval between keepalive pings.
    pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(30);
    /// Timeout for keepalive response.
    pub const KEEPALIVE: Duration = Duration::from_secs(10);
}

// ---------------------------------------------------------------------------
// Protocol State
// ---------------------------------------------------------------------------

/// Protocol finite state machine states.
///
/// Governs which messages are valid at any point in the connection lifecycle.
/// See `PROTOCOL_SPEC.md` Section 3 for the full state transition table.
///
/// # Example
///
/// ```
/// use clankers_gym::protocol::ProtocolState;
///
/// let state = ProtocolState::Disconnected;
/// assert_eq!(format!("{state}"), "DISCONNECTED");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolState {
    /// No TCP connection established.
    Disconnected,
    /// TCP connection established, awaiting handshake.
    Connected,
    /// Init message sent/received, awaiting response.
    Handshaking,
    /// Handshake complete, environment ready for reset or close.
    Ready,
    /// Episode active, accepting step commands.
    EpisodeRunning,
    /// Close requested, awaiting acknowledgment.
    Closing,
    /// Unrecoverable error, connection should be terminated.
    Error,
}

impl std::fmt::Display for ProtocolState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Disconnected => "DISCONNECTED",
            Self::Connected => "CONNECTED",
            Self::Handshaking => "HANDSHAKING",
            Self::Ready => "READY",
            Self::EpisodeRunning => "EPISODE_RUNNING",
            Self::Closing => "CLOSING",
            Self::Error => "ERROR",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Protocol Errors
// ---------------------------------------------------------------------------

/// Protocol error types with numeric codes per `PROTOCOL_SPEC.md` Section 5.
///
/// Error codes are grouped by category:
/// - **1xx** — Protocol errors (version, framing, sequencing)
/// - **2xx** — Validation errors (action, seed, options)
/// - **3xx** — Simulation errors (env state, physics)
/// - **4xx** — Internal errors (resources, timeouts, I/O)
#[derive(Debug, Error)]
pub enum ProtocolError {
    // --- Protocol (1xx) ---
    /// Version mismatch between client and server (100).
    #[error("version mismatch: client={client}, server={server}")]
    VersionMismatch {
        /// Client's protocol version.
        client: String,
        /// Server's protocol version.
        server: String,
    },

    /// Malformed message payload (101).
    #[error("invalid message: {0}")]
    InvalidMessage(String),

    /// Unrecognised message type field (102).
    #[error("unknown message type: {0}")]
    UnknownMessageType(String),

    /// Message received in wrong protocol state (103).
    #[error(
        "unexpected message '{got}' in state {current_state} (expected: {expected:?})"
    )]
    UnexpectedMessage {
        /// Current protocol state.
        current_state: ProtocolState,
        /// Message types valid in this state.
        expected: Vec<String>,
        /// Message type that was actually received.
        got: String,
    },

    /// Payload exceeds size limit (104).
    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge {
        /// Actual payload size.
        size: usize,
        /// Maximum allowed size.
        max: usize,
    },

    /// Messages arrived out of sequence (105).
    #[error("invalid message sequence")]
    InvalidSequence,

    /// Requested capability not supported (106).
    #[error("capability unsupported: {0}")]
    CapabilityUnsupported(String),

    // --- Validation (2xx) ---

    /// Action format is invalid (200).
    #[error("invalid action: {0}")]
    InvalidAction(String),

    /// Action value out of space bounds (201).
    #[error(
        "action out of bounds: index {index}, value {value}, bounds ({}, {})",
        bounds.0,
        bounds.1
    )]
    ActionOutOfBounds {
        /// Index of the out-of-bounds element.
        index: usize,
        /// The offending value.
        value: f32,
        /// The (low, high) bounds.
        bounds: (f32, f32),
    },

    /// Invalid seed value (202).
    #[error("invalid seed: {0}")]
    InvalidSeed(String),

    /// Invalid options in request (203).
    #[error("invalid options: {0}")]
    InvalidOptions(String),

    /// Required field missing from message (204).
    #[error("missing required field: {0}")]
    MissingRequiredField(String),

    /// Field has wrong type (205).
    #[error("type error: {0}")]
    TypeError(String),

    /// Action contains NaN values (206).
    #[error("action contains NaN at indices: {0:?}")]
    ActionContainsNan(Vec<usize>),

    /// Action contains infinity values (207).
    #[error("action contains infinity at indices: {0:?}")]
    ActionContainsInf(Vec<usize>),

    // --- Simulation (3xx) ---

    /// Environment not initialized (300).
    #[error("environment not initialized")]
    EnvNotInitialized,

    /// Environment not reset (301).
    #[error("environment not reset")]
    EnvNotReset,

    /// Environment episode already done (302).
    #[error("environment already done")]
    EnvAlreadyDone,

    /// Agent not found (303).
    #[error("agent not found: {0}")]
    AgentNotFound(String),

    /// Simulation error (304).
    #[error("simulation error: {0}")]
    SimulationError(String),

    /// Physics became unstable (305).
    #[error("physics unstable")]
    PhysicsUnstable,

    // --- Internal (4xx) ---

    /// Internal server error (400).
    #[error("internal error: {0}")]
    InternalError(String),

    /// Server resources exhausted (401).
    #[error("resource exhausted")]
    ResourceExhausted,

    /// Operation timed out (402).
    #[error("timeout")]
    Timeout,

    /// Server shutting down (403).
    #[error("shutdown")]
    Shutdown,

    /// Connection lost (404).
    #[error("connection lost")]
    ConnectionLost,

    // --- I/O ---

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl ProtocolError {
    /// Numeric error code per `PROTOCOL_SPEC.md` Section 5.2.
    #[must_use]
    pub const fn code(&self) -> u16 {
        match self {
            Self::VersionMismatch { .. } => 100,
            Self::InvalidMessage(_) | Self::Json(_) => 101,
            Self::UnknownMessageType(_) => 102,
            Self::UnexpectedMessage { .. } => 103,
            Self::PayloadTooLarge { .. } => 104,
            Self::InvalidSequence => 105,
            Self::CapabilityUnsupported(_) => 106,
            Self::InvalidAction(_) => 200,
            Self::ActionOutOfBounds { .. } => 201,
            Self::InvalidSeed(_) => 202,
            Self::InvalidOptions(_) => 203,
            Self::MissingRequiredField(_) => 204,
            Self::TypeError(_) => 205,
            Self::ActionContainsNan(_) => 206,
            Self::ActionContainsInf(_) => 207,
            Self::EnvNotInitialized => 300,
            Self::EnvNotReset => 301,
            Self::EnvAlreadyDone => 302,
            Self::AgentNotFound(_) => 303,
            Self::SimulationError(_) => 304,
            Self::PhysicsUnstable => 305,
            Self::InternalError(_) | Self::Io(_) => 400,
            Self::ResourceExhausted => 401,
            Self::Timeout => 402,
            Self::Shutdown => 403,
            Self::ConnectionLost => 404,
        }
    }

    /// Whether the error is recoverable (connection can continue).
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(self.code(), 101..=106 | 200..=299 | 301..=303)
    }

    /// Error category string.
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self.code() {
            100..=199 => "protocol",
            200..=299 => "validation",
            300..=399 => "simulation",
            _ => "internal",
        }
    }

    /// Convert this error into a [`Response::Error`].
    #[must_use]
    pub fn into_response(self) -> Response {
        Response::Error {
            message: self.to_string(),
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

    // ---- Protocol version & constants ----

    #[test]
    fn protocol_version_is_semver() {
        let parts: Vec<&str> = PROTOCOL_VERSION.split('.').collect();
        assert_eq!(parts.len(), 3);
        for part in parts {
            part.parse::<u32>().unwrap();
        }
    }

    #[test]
    fn size_limits_are_ordered() {
        let msg = MAX_MESSAGE_SIZE;
        let obs = MAX_OBSERVATION_SIZE;
        let act = MAX_ACTION_SIZE;
        assert_eq!(msg, 16 * 1024 * 1024);
        assert!(obs < msg);
        assert!(act < obs);
    }

    #[test]
    fn timeouts_are_positive() {
        assert!(!timeouts::CONNECT.is_zero());
        assert!(!timeouts::HANDSHAKE.is_zero());
        assert!(!timeouts::REQUEST.is_zero());
        assert!(!timeouts::STEP.is_zero());
        assert!(!timeouts::RESET.is_zero());
        assert!(!timeouts::CLOSE.is_zero());
        assert!(!timeouts::KEEPALIVE_INTERVAL.is_zero());
        assert!(!timeouts::KEEPALIVE.is_zero());
    }

    // ---- ProtocolState ----

    #[test]
    fn protocol_state_display() {
        assert_eq!(ProtocolState::Disconnected.to_string(), "DISCONNECTED");
        assert_eq!(ProtocolState::Connected.to_string(), "CONNECTED");
        assert_eq!(ProtocolState::Handshaking.to_string(), "HANDSHAKING");
        assert_eq!(ProtocolState::Ready.to_string(), "READY");
        assert_eq!(ProtocolState::EpisodeRunning.to_string(), "EPISODE_RUNNING");
        assert_eq!(ProtocolState::Closing.to_string(), "CLOSING");
        assert_eq!(ProtocolState::Error.to_string(), "ERROR");
    }

    #[test]
    fn protocol_state_clone_and_eq() {
        let state = ProtocolState::Ready;
        let cloned = state;
        assert_eq!(state, cloned);
        assert_ne!(ProtocolState::Ready, ProtocolState::Error);
    }

    // ---- ProtocolError ----

    #[test]
    fn error_code_protocol_category() {
        let err = ProtocolError::VersionMismatch {
            client: "1.0.0".into(),
            server: "2.0.0".into(),
        };
        assert_eq!(err.code(), 100);
        assert_eq!(err.category(), "protocol");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn error_code_validation_category() {
        let err = ProtocolError::InvalidAction("bad shape".into());
        assert_eq!(err.code(), 200);
        assert_eq!(err.category(), "validation");
        assert!(err.is_recoverable());
    }

    #[test]
    fn error_code_simulation_category() {
        let err = ProtocolError::EnvNotReset;
        assert_eq!(err.code(), 301);
        assert_eq!(err.category(), "simulation");
        assert!(err.is_recoverable());
    }

    #[test]
    fn error_code_internal_category() {
        let err = ProtocolError::InternalError("oops".into());
        assert_eq!(err.code(), 400);
        assert_eq!(err.category(), "internal");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn error_into_response() {
        let err = ProtocolError::EnvNotReset;
        let resp = err.into_response();
        if let Response::Error { message } = resp {
            assert!(message.contains("not reset"));
        } else {
            panic!("expected Error response");
        }
    }

    #[test]
    fn error_display_includes_details() {
        let err = ProtocolError::PayloadTooLarge {
            size: 20_000_000,
            max: MAX_MESSAGE_SIZE,
        };
        let msg = err.to_string();
        assert!(msg.contains("20000000"));
        assert!(msg.contains("16777216"));
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broken");
        let err = ProtocolError::from(io_err);
        assert_eq!(err.code(), 400);
        assert!(!err.is_recoverable());
    }

    #[test]
    fn error_unexpected_message() {
        let err = ProtocolError::UnexpectedMessage {
            current_state: ProtocolState::Ready,
            expected: vec!["reset".into(), "close".into()],
            got: "step".into(),
        };
        assert_eq!(err.code(), 103);
        assert!(err.is_recoverable());
        let msg = err.to_string();
        assert!(msg.contains("READY"));
        assert!(msg.contains("step"));
    }
}
