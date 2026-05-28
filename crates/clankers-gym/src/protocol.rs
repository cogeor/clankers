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
//! # Wire format
//!
//! Each message on the wire is a **4-byte little-endian `u32` length
//! prefix** followed by that many bytes of UTF-8 JSON payload. When an
//! image observation is sent (negotiated via the `binary_obs`
//! capability), a *second* length-prefixed frame carrying the raw `u8`
//! pixel bytes follows immediately after the JSON response frame. See
//! [`crate::framing`] for the helpers and
//! [`crate::encoding::EncodedObservation`] for the JSON header shape.
//!
//! Wire format unchanged since `1.0.0` for non-batch frames. The
//! `1.0.0 → 1.1.0` bump advertised that image observations now ship on
//! `Reset` as well as `Step`. The `1.1.0 → 1.2.0` bump (W7 PR2)
//! advertises the new `binary_batch` capability: when the client opts
//! in, batched observations ship as a single `BinaryFrameHeader` +
//! flat payload on the existing binary channel, rather than as inline
//! JSON arrays.
//!
//! # Annotated hex example — single-env image reset
//!
//! A `Reset` response carrying a 64×64×3 RGB image looks like:
//!
//! ```text
//! +--------+--------------------------------------------+
//! | 4 B LE | JSON header                                |
//! | length | {"type":"reset","observation":{"data":[]}, |
//! |        |  "info":{...},"obs_encoding":{             |
//! |        |     "type":"RawU8Image","width":64,        |
//! |        |     "height":64,"channels":3,              |
//! |        |     "layout":"Hwc"}}                       |
//! +--------+--------------------------------------------+
//! +--------+--------------------------------------------+
//! | 4 B LE | 12_288 bytes of raw RGB pixel data         |
//! | = 12288| (64 * 64 * 3, row-major HWC)               |
//! +--------+--------------------------------------------+
//! ```
//!
//! # Annotated hex example — batch f32 step (`1.2.0`, opt-in)
//!
//! A `BatchStep` response carrying `num_envs=4, obs_dim=8` f32
//! observations looks like:
//!
//! ```text
//! +--------+--------------------------------------------+
//! | 4 B LE | JSON envelope                              |
//! | length | {"type":"batch_step","observations":[      |
//! |        |   {"data":[]}, {"data":[]},                |
//! |        |   {"data":[]}, {"data":[]}],               |
//! |        |  "rewards":[0,0,0,0], "terminated":[...],  |
//! |        |  "truncated":[...], "infos":[...],         |
//! |        |  "obs_encoding":{                          |
//! |        |     "type":"BatchF32","num_envs":4,        |
//! |        |     "obs_dim":8}}                          |
//! +--------+--------------------------------------------+
//! +--------+--------------------------------------------+
//! | 4 B LE | 24-byte BinaryFrameHeader                  |
//! | = 24 + |   version=1 kind=0 num_envs=4 dim=8        |
//! |  128   | 128 bytes of f32 payload (4 * 8 * 4)       |
//! +--------+--------------------------------------------+
//! ```
//!
//! ```
//! // protocol_doc_matches_wire_format: round-trip a Reset response
//! // through the documented 4-byte LE length-prefixed framing.
//! use clankers_gym::framing::{read_message, write_message};
//! use clankers_gym::protocol::Response;
//! use std::io::Cursor;
//!
//! let resp = Response::Reset {
//!     observation: clankers_core::types::Observation::zeros(0),
//!     info: clankers_core::types::ResetInfo::default(),
//!     obs_encoding: None,
//! };
//! let mut buf = Vec::new();
//! write_message(&mut buf, &resp).unwrap();
//!
//! // First 4 bytes are the little-endian length prefix.
//! let len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
//! assert_eq!(len, buf.len() - 4);
//!
//! let mut cursor = Cursor::new(&buf);
//! let decoded: Response = read_message(&mut cursor).unwrap().unwrap();
//! assert!(matches!(decoded, Response::Reset { .. }));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

/// Typed, internal view of the negotiated capability set.
///
/// `CODE_QUALITY_REVIEW` Finding "Capability Negotiation Uses Raw Strings"
/// / P1.3. The wire format remains a `{ "binary_obs": true, ... }` JSON
/// object so the protocol stays backward-compatible with existing
/// clients (incl. Python). Internal server/session code reads typed
/// fields (`caps.binary_obs`) instead of `caps.get("binary_obs")`.
///
/// Unknown capability keys round-trip unchanged via the [`Self::unknown`]
/// catchall so forward-compatible negotiation still works.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Client supports / server advertises image observations as a
    /// length-prefixed binary frame after the JSON reply.
    pub binary_obs: bool,
    /// Client supports / server advertises batched observations as a
    /// `BinaryFrameHeader` + flat payload (W7 PR2).
    pub binary_batch: bool,
    /// Client supports / server advertises `VecEnv` batched step / reset
    /// requests.
    pub batch_step: bool,
    /// Any wire-level capability key not represented as a typed field.
    /// Preserves forward compatibility with newer / older peers — the
    /// `negotiate_with` server logic treats unknown keys as best-effort
    /// AND of two booleans.
    pub unknown: HashMap<String, bool>,
}

impl Capabilities {
    /// Logical AND of two capability sets, used during handshake.
    ///
    /// Returns a [`Capabilities`] where each typed flag is `self && other`
    /// and unknown keys present on both sides are `ANDed`; keys present on
    /// only one side are dropped (matches the legacy HashMap-based
    /// negotiation behaviour).
    #[must_use]
    pub fn negotiate_with(&self, other: &Self) -> Self {
        let unknown: HashMap<String, bool> = self
            .unknown
            .iter()
            .filter_map(|(k, v)| other.unknown.get(k).map(|ov| (k.clone(), *v && *ov)))
            .collect();
        Self {
            binary_obs: self.binary_obs && other.binary_obs,
            binary_batch: self.binary_batch && other.binary_batch,
            batch_step: self.batch_step && other.batch_step,
            unknown,
        }
    }
}

impl From<HashMap<String, bool>> for Capabilities {
    fn from(mut map: HashMap<String, bool>) -> Self {
        let binary_obs = map.remove("binary_obs").unwrap_or(false);
        let binary_batch = map.remove("binary_batch").unwrap_or(false);
        let batch_step = map.remove("batch_step").unwrap_or(false);
        Self {
            binary_obs,
            binary_batch,
            batch_step,
            unknown: map,
        }
    }
}

// The `unknown` field is typed `HashMap<String, bool>` with the
// default hasher; this impl reuses it, so it can't be generic over
// hashers without re-allocating.
#[allow(clippy::implicit_hasher)]
impl From<Capabilities> for HashMap<String, bool> {
    fn from(caps: Capabilities) -> Self {
        let mut map = caps.unknown;
        map.insert("binary_obs".into(), caps.binary_obs);
        map.insert("binary_batch".into(), caps.binary_batch);
        map.insert("batch_step".into(), caps.batch_step);
        map
    }
}

impl Serialize for Capabilities {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let map: HashMap<String, bool> = self.clone().into();
        map.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Capabilities {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let map: HashMap<String, bool> = HashMap::deserialize(deserializer)?;
        Ok(map.into())
    }
}

use clankers_core::types::{
    Action, ActionSpace, BatchResetResult, BatchStepResult, Observation, ObservationSpace,
    ResetInfo, ResetResult, StepInfo, StepResult,
};

pub use crate::encoding::{EncodedObservation, ImageLayout};

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
    /// Initialize connection (handshake). Must be the first message after TCP connect.
    Init {
        /// Protocol version the client supports.
        protocol_version: String,
        /// Client identifier.
        client_name: String,
        /// Client version string.
        client_version: String,
        /// Capability flags the client supports. Wire-compatible with
        /// the legacy `{ "key": bool, ... }` JSON object — see
        /// [`Capabilities`] for the typed view.
        #[serde(default)]
        capabilities: Capabilities,
        /// Optional seed for deterministic operation.
        #[serde(default)]
        seed: Option<u64>,
    },
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
    /// Reset multiple environments in a `VecEnv` batch.
    ///
    /// Requires the `batch_step` capability to be negotiated.
    BatchReset {
        /// Environment indices to reset.
        env_ids: Vec<u16>,
        /// Optional per-env seeds (must be same length as `env_ids` if present).
        #[serde(default)]
        seeds: Option<Vec<Option<u64>>>,
    },
    /// Step all environments in a `VecEnv` batch.
    ///
    /// Requires the `batch_step` capability to be negotiated.
    BatchStep {
        /// Actions for each environment. Length must equal `num_envs`.
        actions: Vec<Action>,
    },
    /// Keepalive probe.
    Ping {
        /// Client-side timestamp (epoch milliseconds).
        timestamp: u64,
    },
}

// ---------------------------------------------------------------------------
// EnvInfo
// ---------------------------------------------------------------------------

/// Environment metadata sent during the handshake.
///
/// Included in [`Response::InitResponse`] so the client knows the
/// observation/action shapes and agent count before the first reset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvInfo {
    /// Number of agents in the environment.
    pub n_agents: usize,
    /// Observation space descriptor.
    pub observation_space: ObservationSpace,
    /// Action space descriptor.
    pub action_space: ActionSpace,
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
    /// Handshake reply with negotiated capabilities and environment metadata.
    InitResponse {
        /// Negotiated protocol version.
        protocol_version: String,
        /// Environment name.
        env_name: String,
        /// Environment version.
        env_version: String,
        /// Environment metadata (spaces, agent count).
        env_info: EnvInfo,
        /// Negotiated capability flags (logical AND of client and server).
        capabilities: Capabilities,
        /// Whether the requested seed was accepted.
        seed_accepted: bool,
    },
    /// Observation and action space descriptions.
    Spaces {
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    },
    /// Result of a reset operation.
    Reset {
        observation: Observation,
        info: ResetInfo,
        /// Observation encoding used for this response.
        ///
        /// When `Some(EncodedObservation::RawU8Image { .. })`, the
        /// `observation` field contains an empty sentinel and raw pixel
        /// bytes follow as a binary frame immediately after the JSON
        /// message. Added in protocol `1.1.0` so image envs satisfy the
        /// Gymnasium space contract on the first reset.
        #[serde(skip_serializing_if = "Option::is_none")]
        obs_encoding: Option<EncodedObservation>,
    },
    /// Result of a step operation.
    Step {
        observation: Observation,
        /// Scalar reward for this step.
        #[serde(default)]
        reward: f32,
        terminated: bool,
        truncated: bool,
        info: StepInfo,
        /// Observation encoding used for this response.
        ///
        /// When `Some(EncodedObservation::RawU8Image { .. })`, the
        /// `observation` field contains an empty sentinel and raw pixel
        /// bytes follow as a binary frame immediately after the JSON
        /// message.
        #[serde(skip_serializing_if = "Option::is_none")]
        obs_encoding: Option<EncodedObservation>,
    },
    /// Batched reset results for multiple environments.
    BatchReset {
        /// Per-env observations after reset.
        observations: Vec<Observation>,
        /// Per-env reset info.
        infos: Vec<ResetInfo>,
        /// Observation encoding used for this response.
        ///
        /// When `Some(EncodedObservation::BatchF32 { .. })` or
        /// `Some(EncodedObservation::BatchRawU8Image { .. })`, the
        /// `observations` field contains per-env empty sentinels and a
        /// single binary frame (header + flat payload) follows
        /// immediately after the JSON message. Added in protocol
        /// `1.2.0` to advertise the `binary_batch` capability (W7 PR2).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        obs_encoding: Option<EncodedObservation>,
    },
    /// Batched step results for all environments.
    BatchStep {
        /// Per-env observations.
        observations: Vec<Observation>,
        /// Per-env rewards.
        #[serde(default)]
        rewards: Vec<f32>,
        /// Per-env terminated flags.
        terminated: Vec<bool>,
        /// Per-env truncated flags.
        truncated: Vec<bool>,
        /// Per-env step info.
        infos: Vec<StepInfo>,
        /// Observation encoding used for this response. See
        /// [`Response::BatchReset::obs_encoding`] for semantics.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        obs_encoding: Option<EncodedObservation>,
    },
    /// Acknowledgement of close.
    Close,
    /// Error response.
    Error { message: String },
    /// Keepalive reply.
    Pong {
        /// Echo of the client's timestamp.
        timestamp: u64,
        /// Server-side timestamp (epoch milliseconds).
        server_time: u64,
    },
}

impl Response {
    /// Create a reset response from a [`ResetResult`].
    #[must_use]
    pub fn from_reset(result: ResetResult) -> Self {
        Self::Reset {
            observation: result.observation,
            info: result.info,
            obs_encoding: None,
        }
    }

    /// Create a reset response from a [`ResetResult`] with binary observation encoding.
    ///
    /// The `observation` field is set to an empty sentinel. The caller must
    /// send the raw pixel bytes as a binary frame immediately after this JSON message.
    ///
    /// Added in protocol `1.1.0` so image envs satisfy the Gymnasium
    /// space contract on the first reset (W4 PR1).
    #[must_use]
    pub fn from_reset_binary(result: ResetResult, encoding: EncodedObservation) -> Self {
        Self::Reset {
            observation: Observation::zeros(0),
            info: result.info,
            obs_encoding: Some(encoding),
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
            obs_encoding: None,
        }
    }

    /// Create a step response from a [`StepResult`] with binary observation encoding.
    ///
    /// The `observation` field is set to an empty sentinel. The caller must
    /// send the raw pixel bytes as a binary frame immediately after this JSON message.
    #[must_use]
    pub fn from_step_binary(result: StepResult, encoding: EncodedObservation) -> Self {
        Self::Step {
            observation: Observation::zeros(0),
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
            obs_encoding: Some(encoding),
        }
    }

    /// Create a batch reset response from a [`BatchResetResult`].
    #[must_use]
    pub fn from_batch_reset(result: BatchResetResult) -> Self {
        Self::BatchReset {
            observations: result.observations,
            infos: result.infos,
            obs_encoding: None,
        }
    }

    /// Create a batch step response from a [`BatchStepResult`].
    #[must_use]
    pub fn from_batch_step(result: BatchStepResult) -> Self {
        Self::BatchStep {
            observations: result.observations,
            rewards: result.rewards,
            terminated: result.terminated,
            truncated: result.truncated,
            infos: result.infos,
            obs_encoding: None,
        }
    }

    /// Create a batch reset response from a [`BatchResetResult`] with
    /// binary observation encoding.
    ///
    /// The per-env `observations` are replaced with empty-length
    /// sentinels (matches W4's single-env `from_reset_binary`). The
    /// caller MUST send the encoded binary frame as a follow-up
    /// length-prefixed binary frame immediately after this JSON message.
    ///
    /// Added in protocol `1.2.0` (W7 PR2).
    #[must_use]
    pub fn from_batch_reset_binary(result: BatchResetResult, encoding: EncodedObservation) -> Self {
        let num_envs = result.observations.len();
        Self::BatchReset {
            observations: vec![Observation::zeros(0); num_envs],
            infos: result.infos,
            obs_encoding: Some(encoding),
        }
    }

    /// Create a batch step response from a [`BatchStepResult`] with
    /// binary observation encoding. See [`Self::from_batch_reset_binary`].
    #[must_use]
    pub fn from_batch_step_binary(result: BatchStepResult, encoding: EncodedObservation) -> Self {
        let num_envs = result.observations.len();
        Self::BatchStep {
            observations: vec![Observation::zeros(0); num_envs],
            rewards: result.rewards,
            terminated: result.terminated,
            truncated: result.truncated,
            infos: result.infos,
            obs_encoding: Some(encoding),
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
// ObsEncoding
// ---------------------------------------------------------------------------

/// Observation encoding negotiated between client and server.
///
/// When binary observation transfer is active (`binary_obs` capability),
/// the server sends raw pixel bytes after the JSON step response frame.
/// The JSON `observation` field contains an empty sentinel in that case.
///
/// Deprecated in W4 PR1: superseded by
/// [`crate::encoding::EncodedObservation`]. Kept as a `#[deprecated]`
/// enum for source-compat with any external Rust caller that
/// pattern-matches on `ObsEncoding::Json` / `ObsEncoding::RawU8`. The
/// server stops constructing this type internally; it is no longer
/// used by [`Response::Step`] or [`Response::Reset`].
#[deprecated(note = "use EncodedObservation")]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum ObsEncoding {
    /// Standard JSON-encoded observation (default).
    Json,
    /// Raw u8 pixel bytes sent as a binary frame after the JSON response.
    RawU8 {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Number of channels (e.g. 3 for RGB, 1 for grayscale).
        channels: u8,
    },
}

// ---------------------------------------------------------------------------
// Protocol Version & Constants
// ---------------------------------------------------------------------------

/// Protocol version string (semantic versioning) per `PROTOCOL_SPEC.md`.
///
/// Bumped to `1.1.0` in W4 PR1 to advertise that image observations
/// now ship on `Reset` as well as `Step`. Bumped to `1.2.0` in W7 PR2
/// to advertise the `binary_batch` capability: batched observations
/// can ship as a single `BinaryFrameHeader` + flat payload on the
/// existing binary channel. The wire format for non-batch frames and
/// for non-opt-in clients is unchanged since `1.0.0`.
pub const PROTOCOL_VERSION: &str = "1.2.0";

/// Maximum JSON payload size in bytes (16 MiB).
pub const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;

/// Maximum observation data size in bytes (8 MiB).
pub const MAX_OBSERVATION_SIZE: usize = 8 * 1024 * 1024;

/// Maximum action data size in bytes (1 MiB).
pub const MAX_ACTION_SIZE: usize = 1024 * 1024;

/// Negotiate a compatible protocol version per `PROTOCOL_SPEC.md` Section 2.5.
///
/// - Major versions **must** match.
/// - Minor version is the minimum of client and server.
/// - Patch version is the server's.
///
/// Returns the negotiated version string on success, or a
/// [`ProtocolError::VersionMismatch`] on incompatible major versions.
///
/// # Example
///
/// ```
/// use clankers_gym::protocol::negotiate_version;
///
/// let v = negotiate_version("1.0.0", "1.2.1").unwrap();
/// assert_eq!(v, "1.0.1");
///
/// // A 1.1.0 client against a 1.2.0 server downgrades to 1.1.0
/// // (server's patch is 0, so the negotiated patch is 0).
/// let v = negotiate_version("1.1.0", "1.2.0").unwrap();
/// assert_eq!(v, "1.1.0");
///
/// assert!(negotiate_version("2.0.0", "1.0.0").is_err());
/// ```
pub fn negotiate_version(client: &str, server: &str) -> Result<String, ProtocolError> {
    let parse = |v: &str| -> Option<(u32, u32, u32)> {
        let mut parts = v.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        Some((major, minor, patch))
    };

    let (c_major, c_minor, _c_patch) = parse(client).ok_or_else(|| {
        ProtocolError::InvalidMessage(format!("invalid client version: {client}"))
    })?;
    let (s_major, s_minor, s_patch) = parse(server).ok_or_else(|| {
        ProtocolError::InvalidMessage(format!("invalid server version: {server}"))
    })?;

    if c_major != s_major {
        return Err(ProtocolError::VersionMismatch {
            client: client.into(),
            server: server.into(),
        });
    }

    let negotiated_minor = c_minor.min(s_minor);
    Ok(format!("{c_major}.{negotiated_minor}.{s_patch}"))
}

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
    #[error("unexpected message '{got}' in state {current_state} (expected: {expected:?})")]
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

    // ---- Version negotiation ----

    #[test]
    fn negotiate_same_version() {
        let v = negotiate_version("1.0.0", "1.0.0").unwrap();
        assert_eq!(v, "1.0.0");
    }

    #[test]
    fn negotiate_client_higher_minor() {
        let v = negotiate_version("1.2.0", "1.0.3").unwrap();
        assert_eq!(v, "1.0.3");
    }

    #[test]
    fn negotiate_server_higher_minor() {
        let v = negotiate_version("1.0.0", "1.2.1").unwrap();
        assert_eq!(v, "1.0.1");
    }

    #[test]
    fn negotiate_major_mismatch() {
        let err = negotiate_version("2.0.0", "1.0.0").unwrap_err();
        assert!(matches!(err, ProtocolError::VersionMismatch { .. }));
    }

    #[test]
    fn negotiate_invalid_version_string() {
        let err = negotiate_version("bad", "1.0.0").unwrap_err();
        assert!(matches!(err, ProtocolError::InvalidMessage(_)));
    }

    // ---- W7 PR2: 1.2.0 server negotiating against older clients ----

    #[test]
    fn negotiate_v110_client_against_v120_server() {
        // 1.1.0 client + 1.2.0 server: minor downgrade to 1.1.0, server's patch (0).
        let v = negotiate_version("1.1.0", "1.2.0").unwrap();
        assert_eq!(v, "1.1.0");
    }

    #[test]
    fn negotiate_v100_client_against_v120_server() {
        // 1.0.0 client + 1.2.0 server: minor downgrade to 1.0.0, server's patch (0).
        let v = negotiate_version("1.0.0", "1.2.0").unwrap();
        assert_eq!(v, "1.0.0");
    }

    #[test]
    fn negotiate_v120_client_against_v120_server() {
        // Same version: returns exactly 1.2.0.
        let v = negotiate_version("1.2.0", "1.2.0").unwrap();
        assert_eq!(v, "1.2.0");
    }

    // ---- Init / Handshake ----

    #[test]
    fn request_init_roundtrip() {
        let caps = Capabilities {
            batch_step: true,
            ..Default::default()
        };
        let req = Request::Init {
            protocol_version: "1.0.0".into(),
            client_name: "test".into(),
            client_version: "0.1.0".into(),
            capabilities: caps,
            seed: Some(42),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::Init {
            protocol_version,
            seed,
            capabilities,
            ..
        } = req2
        {
            assert_eq!(protocol_version, "1.0.0");
            assert_eq!(seed, Some(42));
            assert!(capabilities.batch_step);
        } else {
            panic!("expected Init");
        }
    }

    #[test]
    fn request_init_from_json() {
        let json = r#"{"type":"init","protocol_version":"1.0.0","client_name":"py","client_version":"0.1.0"}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        if let Request::Init {
            capabilities, seed, ..
        } = req
        {
            assert_eq!(capabilities, Capabilities::default());
            assert_eq!(seed, None);
        } else {
            panic!("expected Init");
        }
    }

    #[test]
    fn response_init_response_roundtrip() {
        let env_info = EnvInfo {
            n_agents: 1,
            observation_space: ObservationSpace::Box {
                low: vec![-1.0; 3],
                high: vec![1.0; 3],
            },
            action_space: ActionSpace::Discrete { n: 4 },
        };
        let resp = Response::InitResponse {
            protocol_version: "1.0.0".into(),
            env_name: "TestEnv".into(),
            env_version: "0.1.0".into(),
            env_info,
            capabilities: Capabilities::default(),
            seed_accepted: true,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::InitResponse {
            protocol_version,
            seed_accepted,
            env_info,
            ..
        } = resp2
        {
            assert_eq!(protocol_version, "1.0.0");
            assert!(seed_accepted);
            assert_eq!(env_info.n_agents, 1);
        } else {
            panic!("expected InitResponse");
        }
    }

    // ---- Ping / Pong ----

    #[test]
    fn request_ping_roundtrip() {
        let req = Request::Ping {
            timestamp: 1_000_000,
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::Ping { timestamp } = req2 {
            assert_eq!(timestamp, 1_000_000);
        } else {
            panic!("expected Ping");
        }
    }

    #[test]
    fn response_pong_roundtrip() {
        let resp = Response::Pong {
            timestamp: 1_000_000,
            server_time: 1_000_001,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Pong {
            timestamp,
            server_time,
        } = resp2
        {
            assert_eq!(timestamp, 1_000_000);
            assert_eq!(server_time, 1_000_001);
        } else {
            panic!("expected Pong");
        }
    }

    // ---- EnvInfo ----

    #[test]
    fn env_info_roundtrip() {
        let info = EnvInfo {
            n_agents: 2,
            observation_space: ObservationSpace::Box {
                low: vec![0.0; 4],
                high: vec![1.0; 4],
            },
            action_space: ActionSpace::Box {
                low: vec![-1.0; 2],
                high: vec![1.0; 2],
            },
        };
        let json = serde_json::to_string(&info).unwrap();
        let info2: EnvInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info2.n_agents, 2);
    }

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
                ..Default::default()
            },
            obs_encoding: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Reset {
            observation, info, ..
        } = resp2
        {
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
            reward: 0.0,
            terminated: false,
            truncated: true,
            info: StepInfo {
                episode_length: 10,
                custom: HashMap::new(),
                ..Default::default()
            },
        };
        let resp = Response::from_step(result);
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Step { truncated, .. } = resp2 {
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

    // ---- Batch protocol ----

    #[test]
    fn request_batch_reset_roundtrip() {
        let req = Request::BatchReset {
            env_ids: vec![0, 2, 4],
            seeds: Some(vec![Some(42), None, Some(7)]),
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::BatchReset { env_ids, seeds } = req2 {
            assert_eq!(env_ids, vec![0, 2, 4]);
            let seeds = seeds.unwrap();
            assert_eq!(seeds[0], Some(42));
            assert_eq!(seeds[1], None);
        } else {
            panic!("expected BatchReset");
        }
    }

    #[test]
    fn request_batch_reset_no_seeds() {
        let json = r#"{"type":"batch_reset","env_ids":[0,1]}"#;
        let req: Request = serde_json::from_str(json).unwrap();
        if let Request::BatchReset { env_ids, seeds } = req {
            assert_eq!(env_ids, vec![0, 1]);
            assert!(seeds.is_none());
        } else {
            panic!("expected BatchReset");
        }
    }

    #[test]
    fn request_batch_step_roundtrip() {
        let req = Request::BatchStep {
            actions: vec![Action::Continuous(vec![0.5, -0.3]), Action::Discrete(2)],
        };
        let json = serde_json::to_string(&req).unwrap();
        let req2: Request = serde_json::from_str(&json).unwrap();
        if let Request::BatchStep { actions } = req2 {
            assert_eq!(actions.len(), 2);
            assert_eq!(actions[0], Action::Continuous(vec![0.5, -0.3]));
            assert_eq!(actions[1], Action::Discrete(2));
        } else {
            panic!("expected BatchStep");
        }
    }

    #[test]
    fn response_batch_reset_roundtrip() {
        let resp = Response::BatchReset {
            observations: vec![
                Observation::new(vec![1.0, 2.0]),
                Observation::new(vec![3.0, 4.0]),
            ],
            infos: vec![
                ResetInfo {
                    seed: Some(42),
                    custom: HashMap::new(),
                    ..Default::default()
                },
                ResetInfo::default(),
            ],
            obs_encoding: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::BatchReset {
            observations,
            infos,
            ..
        } = resp2
        {
            assert_eq!(observations.len(), 2);
            assert_eq!(observations[0].as_slice(), &[1.0, 2.0]);
            assert_eq!(infos[0].seed, Some(42));
        } else {
            panic!("expected BatchReset");
        }
    }

    #[test]
    fn response_batch_step_roundtrip() {
        let resp = Response::BatchStep {
            observations: vec![Observation::zeros(2), Observation::zeros(2)],
            rewards: vec![0.0, 0.0],
            terminated: vec![false, true],
            truncated: vec![false, false],
            infos: vec![StepInfo::default(), StepInfo::default()],
            obs_encoding: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::BatchStep { terminated, .. } = resp2 {
            assert!(terminated[1]);
        } else {
            panic!("expected BatchStep");
        }
    }

    #[test]
    fn response_from_batch_reset() {
        use clankers_core::types::BatchResetResult;
        let result = BatchResetResult {
            observations: vec![Observation::zeros(2)],
            infos: vec![ResetInfo::default()],
        };
        let resp = Response::from_batch_reset(result);
        assert!(matches!(resp, Response::BatchReset { .. }));
    }

    #[test]
    fn response_from_batch_step() {
        use clankers_core::types::BatchStepResult;
        let result = BatchStepResult {
            observations: vec![Observation::zeros(2)],
            rewards: vec![0.0],
            terminated: vec![false],
            truncated: vec![false],
            infos: vec![StepInfo::default()],
        };
        let resp = Response::from_batch_step(result);
        assert!(matches!(resp, Response::BatchStep { .. }));
    }

    // ---- W7 PR2: BatchReset / BatchStep + obs_encoding ----

    #[test]
    fn response_batch_reset_with_encoded_observation_roundtrip() {
        let resp = Response::BatchReset {
            observations: vec![Observation::zeros(0), Observation::zeros(0)],
            infos: vec![ResetInfo::default(), ResetInfo::default()],
            obs_encoding: Some(EncodedObservation::BatchF32 {
                num_envs: 2,
                obs_dim: 4,
                payload: vec![],
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::BatchReset { obs_encoding, .. } = resp2 {
            match obs_encoding.unwrap() {
                EncodedObservation::BatchF32 {
                    num_envs, obs_dim, ..
                } => {
                    assert_eq!(num_envs, 2);
                    assert_eq!(obs_dim, 4);
                }
                _ => panic!("expected BatchF32"),
            }
        } else {
            panic!("expected BatchReset");
        }
    }

    #[test]
    fn response_batch_step_without_obs_encoding_omits_field() {
        use clankers_core::types::BatchStepResult;
        let result = BatchStepResult {
            observations: vec![Observation::zeros(2)],
            rewards: vec![0.0],
            terminated: vec![false],
            truncated: vec![false],
            infos: vec![StepInfo::default()],
        };
        let resp = Response::from_batch_step(result);
        let json = serde_json::to_string(&resp).unwrap();
        // obs_encoding is None → field absent from JSON wire form.
        assert!(!json.contains("obs_encoding"));
    }

    #[test]
    fn response_from_batch_reset_binary_replaces_observations_with_empty_sentinels() {
        use clankers_core::types::BatchResetResult;
        let result = BatchResetResult {
            observations: vec![
                Observation::new(vec![1.0, 2.0]),
                Observation::new(vec![3.0, 4.0]),
                Observation::new(vec![5.0, 6.0]),
            ],
            infos: vec![ResetInfo::default(); 3],
        };
        let enc = EncodedObservation::BatchF32 {
            num_envs: 3,
            obs_dim: 2,
            payload: vec![],
        };
        let resp = Response::from_batch_reset_binary(result, enc);
        if let Response::BatchReset {
            observations,
            obs_encoding,
            ..
        } = resp
        {
            assert_eq!(observations.len(), 3);
            assert!(observations.iter().all(Observation::is_empty));
            assert!(matches!(
                obs_encoding,
                Some(EncodedObservation::BatchF32 {
                    num_envs: 3,
                    obs_dim: 2,
                    ..
                }),
            ));
        } else {
            panic!("expected BatchReset");
        }
    }

    #[test]
    fn response_from_batch_step_binary_replaces_observations_with_empty_sentinels() {
        use clankers_core::types::BatchStepResult;
        let result = BatchStepResult {
            observations: vec![
                Observation::new(vec![1.0, 2.0]),
                Observation::new(vec![3.0, 4.0]),
            ],
            rewards: vec![0.5, -0.5],
            terminated: vec![false, true],
            truncated: vec![false, false],
            infos: vec![StepInfo::default(); 2],
        };
        let enc = EncodedObservation::BatchF32 {
            num_envs: 2,
            obs_dim: 2,
            payload: vec![],
        };
        let resp = Response::from_batch_step_binary(result, enc);
        if let Response::BatchStep {
            observations,
            rewards,
            terminated,
            obs_encoding,
            ..
        } = resp
        {
            assert!(observations.iter().all(Observation::is_empty));
            assert_eq!(rewards, vec![0.5, -0.5]);
            assert_eq!(terminated, vec![false, true]);
            assert!(matches!(
                obs_encoding,
                Some(EncodedObservation::BatchF32 { .. })
            ));
        } else {
            panic!("expected BatchStep");
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

    // ---- EncodedObservation (W4 PR1) ----
    //
    // Migrated in W4 PR1 from the legacy `ObsEncoding::{Json, RawU8}`
    // tests to `EncodedObservation::{FlatF32, RawU8Image, Dict}`. The
    // legacy enum is still present (deprecated) but no longer
    // constructed by the server. Wire-tag rename `RawU8` → `RawU8Image`
    // is intentional and documented in `CHANGELOG.md`.

    #[test]
    fn encoded_observation_flat_f32_roundtrip() {
        let enc = EncodedObservation::FlatF32 {
            data: vec![1.0, 2.0],
        };
        let json = serde_json::to_string(&enc).unwrap();
        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        assert!(json.contains("FlatF32"));
        match back {
            EncodedObservation::FlatF32 { data } => assert_eq!(data, vec![1.0, 2.0]),
            _ => panic!("expected FlatF32"),
        }
    }

    #[test]
    fn encoded_observation_raw_u8_image_roundtrip() {
        let enc = EncodedObservation::RawU8Image {
            width: 320,
            height: 240,
            channels: 3,
            layout: ImageLayout::Hwc,
            payload: vec![],
        };
        let json = serde_json::to_string(&enc).unwrap();
        let back: EncodedObservation = serde_json::from_str(&json).unwrap();
        match back {
            EncodedObservation::RawU8Image {
                width,
                height,
                channels,
                layout,
                payload,
            } => {
                assert_eq!(width, 320);
                assert_eq!(height, 240);
                assert_eq!(channels, 3);
                assert_eq!(layout, ImageLayout::Hwc);
                assert!(payload.is_empty());
            }
            _ => panic!("expected RawU8Image"),
        }
    }

    #[test]
    fn encoded_observation_raw_u8_image_type_tag() {
        let enc = EncodedObservation::RawU8Image {
            width: 64,
            height: 64,
            channels: 1,
            layout: ImageLayout::Hwc,
            payload: vec![],
        };
        let json = serde_json::to_string(&enc).unwrap();
        // The serde tag "type" should be present and use the new
        // variant name.
        assert!(json.contains("RawU8Image"));
        assert!(json.contains("width"));
        assert!(json.contains("height"));
        assert!(json.contains("channels"));
        assert!(json.contains("layout"));
        // payload is #[serde(skip)] — must not appear on the wire.
        assert!(!json.contains("payload"));
    }

    #[test]
    fn response_step_with_encoded_observation_roundtrip() {
        let resp = Response::Step {
            observation: Observation::zeros(0),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo {
                episode_length: 5,
                custom: HashMap::new(),
                ..Default::default()
            },
            obs_encoding: Some(EncodedObservation::RawU8Image {
                width: 84,
                height: 84,
                channels: 3,
                layout: ImageLayout::Hwc,
                payload: vec![],
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Step { obs_encoding, .. } = resp2 {
            let enc = obs_encoding.unwrap();
            assert!(matches!(
                enc,
                EncodedObservation::RawU8Image {
                    width: 84,
                    height: 84,
                    channels: 3,
                    layout: ImageLayout::Hwc,
                    ..
                }
            ));
        } else {
            panic!("expected Step");
        }
    }

    #[test]
    fn response_step_without_obs_encoding_omits_field() {
        let result = StepResult {
            observation: Observation::new(vec![0.5]),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo {
                episode_length: 1,
                custom: HashMap::new(),
                ..Default::default()
            },
        };
        let resp = Response::from_step(result);
        let json = serde_json::to_string(&resp).unwrap();
        // obs_encoding field should be absent when None
        assert!(!json.contains("obs_encoding"));
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Step { obs_encoding, .. } = resp2 {
            assert!(obs_encoding.is_none());
        } else {
            panic!("expected Step");
        }
    }

    #[test]
    fn response_reset_with_encoded_observation_roundtrip() {
        let resp = Response::Reset {
            observation: Observation::zeros(0),
            info: ResetInfo::default(),
            obs_encoding: Some(EncodedObservation::RawU8Image {
                width: 64,
                height: 64,
                channels: 3,
                layout: ImageLayout::Hwc,
                payload: vec![],
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Reset { obs_encoding, .. } = resp2 {
            assert!(matches!(
                obs_encoding,
                Some(EncodedObservation::RawU8Image {
                    width: 64,
                    height: 64,
                    channels: 3,
                    layout: ImageLayout::Hwc,
                    ..
                })
            ));
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn response_reset_without_obs_encoding_omits_field() {
        let result = ResetResult {
            observation: Observation::new(vec![0.5]),
            info: ResetInfo::default(),
        };
        let resp = Response::from_reset(result);
        let json = serde_json::to_string(&resp).unwrap();
        // obs_encoding field should be absent when None.
        assert!(!json.contains("obs_encoding"));
        let resp2: Response = serde_json::from_str(&json).unwrap();
        if let Response::Reset { obs_encoding, .. } = resp2 {
            assert!(obs_encoding.is_none());
        } else {
            panic!("expected Reset");
        }
    }

    #[test]
    fn response_from_reset_binary_constructs_with_empty_observation() {
        let result = ResetResult {
            observation: Observation::new(vec![0.1, 0.2, 0.3]),
            info: ResetInfo::default(),
        };
        let encoding = EncodedObservation::RawU8Image {
            width: 8,
            height: 8,
            channels: 3,
            layout: ImageLayout::Hwc,
            payload: vec![],
        };
        let resp = Response::from_reset_binary(result, encoding);
        if let Response::Reset {
            observation,
            obs_encoding,
            ..
        } = resp
        {
            // observation is replaced by an empty sentinel; bytes go on
            // the binary frame.
            assert_eq!(observation.len(), 0);
            assert!(matches!(
                obs_encoding,
                Some(EncodedObservation::RawU8Image { width: 8, .. })
            ));
        } else {
            panic!("expected Reset");
        }
    }
}
