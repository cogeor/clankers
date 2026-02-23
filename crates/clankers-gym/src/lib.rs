//! TCP server and Gymnasium-compatible training protocol for Clankers.
//!
//! This crate provides the communication layer between a training client
//! (typically Python) and the Clankers simulation:
//!
//! - [`protocol`] — JSON-serialisable request/response types, error codes,
//!   protocol version constants
//! - [`framing`] — Length-prefixed JSON wire format (4-byte LE `u32` + payload)
//! - [`state_machine`] — [`ProtocolStateMachine`] enforcing valid message ordering
//! - [`env`](mod@env) — [`GymEnv`] wrapper that drives a Bevy App with
//!   the standard `step`/`reset` interface
//! - [`server`] — [`GymServer`] TCP server for remote training clients
//!
//! Messages use length-prefixed JSON framing per `PROTOCOL_SPEC.md`.
//! Connections begin with a handshake (`Init`/`InitResponse`) and then
//! follow the standard Gymnasium `reset`/`step`/`close` pattern.

pub mod env;
pub mod framing;
pub mod protocol;
pub mod server;
pub mod state_machine;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use env::GymEnv;
pub use protocol::{
    EnvInfo, ProtocolError, ProtocolState, Request, Response, negotiate_version, PROTOCOL_VERSION,
};
pub use server::{GymServer, ServerConfig};
pub use state_machine::ProtocolStateMachine;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        GymEnv, GymServer, ProtocolStateMachine, ServerConfig,
        protocol::{
            EnvInfo, ProtocolError, ProtocolState, Request, Response, negotiate_version,
            PROTOCOL_VERSION,
        },
    };
}
