//! TCP server and Gymnasium-compatible training protocol for Clankers.
//!
//! This crate provides the communication layer between a training client
//! (typically Python) and the Clankers simulation:
//!
//! - [`protocol`] — JSON-serialisable request/response message types
//! - [`env`] — [`GymEnv`](env::GymEnv) wrapper that drives a Bevy App with
//!   the standard `step`/`reset` interface
//! - [`server`] — [`GymServer`](server::GymServer) TCP server for remote
//!   training clients
//!
//! Messages are newline-delimited JSON sent over TCP. The protocol follows
//! the standard Gymnasium `step`/`reset`/`close` pattern.

pub mod env;
pub mod protocol;
pub mod server;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use env::GymEnv;
pub use protocol::{Request, Response};
pub use server::GymServer;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        GymEnv, GymServer,
        protocol::{Request, Response},
    };
}
