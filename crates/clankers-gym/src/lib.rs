//! TCP server and Gymnasium-compatible training protocol for Clankers.
//!
//! This crate provides the communication layer between a training client
//! (typically Python) and the Clankers simulation:
//!
//! - [`protocol`] — JSON-serialisable request/response message types
//!
//! Messages are newline-delimited JSON sent over TCP. The protocol follows
//! the standard Gymnasium `step`/`reset`/`close` pattern.

pub mod protocol;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use protocol::{Request, Response};

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::protocol::{Request, Response};
}
