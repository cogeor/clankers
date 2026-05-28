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
//! - [`vec_env`] — [`GymVecEnv`] for batched multi-environment training
//! - [`server`] — [`GymServer`] TCP server for remote training clients
//!
//! Messages use length-prefixed JSON framing per `PROTOCOL_SPEC.md`.
//! Connections begin with a handshake (`Init`/`InitResponse`) and then
//! follow the standard Gymnasium `reset`/`step`/`close` pattern.

pub mod binary_frame;
pub mod encoding;
pub mod env;
pub mod framing;
pub mod protocol;
pub mod server;
pub mod state_machine;
pub mod tensor_frame;
pub mod vec_env;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use binary_frame::{
    BinaryFrameError, BinaryFrameHeader, FRAME_VERSION, HEADER_SIZE, KIND_BATCH_F32,
    KIND_BATCH_RAW_U8, decode_batch_f32, decode_batch_raw_u8, encode_batch_f32,
    encode_batch_raw_u8,
};
pub use encoding::{EncodedObservation, ImageLayout};
pub use env::GymEnv;
#[allow(deprecated)]
pub use protocol::ObsEncoding;
pub use protocol::{
    EnvInfo, PROTOCOL_VERSION, ProtocolError, ProtocolState, Request, Response, negotiate_version,
};
pub use server::{GymServer, ServerConfig, VecGymServer};
pub use state_machine::ProtocolStateMachine;
pub use vec_env::GymVecEnv;

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    #[allow(deprecated)]
    pub use crate::protocol::ObsEncoding;
    pub use crate::{
        BinaryFrameError, BinaryFrameHeader, EncodedObservation, FRAME_VERSION, GymEnv, GymServer,
        GymVecEnv, HEADER_SIZE, ImageLayout, KIND_BATCH_F32, KIND_BATCH_RAW_U8,
        ProtocolStateMachine, ServerConfig, VecGymServer, decode_batch_f32, decode_batch_raw_u8,
        encode_batch_f32, encode_batch_raw_u8,
        protocol::{
            EnvInfo, PROTOCOL_VERSION, ProtocolError, ProtocolState, Request, Response,
            negotiate_version,
        },
    };
}
