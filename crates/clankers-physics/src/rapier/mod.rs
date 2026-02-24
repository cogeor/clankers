//! Raw `rapier3d` physics backend.
//!
//! This module implements [`PhysicsBackend`](crate::backend::PhysicsBackend)
//! using the `rapier3d` crate directly (not `bevy_rapier3d`). We own the
//! [`PhysicsPipeline`](rapier3d::pipeline::PhysicsPipeline), call `step()`
//! ourselves, and have full control over scheduling and data flow.

pub mod backend;
pub mod bridge;
pub mod context;
pub mod systems;

pub use backend::RapierBackend;
pub use context::RapierContext;
