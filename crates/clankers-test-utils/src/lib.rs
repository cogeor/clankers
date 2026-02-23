//! Shared test fixtures and utilities for Clankers crates.
//!
//! Provides reusable helpers for building Bevy test apps, spawning joint
//! entities, registering sensors, and deterministic RNG setup.

pub mod app;
pub mod rng;
pub mod spawn;

// ---------------------------------------------------------------------------
// Re-exports for convenience
// ---------------------------------------------------------------------------

pub use app::{full_test_app, minimal_test_app};
pub use rng::seeded_rng;
pub use spawn::{register_state_sensor, spawn_joint, spawn_joints};
