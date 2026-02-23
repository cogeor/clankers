//! Shared test fixtures and utilities for Clankers crates.
//!
//! Provides reusable helpers for building Bevy test apps, spawning joint
//! entities, registering sensors, deterministic RNG setup, episode lifecycle
//! helpers, and mock implementations of core traits.

pub mod app;
pub mod episodes;
pub mod mocks;
pub mod rng;
pub mod spawn;

// ---------------------------------------------------------------------------
// Re-exports for convenience
// ---------------------------------------------------------------------------

pub use app::{full_test_app, minimal_test_app};
pub use episodes::{episode_snapshot, reset_episode, run_until_done, step_n};
pub use mocks::ConstantSensor;
pub use rng::seeded_rng;
pub use spawn::{register_state_sensor, spawn_joint, spawn_joints};
