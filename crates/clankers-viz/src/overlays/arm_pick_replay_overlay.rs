//! `ArmPickReplayOverlayPlugin` — timeline scrubber + frame thumbnails
//! + GIF export glue for the `arm_pick_replay.rs` example bin.
//!
//! # Status (W8 PR1)
//!
//! Skeleton only — Bevy `Plugin` declaration with an empty `build`.
//! Bin migration + plugin body (lifted from `arm_pick_replay.rs`) is
//! deferred per the loop 7 IMPLEMENTATION.md.

use bevy::prelude::{App, Plugin};

/// Egui timeline + GIF export glue for arm-pick MCAP replay.
///
/// W8 PR1 ships an empty stub; loop 8 populates the body.
#[derive(Default)]
pub struct ArmPickReplayOverlayPlugin;

impl Plugin for ArmPickReplayOverlayPlugin {
    fn build(&self, _app: &mut App) {
        // Intentionally empty — overlay body is deferred to the next
        // loop. See module-level docs.
    }
}
