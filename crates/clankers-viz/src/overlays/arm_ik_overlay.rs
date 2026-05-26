//! `ArmIkOverlayPlugin` — egui control panel + EE camera + motor
//! override hooks for the `arm_ik_viz.rs` example bin.
//!
//! # Status (W8 PR1)
//!
//! Skeleton only — Bevy `Plugin` declaration with an empty `build`. The
//! bin migration + overlay body (lifted from `arm_ik_viz.rs:56-end`) is
//! deferred per the loop 7 IMPLEMENTATION.md.

use bevy::prelude::{App, Plugin};

/// Egui sliders, sync systems, motor override hooks for the 6-DOF arm
/// IK viz bin.
///
/// W8 PR1 ships an empty stub; loop 8 populates the body.
#[derive(Default)]
pub struct ArmIkOverlayPlugin;

impl Plugin for ArmIkOverlayPlugin {
    fn build(&self, _app: &mut App) {
        // Intentionally empty — overlay body is deferred to the next
        // loop. See module-level docs.
    }
}
