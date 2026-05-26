//! `ArmPolicyOverlayPlugin` — egui + camera + EE-camera-sync for the
//! `arm_policy_viz.rs` example bin.
//!
//! # Status (W8 PR1)
//!
//! Skeleton only — Bevy `Plugin` declaration with an empty `build`.
//! Bin migration + plugin body (lifted from `arm_policy_viz.rs`) is
//! deferred per the loop 7 IMPLEMENTATION.md.

use bevy::prelude::{App, Plugin};

/// Egui + camera glue for policy-driven 6-DOF arm visualisation.
///
/// W8 PR1 ships an empty stub; loop 8 populates the body.
#[derive(Default)]
pub struct ArmPolicyOverlayPlugin;

impl Plugin for ArmPolicyOverlayPlugin {
    fn build(&self, _app: &mut App) {
        // Intentionally empty — overlay body is deferred to the next
        // loop. See module-level docs.
    }
}
