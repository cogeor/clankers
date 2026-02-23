//! Manual control interfaces for debugging and teleoperation.
//!
//! This crate provides input-source-agnostic teleop infrastructure:
//!
//! - [`TeleopConfig`] — maps named input channels to joint entities with
//!   scaling and dead zones
//! - [`TeleopCommander`] — resource buffering raw input values from any source
//! - [`ClankersTeleopPlugin`] — Bevy plugin that applies commander values to
//!   joint commands each frame
//!
//! The design separates input capture (keyboard, gamepad, network) from
//! command application, so any input source can drive the teleop system.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_teleop::prelude::*;
//!
//! App::new()
//!     .add_plugins(clankers_core::ClankersCorePlugin)
//!     .add_plugins(ClankersTeleopPlugin)
//!     .insert_resource(TeleopConfig::new())
//!     .run();
//! ```

pub mod commander;
pub mod config;
pub mod systems;

use bevy::prelude::*;
use clankers_core::ClankersSet;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use commander::TeleopCommander;
pub use config::{JointMapping, TeleopConfig};

// ---------------------------------------------------------------------------
// ClankersTeleopPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that applies teleop commands to joint entities.
///
/// Reads [`TeleopCommander`] input values and applies them to
/// [`JointCommand`](clankers_actuator::components::JointCommand) components
/// via the mappings in [`TeleopConfig`].
///
/// Runs in [`ClankersSet::Decide`] (same phase as policy, so teleop
/// overrides policy when active).
pub struct ClankersTeleopPlugin;

impl Plugin for ClankersTeleopPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TeleopCommander>()
            .init_resource::<TeleopConfig>()
            .add_systems(
                Update,
                systems::apply_teleop_commands.in_set(ClankersSet::Decide),
            );
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{ClankersTeleopPlugin, TeleopCommander, TeleopConfig, config::JointMapping};
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_actuator::ClankersActuatorPlugin);
        app.add_plugins(ClankersTeleopPlugin);
        app.finish();
        app.cleanup();
        app.update();

        assert!(app.world().get_resource::<TeleopCommander>().is_some());
        assert!(app.world().get_resource::<TeleopConfig>().is_some());
    }
}
