//! Bevy plugin wrapping [`clankers_actuator_core`] for ECS integration.
//!
//! Add [`ClankersActuatorPlugin`] to your Bevy app, then spawn joint entities
//! with actuator components.  The plugin runs the full motor → transmission →
//! friction pipeline each frame in [`ClankersSet::Act`].
//!
//! # Example
//!
//! ```
//! use bevy::prelude::*;
//! use clankers_actuator::prelude::*;
//! use clankers_core::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(ClankersCorePlugin);
//! app.add_plugins(ClankersActuatorPlugin);
//!
//! app.world_mut().spawn((
//!     Actuator::default(),
//!     JointCommand { value: 5.0 },
//!     JointState::default(),
//!     JointTorque::default(),
//! ));
//! ```

pub mod components;
pub mod systems;

/// Re-export the core crate for downstream convenience.
pub use clankers_actuator_core;

use bevy::prelude::*;
use clankers_core::ClankersSet;

// ---------------------------------------------------------------------------
// ClankersActuatorPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that steps all actuators in [`ClankersSet::Act`].
///
/// Requires [`ClankersCorePlugin`](clankers_core::ClankersCorePlugin) to be
/// added first (it provides [`SimConfig`](clankers_core::config::SimConfig)
/// and the system-set ordering).
pub struct ClankersActuatorPlugin;

impl Plugin for ClankersActuatorPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            systems::actuator_step_system.in_set(ClankersSet::Act),
        );
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        ClankersActuatorPlugin,
        components::{Actuator, JointCommand, JointState, JointTorque},
    };
    // Re-export core types so users don't need a separate import.
    pub use clankers_actuator_core::prelude::*;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clankers_core::ClankersCorePlugin;

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);
        app.finish();
        app.cleanup();
        app.update();
    }
}
