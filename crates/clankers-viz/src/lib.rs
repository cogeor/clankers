//! Interactive visualization, teleoperation, and policy inspection for Clankers.
//!
//! `clankers-viz` provides a windowed Bevy application with:
//! - Orbit camera for scene inspection
//! - egui side panel with joint state display
//! - Simulation controls (pause, step, reset, speed)
//! - Mode switching (Teleop, Policy, Paused)
//!
//! # Usage
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_viz::ClankersVizPlugin;
//!
//! App::new()
//!     .add_plugins(DefaultPlugins)
//!     .add_plugins(clankers_sim::ClankersSimPlugin)
//!     .add_plugins(ClankersVizPlugin)
//!     .run();
//! ```

use bevy::prelude::Resource;
use clankers_core::types::RobotId;

pub mod camera;
pub mod config;
pub mod input;
pub mod mode;
pub mod plugin;
pub mod systems;
pub mod ui;

/// Resource tracking which robot is selected in the viz UI.
///
/// `None` means show all robots (useful for single-robot scenes).
#[derive(Resource, Default, Clone, Debug)]
pub struct SelectedRobotId(pub Option<RobotId>);

pub use config::VizConfig;
pub use mode::VizMode;
pub use plugin::ClankersVizPlugin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_robot_id_default_is_none() {
        let selected = SelectedRobotId::default();
        assert!(selected.0.is_none());
    }

    #[test]
    fn selected_robot_id_clone() {
        let a = SelectedRobotId(Some(RobotId(2)));
        let b = a.clone();
        assert_eq!(a.0, b.0);
    }
}
