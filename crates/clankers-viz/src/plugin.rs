//! The main visualization plugin.
//!
//! [`ClankersVizPlugin`] adds orbit camera, egui panels, ground plane,
//! mode-aware simulation stepping, and keyboard teleop to a Bevy app.

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::PanOrbitCameraPlugin;

use clankers_core::ClankersSet;

use crate::camera;
use crate::config::VizConfig;
use crate::input::KeyboardTeleopMap;
use crate::mode::VizMode;
use crate::systems::VizSimGate;
use crate::{input, systems, ui};

/// Bevy plugin for interactive Clankers visualization.
///
/// Adds:
/// - Orbit camera (pan, zoom, rotate)
/// - egui side panel with joint states, episode info, and controls
/// - Ground plane with lighting
/// - Mode resource ([`VizMode`]) for switching between Paused/Teleop/Policy
/// - Keyboard-to-teleop input mapping
/// - Mode-gated simulation stepping (pause, step-once, resume)
///
/// Expects that [`ClankersSimPlugin`](clankers_sim::ClankersSimPlugin) and
/// [`ClankersTeleopPlugin`](clankers_teleop::ClankersTeleopPlugin) are already
/// added to the app.
pub struct ClankersVizPlugin;

impl Plugin for ClankersVizPlugin {
    fn build(&self, app: &mut App) {
        // Resources.
        app.init_resource::<VizConfig>()
            .init_resource::<VizMode>()
            .init_resource::<VizSimGate>()
            .init_resource::<KeyboardTeleopMap>();

        // Third-party plugins.
        app.add_plugins(EguiPlugin::default())
            .add_plugins(PanOrbitCameraPlugin);

        // Startup: camera and scene.
        app.add_systems(Startup, (camera::spawn_camera, camera::spawn_scene));

        // Mode gating: runs before the ClankersSet pipeline.
        app.add_systems(
            Update,
            (systems::mode_gate_system, systems::mode_transition_system)
                .before(ClankersSet::Observe),
        );

        // Run conditions on pipeline stages — skip when paused.
        app.configure_sets(
            Update,
            (
                ClankersSet::Decide.run_if(systems::sim_should_step),
                ClankersSet::Act.run_if(systems::sim_should_step),
                ClankersSet::Simulate.run_if(systems::sim_should_step),
                ClankersSet::Evaluate.run_if(systems::sim_should_step),
            ),
        );

        // Keyboard input: in Decide, before teleop apply.
        app.add_systems(
            Update,
            input::keyboard_teleop_system
                .in_set(ClankersSet::Decide)
                .before(clankers_teleop::systems::apply_teleop_commands),
        );

        // UI: runs in EguiPrimaryContextPass (required by bevy_egui 0.38).
        app.add_systems(EguiPrimaryContextPass, ui::side_panel_system);
    }
}
