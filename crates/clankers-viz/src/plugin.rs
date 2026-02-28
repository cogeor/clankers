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
/// When `fixed_update` is true, simulation systems (mode gate, teleop, pipeline
/// run conditions) are placed on `FixedUpdate` so physics/MPC runs at a fixed
/// rate decoupled from the render frame rate.
///
/// Expects that [`ClankersSimPlugin`](clankers_sim::ClankersSimPlugin) and
/// [`ClankersTeleopPlugin`](clankers_teleop::ClankersTeleopPlugin) are already
/// added to the app.
pub struct ClankersVizPlugin {
    /// When true, gate simulation stages on `FixedUpdate` instead of `Update`.
    pub fixed_update: bool,
}

impl Default for ClankersVizPlugin {
    fn default() -> Self {
        Self { fixed_update: false }
    }
}

impl Plugin for ClankersVizPlugin {
    fn build(&self, app: &mut App) {
        // Resources.
        app.init_resource::<VizConfig>()
            .init_resource::<VizMode>()
            .init_resource::<VizSimGate>()
            .init_resource::<KeyboardTeleopMap>()
            .init_resource::<crate::SelectedRobotId>();

        // Third-party plugins.
        app.add_plugins(EguiPlugin::default())
            .add_plugins(PanOrbitCameraPlugin);

        // Startup: camera and scene.
        app.add_systems(Startup, (camera::spawn_camera, camera::spawn_scene));

        // Disable orbit camera when egui wants pointer (prevents slider drag → orbit).
        app.add_systems(Update, camera::egui_camera_gate);

        if self.fixed_update {
            // FixedUpdate mode: simulation runs at fixed rate, decoupled from render.
            app.add_systems(
                FixedUpdate,
                (systems::mode_gate_system, systems::mode_transition_system)
                    .before(ClankersSet::Observe),
            );

            app.configure_sets(
                FixedUpdate,
                (
                    ClankersSet::Decide.run_if(systems::sim_should_step),
                    ClankersSet::Act.run_if(systems::sim_should_step),
                    ClankersSet::Simulate.run_if(systems::sim_should_step),
                    ClankersSet::Evaluate.run_if(systems::sim_should_step),
                ),
            );

            app.add_systems(
                FixedUpdate,
                (
                    systems::sync_teleop_to_robot,
                    input::keyboard_teleop_system,
                )
                    .chain()
                    .in_set(ClankersSet::Decide)
                    .before(clankers_teleop::systems::apply_teleop_commands),
            );
        } else {
            // Update mode: simulation tied to render frame rate (original behavior).
            app.add_systems(
                Update,
                (systems::mode_gate_system, systems::mode_transition_system)
                    .before(ClankersSet::Observe),
            );

            app.configure_sets(
                Update,
                (
                    ClankersSet::Decide.run_if(systems::sim_should_step),
                    ClankersSet::Act.run_if(systems::sim_should_step),
                    ClankersSet::Simulate.run_if(systems::sim_should_step),
                    ClankersSet::Evaluate.run_if(systems::sim_should_step),
                ),
            );

            app.add_systems(
                Update,
                (
                    systems::sync_teleop_to_robot,
                    input::keyboard_teleop_system,
                )
                    .chain()
                    .in_set(ClankersSet::Decide)
                    .before(clankers_teleop::systems::apply_teleop_commands),
            );
        }

        // UI: runs in EguiPrimaryContextPass (required by bevy_egui 0.38).
        app.add_systems(EguiPrimaryContextPass, ui::side_panel_system);
    }
}
