//! The main visualization plugin.
//!
//! [`ClankersVizPlugin`] adds orbit camera, egui panels, ground plane,
//! and mode-aware simulation stepping to a Bevy app.

use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use bevy_panorbit_camera::PanOrbitCameraPlugin;

use crate::camera;
use crate::config::VizConfig;
use crate::mode::VizMode;
use crate::ui;

/// Bevy plugin for interactive Clankers visualization.
///
/// Adds:
/// - Orbit camera (pan, zoom, rotate)
/// - egui side panel with joint states, episode info, and controls
/// - Ground plane with lighting
/// - Mode resource ([`VizMode`]) for switching between Paused/Teleop/Policy
///
/// Expects that [`ClankersSimPlugin`](clankers_sim::ClankersSimPlugin) is already
/// added to the app.
pub struct ClankersVizPlugin;

impl Plugin for ClankersVizPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VizConfig>()
            .init_resource::<VizMode>()
            .add_plugins(EguiPlugin::default())
            .add_plugins(PanOrbitCameraPlugin)
            .add_systems(Startup, (camera::spawn_camera, camera::spawn_scene))
            .add_systems(bevy::prelude::Update, ui::side_panel_system);
    }
}
