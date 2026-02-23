//! Visualization configuration.

use bevy::prelude::*;

/// Runtime configuration for the visualization mode.
#[derive(Resource, Clone, Debug)]
pub struct VizConfig {
    /// Simulation speed multiplier (1.0 = realtime).
    pub sim_speed: f32,
    /// Show the egui side panel.
    pub show_panel: bool,
    /// When true, advance exactly one simulation step then re-pause.
    pub step_once: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            sim_speed: 1.0,
            show_panel: true,
            step_once: false,
        }
    }
}
