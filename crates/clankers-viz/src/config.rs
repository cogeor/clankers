//! Visualization configuration.

use bevy::prelude::*;

/// Runtime configuration for the visualization mode.
#[derive(Resource, Clone, Debug)]
pub struct VizConfig {
    /// Start the simulation paused.
    pub start_paused: bool,
    /// Simulation speed multiplier (1.0 = realtime).
    pub sim_speed: f32,
    /// Show the egui side panel.
    pub show_panel: bool,
}

impl Default for VizConfig {
    fn default() -> Self {
        Self {
            start_paused: true,
            sim_speed: 1.0,
            show_panel: true,
        }
    }
}
