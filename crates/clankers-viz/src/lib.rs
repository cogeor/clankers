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

pub mod camera;
pub mod config;
pub mod mode;
pub mod plugin;
pub mod ui;

pub use config::VizConfig;
pub use mode::VizMode;
pub use plugin::ClankersVizPlugin;
