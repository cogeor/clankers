//! Visualization mode state machine.
//!
//! Controls whether the simulation is paused, under teleop control,
//! or running a policy.

use bevy::prelude::*;

/// Active visualization mode.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VizMode {
    /// Simulation paused. No stepping occurs.
    #[default]
    Paused,
    /// Manual teleop control via keyboard/gamepad input.
    Teleop,
    /// Autonomous policy inference drives the robot.
    Policy,
}

impl VizMode {
    /// Human-readable label for UI display.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Paused => "Paused",
            Self::Teleop => "Teleop",
            Self::Policy => "Policy",
        }
    }

    /// Whether this mode advances the simulation.
    pub const fn is_simulating(self) -> bool {
        matches!(self, Self::Teleop | Self::Policy)
    }
}
