//! Mode-gating systems for visualization.
//!
//! Controls whether the simulation pipeline steps based on [`VizMode`],
//! and handles mode-transition side effects.

use bevy::prelude::*;

use clankers_policy::runner::PolicyRunner;
use clankers_teleop::{TeleopCommander, TeleopConfig};

use crate::config::VizConfig;
use crate::mode::VizMode;

/// Resource controlling whether the simulation pipeline runs this frame.
#[derive(Resource, Clone, Debug, Default)]
pub struct VizSimGate {
    /// If true, the Decide/Act/Simulate/Evaluate sets will execute.
    pub should_step: bool,
}

/// Run condition: returns true when the simulation should step.
#[allow(clippy::needless_pass_by_value)]
pub fn sim_should_step(gate: Res<VizSimGate>) -> bool {
    gate.should_step
}

/// System that sets [`VizSimGate`] and [`TeleopConfig::enabled`] based on
/// the current [`VizMode`].
///
/// Must run before `ClankersSet::Observe`.
#[allow(clippy::needless_pass_by_value)]
pub fn mode_gate_system(
    mode: Res<VizMode>,
    mut viz_config: ResMut<VizConfig>,
    mut teleop_config: ResMut<TeleopConfig>,
    mut gate: ResMut<VizSimGate>,
) {
    match *mode {
        VizMode::Paused => {
            teleop_config.enabled = false;
            if viz_config.step_once {
                gate.should_step = true;
                viz_config.step_once = false;
            } else {
                gate.should_step = false;
            }
        }
        VizMode::Teleop => {
            teleop_config.enabled = true;
            gate.should_step = true;
        }
        VizMode::Policy => {
            teleop_config.enabled = false;
            gate.should_step = true;
        }
    }
}

/// System that performs one-shot cleanup on mode transitions.
///
/// Clears [`TeleopCommander`] when leaving Teleop mode and resets
/// [`PolicyRunner`] when entering Policy mode.
#[allow(clippy::needless_pass_by_value)]
pub fn mode_transition_system(
    mode: Res<VizMode>,
    mut last_mode: Local<VizMode>,
    mut commander: ResMut<TeleopCommander>,
    mut policy_runner: Option<ResMut<PolicyRunner>>,
) {
    if *mode != *last_mode {
        // Leaving teleop: clear stale commander values.
        if *last_mode == VizMode::Teleop {
            commander.clear();
        }
        // Entering policy: reset the runner's action to zeros.
        if *mode == VizMode::Policy
            && let Some(ref mut runner) = policy_runner
        {
            runner.reset();
        }
        *last_mode = *mode;
    }
}
