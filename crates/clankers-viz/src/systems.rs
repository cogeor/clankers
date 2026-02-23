//! Mode-gating systems for visualization.
//!
//! Controls whether the simulation pipeline steps based on [`VizMode`],
//! and handles mode-transition side effects.

use bevy::prelude::*;

use clankers_policy::runner::PolicyRunner;
use clankers_teleop::{TeleopCommander, TeleopConfig};

use crate::config::VizConfig;
use crate::mode::VizMode;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_app() -> bevy::prelude::App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_teleop::ClankersTeleopPlugin);
        app.init_resource::<VizConfig>();
        app.init_resource::<VizMode>();
        app.init_resource::<VizSimGate>();
        app.add_systems(
            Update,
            (mode_gate_system, mode_transition_system).before(clankers_core::ClankersSet::Observe),
        );
        app.finish();
        app.cleanup();
        app
    }

    #[test]
    fn paused_mode_blocks_stepping() {
        let mut app = build_test_app();
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Paused;
        app.update();
        assert!(!app.world().resource::<VizSimGate>().should_step);
    }

    #[test]
    fn teleop_mode_enables_stepping_and_teleop() {
        let mut app = build_test_app();
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Teleop;
        app.update();
        assert!(app.world().resource::<VizSimGate>().should_step);
        assert!(app.world().resource::<TeleopConfig>().enabled);
    }

    #[test]
    fn policy_mode_enables_stepping_disables_teleop() {
        let mut app = build_test_app();
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Policy;
        app.update();
        assert!(app.world().resource::<VizSimGate>().should_step);
        assert!(!app.world().resource::<TeleopConfig>().enabled);
    }

    #[test]
    fn step_once_allows_single_step_then_blocks() {
        let mut app = build_test_app();
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Paused;
        app.world_mut().resource_mut::<VizConfig>().step_once = true;
        app.update();
        assert!(app.world().resource::<VizSimGate>().should_step);
        assert!(!app.world().resource::<VizConfig>().step_once);

        // Next frame should be blocked again.
        app.update();
        assert!(!app.world().resource::<VizSimGate>().should_step);
    }

    #[test]
    fn leaving_teleop_clears_commander() {
        let mut app = build_test_app();
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Teleop;
        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("joint_0", 0.5);
        app.update();

        // Switch to paused.
        *app.world_mut().resource_mut::<VizMode>() = VizMode::Paused;
        app.update();

        assert_eq!(app.world().resource::<TeleopCommander>().channel_count(), 0);
    }
}

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
