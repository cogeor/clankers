//! Mode-gating systems for visualization.
//!
//! Controls whether the simulation pipeline steps based on [`VizMode`],
//! and handles mode-transition side effects.

use bevy::prelude::*;

use clankers_core::types::{RobotGroup, RobotId};
use clankers_policy::runner::PolicyRunner;
use clankers_teleop::{TeleopCommander, TeleopConfig};

use crate::config::VizConfig;
use crate::input::KeyboardTeleopMap;
use crate::mode::VizMode;
use crate::SelectedRobotId;

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

    // -----------------------------------------------------------------------
    // sync_teleop_to_robot tests
    // -----------------------------------------------------------------------

    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_core::types::RobotGroup;
    use crate::input::KeyboardTeleopMap;
    use crate::SelectedRobotId;

    fn spawn_joint(world: &mut World) -> Entity {
        world
            .spawn((
                Actuator::default(),
                JointCommand::default(),
                JointState::default(),
                JointTorque::default(),
            ))
            .id()
    }

    fn build_sync_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(clankers_teleop::ClankersTeleopPlugin);
        app.init_resource::<VizConfig>();
        app.init_resource::<VizMode>();
        app.init_resource::<VizSimGate>();
        app.init_resource::<KeyboardTeleopMap>();
        app.init_resource::<SelectedRobotId>();
        app.init_resource::<RobotGroup>();
        app.add_systems(Update, sync_teleop_to_robot);
        app.finish();
        app.cleanup();
        app
    }

    #[test]
    fn sync_rebuilds_on_robot_switch() {
        let mut app = build_sync_test_app();

        // Register two robots with different joint counts.
        let j0 = spawn_joint(app.world_mut());
        let j1 = spawn_joint(app.world_mut());
        let j2 = spawn_joint(app.world_mut());

        let mut group = app.world_mut().resource_mut::<RobotGroup>();
        let id_a = group.allocate("arm".to_string(), vec![j0, j1]);
        let _id_b = group.allocate("gripper".to_string(), vec![j2]);

        // Select robot A.
        app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
        app.update();

        // Should have 2 bindings and 2 mappings.
        assert_eq!(app.world().resource::<KeyboardTeleopMap>().bindings.len(), 2);
        assert_eq!(app.world().resource::<TeleopConfig>().mappings.len(), 2);
    }

    #[test]
    fn sync_none_uses_all_joints() {
        let mut app = build_sync_test_app();

        let j0 = spawn_joint(app.world_mut());
        let j1 = spawn_joint(app.world_mut());
        let j2 = spawn_joint(app.world_mut());

        let mut group = app.world_mut().resource_mut::<RobotGroup>();
        group.allocate("arm".to_string(), vec![j0, j1]);
        group.allocate("gripper".to_string(), vec![j2]);

        // No selection => all joints.
        app.world_mut().resource_mut::<SelectedRobotId>().0 = None;
        app.update();

        assert_eq!(app.world().resource::<KeyboardTeleopMap>().bindings.len(), 3);
        assert_eq!(app.world().resource::<TeleopConfig>().mappings.len(), 3);
    }

    #[test]
    fn sync_clears_commander_on_switch() {
        let mut app = build_sync_test_app();

        let j0 = spawn_joint(app.world_mut());
        let j1 = spawn_joint(app.world_mut());

        let mut group = app.world_mut().resource_mut::<RobotGroup>();
        let id_a = group.allocate("arm".to_string(), vec![j0]);
        let id_b = group.allocate("leg".to_string(), vec![j1]);

        // Select A, run, set a commander value.
        app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
        app.update();
        app.world_mut().resource_mut::<TeleopCommander>().set("joint_0", 0.5);

        // Switch to B.
        app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_b);
        app.update();

        assert_eq!(app.world().resource::<TeleopCommander>().channel_count(), 0);
    }

    #[test]
    fn sync_preserves_enabled_flag() {
        let mut app = build_sync_test_app();

        let j0 = spawn_joint(app.world_mut());
        let mut group = app.world_mut().resource_mut::<RobotGroup>();
        let id_a = group.allocate("arm".to_string(), vec![j0]);

        // Disable teleop, then trigger sync.
        app.world_mut().resource_mut::<TeleopConfig>().enabled = false;
        app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
        app.update();

        assert!(!app.world().resource::<TeleopConfig>().enabled);
    }

    #[test]
    fn sync_no_change_is_noop() {
        let mut app = build_sync_test_app();

        let j0 = spawn_joint(app.world_mut());
        let mut group = app.world_mut().resource_mut::<RobotGroup>();
        let id_a = group.allocate("arm".to_string(), vec![j0]);

        app.world_mut().resource_mut::<SelectedRobotId>().0 = Some(id_a);
        app.update();
        // Set a commander value.
        app.world_mut().resource_mut::<TeleopCommander>().set("joint_0", 0.7);

        // Run again with same selection -- commander should NOT be cleared.
        app.update();
        assert!((app.world().resource::<TeleopCommander>().get("joint_0") - 0.7).abs() < f32::EPSILON);
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

/// Rebuilds [`KeyboardTeleopMap`] and [`TeleopConfig`] when the selected
/// robot changes, so keyboard teleop targets the correct joints.
///
/// When `SelectedRobotId` is `None`, maps ALL joints from ALL robots
/// (sorted by `RobotId` index) for backwards compatibility with
/// single-robot scenes.
#[allow(clippy::needless_pass_by_value)]
pub fn sync_teleop_to_robot(
    selected: Res<SelectedRobotId>,
    robot_group: Option<Res<RobotGroup>>,
    mut teleop_config: ResMut<TeleopConfig>,
    mut teleop_map: ResMut<KeyboardTeleopMap>,
    mut commander: ResMut<TeleopCommander>,
    mut last_selected: Local<Option<Option<RobotId>>>,
) {
    // Determine current selection.
    let current = selected.0;

    // Check if this is the first run or if the selection changed.
    if *last_selected == Some(current) {
        return;
    }
    *last_selected = Some(current);

    // Collect the joint entities for the target robot(s).
    let joints: Vec<Entity> = match (current, robot_group.as_deref()) {
        // A specific robot is selected and the group exists.
        (Some(id), Some(group)) => group
            .get(id)
            .map(|info| info.joints.clone())
            .unwrap_or_default(),
        // No robot selected -- gather all joints, sorted by RobotId.
        (None, Some(group)) => {
            let mut robots: Vec<_> = group.iter().collect();
            robots.sort_by_key(|(id, _)| id.index());
            robots
                .iter()
                .flat_map(|(_, info)| info.joints.iter().copied())
                .collect()
        }
        // No RobotGroup resource at all -- nothing to map.
        (_, None) => Vec::new(),
    };

    // Rebuild KeyboardTeleopMap for the joint count.
    *teleop_map = KeyboardTeleopMap::for_joint_count(joints.len());

    // Rebuild TeleopConfig mappings.
    let enabled = teleop_config.enabled;
    let mut new_config = TeleopConfig::new();
    for (i, &entity) in joints.iter().enumerate() {
        new_config = new_config.with_mapping(
            format!("joint_{i}"),
            clankers_teleop::config::JointMapping::new(entity),
        );
    }
    new_config.enabled = enabled;
    *teleop_config = new_config;

    // Clear stale commander values from previous robot.
    commander.clear();
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
