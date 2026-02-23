//! Bevy systems for applying teleop commands to joint entities.

use bevy::prelude::*;
use clankers_actuator::components::JointCommand;

use crate::commander::TeleopCommander;
use crate::config::TeleopConfig;

// ---------------------------------------------------------------------------
// apply_teleop_commands
// ---------------------------------------------------------------------------

/// System that reads the teleop commander and applies scaled values to joints.
///
/// For each mapping in [`TeleopConfig`], reads the corresponding channel
/// from [`TeleopCommander`], applies dead zone and scaling, then writes
/// the result to the joint's [`JointCommand`].
///
/// Skips application when [`TeleopConfig::enabled`] is `false`.
#[allow(clippy::needless_pass_by_value)]
pub fn apply_teleop_commands(
    config: Res<TeleopConfig>,
    commander: Res<TeleopCommander>,
    mut commands: Query<&mut JointCommand>,
) {
    if !config.enabled {
        return;
    }

    for (channel, mapping) in &config.mappings {
        let raw = commander.get(channel);
        let value = mapping.apply(raw);

        if let Ok(mut cmd) = commands.get_mut(mapping.entity) {
            cmd.value = value;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JointMapping;
    use clankers_actuator::components::{Actuator, JointState, JointTorque};
    use clankers_core::ClankersSet;

    fn build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.init_resource::<TeleopCommander>();
        app.init_resource::<TeleopConfig>();
        app.add_systems(Update, apply_teleop_commands.in_set(ClankersSet::Decide));
        app.finish();
        app.cleanup();
        app
    }

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

    #[test]
    fn applies_command_to_mapped_joint() {
        let mut app = build_test_app();
        let entity = spawn_joint(app.world_mut());

        *app.world_mut().resource_mut::<TeleopConfig>() =
            TeleopConfig::new().with_mapping("axis_0", JointMapping::new(entity).with_scale(2.0));

        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("axis_0", 0.5);

        app.update();

        let cmd = app.world().get::<JointCommand>(entity).unwrap();
        assert!((cmd.value - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn respects_dead_zone() {
        let mut app = build_test_app();
        let entity = spawn_joint(app.world_mut());

        *app.world_mut().resource_mut::<TeleopConfig>() = TeleopConfig::new()
            .with_mapping("axis_0", JointMapping::new(entity).with_dead_zone(0.2));

        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("axis_0", 0.1);

        app.update();

        let cmd = app.world().get::<JointCommand>(entity).unwrap();
        assert!((cmd.value).abs() < f32::EPSILON);
    }

    #[test]
    fn skips_when_disabled() {
        let mut app = build_test_app();
        let entity = spawn_joint(app.world_mut());

        *app.world_mut().resource_mut::<TeleopConfig>() = TeleopConfig::new()
            .with_mapping("axis_0", JointMapping::new(entity))
            .with_enabled(false);

        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("axis_0", 5.0);

        app.update();

        let cmd = app.world().get::<JointCommand>(entity).unwrap();
        assert!((cmd.value).abs() < f32::EPSILON);
    }

    #[test]
    fn missing_entity_does_not_panic() {
        let mut app = build_test_app();
        // Create an entity in a separate world so it doesn't exist in the app
        let fake_entity = World::new().spawn_empty().id();

        *app.world_mut().resource_mut::<TeleopConfig>() =
            TeleopConfig::new().with_mapping("axis_0", JointMapping::new(fake_entity));

        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("axis_0", 1.0);

        // Should not panic
        app.update();
    }

    #[test]
    fn unmapped_channel_has_no_effect() {
        let mut app = build_test_app();
        let entity = spawn_joint(app.world_mut());

        *app.world_mut().resource_mut::<TeleopConfig>() =
            TeleopConfig::new().with_mapping("axis_0", JointMapping::new(entity));

        // Set a different channel
        app.world_mut()
            .resource_mut::<TeleopCommander>()
            .set("axis_1", 10.0);

        app.update();

        let cmd = app.world().get::<JointCommand>(entity).unwrap();
        assert!((cmd.value).abs() < f32::EPSILON);
    }

    #[test]
    fn multiple_joints_mapped() {
        let mut app = build_test_app();
        let e1 = spawn_joint(app.world_mut());
        let e2 = spawn_joint(app.world_mut());

        *app.world_mut().resource_mut::<TeleopConfig>() = TeleopConfig::new()
            .with_mapping("left", JointMapping::new(e1).with_scale(3.0))
            .with_mapping("right", JointMapping::new(e2).with_scale(-1.0));

        let mut cmdr = app.world_mut().resource_mut::<TeleopCommander>();
        cmdr.set("left", 1.0);
        cmdr.set("right", 2.0);

        app.update();

        let c1 = app.world().get::<JointCommand>(e1).unwrap();
        let c2 = app.world().get::<JointCommand>(e2).unwrap();
        assert!((c1.value - 3.0).abs() < f32::EPSILON);
        assert!((c2.value - (-2.0)).abs() < f32::EPSILON);
    }
}
