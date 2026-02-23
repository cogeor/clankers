//! Bevy systems for observation collection and episode lifecycle.

use crate::SensorRegistry;
use crate::buffer::ObservationBuffer;
use crate::episode::{Episode, EpisodeConfig};

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// StepReward
// ---------------------------------------------------------------------------

/// Resource holding the reward for the current step.
///
/// External systems (reward functions) write to this each step.
/// The episode system reads and consumes it.
#[derive(Resource, Clone, Debug, Default)]
pub struct StepReward(pub f32);

// ---------------------------------------------------------------------------
// observe_system
// ---------------------------------------------------------------------------

/// Exclusive system that reads all registered sensors into the observation buffer.
///
/// Runs in [`ClankersSet::Observe`](clankers_core::ClankersSet::Observe).
/// Temporarily removes `SensorRegistry` and `ObservationBuffer` from the world
/// so sensors can query via `&mut World`.
pub fn observe_system(world: &mut World) {
    let Some(mut registry) = world.remove_resource::<SensorRegistry>() else {
        return;
    };
    let Some(mut buffer) = world.remove_resource::<ObservationBuffer>() else {
        world.insert_resource(registry);
        return;
    };

    for entry in &mut registry.entries {
        let obs = entry.sensor.read(world);
        buffer.write(entry.slot_index, obs.as_slice());
    }

    world.insert_resource(buffer);
    world.insert_resource(registry);
}

// ---------------------------------------------------------------------------
// episode_step_system
// ---------------------------------------------------------------------------

/// Advances the episode by one step, accumulating reward and checking truncation.
///
/// Runs in [`ClankersSet::Evaluate`](clankers_core::ClankersSet::Evaluate).
#[allow(clippy::needless_pass_by_value)]
pub fn episode_step_system(
    mut episode: ResMut<Episode>,
    config: Res<EpisodeConfig>,
    mut reward: ResMut<StepReward>,
) {
    if episode.is_running() {
        episode.advance(reward.0);
        episode.check_truncation(config.max_episode_steps);
        reward.0 = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensors::{JointCommandSensor, JointStateSensor};
    use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
    use clankers_core::ClankersSet;

    fn build_test_app() -> App {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.init_resource::<Episode>();
        app.init_resource::<EpisodeConfig>();
        app.init_resource::<ObservationBuffer>();
        app.init_resource::<SensorRegistry>();
        app.init_resource::<StepReward>();
        app.add_systems(Update, observe_system.in_set(ClankersSet::Observe));
        app.add_systems(Update, episode_step_system.in_set(ClankersSet::Evaluate));
        app.finish();
        app.cleanup();
        app
    }

    fn spawn_joint(world: &mut World, pos: f32, vel: f32, cmd: f32) {
        world.spawn((
            Actuator::default(),
            JointCommand { value: cmd },
            JointState {
                position: pos,
                velocity: vel,
            },
            JointTorque::default(),
        ));
    }

    #[test]
    fn observe_system_collects_sensor_data() {
        let mut app = build_test_app();

        // Register sensors
        {
            let world = app.world_mut();
            let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
            let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
            registry.register(Box::new(JointStateSensor::new(2)), &mut buffer);
            registry.register(Box::new(JointCommandSensor::new(2)), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);

            spawn_joint(world, 1.0, 2.0, 10.0);
            spawn_joint(world, 3.0, 4.0, 20.0);
        }

        app.update();

        let buffer = app.world().resource::<ObservationBuffer>();
        assert_eq!(buffer.dim(), 6); // 2*2 + 2
        let obs = buffer.as_observation();
        assert_eq!(obs.len(), 6);
        // JointState: [pos, vel, pos, vel], JointCommand: [cmd, cmd]
        // Values should contain our spawned data (order may vary)
        let state_data = buffer.read(0);
        assert_eq!(state_data.len(), 4);
        let cmd_data = buffer.read(1);
        assert_eq!(cmd_data.len(), 2);
    }

    #[test]
    fn observe_system_handles_empty_registry() {
        let mut app = build_test_app();
        // No sensors registered, no entities spawned
        app.update(); // Should not panic
        let buffer = app.world().resource::<ObservationBuffer>();
        assert_eq!(buffer.dim(), 0);
    }

    #[test]
    fn episode_step_advances_and_accumulates_reward() {
        let mut app = build_test_app();
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.world_mut().resource_mut::<StepReward>().0 = 5.0;

        app.update();

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 1);
        assert!((ep.total_reward - 5.0).abs() < f32::EPSILON);
        // Reward should be consumed
        let reward = app.world().resource::<StepReward>();
        assert!((reward.0).abs() < f32::EPSILON);
    }

    #[test]
    fn episode_step_truncates_at_max_steps() {
        let mut app = build_test_app();
        app.world_mut()
            .resource_mut::<EpisodeConfig>()
            .max_episode_steps = 3;
        app.world_mut().resource_mut::<Episode>().reset(None);

        for _ in 0..3 {
            app.update();
        }

        let ep = app.world().resource::<Episode>();
        assert!(ep.is_done());
        assert_eq!(ep.step_count, 3);
    }

    #[test]
    fn episode_step_does_nothing_when_idle() {
        let mut app = build_test_app();
        app.world_mut().resource_mut::<StepReward>().0 = 10.0;

        app.update();

        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 0);
        assert!((ep.total_reward).abs() < f32::EPSILON);
    }

    #[test]
    fn full_observe_then_evaluate_pipeline() {
        let mut app = build_test_app();

        // Register a sensor
        {
            let world = app.world_mut();
            let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
            let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
            registry.register(Box::new(JointStateSensor::new(1)), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);

            spawn_joint(world, 1.0, 2.0, 0.0);
        }

        // Start episode
        app.world_mut().resource_mut::<Episode>().reset(None);
        app.world_mut().resource_mut::<StepReward>().0 = 1.5;

        // Run one frame
        app.update();

        // Check observation was collected
        let buffer = app.world().resource::<ObservationBuffer>();
        let state_data = buffer.read(0);
        assert_eq!(state_data.len(), 2);

        // Check episode advanced
        let ep = app.world().resource::<Episode>();
        assert_eq!(ep.step_count, 1);
        assert!((ep.total_reward - 1.5).abs() < f32::EPSILON);
    }
}
