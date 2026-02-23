//! Entity spawn helpers for tests.

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_env::SensorRegistry;
use clankers_env::buffer::ObservationBuffer;
use clankers_env::sensors::JointStateSensor;

/// Spawn a single joint entity with default actuator and the given state.
pub fn spawn_joint(world: &mut World, position: f32, velocity: f32) -> Entity {
    world
        .spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState { position, velocity },
            JointTorque::default(),
        ))
        .id()
}

/// Spawn `n` joint entities at default state (position=0, velocity=0).
///
/// Returns the entity IDs in spawning order.
pub fn spawn_joints(world: &mut World, n: usize) -> Vec<Entity> {
    (0..n).map(|_| spawn_joint(world, 0.0, 0.0)).collect()
}

/// Register a [`JointStateSensor`] for `n_joints` into the app's sensor
/// registry and observation buffer.
///
/// Must be called after the `ClankersEnvPlugin` has been added (so that
/// `SensorRegistry` and `ObservationBuffer` exist as resources).
pub fn register_state_sensor(app: &mut App, n_joints: usize) {
    let world = app.world_mut();
    let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
    let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
    registry.register(Box::new(JointStateSensor::new(n_joints)), &mut buffer);
    world.insert_resource(buffer);
    world.insert_resource(registry);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::full_test_app;
    use clankers_env::episode::Episode;

    #[test]
    fn spawn_joint_creates_entity() {
        let mut app = full_test_app();
        let entity = spawn_joint(app.world_mut(), 1.0, 2.0);

        let state = app.world().get::<JointState>(entity).unwrap();
        assert!((state.position - 1.0).abs() < f32::EPSILON);
        assert!((state.velocity - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn spawn_joints_creates_n_entities() {
        let mut app = full_test_app();
        let entities = spawn_joints(app.world_mut(), 3);
        assert_eq!(entities.len(), 3);

        for &e in &entities {
            assert!(app.world().get::<Actuator>(e).is_some());
            assert!(app.world().get::<JointCommand>(e).is_some());
        }
    }

    #[test]
    fn register_sensor_works() {
        let mut app = full_test_app();
        spawn_joint(app.world_mut(), 0.0, 0.0);
        register_state_sensor(&mut app, 1);

        app.world_mut().resource_mut::<Episode>().reset(None);
        app.update();

        let buffer = app.world().resource::<ObservationBuffer>();
        assert_eq!(buffer.dim(), 2); // pos + vel
    }
}
