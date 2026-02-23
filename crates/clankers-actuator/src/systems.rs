//! Bevy systems for actuator simulation.

use bevy::prelude::*;
use clankers_core::config::SimConfig;

use crate::components::{Actuator, JointCommand, JointState, JointTorque};

/// Steps all actuators each frame.
///
/// Runs in [`ClankersSet::Act`](clankers_core::ClankersSet::Act).  For each
/// entity with the four actuator components, computes the output torque from
/// the full motor → transmission → friction pipeline.
///
/// Uses [`SimConfig::control_dt`] as the timestep.
#[allow(clippy::cast_possible_truncation)] // f64 → f32 control_dt
#[allow(clippy::needless_pass_by_value)] // Bevy system parameters are extracted by value
pub fn actuator_step_system(
    sim_config: Res<SimConfig>,
    mut query: Query<(&mut Actuator, &JointCommand, &JointState, &mut JointTorque)>,
) {
    let dt = sim_config.control_dt as f32;
    for (mut actuator, command, state, mut torque) in &mut query {
        torque.value = actuator.step(command.value, state.position, state.velocity, dt);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ClankersActuatorPlugin;
    use clankers_actuator_core::prelude::*;
    use clankers_core::ClankersCorePlugin;

    #[test]
    fn system_computes_torque_for_single_entity() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);

        let entity = app
            .world_mut()
            .spawn((
                Actuator::default(),
                JointCommand { value: 5.0 },
                JointState::default(),
                JointTorque::default(),
            ))
            .id();

        app.finish();
        app.cleanup();
        app.update();

        let torque = app.world().get::<JointTorque>(entity).unwrap();
        // Default: IdealMotor(100, 10), torque mode, 5.0 at rest → 5.0.
        assert!((torque.value - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn system_processes_multiple_entities() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);

        let e1 = app
            .world_mut()
            .spawn((
                Actuator::default(),
                JointCommand { value: 10.0 },
                JointState::default(),
                JointTorque::default(),
            ))
            .id();

        let e2 = app
            .world_mut()
            .spawn((
                Actuator::default(),
                JointCommand { value: 20.0 },
                JointState::default(),
                JointTorque::default(),
            ))
            .id();

        app.finish();
        app.cleanup();
        app.update();

        let t1 = app.world().get::<JointTorque>(e1).unwrap();
        let t2 = app.world().get::<JointTorque>(e2).unwrap();
        assert!((t1.value - 10.0).abs() < f32::EPSILON);
        assert!((t2.value - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn system_uses_control_dt() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);

        // Use a DcMotor with time constant so dt matters.
        let motor = MotorType::Dc(DcMotor::new(100.0, 10.0).with_time_constant(0.1));
        let entity = app
            .world_mut()
            .spawn((
                Actuator::new(
                    motor,
                    Transmission::default(),
                    FrictionModel::default(),
                    ControlMode::Torque,
                ),
                JointCommand { value: 50.0 },
                JointState::default(),
                JointTorque::default(),
            ))
            .id();

        app.finish();
        app.cleanup();
        app.update();

        let torque = app.world().get::<JointTorque>(entity).unwrap();
        // DcMotor with tau=0.1, dt=0.02 (default control_dt):
        // alpha = 0.02 / (0.1 + 0.02) ≈ 0.1667
        // state = 0 + 0.1667 × 50 ≈ 8.33
        assert!(torque.value > 0.0);
        assert!(torque.value < 50.0);
    }

    #[test]
    fn system_with_velocity_control() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);

        let entity = app
            .world_mut()
            .spawn((
                Actuator::new(
                    MotorType::Ideal(IdealMotor::new(100.0, 10.0)),
                    Transmission::default(),
                    FrictionModel::default(),
                    ControlMode::Velocity { kp: 10.0, kd: 0.0 },
                ),
                JointCommand { value: 2.0 },
                JointState {
                    position: 0.0,
                    velocity: 1.0,
                },
                JointTorque::default(),
            ))
            .id();

        app.finish();
        app.cleanup();
        app.update();

        let torque = app.world().get::<JointTorque>(entity).unwrap();
        // PD: 10 × (2.0 - 1.0) = 10.0. Motor at vel=1.0:
        // available = 100 × (1 - 1/10) = 90 → passes 10. Friction ≈ 0.
        assert!((torque.value - 10.0).abs() < f32::EPSILON);
    }

    #[test]
    fn entities_without_all_components_are_skipped() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.add_plugins(ClankersActuatorPlugin);

        // Entity missing JointTorque — should not be queried.
        app.world_mut().spawn((
            Actuator::default(),
            JointCommand { value: 5.0 },
            JointState::default(),
        ));

        app.finish();
        app.cleanup();
        // Should not panic.
        app.update();
    }
}
