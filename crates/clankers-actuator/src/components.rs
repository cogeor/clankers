//! ECS components for actuator simulation.
//!
//! Each controlled joint entity should have all four components:
//! [`Actuator`], [`JointCommand`], [`JointState`], and [`JointTorque`].

use bevy::prelude::*;
use clankers_actuator_core::prelude::*;

// ---------------------------------------------------------------------------
// ActuatorController (internal dispatch)
// ---------------------------------------------------------------------------

/// Internal controller state, created from a [`ControlMode`].
#[derive(Clone, Debug, Default)]
pub enum ActuatorController {
    /// Direct torque/voltage passthrough (no feedback loop).
    #[default]
    Torque,
    /// Velocity tracking via PD controller.
    Velocity(PdController),
    /// Position tracking via PID controller.
    Position(PidController),
}

// ---------------------------------------------------------------------------
// Actuator
// ---------------------------------------------------------------------------

/// Primary actuator component.  One per controlled joint.
///
/// Holds motor dynamics, transmission, friction, and control state.
/// The [`step`](Self::step) method runs the full actuator pipeline and returns
/// the net joint torque.
///
/// # Pipeline
///
/// ```text
/// command ──► Controller ──► Motor ──► Transmission ──► Friction ──► torque
/// ```
#[derive(Component, Clone, Debug, Default)]
pub struct Actuator {
    /// Motor model for this joint.
    pub motor: MotorType,
    /// Gear transmission (ratio, efficiency, backlash).
    pub transmission: Transmission,
    /// Joint friction model.
    pub friction: FrictionModel,
    /// Internal controller state.
    controller: ActuatorController,
}

impl Actuator {
    /// Create a new actuator with explicit motor, transmission, friction, and
    /// control mode.
    #[allow(clippy::needless_pass_by_value)] // ControlMode is small and consumed
    pub const fn new(
        motor: MotorType,
        transmission: Transmission,
        friction: FrictionModel,
        control_mode: ControlMode,
    ) -> Self {
        let controller = match control_mode {
            ControlMode::Torque => ActuatorController::Torque,
            ControlMode::Velocity { kp, kd } => {
                ActuatorController::Velocity(PdController::new(kp, kd))
            }
            ControlMode::Position { kp, ki, kd } => {
                ActuatorController::Position(PidController::new(kp, ki, kd))
            }
        };
        Self {
            motor,
            transmission,
            friction,
            controller,
        }
    }

    /// Builder: set transmission.
    pub const fn with_transmission(mut self, transmission: Transmission) -> Self {
        self.transmission = transmission;
        self
    }

    /// Builder: set friction model.
    pub const fn with_friction(mut self, friction: FrictionModel) -> Self {
        self.friction = friction;
        self
    }

    /// Step the full actuator pipeline.
    ///
    /// - `command`: interpreted based on control mode (Nm, rad/s, or rad).
    /// - `position`: current joint position (rad).
    /// - `velocity`: current joint velocity (rad/s).
    /// - `dt`: timestep (seconds).
    ///
    /// Returns the net torque (Nm) to apply at the joint.
    pub fn step(&mut self, command: f32, position: f32, velocity: f32, dt: f32) -> f32 {
        // 1. Control: command → motor command
        let motor_cmd = match &mut self.controller {
            ActuatorController::Torque => command,
            ActuatorController::Velocity(pd) => pd.compute(command, velocity, dt),
            ActuatorController::Position(pid) => pid.compute(command, position, dt),
        };

        // 2. Motor: apply dynamics and limits
        let motor_torque = self.motor.step(motor_cmd, velocity, dt);

        // 3. Transmission: gear ratio, efficiency, backlash
        let joint_torque = self.transmission.motor_to_joint(motor_torque);
        let joint_torque = self.transmission.apply_backlash(position, joint_torque);

        // 4. Friction: opposes motion
        let friction_torque = self.friction.compute(velocity, joint_torque);

        joint_torque + friction_torque
    }

    /// Reset all internal state (motor, transmission backlash, controller).
    pub const fn reset(&mut self) {
        self.motor.reset();
        self.transmission.reset();
        match &mut self.controller {
            ActuatorController::Torque => {}
            ActuatorController::Velocity(pd) => pd.reset(),
            ActuatorController::Position(pid) => pid.reset(),
        }
    }

    /// Returns the current control mode.
    pub const fn control_mode(&self) -> ControlMode {
        match &self.controller {
            ActuatorController::Torque => ControlMode::Torque,
            ActuatorController::Velocity(pd) => ControlMode::Velocity {
                kp: pd.kp,
                kd: pd.kd,
            },
            ActuatorController::Position(pid) => ControlMode::Position {
                kp: pid.kp(),
                ki: pid.ki(),
                kd: pid.kd(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// JointCommand
// ---------------------------------------------------------------------------

/// Command input for a joint actuator.
///
/// Interpretation depends on the [`Actuator`]'s control mode:
/// - **Torque**: `value` is torque (Nm) or voltage (V) for `FullDcMotor`.
/// - **Velocity**: `value` is target velocity (rad/s).
/// - **Position**: `value` is target position (rad).
#[derive(Component, Clone, Debug, Default)]
pub struct JointCommand {
    /// Command value.
    pub value: f32,
}

// ---------------------------------------------------------------------------
// JointState
// ---------------------------------------------------------------------------

/// State feedback for a joint (written by the physics engine).
#[derive(Component, Clone, Debug, Default)]
pub struct JointState {
    /// Joint position (rad).
    pub position: f32,
    /// Joint velocity (rad/s).
    pub velocity: f32,
}

// ---------------------------------------------------------------------------
// JointTorque
// ---------------------------------------------------------------------------

/// Output torque computed by the actuator (read by the physics engine).
#[derive(Component, Clone, Debug, Default)]
pub struct JointTorque {
    /// Net torque at the joint (Nm).
    pub value: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 0.02;

    // -- Actuator --

    #[test]
    fn default_is_torque_mode() {
        let a = Actuator::default();
        assert!(matches!(a.control_mode(), ControlMode::Torque));
    }

    #[test]
    fn torque_mode_passthrough() {
        let mut a = Actuator::default();
        // Default: IdealMotor(100, 10), direct drive, no friction.
        // Command 5.0 at zero velocity → motor output 5.0 → joint 5.0.
        let torque = a.step(5.0, 0.0, 0.0, DT);
        assert!((torque - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn torque_mode_clamped_by_motor() {
        let mut a = Actuator::default();
        // IdealMotor max_torque = 100. Command 200 → clamped to 100.
        let torque = a.step(200.0, 0.0, 0.0, DT);
        assert!((torque - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn velocity_mode_pd() {
        let motor = MotorType::Ideal(IdealMotor::new(100.0, 10.0));
        let mut a = Actuator::new(
            motor,
            Transmission::default(),
            FrictionModel::default(),
            ControlMode::Velocity { kp: 10.0, kd: 0.0 },
        );
        // Target velocity = 5.0, current = 0.0 → PD output = 10 × 5 = 50.
        // Motor passes 50 (within 100 limit). Direct drive → 50.
        let torque = a.step(5.0, 0.0, 0.0, DT);
        assert!((torque - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn position_mode_pid() {
        let motor = MotorType::Ideal(IdealMotor::new(100.0, 10.0));
        let mut a = Actuator::new(
            motor,
            Transmission::default(),
            FrictionModel::default(),
            ControlMode::Position {
                kp: 20.0,
                ki: 0.0,
                kd: 0.0,
            },
        );
        // Target position = 1.0, current = 0.0 → PID output = 20 × 1 = 20.
        let torque = a.step(1.0, 0.0, 0.0, DT);
        assert!((torque - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_transmission_scales_output() {
        let motor = MotorType::Ideal(IdealMotor::new(10.0, 10.0));
        let mut a = Actuator::new(
            motor,
            Transmission::new(50.0).with_efficiency(1.0),
            FrictionModel::default(),
            ControlMode::Torque,
        );
        // Motor outputs 5.0, transmission multiplies by 50 × 1.0 = 250.
        let torque = a.step(5.0, 0.0, 0.0, DT);
        assert!((torque - 250.0).abs() < f32::EPSILON);
    }

    #[test]
    fn with_friction_opposes_motion() {
        let motor = MotorType::Ideal(IdealMotor::new(100.0, 100.0));
        let mut a = Actuator::new(
            motor,
            Transmission::default(),
            FrictionModel::new(1.0, 0.0), // Coulomb only
            ControlMode::Torque,
        );
        // Motor outputs 10.0. At velocity 5.0 (dynamic regime):
        // friction = -coulomb × sign(v) = -1.0 × 1.0 = -1.0.
        // Net = 10.0 + (-1.0) = 9.0.
        let torque = a.step(10.0, 0.0, 5.0, DT);
        assert!((torque - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reset_clears_state() {
        let mut a = Actuator::new(
            MotorType::Dc(DcMotor::new(10.0, 5.0).with_time_constant(0.1)),
            Transmission::default(),
            FrictionModel::default(),
            ControlMode::Position {
                kp: 10.0,
                ki: 10.0,
                kd: 0.0,
            },
        );
        // Run a few steps to accumulate state.
        a.step(1.0, 0.0, 0.0, DT);
        a.step(1.0, 0.0, 0.0, DT);
        a.reset();
        // After reset, motor state and PID integral should be zero.
        // A zero command at rest should produce zero torque.
        let torque = a.step(0.0, 0.0, 0.0, DT);
        assert!(torque.abs() < f32::EPSILON);
    }

    #[test]
    fn builder_pattern() {
        let a = Actuator::default()
            .with_transmission(Transmission::new(10.0))
            .with_friction(FrictionModel::new(0.5, 0.1));
        assert!((a.transmission.gear_ratio - 10.0).abs() < f32::EPSILON);
        assert!((a.friction.coulomb - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn control_mode_getter_torque() {
        let a = Actuator::default();
        assert!(matches!(a.control_mode(), ControlMode::Torque));
    }

    #[test]
    fn control_mode_getter_velocity() {
        let a = Actuator::new(
            MotorType::default(),
            Transmission::default(),
            FrictionModel::default(),
            ControlMode::Velocity { kp: 5.0, kd: 1.0 },
        );
        match a.control_mode() {
            ControlMode::Velocity { kp, kd } => {
                assert!((kp - 5.0).abs() < f32::EPSILON);
                assert!((kd - 1.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected Velocity mode"),
        }
    }

    #[test]
    fn control_mode_getter_position() {
        let a = Actuator::new(
            MotorType::default(),
            Transmission::default(),
            FrictionModel::default(),
            ControlMode::Position {
                kp: 10.0,
                ki: 2.0,
                kd: 0.5,
            },
        );
        match a.control_mode() {
            ControlMode::Position { kp, ki, kd } => {
                assert!((kp - 10.0).abs() < f32::EPSILON);
                assert!((ki - 2.0).abs() < f32::EPSILON);
                assert!((kd - 0.5).abs() < f32::EPSILON);
            }
            _ => panic!("expected Position mode"),
        }
    }

    // -- JointCommand / JointState / JointTorque --

    #[test]
    fn joint_command_default_zero() {
        let cmd = JointCommand::default();
        assert!(cmd.value.abs() < f32::EPSILON);
    }

    #[test]
    fn joint_state_default_zero() {
        let state = JointState::default();
        assert!(state.position.abs() < f32::EPSILON);
        assert!(state.velocity.abs() < f32::EPSILON);
    }

    #[test]
    fn joint_torque_default_zero() {
        let torque = JointTorque::default();
        assert!(torque.value.abs() < f32::EPSILON);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn all_types_are_send_sync() {
        assert_send_sync::<Actuator>();
        assert_send_sync::<JointCommand>();
        assert_send_sync::<JointState>();
        assert_send_sync::<JointTorque>();
    }
}
