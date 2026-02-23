//! Framework-agnostic actuator dynamics and motor models for robotics
//! simulation.
//!
//! Pure Rust library with no game engine dependencies.  Provides realistic
//! motor behavior, gear transmission, joint friction, and PID/PD controllers.
//!
//! # Motor Pipeline
//!
//! ```text
//! Command → Control Logic → Motor Dynamics → Transmission → Friction → Output Torque
//!           (PID/PD)        (torque-speed)   (gear ratio)   (Coulomb)
//! ```
//!
//! # Quick Start
//!
//! ```
//! use clankers_actuator_core::prelude::*;
//!
//! let mut motor = DcMotor::new(10.0, 5.0).with_time_constant(0.02);
//! let transmission = Transmission::new(50.0);
//! let friction = FrictionModel::new(0.5, 0.1);
//!
//! let dt = 0.001;
//! let torque_cmd = 5.0;
//! let velocity = 0.0;
//!
//! let motor_torque = motor.step(torque_cmd, velocity, dt);
//! let joint_torque = transmission.motor_to_joint(motor_torque);
//! let friction_torque = friction.compute_smooth(velocity);
//! let net_torque = joint_torque + friction_torque;
//! ```

pub mod control;
pub mod friction;
pub mod motor;
pub mod presets;
pub mod transmission;

/// Convenience re-exports for common usage.
pub mod prelude {
    pub use crate::control::{ControlMode, PdController, PidController};
    pub use crate::friction::FrictionModel;
    pub use crate::motor::{DcMotor, FullDcMotor, IdealMotor, MotorModel, MotorType};
    pub use crate::presets;
    pub use crate::transmission::Transmission;
}
