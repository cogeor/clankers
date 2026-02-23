//! Common motor and transmission presets based on real robotics hardware.

use crate::motor::DcMotor;
use crate::transmission::Transmission;

/// Common motor configurations.
pub mod motors {
    use super::DcMotor;

    /// Small hobby servo (Dynamixel XL330 class).
    pub const fn hobby_servo() -> DcMotor {
        DcMotor::new(0.52, 61.0).with_time_constant(0.02)
    }

    /// Industrial robot joint (UR5 class).
    #[allow(clippy::approx_constant)] // 3.14 rad/s is the no-load speed, not π
    pub const fn industrial_joint() -> DcMotor {
        DcMotor::new(150.0, 3.14)
            .with_time_constant(0.01)
            .with_torque_limit(100.0)
    }

    /// High-torque servo (Robotis H54 class).
    pub const fn high_torque_servo() -> DcMotor {
        DcMotor::new(44.7, 5.76).with_time_constant(0.015)
    }

    /// Direct-drive motor (high bandwidth, no gearbox).
    pub const fn direct_drive() -> DcMotor {
        DcMotor::new(10.0, 30.0).with_time_constant(0.005)
    }

    /// Quadruped leg motor (MIT Mini Cheetah class).
    pub const fn quadruped_leg() -> DcMotor {
        DcMotor::new(17.0, 40.0).with_time_constant(0.008)
    }
}

/// Common transmission configurations.
pub mod transmissions {
    use super::Transmission;

    /// Harmonic drive (strain wave).  High ratio, low backlash.
    pub const fn harmonic_drive(ratio: f32) -> Transmission {
        Transmission::new(ratio)
            .with_efficiency(0.85)
            .with_backlash(0.001)
    }

    /// Planetary gearbox.  Moderate efficiency, some backlash.
    pub const fn planetary(ratio: f32) -> Transmission {
        Transmission::new(ratio)
            .with_efficiency(0.92)
            .with_backlash(0.005)
    }

    /// Cycloidal drive.  High torque density, minimal backlash.
    pub const fn cycloidal(ratio: f32) -> Transmission {
        Transmission::new(ratio)
            .with_efficiency(0.90)
            .with_backlash(0.0005)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hobby_servo_valid() {
        let m = motors::hobby_servo();
        assert!(m.stall_torque > 0.0);
        assert!(m.no_load_speed > 0.0);
        assert!(m.time_constant > 0.0);
    }

    #[test]
    fn industrial_joint_valid() {
        let m = motors::industrial_joint();
        assert!(m.stall_torque > 0.0);
        assert!(m.max_torque <= m.stall_torque);
    }

    #[test]
    fn high_torque_servo_valid() {
        let m = motors::high_torque_servo();
        assert!(m.stall_torque > 0.0);
    }

    #[test]
    fn direct_drive_valid() {
        let m = motors::direct_drive();
        assert!(m.stall_torque > 0.0);
    }

    #[test]
    fn quadruped_leg_valid() {
        let m = motors::quadruped_leg();
        assert!(m.stall_torque > 0.0);
    }

    #[test]
    fn harmonic_drive_valid() {
        let t = transmissions::harmonic_drive(100.0);
        assert!((t.gear_ratio - 100.0).abs() < f32::EPSILON);
        assert!((t.efficiency - 0.85).abs() < f32::EPSILON);
        assert!((t.backlash - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn planetary_valid() {
        let t = transmissions::planetary(50.0);
        assert!((t.gear_ratio - 50.0).abs() < f32::EPSILON);
        assert!((t.efficiency - 0.92).abs() < f32::EPSILON);
    }

    #[test]
    fn cycloidal_valid() {
        let t = transmissions::cycloidal(30.0);
        assert!((t.gear_ratio - 30.0).abs() < f32::EPSILON);
        assert!((t.efficiency - 0.90).abs() < f32::EPSILON);
        assert!((t.backlash - 0.0005).abs() < f32::EPSILON);
    }
}
