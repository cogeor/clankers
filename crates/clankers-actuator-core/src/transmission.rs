//! Gear transmission model with efficiency and backlash.
//!
//! # Gear Ratio Convention
//!
//! `gear_ratio = N_output / N_input` (tooth count ratio):
//! - `gear_ratio > 1` means torque multiplication / speed reduction.
//! - Joint torque = motor torque × `gear_ratio` × efficiency.
//! - Motor velocity = joint velocity × `gear_ratio`.

// ---------------------------------------------------------------------------
// Transmission
// ---------------------------------------------------------------------------

/// Gear transmission model with efficiency and backlash.
#[derive(Clone, Debug)]
pub struct Transmission {
    /// Gear ratio (output/input).  `> 1` means torque multiplication.
    pub gear_ratio: f32,
    /// Forward efficiency (0..1).  Fraction of torque transmitted motor-to-joint.
    pub efficiency: f32,
    /// Backlash dead zone angle on the joint side (rad).
    pub backlash: f32,
    // Last known joint-side position (rad).
    last_position: f32,
    // Cumulative travel since last direction reversal (rad).
    // Positive while in the dead zone; set to backlash+1 (sentinel) when engaged.
    dead_zone_travel: f32,
}

impl Default for Transmission {
    fn default() -> Self {
        Self::direct()
    }
}

impl Transmission {
    /// Direct drive (1:1 ratio, 100% efficient, no backlash).
    pub const fn direct() -> Self {
        Self {
            gear_ratio: 1.0,
            efficiency: 1.0,
            backlash: 0.0,
            last_position: 0.0,
            dead_zone_travel: f32::MAX,
        }
    }

    /// New transmission with the given gear ratio and 95% default efficiency.
    pub const fn new(gear_ratio: f32) -> Self {
        Self {
            gear_ratio,
            efficiency: 0.95,
            ..Self::direct()
        }
    }

    /// Set forward efficiency (clamped to `[0.0, 1.0]`).
    pub const fn with_efficiency(mut self, eff: f32) -> Self {
        self.efficiency = eff.clamp(0.0, 1.0);
        self
    }

    /// Set backlash dead zone angle (radians, joint side).
    pub const fn with_backlash(mut self, angle: f32) -> Self {
        self.backlash = angle;
        self
    }

    /// Transform motor torque to joint torque (forward direction).
    pub fn motor_to_joint(&self, motor_torque: f32) -> f32 {
        motor_torque * self.gear_ratio * self.efficiency
    }

    /// Transform joint velocity to motor velocity.
    pub fn joint_to_motor_velocity(&self, joint_vel: f32) -> f32 {
        joint_vel * self.gear_ratio
    }

    /// Backdrive efficiency: ability to transmit torque from joint to motor.
    ///
    /// Formula: `backdrive_eff = 2 × efficiency − 1`.
    /// Returns `0.0` if the transmission is non-backdrivable (efficiency < 0.5).
    pub fn backdrive_efficiency(&self) -> f32 {
        2.0f32.mul_add(self.efficiency, -1.0).max(0.0)
    }

    /// Set the internal backlash tracking position (joint-side, radians).
    pub const fn set_position(&mut self, position: f32) {
        self.last_position = position;
    }

    /// Apply backlash model.  Returns effective torque after dead zone.
    ///
    /// When direction reverses, torque is zeroed until the joint has
    /// accumulated enough displacement to cross the full backlash dead zone.
    pub fn apply_backlash(&mut self, position: f32, torque: f32) -> f32 {
        if self.backlash <= 0.0 {
            return torque;
        }

        let delta = position - self.last_position;
        self.last_position = position;

        let in_dead_zone = self.dead_zone_travel < self.backlash;

        if in_dead_zone {
            // Accumulate travel through the dead zone.
            self.dead_zone_travel += delta.abs();
            if self.dead_zone_travel >= self.backlash {
                // Exited dead zone — gears re-engaged.
                torque
            } else {
                0.0
            }
        } else if torque * delta < 0.0 {
            // Direction reversal detected — enter dead zone.
            self.dead_zone_travel = 0.0;
            0.0
        } else {
            torque
        }
    }

    /// Reset internal backlash state.
    pub const fn reset(&mut self) {
        self.last_position = 0.0;
        self.dead_zone_travel = f32::MAX;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_drive_unity() {
        let t = Transmission::direct();
        assert!((t.gear_ratio - 1.0).abs() < f32::EPSILON);
        assert!((t.efficiency - 1.0).abs() < f32::EPSILON);
        assert!((t.backlash).abs() < f32::EPSILON);
    }

    #[test]
    fn new_has_95_percent_efficiency() {
        let t = Transmission::new(50.0);
        assert!((t.efficiency - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn motor_to_joint_multiplies() {
        let t = Transmission::new(50.0).with_efficiency(1.0);
        assert!((t.motor_to_joint(1.0) - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn motor_to_joint_with_efficiency() {
        let t = Transmission::new(50.0).with_efficiency(0.9);
        assert!((t.motor_to_joint(1.0) - 45.0).abs() < 1e-5);
    }

    #[test]
    fn joint_to_motor_velocity() {
        let t = Transmission::new(50.0);
        assert!((t.joint_to_motor_velocity(1.0) - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn backdrive_efficiency_high() {
        let t = Transmission::new(50.0).with_efficiency(0.9);
        assert!((t.backdrive_efficiency() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn backdrive_efficiency_low_is_zero() {
        let t = Transmission::new(50.0).with_efficiency(0.4);
        assert!((t.backdrive_efficiency()).abs() < f32::EPSILON);
    }

    #[test]
    fn no_backlash_passes_torque() {
        let mut t = Transmission::direct();
        assert!((t.apply_backlash(0.1, 5.0) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn backlash_dead_zone_on_reversal() {
        let mut t = Transmission::new(1.0).with_backlash(0.02);
        // Moving forward: delta > 0, torque > 0 → same direction, no backlash.
        t.set_position(0.0);
        let torque = t.apply_backlash(0.1, 5.0);
        assert!((torque - 5.0).abs() < f32::EPSILON);
        // Reverse torque: delta = +0.01 (forward), torque = -5.0 → reversal.
        let torque = t.apply_backlash(0.11, -5.0);
        assert!((torque).abs() < f32::EPSILON, "should be zero in dead zone");
    }

    #[test]
    fn backlash_accumulates_through_dead_zone() {
        let mut t = Transmission::new(1.0).with_backlash(0.02);
        t.set_position(0.0);
        t.apply_backlash(0.1, 5.0); // forward, engaged (delta=+0.1, torque=+5)
        // Reversal: torque reverses but joint still drifts forward (momentum)
        t.apply_backlash(0.11, -5.0); // delta=+0.01, torque=-5 → reversal detected
        // Now in dead zone (travel=0). Moving backward slowly:
        let torque = t.apply_backlash(0.105, -5.0); // travel += 0.005
        assert!((torque).abs() < f32::EPSILON, "still in dead zone");
        // More backward travel (well past threshold to avoid float precision):
        let torque = t.apply_backlash(0.08, -5.0); // travel += 0.025, total > 0.02
        assert!((torque - (-5.0)).abs() < f32::EPSILON, "should re-engage");
    }

    #[test]
    fn backlash_exits_dead_zone() {
        let mut t = Transmission::new(1.0).with_backlash(0.005);
        t.set_position(0.0);
        t.apply_backlash(0.1, 5.0); // forward
        t.apply_backlash(0.09, -5.0); // reverse → dead zone, travel=0.01 >= 0.005
        // Already crossed backlash (0.01 >= 0.005), so should re-engage
        let torque = t.apply_backlash(0.08, -5.0);
        assert!((torque - (-5.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn reset_clears_backlash_state() {
        let mut t = Transmission::new(1.0).with_backlash(0.01);
        t.set_position(1.0);
        t.reset();
        assert!((t.last_position).abs() < f32::EPSILON);
        assert!(
            t.dead_zone_travel >= t.backlash,
            "should be engaged after reset"
        );
    }

    #[test]
    fn efficiency_clamps() {
        let t = Transmission::new(1.0).with_efficiency(1.5);
        assert!((t.efficiency - 1.0).abs() < f32::EPSILON);
        let t = Transmission::new(1.0).with_efficiency(-0.1);
        assert!((t.efficiency).abs() < f32::EPSILON);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn transmission_is_send_sync() {
        assert_send_sync::<Transmission>();
    }
}
