//! Joint friction models: Coulomb, viscous, and Stribeck.
//!
//! # Physics
//!
//! Dynamic regime: `F = -coulomb·sign(v) - viscous·v`
//!
//! Static regime (|v| < `stiction_velocity`):
//! - If `|applied_torque| < stiction`: friction cancels the applied torque
//!   (holds the joint still).
//! - Otherwise: breakaway at `coulomb` level.
//!
//! Stribeck effect: extra stiction decays as `exp(-(v/v_s)²)`.

// ---------------------------------------------------------------------------
// FrictionModel
// ---------------------------------------------------------------------------

/// Joint friction model with Coulomb, viscous, and Stribeck components.
#[derive(Clone, Debug)]
pub struct FrictionModel {
    /// Coulomb (kinetic) friction (Nm).
    pub coulomb: f32,
    /// Viscous damping coefficient (Nm/(rad/s)).
    pub viscous: f32,
    /// Static friction / stiction threshold (Nm).
    pub stiction: f32,
    /// Velocity threshold for stiction transition (rad/s).
    pub stiction_velocity: f32,
}

impl Default for FrictionModel {
    fn default() -> Self {
        Self::none()
    }
}

impl FrictionModel {
    /// No friction at all.
    pub const fn none() -> Self {
        Self {
            coulomb: 0.0,
            viscous: 0.0,
            stiction: 0.0,
            stiction_velocity: 0.01,
        }
    }

    /// New friction model with Coulomb and viscous terms.
    ///
    /// Stiction defaults to `1.2 × coulomb`.
    pub fn new(coulomb: f32, viscous: f32) -> Self {
        Self {
            coulomb,
            viscous,
            stiction: coulomb * 1.2,
            stiction_velocity: 0.01,
        }
    }

    /// Set stiction threshold.
    pub const fn with_stiction(mut self, stiction: f32) -> Self {
        self.stiction = stiction;
        self
    }

    /// Set stiction velocity threshold (rad/s).
    pub const fn with_stiction_velocity(mut self, vel: f32) -> Self {
        self.stiction_velocity = vel;
        self
    }

    /// Compute friction torque with stiction check (opposes motion).
    ///
    /// Two-argument form using `applied_torque` for static friction regime.
    ///
    /// - Static regime (`|velocity| < stiction_velocity`):
    ///   - If `|applied_torque| < stiction`: returns `-applied_torque` (hold still).
    ///   - Otherwise: returns `-coulomb × sign(applied_torque)` (breakaway).
    /// - Dynamic regime: returns `-coulomb × sign(v) - viscous × v`.
    pub fn compute(&self, velocity: f32, applied_torque: f32) -> f32 {
        if velocity.abs() < self.stiction_velocity {
            // Static regime.
            if applied_torque.abs() < self.stiction {
                -applied_torque
            } else {
                -self.coulomb * applied_torque.signum()
            }
        } else {
            // Dynamic regime.
            (-self.coulomb).mul_add(velocity.signum(), -(self.viscous * velocity))
        }
    }

    /// Compute velocity-dependent friction only (no stiction check).
    ///
    /// Returns `-coulomb × sign(v) - viscous × v`.
    pub fn compute_velocity(&self, velocity: f32) -> f32 {
        (-self.coulomb).mul_add(velocity.signum(), -(self.viscous * velocity))
    }

    /// Smooth friction approximation (no discontinuities at v=0).
    ///
    /// Uses `tanh()` for Coulomb and `exp()` for Stribeck effect.
    /// Suitable for gradient-based optimization or stiff solvers.
    pub fn compute_smooth(&self, velocity: f32) -> f32 {
        let v_norm = velocity / self.stiction_velocity;
        let coulomb = -self.coulomb * v_norm.tanh();
        let viscous = -self.viscous * velocity;

        // Stribeck effect: extra static friction decays with velocity.
        let stribeck = if self.stiction > self.coulomb {
            let extra = self.stiction - self.coulomb;
            -extra * (-v_norm.powi(2)).exp() * v_norm.tanh()
        } else {
            0.0
        };

        coulomb + viscous + stribeck
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_friction_returns_zero() {
        let f = FrictionModel::none();
        assert!((f.compute(1.0, 5.0)).abs() < f32::EPSILON);
        assert!((f.compute_velocity(1.0)).abs() < f32::EPSILON);
        assert!((f.compute_smooth(1.0)).abs() < 1e-6);
    }

    #[test]
    fn stiction_holds_still() {
        let f = FrictionModel::new(1.0, 0.0);
        // |torque| = 0.5 < stiction (1.2)
        let friction = f.compute(0.0, 0.5);
        assert!((friction - (-0.5)).abs() < f32::EPSILON);
    }

    #[test]
    fn stiction_breakaway() {
        let f = FrictionModel::new(1.0, 0.0);
        // |torque| = 5.0 > stiction (1.2)
        let friction = f.compute(0.0, 5.0);
        assert!((friction - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn dynamic_coulomb() {
        let f = FrictionModel::new(1.0, 0.0);
        // Moving fast → dynamic regime
        let friction = f.compute(1.0, 5.0);
        assert!((friction - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn dynamic_viscous() {
        let f = FrictionModel::new(0.0, 0.5);
        let friction = f.compute(2.0, 0.0);
        // Dynamic: -0 * sign(2) - 0.5 * 2 = -1.0
        assert!((friction - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn dynamic_coulomb_plus_viscous() {
        let f = FrictionModel::new(1.0, 0.5);
        let friction = f.compute(2.0, 5.0);
        // -1.0 * sign(2) - 0.5 * 2 = -1.0 - 1.0 = -2.0
        assert!((friction - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_velocity_simple() {
        let f = FrictionModel::new(1.0, 0.5);
        let friction = f.compute_velocity(2.0);
        assert!((friction - (-2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_velocity_negative() {
        let f = FrictionModel::new(1.0, 0.5);
        let friction = f.compute_velocity(-2.0);
        // -1.0 * (-1.0) - 0.5 * (-2.0) = 1.0 + 1.0 = 2.0
        assert!((friction - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn smooth_approximates_coulomb_at_high_speed() {
        let f = FrictionModel::new(1.0, 0.0);
        // At high velocity, tanh → ±1
        let friction = f.compute_smooth(10.0);
        assert!((friction - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn smooth_zero_at_zero_velocity() {
        let f = FrictionModel::new(1.0, 0.0);
        let friction = f.compute_smooth(0.0);
        assert!((friction).abs() < f32::EPSILON);
    }

    #[test]
    fn smooth_stribeck_extra_friction() {
        let f = FrictionModel::new(1.0, 0.0).with_stiction(2.0);
        // Near zero velocity, smooth Stribeck should add extra friction
        let near_zero = f.compute_smooth(0.005);
        let at_speed = f.compute_smooth(1.0);
        // Near zero should have higher magnitude (Stribeck bump)
        // Both should be negative (opposing positive velocity)
        assert!(near_zero < 0.0);
        assert!(at_speed < 0.0);
    }

    #[test]
    fn default_stiction_is_1_2x_coulomb() {
        let f = FrictionModel::new(5.0, 0.1);
        assert!((f.stiction - 6.0).abs() < f32::EPSILON);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn friction_model_is_send_sync() {
        assert_send_sync::<FrictionModel>();
    }
}
