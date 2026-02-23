//! Physics property components for domain randomization and simulation.
//!
//! These lightweight Bevy components represent physical properties that
//! domain randomization systems can modify at episode reset. Physics
//! integrations (e.g., Rapier, XPBD) read these components to configure
//! their internal state.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Mass
// ---------------------------------------------------------------------------

/// Mass of a rigid body in kilograms.
///
/// Domain randomization can vary this per-episode to improve sim-to-real
/// transfer.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct Mass(pub f32);

impl Mass {
    /// Create a new mass value.
    #[must_use]
    pub const fn new(kg: f32) -> Self {
        Self(kg)
    }

    /// Mass in kilograms.
    #[must_use]
    pub const fn kg(self) -> f32 {
        self.0
    }
}

impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}

// ---------------------------------------------------------------------------
// SurfaceFriction
// ---------------------------------------------------------------------------

/// Surface friction coefficients for a collider.
///
/// Stores both static and dynamic friction coefficients.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct SurfaceFriction {
    /// Static (stiction) friction coefficient.
    pub static_friction: f32,
    /// Dynamic (kinetic) friction coefficient.
    pub dynamic_friction: f32,
}

impl SurfaceFriction {
    /// Create with explicit static and dynamic coefficients.
    #[must_use]
    pub const fn new(static_friction: f32, dynamic_friction: f32) -> Self {
        Self {
            static_friction,
            dynamic_friction,
        }
    }
}

impl Default for SurfaceFriction {
    fn default() -> Self {
        Self {
            static_friction: 0.5,
            dynamic_friction: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// ExternalForce
// ---------------------------------------------------------------------------

/// An external force applied to a rigid body each physics step.
///
/// Useful for simulating wind, perturbations, or random pushes during
/// domain randomization.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct ExternalForce {
    /// Force vector in world coordinates (Newtons).
    pub force: Vec3,
    /// Torque vector in world coordinates (Newton-meters).
    pub torque: Vec3,
}

impl ExternalForce {
    /// Create with both force and torque.
    #[must_use]
    pub const fn new(force: Vec3, torque: Vec3) -> Self {
        Self { force, torque }
    }

    /// Create with force only (zero torque).
    #[must_use]
    pub const fn from_force(force: Vec3) -> Self {
        Self {
            force,
            torque: Vec3::ZERO,
        }
    }

    /// Zero force and torque.
    pub const ZERO: Self = Self {
        force: Vec3::ZERO,
        torque: Vec3::ZERO,
    };
}

impl Default for ExternalForce {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_default() {
        let m = Mass::default();
        assert!((m.kg() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mass_new() {
        let m = Mass::new(5.0);
        assert!((m.0 - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn surface_friction_default() {
        let f = SurfaceFriction::default();
        assert!((f.static_friction - 0.5).abs() < f32::EPSILON);
        assert!((f.dynamic_friction - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn surface_friction_new() {
        let f = SurfaceFriction::new(0.8, 0.6);
        assert!((f.static_friction - 0.8).abs() < f32::EPSILON);
        assert!((f.dynamic_friction - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn external_force_default_is_zero() {
        let f = ExternalForce::default();
        assert_eq!(f.force, Vec3::ZERO);
        assert_eq!(f.torque, Vec3::ZERO);
    }

    #[test]
    fn external_force_from_force() {
        let f = ExternalForce::from_force(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(f.force, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(f.torque, Vec3::ZERO);
    }

    #[test]
    fn external_force_new() {
        let f = ExternalForce::new(Vec3::X, Vec3::Y);
        assert_eq!(f.force, Vec3::X);
        assert_eq!(f.torque, Vec3::Y);
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn physics_types_are_send_sync() {
        assert_send_sync::<Mass>();
        assert_send_sync::<SurfaceFriction>();
        assert_send_sync::<ExternalForce>();
    }
}
