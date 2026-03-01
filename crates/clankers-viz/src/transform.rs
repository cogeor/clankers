//! Coordinate transforms between physics (Z-up) and Bevy (Y-up) frames.
//!
//! Rapier and most robotics conventions use a right-handed Z-up frame.
//! Bevy uses a right-handed Y-up frame.  The canonical mapping is:
//!
//! ```text
//! physics (x, y, z)  ->  visual (x, z, -y)
//! ```
//!
//! These helpers centralise the conversion so every crate in the workspace
//! uses the same transform.

use bevy::prelude::{Quat, Vec3};

/// Convert a position from physics Z-up to Bevy Y-up coordinates.
///
/// Mapping: `(x, y, z) -> (x, z, -y)`
#[inline]
pub fn phys_to_vis(pos: Vec3) -> Vec3 {
    Vec3::new(pos.x, pos.z, -pos.y)
}

/// Convert a rotation quaternion from physics Z-up to Bevy Y-up coordinates.
///
/// Mapping: `(x, y, z, w) -> (x, z, -y, w)`
#[inline]
pub fn phys_rot_to_vis(r: &Quat) -> Quat {
    Quat::from_xyzw(r.x, r.z, -r.y, r.w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn identity_position() {
        let p = phys_to_vis(Vec3::ZERO);
        assert_eq!(p, Vec3::ZERO);
    }

    #[test]
    fn z_up_maps_to_y_up() {
        // Physics Z-up unit vector -> Bevy Y-up unit vector
        let p = phys_to_vis(Vec3::new(0.0, 0.0, 1.0));
        assert!((p - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn y_forward_maps_to_neg_z() {
        // Physics Y-forward -> Bevy -Z
        let p = phys_to_vis(Vec3::new(0.0, 1.0, 0.0));
        assert!((p - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-6);
    }

    #[test]
    fn identity_rotation() {
        let q = Quat::IDENTITY;
        let v = phys_rot_to_vis(&q);
        assert!((v - Quat::IDENTITY).length() < 1e-6);
    }

    #[test]
    fn rotation_round_trip_preserves_unit() {
        let q = Quat::from_rotation_z(FRAC_PI_2);
        let v = phys_rot_to_vis(&q);
        assert!((v.length() - 1.0).abs() < 1e-6, "quaternion should remain unit");
    }
}
