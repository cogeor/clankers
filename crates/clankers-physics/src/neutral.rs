//! Engine-neutral physics handles and readback structs.
//!
//! CODE_QUALITY_REVIEW § "Engine-Neutral Physics Boundary" / P2.1.
//! These types are the cross-process / cross-backend lingua franca for
//! body / joint identity and per-step readback. They live in
//! `clankers-physics` rather than a new `clankers-physics-core` crate
//! because the dependency graph already has every workspace consumer
//! depending on `clankers-physics`; spinning out a fresh crate would
//! force a workspace-wide Cargo edit for marginal benefit.
//!
//! # What's neutral
//!
//! - [`BodyHandle`] / [`JointHandle`] — opaque 32-bit identifiers. The
//!   backend assigns them; consumers never inspect the bits. A future
//!   GPU / Brax-style backend can produce them just as well as Rapier.
//! - [`BodyPose`] — position + orientation as plain arrays. Same wire
//!   shape as MuJoCo `qpos` / Brax `qp.x`, `qp.rot`.
//! - [`ContactEvent`] — body-pair + force. Backend-independent because
//!   force is a vector, not a `rapier::ContactPair`.
//!
//! # What's NOT here (yet)
//!
//! The readback API (`trait PhysicsReadback`) that consumers will
//! eventually use to fetch these via `body_poses()` / `contacts()` —
//! that lives in P2.2 / P3 work and binds these types to a backend
//! resource. This module is the foundation.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// Opaque engine-neutral identifier for a rigid body.
///
/// Backends translate to their internal handle at the boundary
/// (Rapier: `RigidBodyHandle`). Consumers treat the inner `u32` as
/// opaque — no arithmetic, no inspection. Hash / Eq are derived so
/// it works as a `HashMap` key, but the value itself has no semantic
/// meaning across backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BodyHandle(pub u32);

impl BodyHandle {
    /// Build a handle from a raw `u32`. Backends are the only callers
    /// that should construct one; tests may use it too.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The raw integer value. Exposed so backends can convert back to
    /// their internal handle type.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Opaque engine-neutral identifier for an articulated / impulse joint.
///
/// See [`BodyHandle`] for the rationale; the same pattern applies.
/// Rapier maps this to `ImpulseJointHandle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct JointHandle(pub u32);

impl JointHandle {
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// BodyPose
// ---------------------------------------------------------------------------

/// World-frame pose of a single rigid body.
///
/// Layout matches the field order used by every major physics engine
/// we'd care to back: position xyz, orientation as a quaternion in
/// xyzw order (Rapier / Bevy / glTF convention; NOT wxyz which some
/// older libs use).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyPose {
    /// World-frame position (m).
    pub position: [f32; 3],
    /// World-frame orientation as a quaternion [x, y, z, w].
    pub orientation_xyzw: [f32; 4],
}

impl BodyPose {
    /// Identity pose: at the world origin, no rotation.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Whether every field is finite (no NaN, no Inf). Cheap sanity
    /// check for backend boundary code.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|v| v.is_finite())
            && self.orientation_xyzw.iter().all(|v| v.is_finite())
    }
}

impl Default for BodyPose {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// ContactEvent
// ---------------------------------------------------------------------------

/// A single backend-emitted contact event.
///
/// Two bodies in contact plus the force vector applied at the
/// contact point. The force is a 3-vector (not a magnitude) so
/// directional contacts (oblique impacts, side-loaded grasps) carry
/// the full signal without an extra field.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactEvent {
    /// First body involved in the contact.
    pub body_a: BodyHandle,
    /// Second body involved in the contact.
    pub body_b: BodyHandle,
    /// Contact force vector (N) applied at the contact point. The
    /// sign convention is backend-defined for now; the typical Rapier
    /// reading is force on `body_a` from `body_b` in world frame.
    pub force: [f32; 3],
}

impl ContactEvent {
    /// Magnitude of the contact force in newtons.
    #[must_use]
    pub fn force_magnitude(&self) -> f32 {
        let [x, y, z] = self.force;
        (x * x + y * y + z * z).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_pose_identity_is_finite() {
        assert!(BodyPose::identity().is_finite());
        assert_eq!(BodyPose::identity(), BodyPose::default());
    }

    #[test]
    fn body_pose_nan_position_is_not_finite() {
        let bp = BodyPose {
            position: [f32::NAN, 0.0, 0.0],
            orientation_xyzw: [0.0, 0.0, 0.0, 1.0],
        };
        assert!(!bp.is_finite());
    }

    #[test]
    fn body_pose_inf_orientation_is_not_finite() {
        let bp = BodyPose {
            position: [0.0, 0.0, 0.0],
            orientation_xyzw: [0.0, 0.0, 0.0, f32::INFINITY],
        };
        assert!(!bp.is_finite());
    }

    #[test]
    fn handle_roundtrip() {
        let bh = BodyHandle::new(42);
        assert_eq!(bh.as_u32(), 42);
        let jh = JointHandle::new(7);
        assert_eq!(jh.as_u32(), 7);
    }

    #[test]
    fn handles_serde_roundtrip() {
        let bh = BodyHandle::new(123);
        let json = serde_json::to_string(&bh).unwrap();
        let back: BodyHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(bh, back);
    }

    #[test]
    fn contact_force_magnitude() {
        let c = ContactEvent {
            body_a: BodyHandle::new(0),
            body_b: BodyHandle::new(1),
            force: [3.0, 4.0, 0.0],
        };
        // 3-4-5 triangle.
        assert!((c.force_magnitude() - 5.0).abs() < 1e-6);
    }
}
