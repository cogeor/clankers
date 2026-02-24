//! Physics marker components.
//!
//! Lightweight ECS markers that the physics backend attaches to entities
//! participating in the simulation. These decouple the rest of the system
//! from the concrete physics engine.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// PhysicsBody
// ---------------------------------------------------------------------------

/// Marker on entities with a rigid body representation in the physics backend.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsBody {
    /// Immovable body (e.g., world/base link).
    Fixed,
    /// Body affected by forces and gravity.
    Dynamic,
}

// ---------------------------------------------------------------------------
// PhysicsJoint
// ---------------------------------------------------------------------------

/// Marker on joint entities that have a physics joint representation.
///
/// Stores the parent and child body entities so the backend can look up
/// the corresponding rigid bodies.
#[derive(Component, Debug, Clone)]
pub struct PhysicsJoint {
    /// Entity with the parent [`PhysicsBody`].
    pub parent_body: Entity,
    /// Entity with the child [`PhysicsBody`].
    pub child_body: Entity,
}

// ---------------------------------------------------------------------------
// GroundPlane
// ---------------------------------------------------------------------------

/// Marker for the static ground plane body.
#[derive(Component, Debug, Default)]
pub struct GroundPlane;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn components_are_send_sync() {
        assert_send_sync::<PhysicsBody>();
        assert_send_sync::<PhysicsJoint>();
        assert_send_sync::<GroundPlane>();
    }

    #[test]
    fn physics_body_variants() {
        let fixed = PhysicsBody::Fixed;
        let dynamic = PhysicsBody::Dynamic;
        assert_ne!(fixed, dynamic);
        assert_eq!(fixed, PhysicsBody::Fixed);
        assert_eq!(dynamic, PhysicsBody::Dynamic);
    }

    #[test]
    fn physics_joint_construction() {
        let joint = PhysicsJoint {
            parent_body: Entity::from_bits(1),
            child_body: Entity::from_bits(2),
        };
        assert_ne!(joint.parent_body, joint.child_body);
    }

    #[test]
    fn ground_plane_default() {
        let _gp = GroundPlane::default();
    }
}
