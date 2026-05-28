//! Engine-neutral scene-object spawning (P2.3).
//!
//! CODE_QUALITY_REVIEW § P2.3 — `clankers-sim` builds scenes without
//! returning Rapier handles in public structs. The shape-to-collider
//! mapping and the rapier-body insertion live here; the only handle
//! a consumer ever sees is the opaque [`crate::neutral::BodyHandle`].
//!
//! `RapierContext` keeps an internal `object_handle_table` mapping
//! `BodyHandle(idx)` → `RigidBodyHandle`, so future readback /
//! command-buffer code can translate the neutral handle back to the
//! concrete one without leaking rapier types.

use bevy::log::warn;
use bevy::math::Vec3;
use clankers_core::config::{ObjectConfig, Shape};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle};

use crate::neutral::BodyHandle;
use crate::rapier::RapierContext;

// ---------------------------------------------------------------------------
// shape_to_collider
// ---------------------------------------------------------------------------

/// Convert a [`Shape`] config into a Rapier collider builder.
///
/// Internal to `clankers-physics` so `clankers-sim` and downstream code
/// never import `rapier3d` just to choose a shape.
fn shape_to_collider(shape: &Shape) -> ColliderBuilder {
    match shape {
        Shape::Sphere(radius) => ColliderBuilder::ball(*radius),
        Shape::Box(half_extents) => {
            ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2])
        }
        Shape::Cylinder {
            radius,
            half_height,
        } => ColliderBuilder::cylinder(*half_height, *radius),
        Shape::Capsule {
            radius,
            half_height,
        } => ColliderBuilder::capsule_y(*half_height, *radius),
        // Mesh shapes are not yet supported; fall back to a small sphere
        // so the build does not panic.
        Shape::ConvexMesh(_) | Shape::TriMesh(_) => {
            warn!(
                "mesh shapes are not yet supported in with_object(); falling back to unit sphere collider"
            );
            ColliderBuilder::ball(0.01)
        }
    }
}

// ---------------------------------------------------------------------------
// Spawning
// ---------------------------------------------------------------------------

impl RapierContext {
    /// Add a static or dynamic scene object and return the engine-neutral
    /// [`BodyHandle`] under which it is tracked.
    ///
    /// The returned handle is opaque: `clankers-sim` and downstream callers
    /// store it without depending on rapier3d. Internally the context keeps
    /// a side-table mapping handle → `RigidBodyHandle` so future
    /// boundary-mirroring code can resolve it back at the right layer.
    pub fn add_scene_object(&mut self, obj: &ObjectConfig) -> BodyHandle {
        let body_builder = if obj.is_static {
            RigidBodyBuilder::fixed()
        } else {
            RigidBodyBuilder::dynamic().can_sleep(false)
        };
        let body = self.rigid_body_set.insert(
            body_builder
                .translation(Vec3::new(obj.position[0], obj.position[1], obj.position[2]))
                .build(),
        );

        let collider = shape_to_collider(&obj.shape)
            .density(obj.mass)
            .friction(obj.friction)
            .restitution(obj.restitution)
            .sensor(obj.is_sensor)
            .build();
        self.collider_set
            .insert_with_parent(collider, body, &mut self.rigid_body_set);

        self.body_handles.insert(obj.name.clone(), body);
        let id = self.object_handle_table.len() as u32;
        self.object_handle_table.push(body);
        BodyHandle::new(id)
    }

    /// Resolve a neutral [`BodyHandle`] (returned by [`Self::add_scene_object`])
    /// to the underlying rapier handle. Returns `None` if the handle did not
    /// originate from this context.
    #[must_use]
    pub fn resolve_object_body(&self, h: BodyHandle) -> Option<RigidBodyHandle> {
        self.object_handle_table.get(h.as_u32() as usize).copied()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn obj(name: &str) -> ObjectConfig {
        ObjectConfig {
            name: name.to_string(),
            shape: Shape::Sphere(0.5),
            position: [1.0, 2.0, 3.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            is_static: false,
            mass: 1.0,
            friction: 0.5,
            restitution: 0.0,
            is_sensor: false,
        }
    }

    #[test]
    fn add_scene_object_assigns_monotonic_handles() {
        let mut ctx = RapierContext::new(Vec3::ZERO, 0.01, 1);
        let h0 = ctx.add_scene_object(&obj("a"));
        let h1 = ctx.add_scene_object(&obj("b"));
        assert_eq!(h0.as_u32(), 0);
        assert_eq!(h1.as_u32(), 1);
    }

    #[test]
    fn resolve_roundtrips_through_side_table() {
        let mut ctx = RapierContext::new(Vec3::ZERO, 0.01, 1);
        let h = ctx.add_scene_object(&obj("box"));
        let rapier = ctx.resolve_object_body(h).expect("handle in table");
        // The handle should refer to an existing body in the set.
        assert!(ctx.rigid_body_set.get(rapier).is_some());
    }

    #[test]
    fn resolve_unknown_handle_returns_none() {
        let ctx = RapierContext::new(Vec3::ZERO, 0.01, 1);
        assert!(ctx.resolve_object_body(BodyHandle::new(99)).is_none());
    }

    #[test]
    fn body_handles_named_map_updated() {
        let mut ctx = RapierContext::new(Vec3::ZERO, 0.01, 1);
        ctx.add_scene_object(&obj("ball"));
        assert!(ctx.body_handles.contains_key("ball"));
    }
}
