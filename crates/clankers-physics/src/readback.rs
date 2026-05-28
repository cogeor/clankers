//! Engine-neutral physics readback (P2.2).
//!
//! `CODE_QUALITY_REVIEW` § Phase 2.2 — "`GymEnv` no longer reads `RapierContext`
//! directly". Provides a single backend-dispatching entry point so consumers
//! (gym, sensors, recorder) collect named body poses + contact events without
//! importing concrete backend types or poking their internal sets.
//!
//! Adding a new backend (XPBD, Brax, GPU) means implementing this dispatch in
//! one place — the consumer call site does not change.
//!
//! The shape here is protocol-friendly (string names + `[f32; 7]` poses,
//! force-magnitude scalars). It is intentionally distinct from
//! [`crate::neutral`], which carries opaque `BodyHandle` ids and full 3-vector
//! force readings for in-process hot loops.

use std::collections::HashMap;

use bevy::prelude::World;
use clankers_core::types::ContactEvent;

use crate::rapier::RapierContext;

// ---------------------------------------------------------------------------
// StepReadback
// ---------------------------------------------------------------------------

/// Per-step physics readback in the wire-protocol shape.
///
/// `body_poses` are keyed by URDF link name; the value is the standard
/// `[x, y, z, qx, qy, qz, qw]` order used everywhere on the wire.
///
/// `contact_events` carry body-pair names + scalar force magnitude in
/// newtons. The contact filter is "any active contact this step".
#[derive(Debug, Default, Clone)]
pub struct StepReadback {
    /// Named link → world-frame `[x, y, z, qx, qy, qz, qw]`.
    pub body_poses: HashMap<String, [f32; 7]>,
    /// Active contact events this step.
    pub contact_events: Vec<ContactEvent>,
}

// ---------------------------------------------------------------------------
// PhysicsReadback
// ---------------------------------------------------------------------------

/// Backend-side ability to produce a [`StepReadback`].
///
/// Implemented by concrete backend resources (currently
/// [`RapierContext`]); consumers should call the free function
/// [`collect_step_readback`] instead of reaching for the resource type
/// themselves.
pub trait PhysicsReadback {
    /// Collect the current step's named body poses + active contact events.
    fn step_readback(&self) -> StepReadback;
}

// ---------------------------------------------------------------------------
// Rapier impl
// ---------------------------------------------------------------------------

impl PhysicsReadback for RapierContext {
    fn step_readback(&self) -> StepReadback {
        let mut body_poses = HashMap::new();
        for (name, &handle) in &self.body_handles {
            if let Some(body) = self.rigid_body_set.get(handle) {
                let t = body.translation();
                let r = body.rotation();
                body_poses.insert(name.clone(), [t.x, t.y, t.z, r.x, r.y, r.z, r.w]);
            }
        }

        let mut collider_to_name: HashMap<rapier3d::prelude::ColliderHandle, String> =
            HashMap::new();
        for (name, &body_handle) in &self.body_handles {
            if let Some(body) = self.rigid_body_set.get(body_handle) {
                for &collider_handle in body.colliders() {
                    collider_to_name.insert(collider_handle, name.clone());
                }
            }
        }

        let dt = self.integration_parameters.dt;
        let mut contact_events = Vec::new();
        for pair in self.narrow_phase.contact_pairs() {
            if !pair.has_any_active_contact() {
                continue;
            }
            let name_a = collider_to_name
                .get(&pair.collider1)
                .cloned()
                .unwrap_or_default();
            let name_b = collider_to_name
                .get(&pair.collider2)
                .cloned()
                .unwrap_or_default();
            let impulse = pair.total_impulse();
            let force_magnitude = if dt > 0.0 {
                impulse.length() / dt
            } else {
                impulse.length()
            };
            if !name_a.is_empty() || !name_b.is_empty() {
                contact_events.push(ContactEvent {
                    body_a: name_a,
                    body_b: name_b,
                    force_magnitude,
                });
            }
        }

        StepReadback {
            body_poses,
            contact_events,
        }
    }
}

// ---------------------------------------------------------------------------
// World-level entry point
// ---------------------------------------------------------------------------

/// Collect a [`StepReadback`] from whichever backend is registered in `world`.
///
/// Returns an empty readback when no recognised backend is present (e.g.
/// headless tests with no physics).
#[must_use]
pub fn collect_step_readback(world: &World) -> StepReadback {
    if let Some(ctx) = world.get_resource::<RapierContext>() {
        return ctx.step_readback();
    }
    StepReadback::default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::App;

    #[test]
    fn empty_world_returns_empty_readback() {
        let app = App::new();
        let r = collect_step_readback(app.world());
        assert!(r.body_poses.is_empty());
        assert!(r.contact_events.is_empty());
    }

    #[test]
    fn step_readback_default_is_empty() {
        let r = StepReadback::default();
        assert!(r.body_poses.is_empty());
        assert!(r.contact_events.is_empty());
    }

    #[test]
    fn rapier_context_with_no_bodies_returns_empty_readback() {
        let ctx = RapierContext::new(bevy::math::Vec3::ZERO, 0.01, 1);
        let r = ctx.step_readback();
        assert!(r.body_poses.is_empty());
        assert!(r.contact_events.is_empty());
    }
}
