//! Dense, slot-indexed runtime views over Rapier handles and motor
//! params, compiled once at scene-build time and read by every
//! [`rapier_step_system`](super::systems::rapier_step_system) frame.
//!
//! W7 PR3 — `perf(physics): compile setup HashMaps to dense runtime
//! vectors`. The setup-time `HashMaps` in
//! [`RapierContext`](super::context::RapierContext)
//! (`joint_handles`, `joint_info`) and the per-entity override map in
//! [`MotorOverrides`](super::systems::MotorOverrides) remain the
//! source of truth for scene construction. This module defines the
//! dense, layout-slot-indexed views that the hot path iterates instead
//! of probing those `HashMaps` per joint per frame.
//!
//! Population happens in
//! [`clankers_sim::builder::compile_runtime`](https://docs.rs/clankers-sim)
//! from the [`JointLayout`](clankers_core::layout::JointLayout) bound
//! to the scene's spawned entities. The free function lives in
//! `clankers-sim` rather than `clankers-physics` so the dep edge stays
//! `clankers-physics ← clankers-sim` (no cycle through
//! `clankers-core::RobotGroup`).
//!
//! # Behaviour invariant
//!
//! `JointRuntime.info` is a **snapshot** of
//! [`JointInfo`] taken at compile time. If callers mutate
//! `ctx.joint_info` post-scene-build they must rebuild `JointRuntimes`
//! to see the change. The hot path never reads `ctx.joint_info` when
//! the dense path is active.

use std::collections::HashMap;

use bevy::prelude::{Entity, Resource};
use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};

use crate::rapier::context::JointInfo;
use crate::rapier::systems::MotorOverrideParams;

// ---------------------------------------------------------------------------
// BodyRuntime
// ---------------------------------------------------------------------------

/// Slot-indexed view of one rigid body.
///
/// Populated by `clankers_sim::builder::compile_runtime` for callers that
/// want a layout-ordered handle table for body-level lookups (the W7 PR3
/// hot path currently does not iterate this — `rapier_step_system` walks
/// joints. Provided here for future per-body migrations.)
#[derive(Clone, Debug)]
pub struct BodyRuntime {
    /// Bevy entity owning this body.
    pub entity: Entity,
    /// Rapier rigid body handle.
    pub handle: RigidBodyHandle,
    /// Position in the source [`JointLayout`](clankers_core::layout::JointLayout).
    pub layout_slot: usize,
}

// ---------------------------------------------------------------------------
// JointRuntime
// ---------------------------------------------------------------------------

/// Slot-indexed view of one impulse joint.
///
/// `info` is a **snapshot** at compile time — see module docs.
#[derive(Clone, Debug)]
pub struct JointRuntime {
    /// Bevy entity owning this joint.
    pub entity: Entity,
    /// Rapier impulse joint handle.
    pub handle: ImpulseJointHandle,
    /// Position in the source [`JointLayout`](clankers_core::layout::JointLayout).
    pub layout_slot: usize,
    /// Joint metadata snapshot (axis + `is_prismatic` + parent/child body
    /// handles). Snapshotted from
    /// [`RapierContext::joint_info`](super::context::RapierContext::joint_info)
    /// at compile time.
    pub info: JointInfo,
    /// Motor override params for this slot, if the joint is actuated.
    ///
    /// `Some` for every actuated joint after a successful
    /// `compile_runtime` (the MEMORY.md invariant promotion). Non-
    /// actuated joints never make it into the runtime vec, so any
    /// `JointRuntime` you obtain from a compiled `JointRuntimes` has
    /// `motor: Some(_)`.
    pub motor: Option<MotorOverrideParams>,
}

// ---------------------------------------------------------------------------
// MotorRuntime
// ---------------------------------------------------------------------------

/// Hot-loop ergonomic alias — when callers only need motor application
/// they can index by slot the same way they index for joint state.
pub type MotorRuntime = JointRuntime;

// ---------------------------------------------------------------------------
// JointRuntimes resource
// ---------------------------------------------------------------------------

/// Bevy resource carrying the compiled dense runtime.
///
/// Empty by default; populated by
/// `clankers_sim::builder::compile_runtime` from
/// [`SceneBuilder::try_build`](https://docs.rs/clankers-sim) at
/// scene-build time. The hot path iterates `self.joints` in slot
/// order, not by entity.
#[derive(Resource, Default, Debug)]
pub struct JointRuntimes {
    /// Dense layout-slot-ordered joint runtime entries.
    pub joints: Vec<JointRuntime>,
    /// Cached entity → slot mapping. Built once at compile time, never
    /// written by the hot path. Used by callers that already have an
    /// `Entity` in hand and need to project it back into the dense
    /// vec.
    pub entity_to_slot: HashMap<Entity, usize>,
}

impl JointRuntimes {
    /// Project a Bevy entity into its layout-slot index, if present.
    #[must_use]
    pub fn slot_for(&self, entity: Entity) -> Option<usize> {
        self.entity_to_slot.get(&entity).copied()
    }

    /// Number of dense runtime entries (actuated joint count).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.joints.len()
    }

    /// Whether the runtime is empty (no entries compiled).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::prelude::Vec3;
    use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};

    fn dummy_info() -> JointInfo {
        JointInfo {
            parent_body: RigidBodyHandle::invalid(),
            child_body: RigidBodyHandle::invalid(),
            axis: Vec3::Z,
            is_prismatic: false,
        }
    }

    fn dummy_motor() -> MotorOverrideParams {
        MotorOverrideParams {
            target_pos: 0.0,
            target_vel: 0.0,
            stiffness: 100.0,
            damping: 10.0,
            max_force: 50.0,
        }
    }

    /// Pinned struct size — drift sentinel. If this fires, audit
    /// `JointRuntime` for an inadvertent type-size change (e.g. a
    /// stray `String` field) that would slow the hot path's cache
    /// behaviour. The exact number is platform-dependent; the
    /// expectation is just that the size stays bounded — pinning
    /// `mem::size_of::<JointRuntime>()` would fail under any padding
    /// regression. We pin a generous upper bound rather than a literal
    /// equality to remain CI-portable across Rust toolchain versions.
    #[test]
    fn joint_runtime_size_pinned() {
        let size = std::mem::size_of::<JointRuntime>();
        // Sanity: should be ≤ 256 bytes on 64-bit. Real value at
        // landing time is ~120-176 depending on `MotorOverrideParams`
        // padding. The bound flags any inadvertent inflation.
        assert!(
            size <= 256,
            "JointRuntime grew unexpectedly: {size} bytes — audit for new fields"
        );
    }

    #[test]
    fn slot_for_returns_layout_position() {
        let mut runtimes = JointRuntimes::default();
        let e0 = Entity::from_bits(10);
        let e1 = Entity::from_bits(11);
        let e2 = Entity::from_bits(12);
        let e3 = Entity::from_bits(13);

        for (slot, entity) in [e0, e1, e2, e3].into_iter().enumerate() {
            runtimes.joints.push(JointRuntime {
                entity,
                handle: ImpulseJointHandle::invalid(),
                layout_slot: slot,
                info: dummy_info(),
                motor: Some(dummy_motor()),
            });
            runtimes.entity_to_slot.insert(entity, slot);
        }

        assert_eq!(runtimes.slot_for(e0), Some(0));
        assert_eq!(runtimes.slot_for(e1), Some(1));
        assert_eq!(runtimes.slot_for(e2), Some(2));
        assert_eq!(runtimes.slot_for(e3), Some(3));
        // Unknown entity → None
        assert_eq!(runtimes.slot_for(Entity::from_bits(99)), None);

        assert_eq!(runtimes.len(), 4);
        assert!(!runtimes.is_empty());
    }

    #[test]
    fn body_runtime_has_clone_and_debug() {
        let b = BodyRuntime {
            entity: Entity::from_bits(1),
            handle: RigidBodyHandle::invalid(),
            layout_slot: 0,
        };
        let _c = b.clone();
        let s = format!("{b:?}");
        assert!(s.contains("BodyRuntime"));
    }

    #[test]
    fn empty_runtime_default_is_clean() {
        let r = JointRuntimes::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.slot_for(Entity::from_bits(1)), None);
    }
}
