//! Buffer ↔ ECS mirroring at the physics boundary (P3.3 / P3.4).
//!
//! `CODE_QUALITY_REVIEW` § Phase 3.3 — "mirror buffers to ECS only at
//! boundaries"; § Phase 3.4 — "sensors / protocol / recorder read
//! buffer views". These helpers convert between the `SoA` buffers from
//! [`crate::buffers`] and the per-entity ECS components that
//! observation buffers / sensors / recorders read.
//!
//! ## Mirroring direction
//!
//! - [`mirror_joint_state_to_ecs`] — for each layout slot, hand
//!   `(entity, position, velocity)` to a caller-supplied closure that
//!   writes into the ECS `JointState` component. Called once at the
//!   end of each physics step.
//! - [`snapshot_joint_torque_from_ecs`] — for each layout slot, call
//!   a caller-supplied closure that yields the live torque from ECS,
//!   then store the result in `buffer.torque[slot]`. Source-of-truth
//!   stays in ECS; this is a snapshot for downstream consumers.
//!
//! ## Indexing
//!
//! Both helpers walk `runtimes.joints` in layout-slot order so the
//! buffer slot index matches everywhere downstream (sensors, protocol
//! framing, MCAP recorder, telemetry).
//!
//! ## Why closure-based
//!
//! Bevy `Query<&mut T>` is not `'static`, which makes it awkward to
//! call from generic helper code. Threading the per-entity access
//! through a closure keeps these helpers pure (no Bevy dep in the
//! signature), testable without a `World`, and equally callable from
//! a Bevy system that owns the query.

use bevy::prelude::Entity;

use crate::buffers::JointStateBuffer;
use crate::rapier::runtime::JointRuntimes;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures returned by the mirroring helpers.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MirrorError {
    /// Buffer length didn't match the runtime's joint count.
    #[error("buffer length {buffer} does not match joint runtime count {runtime}")]
    LengthMismatch { buffer: usize, runtime: usize },
}

// ---------------------------------------------------------------------------
// mirror_joint_state_to_ecs
// ---------------------------------------------------------------------------

/// Publish slot-indexed `JointStateBuffer` values to ECS at the end of
/// the physics step.
///
/// For each layout slot the helper invokes `write(entity, position,
/// velocity)`. Bevy systems pass a closure that calls `query.get_mut`;
/// tests pass a closure that records into a `HashMap`. P3.3 — the
/// boundary is here; the source-of-truth stays in the buffer.
pub fn mirror_joint_state_to_ecs(
    runtimes: &JointRuntimes,
    buffer: &JointStateBuffer,
    mut write: impl FnMut(Entity, f32, f32),
) -> Result<(), MirrorError> {
    if buffer.len() != runtimes.joints.len() {
        return Err(MirrorError::LengthMismatch {
            buffer: buffer.len(),
            runtime: runtimes.joints.len(),
        });
    }
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        write(jr.entity, buffer.position[slot], buffer.velocity[slot]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// snapshot_joint_torque_from_ecs
// ---------------------------------------------------------------------------

/// Take a slot-indexed snapshot of joint torques from ECS into
/// `buffer.torque`.
///
/// For each layout slot the helper invokes `read(entity) -> Option<f32>`.
/// `None` means "no torque component on that entity"; the buffer slot
/// is left unchanged in that case. P3.4 — downstream consumers
/// (recorder, protocol, sensors) read `buffer.torque[slot]` instead of
/// probing ECS per joint per step.
pub fn snapshot_joint_torque_from_ecs(
    runtimes: &JointRuntimes,
    buffer: &mut JointStateBuffer,
    mut read: impl FnMut(Entity) -> Option<f32>,
) -> Result<(), MirrorError> {
    if buffer.len() != runtimes.joints.len() {
        return Err(MirrorError::LengthMismatch {
            buffer: buffer.len(),
            runtime: runtimes.joints.len(),
        });
    }
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        if let Some(t) = read(jr.entity) {
            buffer.torque[slot] = t;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use bevy::prelude::*;
    use rapier3d::prelude::{ImpulseJointHandle, RigidBodyHandle};

    use crate::rapier::context::JointInfo;
    use crate::rapier::runtime::JointRuntime;
    use crate::rapier::systems::MotorOverrideParams;

    fn jr(entity: Entity, slot: usize) -> JointRuntime {
        JointRuntime {
            entity,
            handle: ImpulseJointHandle::invalid(),
            layout_slot: slot,
            info: JointInfo {
                parent_body: RigidBodyHandle::invalid(),
                child_body: RigidBodyHandle::invalid(),
                axis: Vec3::Z,
                is_prismatic: false,
            },
            motor: Some(MotorOverrideParams {
                target_pos: 0.0,
                target_vel: 0.0,
                stiffness: 100.0,
                damping: 10.0,
                max_force: 50.0,
            }),
        }
    }

    #[test]
    fn mirror_writes_slot_values_through_closure() {
        let e0 = Entity::from_bits(100);
        let e1 = Entity::from_bits(101);
        let mut runtimes = JointRuntimes::default();
        runtimes.joints.push(jr(e0, 0));
        runtimes.joints.push(jr(e1, 1));

        let mut buffer = JointStateBuffer::with_capacity(2);
        buffer.position[0] = 1.5;
        buffer.position[1] = -0.25;
        buffer.velocity[0] = 0.1;
        buffer.velocity[1] = -0.2;

        let mut sink: HashMap<Entity, (f32, f32)> = HashMap::new();
        mirror_joint_state_to_ecs(&runtimes, &buffer, |e, p, v| {
            sink.insert(e, (p, v));
        })
        .unwrap();

        assert_eq!(sink.len(), 2);
        assert_eq!(sink[&e0], (1.5, 0.1));
        assert_eq!(sink[&e1], (-0.25, -0.2));
    }

    #[test]
    fn mirror_length_mismatch_is_rejected() {
        let mut runtimes = JointRuntimes::default();
        runtimes.joints.push(jr(Entity::from_bits(1), 0));
        let buffer = JointStateBuffer::with_capacity(3);
        let err = mirror_joint_state_to_ecs(&runtimes, &buffer, |_, _, _| {}).unwrap_err();
        assert_eq!(
            err,
            MirrorError::LengthMismatch {
                buffer: 3,
                runtime: 1
            }
        );
    }

    #[test]
    fn snapshot_torque_reads_through_closure() {
        let e0 = Entity::from_bits(200);
        let e1 = Entity::from_bits(201);
        let mut runtimes = JointRuntimes::default();
        runtimes.joints.push(jr(e0, 0));
        runtimes.joints.push(jr(e1, 1));

        let mut buffer = JointStateBuffer::with_capacity(2);
        buffer.torque[0] = 999.0;
        buffer.torque[1] = 999.0;

        let source: HashMap<Entity, f32> = HashMap::from([(e0, 7.5), (e1, -3.0)]);
        snapshot_joint_torque_from_ecs(&runtimes, &mut buffer, |e| source.get(&e).copied())
            .unwrap();
        assert!((buffer.torque[0] - 7.5).abs() < 1e-6);
        assert!((buffer.torque[1] + 3.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_torque_leaves_missing_entities_unchanged() {
        let e0 = Entity::from_bits(300);
        let mut runtimes = JointRuntimes::default();
        runtimes.joints.push(jr(e0, 0));
        let mut buffer = JointStateBuffer::with_capacity(1);
        buffer.torque[0] = 42.0;
        // Closure always returns None — entity not present.
        snapshot_joint_torque_from_ecs(&runtimes, &mut buffer, |_| None).unwrap();
        assert!((buffer.torque[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn snapshot_length_mismatch_is_rejected() {
        let runtimes = JointRuntimes::default();
        let mut buffer = JointStateBuffer::with_capacity(2);
        let err =
            snapshot_joint_torque_from_ecs(&runtimes, &mut buffer, |_| Some(0.0)).unwrap_err();
        assert_eq!(
            err,
            MirrorError::LengthMismatch {
                buffer: 2,
                runtime: 0
            }
        );
    }
}
