//! Buffer-based physics step entry point (P3.2).
//!
//! CODE_QUALITY_REVIEW § Phase 3.2 — the physics backend exposes a
//! `step(commands, &mut state, &mut body_state)` API that consumes a
//! [`JointCommandBuffer`] and produces a [`JointStateBuffer`] /
//! [`BodyStateBuffer`]. The Bevy ECS systems remain in place; this
//! free function is the engine-neutral entry point that future
//! migrations (P3.3) will route the hot loop through.
//!
//! ## Indexing
//!
//! `commands`, `state`, and `runtimes.joints` are all indexed by
//! layout slot. `body_state` is indexed by body slot — the caller
//! decides which bodies it cares about and passes a `body_handles`
//! slice mapping body slot → [`crate::neutral::BodyHandle`].
//!
//! ## NaN / Inf semantics
//!
//! `target_pos[i].is_nan()` skips the position term for joint `i`.
//! `target_vel[i].is_nan()` skips velocity. `max_force[i].is_infinite()`
//! removes the force ceiling.

use rapier3d::prelude::JointAxis;

use crate::buffers::{BodyStateBuffer, JointCommandBuffer, JointStateBuffer};
use crate::neutral::BodyHandle;
use crate::rapier::RapierContext;
use crate::rapier::runtime::JointRuntimes;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failures returned by [`step_with_buffers`].
///
/// Every variant indicates a programmer-side wiring mistake; the buffers
/// must match the runtime they pair with.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StepError {
    /// `commands.len()` did not match `runtimes.joints.len()`.
    #[error(
        "command buffer length {commands} does not match joint runtime count {runtimes}; \
         buffers must be sized to the active layout slot count"
    )]
    CommandLengthMismatch { commands: usize, runtimes: usize },
    /// `state.len()` did not match `runtimes.joints.len()`.
    #[error(
        "joint state buffer length {state} does not match joint runtime count {runtimes}; \
         resize the buffer before stepping"
    )]
    StateLengthMismatch { state: usize, runtimes: usize },
    /// `body_state.len()` did not match `body_handles.len()`.
    #[error(
        "body state buffer length {state} does not match body handle slice length {handles}; \
         resize body_state before stepping"
    )]
    BodyStateLengthMismatch { state: usize, handles: usize },
}

// ---------------------------------------------------------------------------
// step_with_buffers
// ---------------------------------------------------------------------------

/// Step `ctx` by reading commands from `commands`, then writing joint
/// readback into `state` and body readback into `body_state`.
///
/// The Bevy ECS-driven hot path
/// ([`crate::rapier::systems::rapier_step_system`]) remains the default.
/// This entry point is for engine-neutral callers that already own a
/// command buffer and want to skip ECS round-tripping (P3.3 / P3.4).
///
/// ## Behaviour
///
/// 1. For each joint slot `i`, set the motor target according to
///    `commands.target_pos[i]` / `target_vel[i]` / `max_force[i]`.
///    NaN / Inf sentinels follow the [`JointCommandBuffer`] contract.
/// 2. Run `ctx.substeps` physics substeps via `ctx.step()`.
/// 3. For each joint slot, read back position / velocity into `state`
///    (torque is not recomputed here — it's set by the controller).
/// 4. For each `body_handles[i]`, read pose + velocity into
///    `body_state[i]`. Slots whose handle does not resolve get
///    identity values (zeros + unit quat).
///
/// Errors are returned without mutating the buffers.
pub fn step_with_buffers(
    ctx: &mut RapierContext,
    runtimes: &JointRuntimes,
    commands: &JointCommandBuffer,
    state: &mut JointStateBuffer,
    body_handles: &[BodyHandle],
    body_state: &mut BodyStateBuffer,
) -> Result<(), StepError> {
    if commands.len() != runtimes.joints.len() {
        return Err(StepError::CommandLengthMismatch {
            commands: commands.len(),
            runtimes: runtimes.joints.len(),
        });
    }
    if state.len() != runtimes.joints.len() {
        return Err(StepError::StateLengthMismatch {
            state: state.len(),
            runtimes: runtimes.joints.len(),
        });
    }
    if body_state.len() != body_handles.len() {
        return Err(StepError::BodyStateLengthMismatch {
            state: body_state.len(),
            handles: body_handles.len(),
        });
    }

    // 1. Apply commands.
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        let axis = if jr.info.is_prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        };
        let Some(joint) = ctx.impulse_joint_set.get_mut(jr.handle, true) else {
            continue;
        };

        let want_pos = commands.target_pos[slot];
        let want_vel = commands.target_vel[slot];
        let max_force = commands.max_force[slot];

        // Stiffness / damping come from the compile-time motor snapshot;
        // if no motor snapshot exists this is a torque-trick joint and
        // we leave it alone here (the legacy system handles those).
        let Some(motor) = jr.motor.as_ref() else {
            continue;
        };

        let target_pos = if want_pos.is_nan() {
            motor.target_pos
        } else {
            want_pos
        };
        let target_vel = if want_vel.is_nan() {
            motor.target_vel
        } else {
            want_vel
        };
        let force_ceiling = if max_force.is_infinite() {
            motor.max_force
        } else {
            max_force
        };
        joint
            .data
            .set_motor(axis, target_pos, target_vel, motor.stiffness, motor.damping);
        joint.data.set_motor_max_force(axis, force_ceiling);
    }

    // 2. Step physics.
    for _ in 0..ctx.substeps {
        ctx.step();
    }

    // 3. Read joint state back.
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        let info = &jr.info;
        let Some(parent) = ctx.rigid_body_set.get(info.parent_body) else {
            continue;
        };
        let Some(child) = ctx.rigid_body_set.get(info.child_body) else {
            continue;
        };

        if info.is_prismatic {
            let parent_pos = parent.position().translation;
            let child_pos = child.position().translation;
            let rel = child_pos - parent_pos;
            state.position[slot] = rel.x * info.axis.x + rel.y * info.axis.y + rel.z * info.axis.z;
            let rel_v = child.linvel() - parent.linvel();
            state.velocity[slot] =
                rel_v.x * info.axis.x + rel_v.y * info.axis.y + rel_v.z * info.axis.z;
        } else {
            // Project the relative rotation onto the joint axis to get
            // the scalar angle. Matches `rapier_step_system_dense` /
            // `_hashmap` readback at the algebraic level.
            let parent_rot = parent.position().rotation;
            let child_rot = child.position().rotation;
            let rel = parent_rot.inverse() * child_rot;
            let sin_half_x = rel.x;
            let sin_half_y = rel.y;
            let sin_half_z = rel.z;
            let cos_half = rel.w;
            let dot =
                sin_half_x * info.axis.x + sin_half_y * info.axis.y + sin_half_z * info.axis.z;
            state.position[slot] = 2.0 * dot.atan2(cos_half);
            let rel_omega = child.angvel() - parent.angvel();
            state.velocity[slot] =
                rel_omega.x * info.axis.x + rel_omega.y * info.axis.y + rel_omega.z * info.axis.z;
        }
    }

    // 4. Read body state back.
    for (i, &h) in body_handles.iter().enumerate() {
        let Some(rh) = ctx.resolve_object_body(h) else {
            body_state.position_xyz[i] = [0.0; 3];
            body_state.orientation_xyzw[i] = [0.0, 0.0, 0.0, 1.0];
            body_state.linvel_xyz[i] = [0.0; 3];
            body_state.angvel_xyz[i] = [0.0; 3];
            continue;
        };
        let Some(body) = ctx.rigid_body_set.get(rh) else {
            continue;
        };
        let t = body.translation();
        let r = body.rotation();
        let lv = body.linvel();
        let av = body.angvel();
        body_state.position_xyz[i] = [t.x, t.y, t.z];
        body_state.orientation_xyzw[i] = [r.x, r.y, r.z, r.w];
        body_state.linvel_xyz[i] = [lv.x, lv.y, lv.z];
        body_state.angvel_xyz[i] = [av.x, av.y, av.z];
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use clankers_core::config::{ObjectConfig, Shape};

    fn obj(name: &str) -> ObjectConfig {
        ObjectConfig {
            name: name.to_string(),
            shape: Shape::Sphere(0.1),
            position: [0.0, 0.0, 1.0],
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
    fn command_length_mismatch_is_rejected() {
        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 0.01, 1);
        let runtimes = JointRuntimes::default();
        let commands = JointCommandBuffer::with_capacity(2); // mismatched: runtime is empty
        let mut state = JointStateBuffer::with_capacity(0);
        let mut body_state = BodyStateBuffer::with_capacity(0);
        let err = step_with_buffers(
            &mut ctx,
            &runtimes,
            &commands,
            &mut state,
            &[],
            &mut body_state,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StepError::CommandLengthMismatch {
                commands: 2,
                runtimes: 0
            }
        );
    }

    #[test]
    fn state_length_mismatch_is_rejected() {
        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 0.01, 1);
        let runtimes = JointRuntimes::default();
        let commands = JointCommandBuffer::with_capacity(0);
        let mut state = JointStateBuffer::with_capacity(3);
        let mut body_state = BodyStateBuffer::with_capacity(0);
        let err = step_with_buffers(
            &mut ctx,
            &runtimes,
            &commands,
            &mut state,
            &[],
            &mut body_state,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StepError::StateLengthMismatch {
                state: 3,
                runtimes: 0
            }
        );
    }

    #[test]
    fn body_state_mismatch_is_rejected() {
        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 0.01, 1);
        let runtimes = JointRuntimes::default();
        let commands = JointCommandBuffer::with_capacity(0);
        let mut state = JointStateBuffer::with_capacity(0);
        let mut body_state = BodyStateBuffer::with_capacity(2);
        let handles = [BodyHandle::new(0)];
        let err = step_with_buffers(
            &mut ctx,
            &runtimes,
            &commands,
            &mut state,
            &handles,
            &mut body_state,
        )
        .unwrap_err();
        assert_eq!(
            err,
            StepError::BodyStateLengthMismatch {
                state: 2,
                handles: 1
            }
        );
    }

    #[test]
    fn step_falls_under_gravity_writes_pose_into_body_state() {
        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 0.01, 1);
        let h = ctx.add_scene_object(&obj("ball"));
        let runtimes = JointRuntimes::default();
        let commands = JointCommandBuffer::with_capacity(0);
        let mut state = JointStateBuffer::with_capacity(0);
        let handles = [h];
        let mut body_state = BodyStateBuffer::with_capacity(1);

        step_with_buffers(
            &mut ctx,
            &runtimes,
            &commands,
            &mut state,
            &handles,
            &mut body_state,
        )
        .expect("step succeeds");

        // After a single 10ms step under gravity the ball should still
        // be near its starting height but no longer exactly at it.
        let z = body_state.position_xyz[0][2];
        assert!(
            (0.99..=1.0).contains(&z),
            "expected slight downward motion, got z = {z}"
        );
        // Negative linear z velocity from gravity.
        assert!(
            body_state.linvel_xyz[0][2] < 0.0,
            "expected downward linvel, got {:?}",
            body_state.linvel_xyz[0]
        );
    }

    #[test]
    fn unresolved_handle_writes_identity() {
        let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, 0.0), 0.01, 1);
        let runtimes = JointRuntimes::default();
        let commands = JointCommandBuffer::with_capacity(0);
        let mut state = JointStateBuffer::with_capacity(0);
        let bogus = BodyHandle::new(99);
        let handles = [bogus];
        let mut body_state = BodyStateBuffer::with_capacity(1);
        // Pre-populate with garbage to confirm the function overwrites.
        body_state.position_xyz[0] = [7.0, 7.0, 7.0];
        step_with_buffers(
            &mut ctx,
            &runtimes,
            &commands,
            &mut state,
            &handles,
            &mut body_state,
        )
        .unwrap();
        assert_eq!(body_state.position_xyz[0], [0.0, 0.0, 0.0]);
        assert_eq!(body_state.orientation_xyzw[0], [0.0, 0.0, 0.0, 1.0]);
    }
}
