//! Rapier physics step system.

use bevy::prelude::*;
use rapier3d::prelude::JointAxis;

use clankers_actuator::components::{JointState, JointTorque};

use super::context::RapierContext;

/// Apply joint torques, step physics, read back joint state.
#[allow(clippy::needless_pass_by_value)]
pub fn rapier_step_system(
    mut context: ResMut<RapierContext>,
    mut joints: Query<(Entity, &JointTorque, &mut JointState)>,
) {
    // 1. Apply torques to rapier joints via motor trick
    for (entity, torque, _) in &joints {
        let Some(&joint_handle) = context.joint_handles.get(&entity) else {
            continue;
        };
        let Some(info) = context.joint_info.get(&entity) else {
            continue;
        };

        let axis = if info.is_prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        };

        let t = torque.value;
        if let Some(joint) = context.impulse_joint_set.get_mut(joint_handle, true) {
            // Motor trick: ForceBased motor with huge target velocity,
            // clamped to desired torque magnitude.
            let target_vel = if t.abs() < 1e-10 { 0.0 } else { t.signum() * 1e10 };
            joint.data.set_motor(axis, target_vel, 0.0, 0.0, 1.0);
            joint.data.set_motor_max_force(axis, t.abs());
        }
    }

    // 2. Step physics (substeps)
    let substeps = context.substeps;
    for _ in 0..substeps {
        context.step();
    }

    // 3. Read back joint state from rigid body transforms
    for (entity, _, mut state) in &mut joints {
        let Some(info) = context.joint_info.get(&entity) else {
            continue;
        };

        let Some(parent_body) = context.rigid_body_set.get(info.parent_body) else {
            continue;
        };
        let Some(child_body) = context.rigid_body_set.get(info.child_body) else {
            continue;
        };

        if info.is_prismatic {
            // Prismatic: displacement along joint axis
            let parent_pos = parent_body.position().translation;
            let child_pos = child_body.position().translation;
            let relative_pos = child_pos - parent_pos;
            state.position = relative_pos.dot(info.axis);

            // Velocity along axis
            let relative_vel = child_body.linvel() - parent_body.linvel();
            state.velocity = relative_vel.dot(info.axis);
        } else {
            // Revolute: rotation around joint axis
            let parent_rot = parent_body.position().rotation;
            let child_rot = child_body.position().rotation;
            let relative_rotation = parent_rot.inverse() * child_rot;

            // Extract angle around joint axis from quaternion
            let sin_half = Vec3::new(
                relative_rotation.x,
                relative_rotation.y,
                relative_rotation.z,
            );
            let cos_half = relative_rotation.w;
            let sin_half_proj = sin_half.dot(info.axis);
            state.position = 2.0 * f32::atan2(sin_half_proj, cos_half);

            // Angular velocity around axis
            let relative_angvel = child_body.angvel() - parent_body.angvel();
            state.velocity = relative_angvel.dot(info.axis);
        }
    }
}
