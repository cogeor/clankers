//! Rapier physics step system.

use std::collections::HashMap;

use bevy::prelude::*;
use rapier3d::prelude::JointAxis;

use clankers_actuator::components::{JointState, JointTorque};

use super::context::RapierContext;

/// Position motor parameters for a single joint.
///
/// When a joint entity has an entry in [`MotorOverrides`], the physics step
/// system uses these PD motor parameters (evaluated at physics rate by Rapier)
/// instead of the torque motor trick (constant torque ZOH).
#[derive(Clone, Debug)]
pub struct MotorOverrideParams {
    /// Target joint position (radians for revolute, meters for prismatic).
    pub target_pos: f32,
    /// Target joint velocity (also encodes feedforward: `target_vel = ff_torque / damping`).
    pub target_vel: f32,
    /// Position gain (spring stiffness).
    pub stiffness: f32,
    /// Velocity gain (damping).
    pub damping: f32,
    /// Maximum motor force/torque.
    pub max_force: f32,
}

/// Per-joint position motor overrides.
///
/// Joints listed here bypass the torque motor trick in [`rapier_step_system`]
/// and instead use Rapier's built-in PD motor evaluated at the physics rate.
/// This is essential for stiff PD gains on light links where ZOH torque
/// control at the frame rate would cause oscillation.
#[derive(Resource, Default)]
pub struct MotorOverrides {
    /// Map from joint entity to position motor parameters.
    pub joints: HashMap<Entity, MotorOverrideParams>,
}

/// Per-joint motor command rate limiting.
///
/// Clamps the change in motor target position between consecutive control
/// steps: `target = clamp(target, prev - delta_max, prev + delta_max)`.
/// Applied at the actuator output, NOT inside the MPC QP.
#[derive(Resource)]
pub struct MotorRateLimits {
    /// Maximum position change per control step (radians for revolute).
    pub delta_max: f32,
    /// Previous target positions, keyed by entity.
    pub prev_targets: HashMap<Entity, f32>,
}

impl MotorRateLimits {
    /// Create rate limits with the given maximum position delta per step.
    pub fn new(delta_max: f32) -> Self {
        Self {
            delta_max,
            prev_targets: HashMap::new(),
        }
    }
}

/// High-frequency inner PD interpolation state.
///
/// When this resource is present, motor target positions are linearly
/// interpolated across physics substeps instead of being set once (ZOH).
/// This provides effective 1000Hz PD control while the MPC runs at 50Hz.
#[derive(Resource, Default)]
pub struct InnerPdState {
    /// Previous control step's target positions per entity.
    prev_targets: HashMap<Entity, f32>,
}

/// Apply joint torques, step physics, read back joint state.
///
/// When [`InnerPdState`] is present, motor target positions are linearly
/// interpolated across substeps for effective 1000Hz PD control.
#[allow(clippy::needless_pass_by_value)]
pub fn rapier_step_system(
    mut context: ResMut<RapierContext>,
    mut joints: Query<(Entity, &JointTorque, &mut JointState)>,
    motor_overrides: Option<Res<MotorOverrides>>,
    mut rate_limits: Option<ResMut<MotorRateLimits>>,
    mut inner_pd: Option<ResMut<InnerPdState>>,
) {
    let substeps = context.substeps;
    let use_inner_pd = inner_pd.is_some() && motor_overrides.is_some();

    // Collect override data for interpolation (needed if inner PD is active)
    struct OverrideEntry {
        joint_handle: rapier3d::dynamics::ImpulseJointHandle,
        axis: JointAxis,
        prev_pos: f32,
        target_pos: f32,
        target_vel: f32,
        stiffness: f32,
        damping: f32,
        max_force: f32,
    }

    let mut override_entries: Vec<OverrideEntry> = Vec::new();

    // 1. Apply torques to rapier joints via motor trick (or position motor override)
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

        if let Some(joint) = context.impulse_joint_set.get_mut(joint_handle, true) {
            // Check for position motor override first
            if let Some(ref overrides) = motor_overrides
                && let Some(mo) = overrides.joints.get(&entity)
            {
                // Apply rate limiting if configured
                let target_pos = if let Some(ref mut limits) = rate_limits {
                    let prev = limits.prev_targets.get(&entity).copied().unwrap_or(mo.target_pos);
                    let clamped = mo.target_pos.clamp(
                        prev - limits.delta_max,
                        prev + limits.delta_max,
                    );
                    limits.prev_targets.insert(entity, clamped);
                    clamped
                } else {
                    mo.target_pos
                };

                if use_inner_pd {
                    // Store for substep interpolation
                    let pd = inner_pd.as_mut().unwrap();
                    let prev_pos = pd.prev_targets.get(&entity).copied().unwrap_or(target_pos);
                    pd.prev_targets.insert(entity, target_pos);
                    override_entries.push(OverrideEntry {
                        joint_handle,
                        axis,
                        prev_pos,
                        target_pos,
                        target_vel: mo.target_vel,
                        stiffness: mo.stiffness,
                        damping: mo.damping,
                        max_force: mo.max_force,
                    });
                    // Set initial interpolated target (substep 0)
                    let alpha = 1.0 / substeps as f32;
                    let interp = prev_pos + (target_pos - prev_pos) * alpha;
                    joint.data.set_motor(axis, interp, mo.target_vel, mo.stiffness, mo.damping);
                    joint.data.set_motor_max_force(axis, mo.max_force);
                } else {
                    joint.data.set_motor(axis, target_pos, mo.target_vel, mo.stiffness, mo.damping);
                    joint.data.set_motor_max_force(axis, mo.max_force);
                }
            } else {
                // Motor trick: ForceBased motor with huge target velocity,
                // clamped to desired torque magnitude.
                let t = torque.value;
                if t.abs() > 1e-10 {
                    let target_vel = t.signum() * 1e10;
                    joint.data.set_motor(axis, 0.0, target_vel, 0.0, 1.0);
                    joint.data.set_motor_max_force(axis, t.abs());
                } else {
                    joint.data.set_motor(axis, 0.0, 0.0, 0.0, 0.0);
                    joint.data.set_motor_max_force(axis, 0.0);
                }
            }
        }
    }

    // 2. Step physics with inner PD interpolation
    if use_inner_pd && !override_entries.is_empty() {
        // First substep already has interpolated target set above
        context.step();

        // Remaining substeps: update interpolated targets
        for sub in 1..substeps {
            let alpha = (sub + 1) as f32 / substeps as f32;
            for entry in &override_entries {
                if let Some(joint) = context.impulse_joint_set.get_mut(entry.joint_handle, true) {
                    let interp = entry.prev_pos + (entry.target_pos - entry.prev_pos) * alpha;
                    joint.data.set_motor(entry.axis, interp, entry.target_vel, entry.stiffness, entry.damping);
                    joint.data.set_motor_max_force(entry.axis, entry.max_force);
                }
            }
            context.step();
        }
    } else {
        for _ in 0..substeps {
            context.step();
        }
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
