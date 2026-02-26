//! Whole Body Controller: maps foot forces to joint torques.
//!
//! For stance legs, uses Jacobian transpose: τ = J_c^T F
//! For swing legs, computes desired joint positions from foot trajectories.

use nalgebra::{DMatrix, UnitQuaternion, Vector3};

/// Per-leg control output from the WBC.
#[derive(Clone, Debug)]
pub struct LegCommand {
    /// Joint torques (stance) or positions (swing) for this leg.
    pub values: Vec<f64>,
    /// Whether this leg is in stance (torque mode) or swing (position mode).
    pub is_stance: bool,
}

/// Compute stance leg damping torques.
///
/// Adds joint-space velocity damping alongside the MPC force feedforward.
/// This stabilizes stance legs against vibrations. MIT Cheetah uses
/// `kd_joint = 0.2` for all stance joints.
///
/// `joint_velocities`: current joint velocities in rad/s
/// `kd_joint`: damping gain per joint (scalar, applied uniformly)
///
/// Returns damping torques: `tau_damp = -kd_joint * qdot`
pub fn stance_damping_torques(joint_velocities: &[f64], kd_joint: f64) -> Vec<f64> {
    joint_velocities
        .iter()
        .map(|&qd| -kd_joint * qd)
        .collect()
}

/// Compute joint torques for a stance leg using Jacobian transpose.
///
/// τ = J_foot^T F_foot
///
/// `jacobian` is a 3×n matrix where n is the number of joints in the leg.
/// `force` is the 3D ground reaction force at the foot.
///
/// Returns n joint torques.
pub fn jacobian_transpose_torques(
    jacobian: &DMatrix<f64>,
    force: &Vector3<f64>,
) -> Vec<f64> {
    let n_joints = jacobian.ncols();
    let mut torques = vec![0.0; n_joints];

    // τ = J^T F
    for j in 0..n_joints {
        torques[j] = jacobian[(0, j)] * force.x
            + jacobian[(1, j)] * force.y
            + jacobian[(2, j)] * force.z;
    }
    torques
}

/// Compute the linear (translational) Jacobian for a leg chain.
///
/// Given joint origins, axes, and end-effector position (all in world frame),
/// computes the 3×n Jacobian mapping joint velocities to foot linear velocity.
///
/// For revolute joints: J_col = axis × (ee_pos - joint_origin)
/// For prismatic joints: J_col = axis
pub fn compute_leg_jacobian(
    origins: &[Vector3<f64>],
    axes: &[Vector3<f64>],
    ee_pos: &Vector3<f64>,
    is_prismatic: &[bool],
) -> DMatrix<f64> {
    let n = origins.len();
    let mut j = DMatrix::zeros(3, n);

    for i in 0..n {
        if is_prismatic[i] {
            j[(0, i)] = axes[i].x;
            j[(1, i)] = axes[i].y;
            j[(2, i)] = axes[i].z;
        } else {
            let r = ee_pos - origins[i];
            let cross = axes[i].cross(&r);
            j[(0, i)] = cross.x;
            j[(1, i)] = cross.y;
            j[(2, i)] = cross.z;
        }
    }
    j
}

/// Transform body-frame joint origins and axes to world frame.
///
/// `body_rotation` is a unit quaternion representing body orientation.
/// `body_translation` is the body position in world frame.
pub fn transform_frames_to_world(
    origins_body: &[Vector3<f64>],
    axes_body: &[Vector3<f64>],
    body_rotation: &UnitQuaternion<f64>,
    body_translation: &Vector3<f64>,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>) {
    let origins_world = origins_body
        .iter()
        .map(|o| body_rotation * o + body_translation)
        .collect();
    let axes_world = axes_body.iter().map(|a| body_rotation * a).collect();
    (origins_world, axes_world)
}

/// Convert f32 joint frames from the IK chain to f64 for WBC computation.
pub fn frames_f32_to_f64(
    origins: &[Vector3<f32>],
    axes: &[Vector3<f32>],
    ee_pos: &Vector3<f32>,
) -> (Vec<Vector3<f64>>, Vec<Vector3<f64>>, Vector3<f64>) {
    let origins_f64 = origins.iter().map(|v| v.cast::<f64>()).collect();
    let axes_f64 = axes.iter().map(|v| v.cast::<f64>()).collect();
    let ee_f64 = ee_pos.cast::<f64>();
    (origins_f64, axes_f64, ee_f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn jacobian_transpose_simple() {
        // Simple 2-joint planar leg in XZ plane
        // Joint 0 at origin, axis Y, upper leg length 0.15
        // Joint 1 at (0,0,-0.15), axis Y, lower leg length 0.15
        // At q=0, foot at (0,0,-0.3)

        let origins = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -0.15),
        ];
        let axes = vec![
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let ee_pos = Vector3::new(0.0, 0.0, -0.3);
        let is_prismatic = vec![false, false];

        let j = compute_leg_jacobian(&origins, &axes, &ee_pos, &is_prismatic);
        assert_eq!(j.nrows(), 3);
        assert_eq!(j.ncols(), 2);

        // J[0, :] = axis × (ee - origin) for each joint
        // Joint 0: (0,1,0) × (0,0,-0.3) = (1*(-0.3)-0*0, 0*0-0*(-0.3), 0*0-1*0) = (-0.3, 0, 0)
        // Wait: y_hat × (0,0,-0.3) = |i  j  k |
        //                             |0  1  0 |
        //                             |0  0 -0.3|
        // = i(1*(-0.3) - 0*0) - j(0*(-0.3) - 0*0) + k(0*0 - 1*0)
        // = (-0.3, 0, 0)
        assert_relative_eq!(j[(0, 0)], -0.3, epsilon = 1e-10);
        assert_relative_eq!(j[(2, 0)], 0.0, epsilon = 1e-10);

        // Joint 1: (0,1,0) × (0,0,-0.15) = (-0.15, 0, 0)
        assert_relative_eq!(j[(0, 1)], -0.15, epsilon = 1e-10);

        // Apply vertical force: F = (0, 0, mg)
        let force = Vector3::new(0.0, 0.0, 50.0);
        let torques = jacobian_transpose_torques(&j, &force);

        // τ = J^T F
        // τ[0] = J[0,0]*0 + J[1,0]*0 + J[2,0]*50 = 0*50 = 0
        // τ[1] = J[0,1]*0 + J[1,1]*0 + J[2,1]*50 = 0*50 = 0
        // With straight legs, vertical force produces zero torque (expected!)
        assert_relative_eq!(torques[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(torques[1], 0.0, epsilon = 1e-10);

        // Now apply horizontal force (push forward):
        let force_x = Vector3::new(10.0, 0.0, 0.0);
        let torques_x = jacobian_transpose_torques(&j, &force_x);

        // τ[0] = J[0,0]*10 = -0.3 * 10 = -3.0
        // τ[1] = J[0,1]*10 = -0.15 * 10 = -1.5
        assert_relative_eq!(torques_x[0], -3.0, epsilon = 1e-10);
        assert_relative_eq!(torques_x[1], -1.5, epsilon = 1e-10);
    }

    #[test]
    fn jacobian_with_bent_leg() {
        // Leg bent at knee: hip at origin, knee at (0.1, 0, -0.11), foot at (0.15, 0, -0.24)
        let origins = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.1, 0.0, -0.11),
        ];
        let axes = vec![
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let ee_pos = Vector3::new(0.15, 0.0, -0.24);
        let is_prismatic = vec![false, false];

        let j = compute_leg_jacobian(&origins, &axes, &ee_pos, &is_prismatic);

        // Vertical force through foot should now produce nonzero torques
        // because the foot is offset horizontally from the joints
        let force = Vector3::new(0.0, 0.0, 50.0);
        let torques = jacobian_transpose_torques(&j, &force);

        // Both torques should be nonzero (foot is not directly below joints)
        assert!(torques[0].abs() > 1.0, "Hip torque should be nonzero");
        assert!(torques[1].abs() > 1.0, "Knee torque should be nonzero");
    }

    #[test]
    fn transform_frames_identity() {
        let origins = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.5, 0.0, -0.1)];
        let axes = vec![Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        let identity = nalgebra::UnitQuaternion::identity();
        let translation = Vector3::new(10.0, 20.0, 30.0);

        let (origins_w, axes_w) = super::transform_frames_to_world(&origins, &axes, &identity, &translation);

        // Origins should be translated but not rotated
        assert_relative_eq!(origins_w[0].x, 11.0, epsilon = 1e-10);
        assert_relative_eq!(origins_w[0].y, 22.0, epsilon = 1e-10);
        assert_relative_eq!(origins_w[0].z, 33.0, epsilon = 1e-10);
        assert_relative_eq!(origins_w[1].x, 10.5, epsilon = 1e-10);

        // Axes should be unchanged (identity rotation, no translation)
        assert_relative_eq!(axes_w[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(axes_w[0].y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(axes_w[0].z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn transform_frames_90deg_yaw() {
        // 90-degree rotation about Z axis: X -> Y, Y -> -X
        let rot = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f64::consts::FRAC_PI_2,
        );
        let translation = Vector3::zeros();

        let origins = vec![Vector3::new(1.0, 0.0, 0.0)];
        let axes = vec![Vector3::new(0.0, 1.0, 0.0)];

        let (origins_w, axes_w) = super::transform_frames_to_world(&origins, &axes, &rot, &translation);

        // (1,0,0) rotated 90 about Z -> (0,1,0)
        assert_relative_eq!(origins_w[0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(origins_w[0].y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(origins_w[0].z, 0.0, epsilon = 1e-10);

        // Y-axis rotated 90 about Z -> (-1,0,0)
        assert_relative_eq!(axes_w[0].x, -1.0, epsilon = 1e-10);
        assert_relative_eq!(axes_w[0].y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(axes_w[0].z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn jacobian_frame_consistency() {
        // Verify that transforming body-frame Jacobian inputs to world frame
        // produces physically correct torques.
        //
        // Setup: straight 2-joint leg along -Z, body rotated 90 deg about Z.
        let origins_body = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -0.15),
        ];
        let axes_body = vec![
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let ee_body = Vector3::new(0.0, 0.0, -0.3);
        let is_prismatic = vec![false, false];

        let body_rot = nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f64::consts::FRAC_PI_2,
        );
        let body_pos = Vector3::new(0.0, 0.0, 0.35);

        // Transform to world frame
        let (origins_w, axes_w) =
            super::transform_frames_to_world(&origins_body, &axes_body, &body_rot, &body_pos);
        let ee_world = body_rot * ee_body + body_pos;

        let j = compute_leg_jacobian(&origins_w, &axes_w, &ee_world, &is_prismatic);

        // Vertical support force at the foot
        let force = Vector3::new(0.0, 0.0, 50.0);
        let torques = jacobian_transpose_torques(&j, &force);

        // Straight leg directly below body: vertical force should produce
        // near-zero torques regardless of body yaw.
        assert_relative_eq!(torques[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(torques[1], 0.0, epsilon = 1e-8);
    }

    #[test]
    fn frames_conversion() {
        let origins_f32 = vec![Vector3::new(1.0f32, 2.0, 3.0)];
        let axes_f32 = vec![Vector3::new(0.0f32, 1.0, 0.0)];
        let ee_f32 = Vector3::new(4.0f32, 5.0, 6.0);

        let (origins, axes, ee) = frames_f32_to_f64(&origins_f32, &axes_f32, &ee_f32);
        assert_relative_eq!(origins[0].x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(axes[0].y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(ee.z, 6.0, epsilon = 1e-10);
    }
}
