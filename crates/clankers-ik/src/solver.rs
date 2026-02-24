//! Damped Least Squares (Levenberg-Marquardt) IK solver.
//!
//! Iteratively solves for joint positions that place the end-effector
//! at a target pose, using the geometric Jacobian and DLS pseudoinverse.

use nalgebra::{DMatrix, DVector, Isometry3, UnitQuaternion, Vector3};

use crate::chain::KinematicChain;

/// What the solver should target.
#[derive(Debug, Clone)]
pub enum IkTarget {
    /// Target position only (3-DOF constraint).
    Position(Vector3<f32>),
    /// Target full pose: position + orientation (6-DOF constraint).
    Pose(Isometry3<f32>),
}

/// Configuration for the DLS solver.
#[derive(Debug, Clone)]
pub struct DlsConfig {
    /// Maximum solver iterations.
    pub max_iterations: u32,
    /// Position error tolerance (meters).
    pub position_tolerance: f32,
    /// Orientation error tolerance (radians).
    pub angle_tolerance: f32,
    /// Damping factor (lambda). Higher = more robust near singularities,
    /// but slower convergence.
    pub damping: f32,
}

impl Default for DlsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            position_tolerance: 1e-4,
            angle_tolerance: 1e-3,
            damping: 0.01,
        }
    }
}

/// Result of an IK solve.
#[derive(Debug, Clone)]
pub struct IkResult {
    /// Solved joint positions.
    pub joint_positions: Vec<f32>,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations used.
    pub iterations: u32,
    /// Final position error (meters).
    pub position_error: f32,
    /// Final orientation error (radians). Zero if target is position-only.
    pub orientation_error: f32,
}

/// Damped Least Squares IK solver.
pub struct DlsSolver {
    config: DlsConfig,
}

impl DlsSolver {
    /// Create a new solver with the given configuration.
    pub const fn new(config: DlsConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DlsConfig::default())
    }

    /// Solve IK for the given chain and target.
    ///
    /// `q_init` is the starting joint configuration (warm-start from current state).
    pub fn solve(
        &self,
        chain: &KinematicChain,
        target: &IkTarget,
        q_init: &[f32],
    ) -> IkResult {
        assert_eq!(q_init.len(), chain.dof());

        let mut q: Vec<f32> = q_init.to_vec();
        let n = chain.dof();

        for iteration in 0..self.config.max_iterations {
            let ee_pose = chain.forward_kinematics(&q);
            let (pos_err, ori_err, error_vec) = compute_error(&ee_pose, target);

            // Check convergence
            let converged = match target {
                IkTarget::Position(_) => pos_err < self.config.position_tolerance,
                IkTarget::Pose(_) => {
                    pos_err < self.config.position_tolerance
                        && ori_err < self.config.angle_tolerance
                }
            };

            if converged {
                return IkResult {
                    joint_positions: q,
                    converged: true,
                    iterations: iteration,
                    position_error: pos_err,
                    orientation_error: ori_err,
                };
            }

            // Compute Jacobian
            let jacobian = compute_jacobian(chain, &q, target);
            let m = jacobian.nrows();

            // DLS: dq = J^T (J J^T + lambda^2 I)^{-1} * error
            let jjt = &jacobian * jacobian.transpose();
            let damped = jjt + DMatrix::identity(m, m) * (self.config.damping * self.config.damping);
            let Some(damped_inv) = damped.try_inverse() else {
                // Matrix is singular even with damping — give up
                return IkResult {
                    joint_positions: q,
                    converged: false,
                    iterations: iteration,
                    position_error: pos_err,
                    orientation_error: ori_err,
                };
            };

            let dq = jacobian.transpose() * damped_inv * error_vec;

            // Update joint positions
            for i in 0..n {
                q[i] += dq[i];
            }

            // Clamp to joint limits
            chain.clamp_joints(&mut q);
        }

        // Didn't converge
        let ee_pose = chain.forward_kinematics(&q);
        let (pos_err, ori_err, _) = compute_error(&ee_pose, target);

        IkResult {
            joint_positions: q,
            converged: false,
            iterations: self.config.max_iterations,
            position_error: pos_err,
            orientation_error: ori_err,
        }
    }
}

/// Compute the error vector between current EE pose and target.
///
/// Returns (position_error_norm, orientation_error_norm, error_vector).
fn compute_error(
    ee_pose: &Isometry3<f32>,
    target: &IkTarget,
) -> (f32, f32, DVector<f32>) {
    match target {
        IkTarget::Position(target_pos) => {
            let pos_err = target_pos - ee_pose.translation.vector;
            let pos_err_norm = pos_err.norm();
            let error = DVector::from_column_slice(&[pos_err.x, pos_err.y, pos_err.z]);
            (pos_err_norm, 0.0, error)
        }
        IkTarget::Pose(target_pose) => {
            let pos_err = target_pose.translation.vector - ee_pose.translation.vector;
            let pos_err_norm = pos_err.norm();

            // Orientation error as axis-angle
            let rot_err = target_pose.rotation * ee_pose.rotation.inverse();
            let ori_err_vec = orientation_error(&rot_err);
            let ori_err_norm = ori_err_vec.norm();

            let error = DVector::from_column_slice(&[
                pos_err.x, pos_err.y, pos_err.z,
                ori_err_vec.x, ori_err_vec.y, ori_err_vec.z,
            ]);
            (pos_err_norm, ori_err_norm, error)
        }
    }
}

/// Extract orientation error as a 3-vector (axis * angle) from a unit quaternion.
fn orientation_error(q: &UnitQuaternion<f32>) -> Vector3<f32> {
    if let Some(axis) = q.axis() {
        axis.into_inner() * q.angle()
    } else {
        Vector3::zeros()
    }
}

/// Compute the geometric Jacobian for the current configuration.
///
/// For position-only targets, returns a 3xN matrix.
/// For full-pose targets, returns a 6xN matrix (linear + angular rows).
fn compute_jacobian(
    chain: &KinematicChain,
    q: &[f32],
    target: &IkTarget,
) -> DMatrix<f32> {
    let n = chain.dof();
    let (origins, axes, ee_pos) = chain.joint_frames(q);

    let rows = match target {
        IkTarget::Position(_) => 3,
        IkTarget::Pose(_) => 6,
    };

    let mut jacobian = DMatrix::zeros(rows, n);

    for i in 0..n {
        let joint = &chain.joints()[i];
        let z_i = &axes[i]; // joint axis in base frame
        let o_i = &origins[i]; // joint origin in base frame

        if joint.is_prismatic {
            // Linear velocity: z_i
            jacobian[(0, i)] = z_i.x;
            jacobian[(1, i)] = z_i.y;
            jacobian[(2, i)] = z_i.z;
            // Angular velocity: 0 (prismatic doesn't rotate)
            // rows 3-5 stay zero
        } else {
            // Revolute joint
            // Linear velocity: z_i x (ee_pos - o_i)
            let r = ee_pos - o_i;
            let cross = z_i.cross(&r);
            jacobian[(0, i)] = cross.x;
            jacobian[(1, i)] = cross.y;
            jacobian[(2, i)] = cross.z;

            if rows == 6 {
                // Angular velocity: z_i
                jacobian[(3, i)] = z_i.x;
                jacobian[(4, i)] = z_i.y;
                jacobian[(5, i)] = z_i.z;
            }
        }
    }

    jacobian
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use clankers_urdf::parse_string;

    const TWO_LINK_ARM: &str = r#"
        <robot name="two_link_arm">
            <link name="base"><inertial><mass value="10.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial></link>
            <link name="upper_arm"><inertial><mass value="2.0"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.002"/></inertial></link>
            <link name="forearm"><inertial><mass value="1.0"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/></inertial></link>
            <link name="end_effector"><inertial><mass value="0.1"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial></link>
            <joint name="shoulder" type="revolute">
                <parent link="base"/><child link="upper_arm"/>
                <origin xyz="0 0 0.05" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2.617" upper="2.617" effort="50" velocity="3"/>
            </joint>
            <joint name="elbow" type="revolute">
                <parent link="upper_arm"/><child link="forearm"/>
                <origin xyz="0 0 0.3" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2.094" upper="2.094" effort="30" velocity="5"/>
            </joint>
            <joint name="ee_fixed" type="fixed">
                <parent link="forearm"/><child link="end_effector"/>
                <origin xyz="0 0 0.25"/>
            </joint>
        </robot>
    "#;

    const SIX_DOF_ARM: &str = r#"
        <robot name="six_dof_arm">
            <link name="base"><inertial><mass value="20.0"/><inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/></inertial></link>
            <link name="shoulder_link"><inertial><mass value="3.0"/><inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/></inertial></link>
            <link name="upper_arm"><inertial><mass value="2.5"/><inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.003"/></inertial></link>
            <link name="elbow_link"><inertial><mass value="1.5"/><inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.002"/></inertial></link>
            <link name="forearm"><inertial><mass value="1.0"/><inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.001"/></inertial></link>
            <link name="wrist_link"><inertial><mass value="0.5"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0005"/></inertial></link>
            <link name="end_effector"><inertial><mass value="0.2"/><inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/></inertial></link>
            <joint name="j1_base_yaw" type="revolute">
                <parent link="base"/><child link="shoulder_link"/>
                <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="80" velocity="2"/>
            </joint>
            <joint name="j2_shoulder_pitch" type="revolute">
                <parent link="shoulder_link"/><child link="upper_arm"/>
                <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
                <limit lower="-1.5708" upper="2.356" effort="60" velocity="2"/>
            </joint>
            <joint name="j3_elbow_pitch" type="revolute">
                <parent link="upper_arm"/><child link="elbow_link"/>
                <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
                <limit lower="-2.356" upper="2.356" effort="40" velocity="3"/>
            </joint>
            <joint name="j4_forearm_roll" type="revolute">
                <parent link="elbow_link"/><child link="forearm"/>
                <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="20" velocity="5"/>
            </joint>
            <joint name="j5_wrist_pitch" type="revolute">
                <parent link="forearm"/><child link="wrist_link"/>
                <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
                <limit lower="-2.094" upper="2.094" effort="10" velocity="5"/>
            </joint>
            <joint name="j6_wrist_roll" type="revolute">
                <parent link="wrist_link"/><child link="end_effector"/>
                <origin xyz="0 0 0.06"/><axis xyz="0 0 1"/>
                <limit lower="-3.14159" upper="3.14159" effort="5" velocity="8"/>
            </joint>
        </robot>
    "#;

    #[test]
    fn ik_roundtrip_two_link() {
        // FK at known angles, then IK to recover them
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let q_target = [0.3, -0.5];
        let ee_target = chain.forward_kinematics(&q_target);
        let target = IkTarget::Position(ee_target.translation.vector);

        let solver = DlsSolver::with_defaults();
        let result = solver.solve(&chain, &target, &[0.0, 0.0]);

        assert!(result.converged, "IK did not converge: pos_err={}", result.position_error);
        assert!(result.position_error < 1e-3);

        // Verify FK at solved angles matches target
        let ee_solved = chain.forward_kinematics(&result.joint_positions);
        assert_relative_eq!(
            ee_solved.translation.x, ee_target.translation.x, epsilon = 1e-3
        );
        assert_relative_eq!(
            ee_solved.translation.y, ee_target.translation.y, epsilon = 1e-3
        );
        assert_relative_eq!(
            ee_solved.translation.z, ee_target.translation.z, epsilon = 1e-3
        );
    }

    #[test]
    fn ik_six_dof_position() {
        let model = parse_string(SIX_DOF_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        // Target: reach forward
        let target = IkTarget::Position(Vector3::new(0.3, 0.0, 0.5));
        let solver = DlsSolver::with_defaults();
        let result = solver.solve(&chain, &target, &[0.0; 6]);

        assert!(result.converged, "IK did not converge: pos_err={}", result.position_error);

        let ee = chain.forward_kinematics(&result.joint_positions);
        assert_relative_eq!(ee.translation.x, 0.3, epsilon = 1e-3);
        assert_relative_eq!(ee.translation.y, 0.0, epsilon = 1e-3);
        assert_relative_eq!(ee.translation.z, 0.5, epsilon = 1e-3);
    }

    #[test]
    fn ik_six_dof_full_pose_roundtrip() {
        let model = parse_string(SIX_DOF_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let q_target = [0.5, 0.3, -0.4, 0.2, 0.1, -0.3];
        let ee_target = chain.forward_kinematics(&q_target);
        let target = IkTarget::Pose(ee_target);

        let solver = DlsSolver::new(DlsConfig {
            max_iterations: 200,
            ..DlsConfig::default()
        });
        let result = solver.solve(&chain, &target, &[0.0; 6]);

        assert!(result.converged, "IK did not converge: pos_err={}, ori_err={}",
            result.position_error, result.orientation_error);
        assert!(result.position_error < 1e-3);
        assert!(result.orientation_error < 1e-2);
    }

    #[test]
    fn ik_unreachable_target() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        // Target way outside workspace (arm reach is ~0.55m from base)
        let target = IkTarget::Position(Vector3::new(5.0, 5.0, 5.0));
        let solver = DlsSolver::new(DlsConfig {
            max_iterations: 50,
            ..DlsConfig::default()
        });
        let result = solver.solve(&chain, &target, &[0.0, 0.0]);

        // Should NOT converge but should not panic
        assert!(!result.converged);
        assert!(result.position_error > 1.0);
    }

    #[test]
    fn ik_warm_start() {
        let model = parse_string(SIX_DOF_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let target = IkTarget::Position(Vector3::new(0.2, 0.1, 0.6));
        let solver = DlsSolver::with_defaults();

        // Cold start
        let cold = solver.solve(&chain, &target, &[0.0; 6]);
        assert!(cold.converged);

        // Warm start from cold solution — should converge faster
        let warm = solver.solve(&chain, &target, &cold.joint_positions);
        assert!(warm.converged);
        assert!(warm.iterations <= cold.iterations);
    }

    #[test]
    fn ik_respects_joint_limits() {
        let model = parse_string(TWO_LINK_ARM).unwrap();
        let chain = KinematicChain::from_model(&model, "end_effector").unwrap();

        let target = IkTarget::Position(Vector3::new(0.0, 0.0, 0.4));
        let solver = DlsSolver::with_defaults();
        let result = solver.solve(&chain, &target, &[0.0, 0.0]);

        // All solved positions should be within limits
        for (i, &q) in result.joint_positions.iter().enumerate() {
            let joint = &chain.joints()[i];
            assert!(
                q >= joint.lower_limit - 1e-6 && q <= joint.upper_limit + 1e-6,
                "Joint {} ({}) out of limits: {} not in [{}, {}]",
                i, joint.name, q, joint.lower_limit, joint.upper_limit
            );
        }
    }
}
