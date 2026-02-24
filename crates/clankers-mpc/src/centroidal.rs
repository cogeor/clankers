//! Centroidal dynamics model for the MPC.
//!
//! Treats the robot as a single floating rigid body with point-contact feet.
//! The state vector is 13-dimensional:
//!
//! ```text
//! x = [Θ_rpy(3), p_xyz(3), ω_xyz(3), v_xyz(3), g(1)]
//! ```
//!
//! The continuous dynamics are:
//! - Θ̇ = R_z(yaw)^{-1} ω   (Euler angle kinematics)
//! - ṗ = v
//! - ω̇ = I^{-1} Σ (r_i × f_i)
//! - v̇ = (1/m) Σ f_i - g e_z
//! - ġ = 0
//!
//! where r_i = foot_i_pos - CoM, f_i is the ground reaction force at foot i,
//! and the gravity term is absorbed into the 13th state.

use nalgebra::{DMatrix, Matrix3, Vector3};

use crate::types::STATE_DIM;

/// Build the continuous-time dynamics matrices A_c (13×13) and B_c (13×3n).
///
/// `yaw`: current yaw angle (radians) for the Euler angle kinematics.
/// `foot_positions`: world-frame position of each foot.
/// `com`: world-frame center of mass position.
/// `inertia_inv`: inverse of the body-frame inertia tensor.
/// `mass`: total robot mass.
#[allow(clippy::similar_names)]
pub fn build_continuous_dynamics(
    yaw: f64,
    foot_positions: &[Vector3<f64>],
    com: &Vector3<f64>,
    inertia_inv: &Matrix3<f64>,
    mass: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_feet = foot_positions.len();
    let n_u = 3 * n_feet;

    let mut a_c = DMatrix::zeros(STATE_DIM, STATE_DIM);
    let mut b_c = DMatrix::zeros(STATE_DIM, n_u);

    // --- A_c ---

    // Θ̇ = R_z(yaw)^{-1} ω : rows 0-2, cols 6-8
    let (cy, sy) = (yaw.cos(), yaw.sin());
    // R_z(yaw)^{-1} = R_z(-yaw) = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
    a_c[(0, 6)] = cy;
    a_c[(0, 7)] = sy;
    a_c[(1, 6)] = -sy;
    a_c[(1, 7)] = cy;
    a_c[(2, 8)] = 1.0;

    // ṗ = v : rows 3-5, cols 9-11
    a_c[(3, 9)] = 1.0;
    a_c[(4, 10)] = 1.0;
    a_c[(5, 11)] = 1.0;

    // v̇ gravity term via constant state: row 11, col 12
    // v_z_dot includes -g contribution: A_c[11, 12] = -1
    a_c[(11, 12)] = -1.0;

    // --- B_c ---

    let inv_mass = 1.0 / mass;

    for (i, foot_pos) in foot_positions.iter().enumerate() {
        let r = foot_pos - com; // foot position relative to CoM
        let r_cross = skew_symmetric(&r);

        // ω̇ = I^{-1} (r × f) : rows 6-8
        let i_inv_r_cross = inertia_inv * r_cross;
        for row in 0..3 {
            for col in 0..3 {
                b_c[(6 + row, 3 * i + col)] = i_inv_r_cross[(row, col)];
            }
        }

        // v̇ = (1/m) f : rows 9-11
        b_c[(9, 3 * i)] = inv_mass;
        b_c[(10, 3 * i + 1)] = inv_mass;
        b_c[(11, 3 * i + 2)] = inv_mass;
    }

    (a_c, b_c)
}

/// Discretize continuous dynamics using first-order Euler (ZOH approximation).
///
/// A_d = I + A_c dt,  B_d = B_c dt
///
/// This is accurate for small dt (< 0.05s), which is always the case for MPC.
pub fn discretize(
    a_c: &DMatrix<f64>,
    b_c: &DMatrix<f64>,
    dt: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a_c.nrows();
    let a_d = DMatrix::identity(n, n) + a_c * dt;
    let b_d = b_c * dt;
    (a_d, b_d)
}

/// Compute the skew-symmetric (cross product) matrix of a 3D vector.
///
/// ```text
/// [v]_× = [ 0   -vz   vy ]
///         [ vz   0   -vx ]
///         [-vy   vx   0  ]
/// ```
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_inertia() -> Matrix3<f64> {
        Matrix3::new(
            0.07, 0.0, 0.0,
            0.0, 0.26, 0.0,
            0.0, 0.0, 0.28,
        )
    }

    fn test_inertia_inv() -> Matrix3<f64> {
        test_inertia().try_inverse().unwrap()
    }

    fn quadruped_feet() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.15, 0.08, 0.0),   // FL
            Vector3::new(0.15, -0.08, 0.0),  // FR
            Vector3::new(-0.15, 0.08, 0.0),  // RL
            Vector3::new(-0.15, -0.08, 0.0), // RR
        ]
    }

    #[test]
    fn skew_symmetric_properties() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = skew_symmetric(&v);

        // Skew-symmetric: S^T = -S
        assert_relative_eq!(s, -s.transpose(), epsilon = 1e-12);

        // Cross product: s * w = v × w
        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross = s * w;
        let expected = v.cross(&w);
        assert_relative_eq!(cross, expected, epsilon = 1e-12);
    }

    #[test]
    fn a_matrix_dimensions_and_structure() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia_inv(), 8.6);

        assert_eq!(a_c.nrows(), 13);
        assert_eq!(a_c.ncols(), 13);
        assert_eq!(b_c.nrows(), 13);
        assert_eq!(b_c.ncols(), 12); // 4 feet × 3 forces

        // At yaw=0, R_z^{-1} = I_3, so A_c[0:3, 6:9] should be identity
        assert_relative_eq!(a_c[(0, 6)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 7)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(2, 8)], 1.0, epsilon = 1e-12);

        // ṗ = v block: A_c[3:6, 9:12] = I_3
        assert_relative_eq!(a_c[(3, 9)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(4, 10)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(5, 11)], 1.0, epsilon = 1e-12);

        // Gravity: A_c[11, 12] = -1
        assert_relative_eq!(a_c[(11, 12)], -1.0, epsilon = 1e-12);

        // Last row is all zeros (constant state)
        for j in 0..13 {
            assert_relative_eq!(a_c[(12, j)], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn b_matrix_linear_force() {
        let feet = vec![Vector3::new(0.0, 0.0, 0.0)];
        let com = Vector3::new(0.0, 0.0, 0.35);
        let mass = 10.0;
        let (_, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia_inv(), mass);

        // Linear acceleration rows (9-11): (1/m) I_3
        assert_relative_eq!(b_c[(9, 0)], 0.1, epsilon = 1e-12);
        assert_relative_eq!(b_c[(10, 1)], 0.1, epsilon = 1e-12);
        assert_relative_eq!(b_c[(11, 2)], 0.1, epsilon = 1e-12);
    }

    #[test]
    fn b_matrix_torque_from_force() {
        // Foot at (1, 0, 0), CoM at origin → r = (1, 0, 0)
        // Force in z → torque = r × f = (1,0,0) × (0,0,fz) = (0, -fz, 0)
        let feet = vec![Vector3::new(1.0, 0.0, 0.0)];
        let com = Vector3::zeros();
        // Use identity inertia for simplicity
        let i_inv = Matrix3::identity();
        let (_, b_c) = build_continuous_dynamics(0.0, &feet, &com, &i_inv, 10.0);

        // ω̇ = I^{-1} (r × f), with I=I_3 and r=(1,0,0):
        // fz force (col 2): torque = (1,0,0)×(0,0,1) = (0,-1,0)
        // So B_c[7, 2] = -1.0 (ω_y from fz)
        assert_relative_eq!(b_c[(7, 2)], -1.0, epsilon = 1e-12);
        // B_c[6, 2] = 0.0 (ω_x from fz)
        assert_relative_eq!(b_c[(6, 2)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn yaw_rotation_in_a_matrix() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let yaw = std::f64::consts::FRAC_PI_4; // 45 degrees
        let (a_c, _) = build_continuous_dynamics(yaw, &feet, &com, &test_inertia_inv(), 8.6);

        let c = yaw.cos();
        let s = yaw.sin();
        assert_relative_eq!(a_c[(0, 6)], c, epsilon = 1e-12);
        assert_relative_eq!(a_c[(0, 7)], s, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 6)], -s, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 7)], c, epsilon = 1e-12);
    }

    #[test]
    fn discretize_identity_at_zero_dt() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia_inv(), 8.6);

        let (a_d, b_d) = discretize(&a_c, &b_c, 0.0);

        // A_d should be identity when dt=0
        let identity = DMatrix::identity(13, 13);
        assert_relative_eq!(a_d, identity, epsilon = 1e-12);

        // B_d should be zero when dt=0
        let zeros = DMatrix::zeros(13, 12);
        assert_relative_eq!(b_d, zeros, epsilon = 1e-12);
    }

    #[test]
    fn discretize_gravity_propagation() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia_inv(), 8.6);

        let dt = 0.02;
        let (a_d, _) = discretize(&a_c, &b_c, dt);

        // With zero forces, state propagation should show gravity:
        // v_z_{k+1} = v_z_k + A_d[11,12] * g
        // A_d[11,12] = 0 + (-1) * dt = -0.02
        assert_relative_eq!(a_d[(11, 12)], -dt, epsilon = 1e-12);

        // So if g=9.81 and vz=0: vz_next = 0 + (-0.02)*9.81 = -0.1962
        // This is correct: free-fall velocity after 0.02s
    }

    #[test]
    fn free_fall_state_propagation() {
        use nalgebra::DVector;

        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia_inv(), 8.6);
        let dt = 0.01;
        let (a_d, _b_d) = discretize(&a_c, &b_c, dt);

        // Initial state: standing still at height 0.35, g=9.81
        let mut x = DVector::zeros(13);
        x[5] = 0.35; // p_z
        x[12] = 9.81; // gravity

        // Zero forces (free fall) — propagation only uses A_d
        let _u: DVector<f64> = DVector::zeros(12);

        // Propagate 10 steps (0.1s of free fall)
        for _ in 0..10 {
            x = &a_d * &x;
        }

        // After 0.1s free fall:
        // v_z ≈ -g*t = -9.81*0.1 = -0.981
        // p_z ≈ 0.35 - 0.5*g*t² = 0.35 - 0.049 = 0.301
        assert_relative_eq!(x[11], -0.981, epsilon = 0.01); // v_z
        assert_relative_eq!(x[5], 0.301, epsilon = 0.005); // p_z
    }
}
