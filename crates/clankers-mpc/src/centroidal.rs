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
//! - Θ̇ = R_z(yaw)^{-1} ω   (Euler angle kinematics, small angle approx)
//! - ṗ = v
//! - ω̇ = I_world^{-1} Σ (r_i × f_i)
//! - v̇ = (1/m) Σ f_i - g e_z
//! - ġ = 0
//!
//! Key convexification assumptions (Di Carlo et al., IROS 2018):
//! 1. Small roll/pitch: Euler rate matrix uses only yaw rotation
//! 2. Yaw-only inertia rotation: I_world = R_yaw I_body R_yaw^T
//! 3. Foot positions treated as parameters, not state

use nalgebra::{DMatrix, Matrix3, Vector3};

use crate::types::STATE_DIM;

/// Build the continuous-time dynamics matrices A_c (13×13) and B_c (13×3n).
///
/// Matches the MIT Cheetah `ct_ss_mats()` formulation:
/// - Inertia is rotated to world frame using yaw-only rotation
/// - Euler angle kinematics use R_z(yaw)^{-1}
///
/// # Arguments
/// * `yaw` - current yaw angle (radians)
/// * `foot_positions` - world-frame position of each foot
/// * `com` - world-frame center of mass position
/// * `body_inertia` - body-frame inertia tensor (NOT pre-inverted)
/// * `mass` - total robot mass
#[allow(clippy::similar_names)]
pub fn build_continuous_dynamics(
    yaw: f64,
    foot_positions: &[Vector3<f64>],
    com: &Vector3<f64>,
    body_inertia: &Matrix3<f64>,
    mass: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_feet = foot_positions.len();
    let n_u = 3 * n_feet;

    let mut a_c = DMatrix::zeros(STATE_DIM, STATE_DIM);
    let mut b_c = DMatrix::zeros(STATE_DIM, n_u);

    // --- Rotate inertia to world frame using yaw-only rotation ---
    // I_world = R_yaw * I_body * R_yaw^T (MIT Cheetah convexification)
    let r_yaw = yaw_rotation_matrix(yaw);
    let i_world = &r_yaw * body_inertia * r_yaw.transpose();
    let i_world_inv = i_world
        .try_inverse()
        .expect("world-frame inertia tensor must be invertible");

    // --- A_c ---

    // Θ̇ = R_z(yaw)^{-1} ω : rows 0-2, cols 6-8
    let (cy, sy) = (yaw.cos(), yaw.sin());
    // R_z(yaw)^{-1} = R_z(-yaw)
    a_c[(0, 6)] = cy;
    a_c[(0, 7)] = sy;
    a_c[(1, 6)] = -sy;
    a_c[(1, 7)] = cy;
    a_c[(2, 8)] = 1.0;

    // ṗ = v : rows 3-5, cols 9-11
    a_c[(3, 9)] = 1.0;
    a_c[(4, 10)] = 1.0;
    a_c[(5, 11)] = 1.0;

    // v̇ gravity term: row 11, col 12 = -1 (v_z_dot includes -g)
    a_c[(11, 12)] = -1.0;

    // --- B_c ---

    let inv_mass = 1.0 / mass;

    for (i, foot_pos) in foot_positions.iter().enumerate() {
        let r = foot_pos - com; // foot position relative to CoM
        let r_cross = skew_symmetric(&r);

        // ω̇ = I_world^{-1} (r × f) : rows 6-8
        let i_inv_r_cross = &i_world_inv * r_cross;
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

/// Discretize using the matrix exponential (exact for LTI systems).
///
/// Uses the augmented matrix approach from MIT Cheetah `c2qp()`:
/// ```text
/// [A_d  B_d] = expm(dt * [A_c  B_c])
/// [ 0    I ]             [ 0    0 ]
/// ```
///
/// This is more accurate than forward Euler, especially for the angular
/// dynamics coupling in the A matrix.
pub fn discretize_matrix_exp(
    a_c: &DMatrix<f64>,
    b_c: &DMatrix<f64>,
    dt: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_x = a_c.nrows();
    let n_u = b_c.ncols();
    let n_aug = n_x + n_u;

    // Build augmented matrix [A_c  B_c; 0  0] * dt
    let mut aug = DMatrix::zeros(n_aug, n_aug);
    aug.view_mut((0, 0), (n_x, n_x)).copy_from(a_c);
    aug.view_mut((0, n_x), (n_x, n_u)).copy_from(b_c);
    aug *= dt;

    let exp_aug = matrix_exp(&aug);

    let a_d = exp_aug.view((0, 0), (n_x, n_x)).clone_owned();
    let b_d = exp_aug.view((0, n_x), (n_x, n_u)).clone_owned();

    (a_d, b_d)
}

/// Discretize using first-order Euler (ZOH approximation).
///
/// A_d = I + A_c dt,  B_d = B_c dt
///
/// Kept for testing/comparison. Prefer `discretize_matrix_exp` for production.
pub fn discretize_euler(
    a_c: &DMatrix<f64>,
    b_c: &DMatrix<f64>,
    dt: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n = a_c.nrows();
    let a_d = DMatrix::identity(n, n) + a_c * dt;
    let b_d = b_c * dt;
    (a_d, b_d)
}

/// Compute the yaw-only rotation matrix R_z(yaw).
///
/// Used for rotating inertia to world frame (convexification).
fn yaw_rotation_matrix(yaw: f64) -> Matrix3<f64> {
    let (cy, sy) = (yaw.cos(), yaw.sin());
    Matrix3::new(cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0)
}

/// Compute the matrix exponential e^M using scaling-and-squaring with
/// Taylor series.
///
/// For matrices with ||M|| < 1 (typical for dt * dynamics), converges
/// in ~8 terms to machine precision.
fn matrix_exp(m: &DMatrix<f64>) -> DMatrix<f64> {
    let n = m.nrows();

    // Scaling: find s such that ||M/2^s||_inf < 1
    let norm_inf = m
        .row_iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);

    let s = if norm_inf > 1.0 {
        (norm_inf.log2().ceil() as u32).max(1)
    } else {
        0
    };

    let m_scaled = if s > 0 {
        m / f64::from(2u32.pow(s))
    } else {
        m.clone()
    };

    // Taylor series: e^M = I + M + M^2/2! + ... + M^N/N!
    let mut result = DMatrix::identity(n, n);
    let mut term = DMatrix::identity(n, n);

    for k in 1..=13 {
        term = &term * &m_scaled / (k as f64);
        result += &term;
        // Early exit when terms become negligible
        let term_norm = term
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        if term_norm < 1e-16 {
            break;
        }
    }

    // Squaring: e^M = (e^(M/2^s))^(2^s)
    for _ in 0..s {
        result = &result * &result;
    }

    result
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
        0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_inertia() -> Matrix3<f64> {
        Matrix3::new(
            0.07, 0.0, 0.0, 0.0, 0.26, 0.0, 0.0, 0.0, 0.242,
        )
    }

    fn quadruped_feet() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.15, 0.08, 0.0),
            Vector3::new(0.15, -0.08, 0.0),
            Vector3::new(-0.15, 0.08, 0.0),
            Vector3::new(-0.15, -0.08, 0.0),
        ]
    }

    #[test]
    fn skew_symmetric_properties() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = skew_symmetric(&v);

        assert_relative_eq!(s, -s.transpose(), epsilon = 1e-12);

        let w = Vector3::new(4.0, 5.0, 6.0);
        let cross = s * w;
        let expected = v.cross(&w);
        assert_relative_eq!(cross, expected, epsilon = 1e-12);
    }

    #[test]
    fn a_matrix_structure_at_zero_yaw() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia(), 9.0);

        assert_eq!(a_c.nrows(), 13);
        assert_eq!(a_c.ncols(), 13);
        assert_eq!(b_c.nrows(), 13);
        assert_eq!(b_c.ncols(), 12);

        // At yaw=0, R_z^{-1} = I_3
        assert_relative_eq!(a_c[(0, 6)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 7)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(2, 8)], 1.0, epsilon = 1e-12);

        // ṗ = v
        assert_relative_eq!(a_c[(3, 9)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(4, 10)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a_c[(5, 11)], 1.0, epsilon = 1e-12);

        // Gravity
        assert_relative_eq!(a_c[(11, 12)], -1.0, epsilon = 1e-12);

        // Last row all zeros
        for j in 0..13 {
            assert_relative_eq!(a_c[(12, j)], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn b_matrix_linear_force() {
        let feet = vec![Vector3::new(0.0, 0.0, 0.0)];
        let com = Vector3::new(0.0, 0.0, 0.35);
        let mass = 10.0;
        let (_, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia(), mass);

        assert_relative_eq!(b_c[(9, 0)], 0.1, epsilon = 1e-12);
        assert_relative_eq!(b_c[(10, 1)], 0.1, epsilon = 1e-12);
        assert_relative_eq!(b_c[(11, 2)], 0.1, epsilon = 1e-12);
    }

    #[test]
    fn b_matrix_torque_from_force() {
        let feet = vec![Vector3::new(1.0, 0.0, 0.0)];
        let com = Vector3::zeros();
        let i_body = Matrix3::identity();
        let (_, b_c) = build_continuous_dynamics(0.0, &feet, &com, &i_body, 10.0);

        // r=(1,0,0), f_z: torque = (1,0,0)×(0,0,1) = (0,-1,0)
        assert_relative_eq!(b_c[(7, 2)], -1.0, epsilon = 1e-12);
        assert_relative_eq!(b_c[(6, 2)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn yaw_rotation_in_a_matrix() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let yaw = std::f64::consts::FRAC_PI_4;
        let (a_c, _) = build_continuous_dynamics(yaw, &feet, &com, &test_inertia(), 9.0);

        let c = yaw.cos();
        let s = yaw.sin();
        assert_relative_eq!(a_c[(0, 6)], c, epsilon = 1e-12);
        assert_relative_eq!(a_c[(0, 7)], s, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 6)], -s, epsilon = 1e-12);
        assert_relative_eq!(a_c[(1, 7)], c, epsilon = 1e-12);
    }

    #[test]
    fn inertia_rotation_at_nonzero_yaw() {
        // At yaw=pi/2, I_world should swap Ixx and Iyy for diagonal inertia
        let i_body = Matrix3::new(1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0);
        let feet = vec![Vector3::new(1.0, 0.0, 0.0)];
        let com = Vector3::zeros();

        let (_, b_0) = build_continuous_dynamics(0.0, &feet, &com, &i_body, 10.0);
        let (_, b_90) =
            build_continuous_dynamics(std::f64::consts::FRAC_PI_2, &feet, &com, &i_body, 10.0);

        // At yaw=0: I_world = diag(1,4,9), I^{-1} = diag(1, 1/4, 1/9)
        // At yaw=90: I_world = diag(4,1,9), I^{-1} = diag(1/4, 1, 1/9)
        // With r=(1,0,0), [r]_x = [0,0,0; 0,0,-1; 0,1,0]
        // B[7,2] = I^{-1}[1,1] * (-1): at yaw=0 → -0.25, at yaw=90 → -1.0
        assert!(
            (b_0[(7, 2)] - b_90[(7, 2)]).abs() > 0.1,
            "Inertia rotation should affect angular dynamics: b0={}, b90={}",
            b_0[(7, 2)],
            b_90[(7, 2)]
        );
    }

    #[test]
    fn matrix_exp_identity_for_zero() {
        let zero = DMatrix::zeros(5, 5);
        let result = matrix_exp(&zero);
        let identity = DMatrix::identity(5, 5);
        assert_relative_eq!(result, identity, epsilon = 1e-14);
    }

    #[test]
    fn matrix_exp_scalar_case() {
        // For a 1x1 matrix [a], exp([a]) = [e^a]
        let mut m = DMatrix::zeros(1, 1);
        m[(0, 0)] = 1.0;
        let result = matrix_exp(&m);
        assert_relative_eq!(result[(0, 0)], std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn matrix_exp_discretization_vs_euler() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia(), 9.0);

        let dt = 0.02;
        let (a_euler, b_euler) = discretize_euler(&a_c, &b_c, dt);
        let (a_exp, b_exp) = discretize_matrix_exp(&a_c, &b_c, dt);

        // For small dt, both should be close but not identical
        assert_relative_eq!(a_euler, a_exp, epsilon = 0.01);
        assert_relative_eq!(b_euler, b_exp, epsilon = 0.001);

        // Matrix exp should preserve the structure better
        // A_d should still have the gravity row as [0,...,0,1]
        assert_relative_eq!(a_exp[(12, 12)], 1.0, epsilon = 1e-12);
        for j in 0..12 {
            assert_relative_eq!(a_exp[(12, j)], 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn free_fall_state_propagation() {
        use nalgebra::DVector;

        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia(), 9.0);
        let dt = 0.01;
        let (a_d, _b_d) = discretize_matrix_exp(&a_c, &b_c, dt);

        let mut x = DVector::zeros(13);
        x[5] = 0.35; // p_z
        x[12] = 9.81; // gravity

        // Propagate 10 steps (0.1s of free fall)
        for _ in 0..10 {
            x = &a_d * &x;
        }

        // v_z ≈ -g*t = -9.81*0.1 = -0.981
        // p_z ≈ 0.35 - 0.5*g*t² = 0.35 - 0.049 = 0.301
        assert_relative_eq!(x[11], -0.981, epsilon = 0.01);
        assert_relative_eq!(x[5], 0.301, epsilon = 0.005);
    }

    #[test]
    fn discretize_euler_identity_at_zero_dt() {
        let feet = quadruped_feet();
        let com = Vector3::new(0.0, 0.0, 0.35);
        let (a_c, b_c) = build_continuous_dynamics(0.0, &feet, &com, &test_inertia(), 9.0);

        let (a_d, b_d) = discretize_euler(&a_c, &b_c, 0.0);

        let identity = DMatrix::identity(13, 13);
        assert_relative_eq!(a_d, identity, epsilon = 1e-12);

        let zeros = DMatrix::zeros(13, 12);
        assert_relative_eq!(b_d, zeros, epsilon = 1e-12);
    }
}
