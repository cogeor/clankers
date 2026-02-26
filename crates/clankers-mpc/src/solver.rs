//! MPC solver: condensed QP formulation for centroidal dynamics.
//!
//! Uses the MIT Cheetah condensed (single-shooting) approach where state
//! variables are eliminated, leaving only forces as decision variables:
//!
//! ```text
//! X = A_qp x_0 + B_qp U          (state prediction)
//! min  (1/2) U^T H U + g^T U     (condensed QP)
//! H = 2 (B_qp^T S B_qp + α I)
//! g = 2 B_qp^T S (A_qp x_0 - X_ref)
//! ```
//!
//! Subject to linearized friction pyramid constraints per stance foot
//! and zero-force equality constraints for swing feet.

use std::time::Instant;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT::{NonnegativeConeT, ZeroConeT},
};
use nalgebra::{DMatrix, DVector, Vector3};

use crate::centroidal::{build_continuous_dynamics, discretize_matrix_exp};
use crate::types::{MpcConfig, MpcSolution, ReferenceTrajectory, STATE_DIM};

/// Centroidal MPC solver using condensed QP formulation.
pub struct MpcSolver {
    config: MpcConfig,
}

impl MpcSolver {
    /// Create a new MPC solver with the given configuration.
    pub const fn new(config: MpcConfig) -> Self {
        Self { config }
    }

    /// Access the solver configuration.
    pub const fn config(&self) -> &MpcConfig {
        &self.config
    }

    /// Solve the MPC problem for optimal foot forces.
    ///
    /// Uses the condensed (single-shooting) formulation from MIT Cheetah:
    /// decision variables are only forces U (12*H), states are eliminated.
    ///
    /// # Arguments
    /// * `x0` - Current 13D state vector
    /// * `foot_positions` - World-frame foot positions (one per foot)
    /// * `contacts` - Contact flags: `contacts[step][foot]`, size horizon × n_feet
    /// * `reference` - Desired state trajectory over the horizon
    pub fn solve(
        &self,
        x0: &DVector<f64>,
        foot_positions: &[Vector3<f64>],
        contacts: &[Vec<bool>],
        reference: &ReferenceTrajectory,
    ) -> MpcSolution {
        let start = Instant::now();

        let h = self.config.horizon;
        let n_feet = foot_positions.len();
        let n_u_step = 3 * n_feet;
        let n_u_total = n_u_step * h;

        // 1. Build dynamics matrices at current linearization point
        let yaw = x0[2];
        let com = Vector3::new(x0[3], x0[4], x0[5]);
        let (a_c, b_c) = build_continuous_dynamics(
            yaw,
            foot_positions,
            &com,
            &self.config.inertia,
            self.config.mass,
        );
        let (a_d, b_d) = discretize_matrix_exp(&a_c, &b_c, self.config.dt);

        // 2. Build condensed prediction matrices
        //    A_qp = [A_d; A_d^2; ...; A_d^H]   (13H × 13)
        //    B_qp = lower block-triangular       (13H × 12H)
        let (a_qp, b_qp) = build_prediction_matrices(&a_d, &b_d, h);

        // 3. Build QP cost: H = 2(B_qp^T S B_qp + alpha I), g = 2 B_qp^T S (A_qp x0 - Xref)
        let (p_mat, q_vec) =
            self.build_condensed_cost(&a_qp, &b_qp, x0, reference, h, n_u_total);

        // 4. Build constraints (friction cone + swing zero-force)
        let (a_con, b_con, n_eq, n_ineq) =
            self.build_constraints(contacts, h, n_feet, n_u_total);

        // 5. Convert to Clarabel CSC format
        let p_csc = dmatrix_to_csc_upper_tri(&p_mat);
        let a_csc = dmatrix_to_csc(&a_con);

        // 6. Define cones: equalities first, then inequalities
        let cones = vec![ZeroConeT(n_eq), NonnegativeConeT(n_ineq)];

        // 7. Solve
        let settings = DefaultSettingsBuilder::default()
            .max_iter(self.config.max_solver_iters)
            .verbose(false)
            .tol_gap_abs(1e-6)
            .tol_gap_rel(1e-6)
            .tol_feas(1e-6)
            .build()
            .expect("valid solver settings");

        let q_slice: Vec<f64> = q_vec.iter().copied().collect();
        let b_slice: Vec<f64> = b_con.iter().copied().collect();

        let solver_result =
            DefaultSolver::new(&p_csc, &q_slice, &a_csc, &b_slice, &cones, settings);

        let converged;
        let mut forces = vec![Vector3::zeros(); n_feet];
        let mut force_trajectory = DVector::zeros(n_u_total);
        let mut state_trajectory = DVector::zeros(STATE_DIM * h);

        match solver_result {
            Ok(mut solver) => {
                solver.solve();
                let sol = &solver.solution;

                converged = matches!(
                    sol.status,
                    SolverStatus::Solved | SolverStatus::AlmostSolved
                );

                if converged {
                    // Solution is U = [f_0, f_1, ..., f_{H-1}]
                    for i in 0..n_u_total {
                        force_trajectory[i] = sol.x[i];
                    }

                    // Extract first-step foot forces
                    for (foot, force) in forces.iter_mut().enumerate() {
                        *force = Vector3::new(
                            sol.x[3 * foot],
                            sol.x[3 * foot + 1],
                            sol.x[3 * foot + 2],
                        );
                    }

                    // Reconstruct state trajectory: X = A_qp x0 + B_qp U
                    let u_vec = DVector::from_column_slice(&sol.x[..n_u_total]);
                    state_trajectory = &a_qp * x0 + &b_qp * u_vec;
                }
            }
            Err(_) => {
                converged = false;
            }
        }

        let elapsed = start.elapsed();

        MpcSolution {
            forces,
            force_trajectory,
            state_trajectory,
            converged,
            solve_time_us: u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        }
    }

    /// Build the condensed QP cost matrices.
    ///
    /// P = 2 (B_qp^T S B_qp + alpha I)
    /// q = 2 B_qp^T S (A_qp x0 - X_ref)
    fn build_condensed_cost(
        &self,
        a_qp: &DMatrix<f64>,
        b_qp: &DMatrix<f64>,
        x0: &DVector<f64>,
        reference: &ReferenceTrajectory,
        h: usize,
        n_u_total: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let n_x_total = STATE_DIM * h;

        // Build diagonal weight vector for S (block-diagonal Q repeated H times)
        let mut s_diag = DVector::zeros(n_x_total);
        for k in 0..h {
            let off = k * STATE_DIM;
            for i in 0..12 {
                s_diag[off + i] = self.config.q_weights[i];
            }
            // s_diag[off + 12] = 0.0; // gravity state: zero weight
        }

        // Compute S * B_qp efficiently (scale each row by diagonal weight)
        let mut sb = b_qp.clone();
        for i in 0..n_x_total {
            let w = s_diag[i];
            if w.abs() > 1e-20 {
                for j in 0..n_u_total {
                    sb[(i, j)] *= w;
                }
            } else {
                for j in 0..n_u_total {
                    sb[(i, j)] = 0.0;
                }
            }
        }

        // P = 2 * (B_qp^T * S * B_qp + alpha * I)
        let btsb = b_qp.transpose() * &sb;
        let p_mat = 2.0 * (btsb + DMatrix::identity(n_u_total, n_u_total) * self.config.r_weight);

        // q = 2 * B_qp^T * S * (A_qp * x0 - X_ref)
        let error = a_qp * x0 - &reference.states;
        let mut s_error = error;
        for i in 0..n_x_total {
            s_error[i] *= s_diag[i];
        }
        let q_vec = 2.0 * b_qp.transpose() * s_error;

        (p_mat, q_vec)
    }

    /// Build constraint matrices for friction cone and swing feet.
    ///
    /// Equalities (ZeroCone) first, then inequalities (NonnegativeCone).
    fn build_constraints(
        &self,
        contacts: &[Vec<bool>],
        h: usize,
        n_feet: usize,
        n_u_total: usize,
    ) -> (DMatrix<f64>, DVector<f64>, usize, usize) {
        let n_u_step = 3 * n_feet;

        // Count constraints
        let mut n_swing_eq = 0;
        let mut n_friction_ineq = 0;

        for step_contacts in contacts.iter().take(h) {
            for &in_contact in step_contacts {
                if in_contact {
                    n_friction_ineq += 6; // 4 friction pyramid + fz>=0 + fz<=fmax
                } else {
                    n_swing_eq += 3; // fx=fy=fz=0
                }
            }
        }

        let n_eq = n_swing_eq;
        let n_ineq = n_friction_ineq;
        let n_constraints = n_eq + n_ineq;

        let mut a_con = DMatrix::zeros(n_constraints, n_u_total);
        let mut b_con = DVector::zeros(n_constraints);

        let mut row = 0;

        // --- Swing foot equality constraints (f = 0) ---
        for (k, step_contacts) in contacts.iter().enumerate().take(h) {
            for (foot, &in_contact) in step_contacts.iter().enumerate() {
                if !in_contact {
                    let u_off = k * n_u_step + 3 * foot;
                    for j in 0..3 {
                        a_con[(row, u_off + j)] = 1.0;
                        // b_con[row] = 0.0 (already zero)
                        row += 1;
                    }
                }
            }
        }

        assert_eq!(row, n_eq, "Equality constraint count mismatch");

        // --- Friction cone inequality constraints ---
        // Clarabel NonnegativeCone: A z + s = b, s >= 0  ⟹  A z <= b
        let mu = self.config.friction_coeff;
        let f_max = self.config.f_max;

        for (k, step_contacts) in contacts.iter().enumerate().take(h) {
            for (foot, &in_contact) in step_contacts.iter().enumerate() {
                if in_contact {
                    let fx_idx = k * n_u_step + 3 * foot;
                    let fy_idx = fx_idx + 1;
                    let fz_idx = fx_idx + 2;

                    // 1. fx - mu*fz <= 0
                    a_con[(row, fx_idx)] = 1.0;
                    a_con[(row, fz_idx)] = -mu;
                    row += 1;

                    // 2. -fx - mu*fz <= 0
                    a_con[(row, fx_idx)] = -1.0;
                    a_con[(row, fz_idx)] = -mu;
                    row += 1;

                    // 3. fy - mu*fz <= 0
                    a_con[(row, fy_idx)] = 1.0;
                    a_con[(row, fz_idx)] = -mu;
                    row += 1;

                    // 4. -fy - mu*fz <= 0
                    a_con[(row, fy_idx)] = -1.0;
                    a_con[(row, fz_idx)] = -mu;
                    row += 1;

                    // 5. -fz <= 0  (fz >= 0)
                    a_con[(row, fz_idx)] = -1.0;
                    row += 1;

                    // 6. fz <= f_max
                    a_con[(row, fz_idx)] = 1.0;
                    b_con[row] = f_max;
                    row += 1;
                }
            }
        }

        assert_eq!(row, n_constraints, "Total constraint count mismatch");

        (a_con, b_con, n_eq, n_ineq)
    }
}

/// Build the condensed prediction matrices A_qp and B_qp.
///
/// A_qp stacks powers of A_d:
/// ```text
/// A_qp = [A_d; A_d^2; ...; A_d^H]     (13H × 13)
/// ```
///
/// B_qp is lower block-triangular:
/// ```text
/// B_qp = [B_d,      0,      ...  0    ]
///        [A_d B_d,   B_d,    ...  0    ]
///        [A_d^2 B_d, A_d B_d, ... 0    ]
///        [ ...                    B_d  ]     (13H × 12H)
/// ```
fn build_prediction_matrices(
    a_d: &DMatrix<f64>,
    b_d: &DMatrix<f64>,
    horizon: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_x = a_d.nrows();
    let n_u = b_d.ncols();

    // Precompute A^k for k = 0..H
    let mut a_powers: Vec<DMatrix<f64>> = Vec::with_capacity(horizon + 1);
    a_powers.push(DMatrix::identity(n_x, n_x)); // A^0 = I
    for i in 1..=horizon {
        a_powers.push(a_d * &a_powers[i - 1]);
    }

    // Precompute A^k * B for k = 0..H-1
    let mut ab_products: Vec<DMatrix<f64>> = Vec::with_capacity(horizon);
    for i in 0..horizon {
        ab_products.push(&a_powers[i] * b_d);
    }

    // Build A_qp (13H × 13)
    let mut a_qp = DMatrix::zeros(n_x * horizon, n_x);
    for k in 0..horizon {
        a_qp.view_mut((k * n_x, 0), (n_x, n_x))
            .copy_from(&a_powers[k + 1]);
    }

    // Build B_qp (13H × 12H) - lower block triangular
    let mut b_qp = DMatrix::zeros(n_x * horizon, n_u * horizon);
    for k in 0..horizon {
        for j in 0..=k {
            let power = k - j; // A^(k-j) * B
            b_qp.view_mut((k * n_x, j * n_u), (n_x, n_u))
                .copy_from(&ab_products[power]);
        }
    }

    (a_qp, b_qp)
}

/// Convert a nalgebra `DMatrix<f64>` to a Clarabel `CscMatrix<f64>` (full matrix).
fn dmatrix_to_csc(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let (nrows, ncols) = m.shape();
    let mut colptr = vec![0usize; ncols + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for j in 0..ncols {
        for i in 0..nrows {
            let v = m[(i, j)];
            if v.abs() > 1e-15 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr[j + 1] = rowval.len();
    }

    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

/// Convert a symmetric `DMatrix<f64>` to upper-triangular `CscMatrix<f64>`.
fn dmatrix_to_csc_upper_tri(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let (nrows, ncols) = m.shape();
    let mut colptr = vec![0usize; ncols + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for j in 0..ncols {
        for i in 0..=j.min(nrows - 1) {
            let v = m[(i, j)];
            if v.abs() > 1e-15 {
                rowval.push(i);
                nzval.push(v);
            }
        }
        colptr[j + 1] = rowval.len();
    }

    CscMatrix::new(nrows, ncols, colptr, rowval, nzval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BodyState;
    use approx::assert_relative_eq;
    fn test_config() -> MpcConfig {
        MpcConfig::default()
    }

    fn quadruped_feet_standing() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(0.15, 0.08, 0.0),
            Vector3::new(0.15, -0.08, 0.0),
            Vector3::new(-0.15, 0.08, 0.0),
            Vector3::new(-0.15, -0.08, 0.0),
        ]
    }

    #[test]
    fn standing_balance_forces_support_gravity() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.30),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);

        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.30,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);

        assert!(solution.converged, "QP must converge for standing balance");

        // Total vertical force should closely balance gravity: m*g = 9.0*9.81 ≈ 88.3N
        // Slight overshoot is expected since the MPC also tracks the reference trajectory
        let total_fz: f64 = solution.forces.iter().map(|f| f.z).sum();
        assert_relative_eq!(total_fz, config.mass * config.gravity, epsilon = 10.0);

        // Each foot should carry roughly equal load
        for (i, f) in solution.forces.iter().enumerate() {
            assert!(
                f.z > 10.0,
                "Foot {i}: fz={:.1} should be positive (supporting weight)",
                f.z
            );
        }

        // Horizontal forces should be near zero
        let total_fx: f64 = solution.forces.iter().map(|f| f.x.abs()).sum();
        let total_fy: f64 = solution.forces.iter().map(|f| f.y.abs()).sum();
        assert!(total_fx < 5.0, "fx should be small: {total_fx}");
        assert!(total_fy < 5.0, "fy should be small: {total_fy}");
    }

    #[test]
    fn forces_satisfy_friction_cone() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.29),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::new(0.3, 0.0, 0.0),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::new(0.5, 0.0, 0.0),
            0.29,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        let mu = config.friction_coeff;
        for (i, f) in solution.forces.iter().enumerate() {
            assert!(f.z >= -1e-3, "Foot {i}: fz={} must be >= 0", f.z);
            assert!(
                f.x.abs() <= mu * f.z + 1e-3,
                "Foot {i}: |fx|={} > mu*fz={}",
                f.x.abs(),
                mu * f.z
            );
            assert!(
                f.y.abs() <= mu * f.z + 1e-3,
                "Foot {i}: |fy|={} > mu*fz={}",
                f.y.abs(),
                mu * f.z
            );
        }
    }

    #[test]
    fn swing_feet_zero_force() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.29),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();

        // Trot: FL+RR stance, FR+RL swing
        let contacts: Vec<Vec<bool>> = vec![vec![true, false, false, true]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.29,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        // Swing feet should have zero force
        assert!(solution.forces[1].norm() < 1e-3, "FR swing force should be ~0");
        assert!(solution.forces[2].norm() < 1e-3, "RL swing force should be ~0");

        // Stance feet should support the robot (2 feet carry all weight)
        let stance_fz = solution.forces[0].z + solution.forces[3].z;
        assert_relative_eq!(stance_fz, config.mass * config.gravity, epsilon = 15.0);
    }

    #[test]
    fn solve_time_reasonable() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.29),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.29,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        assert!(
            solution.solve_time_us < 200_000,
            "Solve took {}us, expected < 200000us (debug mode is slow)",
            solution.solve_time_us
        );
    }

    #[test]
    fn prediction_matrices_dimensions() {
        let n_x = STATE_DIM;
        let n_u = 12;
        let h = 5;

        let a_d = DMatrix::identity(n_x, n_x);
        let b_d = DMatrix::zeros(n_x, n_u);

        let (a_qp, b_qp) = build_prediction_matrices(&a_d, &b_d, h);

        assert_eq!(a_qp.nrows(), n_x * h);
        assert_eq!(a_qp.ncols(), n_x);
        assert_eq!(b_qp.nrows(), n_x * h);
        assert_eq!(b_qp.ncols(), n_u * h);
    }

    #[test]
    fn prediction_with_identity_dynamics() {
        // With A_d = I and B_d = 0, A_qp should be stacked identities
        let n_x = STATE_DIM;
        let n_u = 12;
        let h = 3;

        let a_d = DMatrix::identity(n_x, n_x);
        let b_d = DMatrix::zeros(n_x, n_u);

        let (a_qp, _b_qp) = build_prediction_matrices(&a_d, &b_d, h);

        // Each block of A_qp should be identity (I^k = I)
        for k in 0..h {
            let block = a_qp.view((k * n_x, 0), (n_x, n_x));
            let identity = DMatrix::identity(n_x, n_x);
            assert_relative_eq!(block.clone_owned(), identity, epsilon = 1e-12);
        }
    }

    #[test]
    fn forward_velocity_produces_positive_fx() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.29),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(), // starting from rest
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::new(0.5, 0.0, 0.0), // want to go forward
            0.29,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        // Total forward force should be positive (accelerating forward)
        let total_fx: f64 = solution.forces.iter().map(|f| f.x).sum();
        assert!(
            total_fx > 0.0,
            "Total fx={total_fx} should be positive for forward acceleration"
        );
    }
}
