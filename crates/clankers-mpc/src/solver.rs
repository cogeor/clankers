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
///
/// Pre-allocates workspace matrices (~600KB for a quadruped with horizon=10)
/// to avoid per-solve heap allocation.
pub struct MpcSolver {
    config: MpcConfig,
    n_feet: usize,
    n_u_step: usize,
    n_u_total: usize,
    n_x_total: usize,
    // Prediction workspace
    a_qp: DMatrix<f64>,
    b_qp: DMatrix<f64>,
    // Cost workspace
    b_qp_t: DMatrix<f64>,
    sb: DMatrix<f64>,
    p_mat: DMatrix<f64>,
    q_vec: DVector<f64>,
    s_diag: DVector<f64>,
    // Matrix power workspace
    a_powers: Vec<DMatrix<f64>>,
    ab_products: Vec<DMatrix<f64>>,
}

impl MpcSolver {
    /// Create a new MPC solver with pre-allocated workspace.
    ///
    /// `n_feet` determines workspace dimensions and must match the number of
    /// foot positions passed to [`solve`].
    pub fn new(config: MpcConfig, n_feet: usize) -> Self {
        let h = config.horizon;
        let n_u_step = 3 * n_feet;
        let n_u_total = n_u_step * h;
        let n_x_total = STATE_DIM * h;

        // Build s_diag once (only depends on config weights)
        let mut s_diag = DVector::zeros(n_x_total);
        for k in 0..h {
            let off = k * STATE_DIM;
            for i in 0..12 {
                s_diag[off + i] = config.q_weights[i];
            }
        }

        Self {
            config,
            n_feet,
            n_u_step,
            n_u_total,
            n_x_total,
            a_qp: DMatrix::zeros(n_x_total, STATE_DIM),
            b_qp: DMatrix::zeros(n_x_total, n_u_total),
            b_qp_t: DMatrix::zeros(n_u_total, n_x_total),
            sb: DMatrix::zeros(n_x_total, n_u_total),
            p_mat: DMatrix::zeros(n_u_total, n_u_total),
            q_vec: DVector::zeros(n_u_total),
            s_diag,
            a_powers: (0..=h)
                .map(|_| DMatrix::zeros(STATE_DIM, STATE_DIM))
                .collect(),
            ab_products: (0..h)
                .map(|_| DMatrix::zeros(STATE_DIM, n_u_step))
                .collect(),
        }
    }

    /// Access the solver configuration.
    pub fn config(&self) -> &MpcConfig {
        &self.config
    }

    /// Solve the MPC problem for optimal foot forces.
    ///
    /// Uses the condensed (single-shooting) formulation from MIT Cheetah:
    /// decision variables are only forces U (12*H), states are eliminated.
    ///
    /// Workspace matrices are reused between calls to avoid allocation.
    ///
    /// # Arguments
    /// * `x0` - Current 13D state vector
    /// * `foot_positions` - World-frame foot positions (one per foot)
    /// * `contacts` - Contact flags: `contacts[step][foot]`, size horizon × n_feet
    /// * `reference` - Desired state trajectory over the horizon
    pub fn solve(
        &mut self,
        x0: &DVector<f64>,
        foot_positions: &[Vector3<f64>],
        contacts: &[Vec<bool>],
        reference: &ReferenceTrajectory,
    ) -> MpcSolution {
        let start = Instant::now();

        let h = self.config.horizon;
        let n_u_total = self.n_u_total;

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

        // 2. Fill prediction matrices into workspace (no allocation)
        self.fill_prediction_matrices(&a_d, &b_d);

        // 3. Fill cost matrices into workspace (no allocation for big matrices)
        self.fill_condensed_cost(x0, reference);

        // 4. Build constraints (still allocates — size varies with contact pattern)
        let (a_con, b_con, n_eq, n_ineq) = build_constraints(
            &self.config,
            contacts,
            h,
            self.n_feet,
            n_u_total,
        );

        // 5. Convert to Clarabel CSC format
        let p_csc = dmatrix_to_csc_upper_tri(&self.p_mat);
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

        let solver_result = DefaultSolver::new(
            &p_csc,
            self.q_vec.as_slice(),
            &a_csc,
            b_con.as_slice(),
            &cones,
            settings,
        );

        let converged;
        let mut forces = vec![Vector3::zeros(); self.n_feet];
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
                    for i in 0..n_u_total {
                        force_trajectory[i] = sol.x[i];
                    }

                    for (foot, force) in forces.iter_mut().enumerate() {
                        *force = Vector3::new(
                            sol.x[3 * foot],
                            sol.x[3 * foot + 1],
                            sol.x[3 * foot + 2],
                        );
                    }

                    // Reconstruct state trajectory: X = A_qp x0 + B_qp U
                    let u_vec = DVector::from_column_slice(&sol.x[..n_u_total]);
                    state_trajectory = &self.a_qp * x0 + &self.b_qp * u_vec;
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

    /// Fill prediction matrices A_qp and B_qp into workspace.
    ///
    /// Uses `split_at_mut` on `a_powers` to avoid aliasing, and `gemm`
    /// for in-place matrix multiplication.
    fn fill_prediction_matrices(&mut self, a_d: &DMatrix<f64>, b_d: &DMatrix<f64>) {
        let h = self.config.horizon;
        let n_x = STATE_DIM;
        let n_u = self.n_u_step;

        // A^0 = I
        self.a_powers[0].fill(0.0);
        for i in 0..n_x {
            self.a_powers[0][(i, i)] = 1.0;
        }

        // A^k = A_d * A^(k-1), using split_at_mut to satisfy borrow checker
        for i in 1..=h {
            let (left, right) = self.a_powers.split_at_mut(i);
            right[0].gemm(1.0, a_d, &left[i - 1], 0.0);
        }

        // A^k * B_d for k = 0..H
        for i in 0..h {
            self.ab_products[i].gemm(1.0, &self.a_powers[i], b_d, 0.0);
        }

        // Fill A_qp: [A_d; A_d^2; ...; A_d^H]
        for k in 0..h {
            self.a_qp
                .view_mut((k * n_x, 0), (n_x, n_x))
                .copy_from(&self.a_powers[k + 1]);
        }

        // Fill B_qp: lower block triangular
        for k in 0..h {
            for j in 0..=k {
                let power = k - j;
                self.b_qp
                    .view_mut((k * n_x, j * n_u), (n_x, n_u))
                    .copy_from(&self.ab_products[power]);
            }
        }
    }

    /// Fill the condensed QP cost matrices P and q into workspace.
    ///
    /// P = 2 (B_qp^T S B_qp + alpha I)
    /// q = 2 B_qp^T S (A_qp x0 - X_ref)
    fn fill_condensed_cost(&mut self, x0: &DVector<f64>, reference: &ReferenceTrajectory) {
        let n_x_total = self.n_x_total;
        let n_u_total = self.n_u_total;

        // sb = S * B_qp (scale rows of b_qp by diagonal state weights)
        self.sb.copy_from(&self.b_qp);
        for i in 0..n_x_total {
            let w = self.s_diag[i];
            if w.abs() > 1e-20 {
                for j in 0..n_u_total {
                    self.sb[(i, j)] *= w;
                }
            } else {
                for j in 0..n_u_total {
                    self.sb[(i, j)] = 0.0;
                }
            }
        }

        // P = 2 * (B^T * S * B + alpha * I)
        self.b_qp.transpose_to(&mut self.b_qp_t);
        self.p_mat.gemm(2.0, &self.b_qp_t, &self.sb, 0.0);
        let two_alpha = 2.0 * self.config.r_weight;
        for i in 0..n_u_total {
            self.p_mat[(i, i)] += two_alpha;
        }

        // q = 2 * B^T * S * (A_qp * x0 - X_ref)
        let error = &self.a_qp * x0 - &reference.states;
        let mut s_error = error;
        for i in 0..n_x_total {
            s_error[i] *= self.s_diag[i];
        }
        self.q_vec.gemv(2.0, &self.b_qp_t, &s_error, 0.0);
    }
}

/// Build constraint matrices for friction cone and swing feet.
///
/// Equalities (ZeroCone) first, then inequalities (NonnegativeCone).
fn build_constraints(
    config: &MpcConfig,
    contacts: &[Vec<bool>],
    h: usize,
    n_feet: usize,
    n_u_total: usize,
) -> (DMatrix<f64>, DVector<f64>, usize, usize) {
    let n_u_step = 3 * n_feet;

    let mut n_swing_eq = 0;
    let mut n_friction_ineq = 0;

    for step_contacts in contacts.iter().take(h) {
        for &in_contact in step_contacts {
            if in_contact {
                n_friction_ineq += 6;
            } else {
                n_swing_eq += 3;
            }
        }
    }

    let n_eq = n_swing_eq;
    let n_ineq = n_friction_ineq;
    let n_constraints = n_eq + n_ineq;

    let mut a_con = DMatrix::zeros(n_constraints, n_u_total);
    let mut b_con = DVector::zeros(n_constraints);

    let mut row = 0;

    // Swing foot equality constraints (f = 0)
    for (k, step_contacts) in contacts.iter().enumerate().take(h) {
        for (foot, &in_contact) in step_contacts.iter().enumerate() {
            if !in_contact {
                let u_off = k * n_u_step + 3 * foot;
                for j in 0..3 {
                    a_con[(row, u_off + j)] = 1.0;
                    row += 1;
                }
            }
        }
    }

    assert_eq!(row, n_eq, "Equality constraint count mismatch");

    // Friction cone inequality constraints
    let mu = config.friction_coeff;
    let f_max = config.f_max;

    for (k, step_contacts) in contacts.iter().enumerate().take(h) {
        for (foot, &in_contact) in step_contacts.iter().enumerate() {
            if in_contact {
                let fx_idx = k * n_u_step + 3 * foot;
                let fy_idx = fx_idx + 1;
                let fz_idx = fx_idx + 2;

                a_con[(row, fx_idx)] = 1.0;
                a_con[(row, fz_idx)] = -mu;
                row += 1;

                a_con[(row, fx_idx)] = -1.0;
                a_con[(row, fz_idx)] = -mu;
                row += 1;

                a_con[(row, fy_idx)] = 1.0;
                a_con[(row, fz_idx)] = -mu;
                row += 1;

                a_con[(row, fy_idx)] = -1.0;
                a_con[(row, fz_idx)] = -mu;
                row += 1;

                a_con[(row, fz_idx)] = -1.0;
                row += 1;

                a_con[(row, fz_idx)] = 1.0;
                b_con[row] = f_max;
                row += 1;
            }
        }
    }

    assert_eq!(row, n_constraints, "Total constraint count mismatch");

    (a_con, b_con, n_eq, n_ineq)
}

/// Build prediction matrices (allocating version, for tests only).
#[cfg(test)]
fn build_prediction_matrices(
    a_d: &DMatrix<f64>,
    b_d: &DMatrix<f64>,
    horizon: usize,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let n_x = a_d.nrows();
    let n_u = b_d.ncols();

    let mut a_powers: Vec<DMatrix<f64>> = Vec::with_capacity(horizon + 1);
    a_powers.push(DMatrix::identity(n_x, n_x));
    for i in 1..=horizon {
        a_powers.push(a_d * &a_powers[i - 1]);
    }

    let mut ab_products: Vec<DMatrix<f64>> = Vec::with_capacity(horizon);
    for i in 0..horizon {
        ab_products.push(&a_powers[i] * b_d);
    }

    let mut a_qp = DMatrix::zeros(n_x * horizon, n_x);
    for k in 0..horizon {
        a_qp.view_mut((k * n_x, 0), (n_x, n_x))
            .copy_from(&a_powers[k + 1]);
    }

    let mut b_qp = DMatrix::zeros(n_x * horizon, n_u * horizon);
    for k in 0..horizon {
        for j in 0..=k {
            let power = k - j;
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
        let mut solver = MpcSolver::new(config.clone(), 4);

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

        let total_fz: f64 = solution.forces.iter().map(|f| f.z).sum();
        assert_relative_eq!(total_fz, config.mass * config.gravity, epsilon = 10.0);

        for (i, f) in solution.forces.iter().enumerate() {
            assert!(
                f.z > 10.0,
                "Foot {i}: fz={:.1} should be positive (supporting weight)",
                f.z
            );
        }

        let total_fx: f64 = solution.forces.iter().map(|f| f.x.abs()).sum();
        let total_fy: f64 = solution.forces.iter().map(|f| f.y.abs()).sum();
        assert!(total_fx < 5.0, "fx should be small: {total_fx}");
        assert!(total_fy < 5.0, "fy should be small: {total_fy}");
    }

    #[test]
    fn forces_satisfy_friction_cone() {
        let config = test_config();
        let mut solver = MpcSolver::new(config.clone(), 4);

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
        let mut solver = MpcSolver::new(config.clone(), 4);

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

        assert!(solution.forces[1].norm() < 1e-3, "FR swing force should be ~0");
        assert!(solution.forces[2].norm() < 1e-3, "RL swing force should be ~0");

        let stance_fz = solution.forces[0].z + solution.forces[3].z;
        assert_relative_eq!(stance_fz, config.mass * config.gravity, epsilon = 15.0);
    }

    #[test]
    fn solve_time_reasonable() {
        let config = test_config();
        let mut solver = MpcSolver::new(config.clone(), 4);

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
        let n_x = STATE_DIM;
        let n_u = 12;
        let h = 3;

        let a_d = DMatrix::identity(n_x, n_x);
        let b_d = DMatrix::zeros(n_x, n_u);

        let (a_qp, _b_qp) = build_prediction_matrices(&a_d, &b_d, h);

        for k in 0..h {
            let block = a_qp.view((k * n_x, 0), (n_x, n_x));
            let identity = DMatrix::identity(n_x, n_x);
            assert_relative_eq!(block.clone_owned(), identity, epsilon = 1e-12);
        }
    }

    #[test]
    fn forward_velocity_produces_positive_fx() {
        let config = test_config();
        let mut solver = MpcSolver::new(config.clone(), 4);

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
            &Vector3::new(0.5, 0.0, 0.0),
            0.29,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        let total_fx: f64 = solution.forces.iter().map(|f| f.x).sum();
        assert!(
            total_fx > 0.0,
            "Total fx={total_fx} should be positive for forward acceleration"
        );
    }
}
