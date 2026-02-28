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
    DefaultSettings, DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus,
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
    // Cached Clarabel settings (avoids rebuilding each solve)
    clarabel_settings: DefaultSettings<f64>,
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

        let clarabel_settings = DefaultSettingsBuilder::default()
            .max_iter(config.max_solver_iters)
            .verbose(false)
            .tol_gap_abs(1e-6)
            .tol_gap_rel(1e-6)
            .tol_feas(1e-6)
            .build()
            .expect("valid solver settings");

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
            clarabel_settings,
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
        let orientation = Vector3::new(x0[0], x0[1], x0[2]);
        let com = Vector3::new(x0[3], x0[4], x0[5]);
        let (a_c, b_c) = build_continuous_dynamics(
            &orientation,
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

        // 4. Build constraint CSC directly (avoids dense intermediate matrix)
        let (a_csc, b_con, n_eq, n_ineq) = build_constraints_csc(
            &self.config,
            contacts,
            h,
            self.n_feet,
            n_u_total,
        );

        // 5. Convert P to Clarabel CSC format
        let p_csc = dmatrix_to_csc_upper_tri(&self.p_mat);

        // 6. Define cones: equalities first, then inequalities
        let cones = vec![ZeroConeT(n_eq), NonnegativeConeT(n_ineq)];

        // 7. Solve (reuse cached settings)
        let solver_result = DefaultSolver::new(
            &p_csc,
            self.q_vec.as_slice(),
            &a_csc,
            &b_con,
            &cones,
            self.clarabel_settings.clone(),
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

        // Force rate regularization: P += 2 * w_df * D^T D
        // D^T D has block-tridiagonal structure:
        //   diag blocks: I (first/last), 2I (middle)
        //   off-diag blocks: -I (adjacent)
        let w_df = self.config.delta_f_weight;
        if w_df > 0.0 {
            let h = self.config.horizon;
            let n_u = self.n_u_step;
            let two_w = 2.0 * w_df;
            for k in 0..h {
                let off = k * n_u;
                // Diagonal block coefficient: 1 for first/last, 2 for middle
                let diag_coeff = if k == 0 || k == h - 1 { 1.0 } else { 2.0 };
                for i in 0..n_u {
                    self.p_mat[(off + i, off + i)] += two_w * diag_coeff;
                }
                // Off-diagonal block: -I between k and k+1
                if k + 1 < h {
                    let off_next = (k + 1) * n_u;
                    for i in 0..n_u {
                        self.p_mat[(off + i, off_next + i)] -= two_w;
                        self.p_mat[(off_next + i, off + i)] -= two_w;
                    }
                }
            }
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

/// Build constraint matrices directly in CSC format for friction cone and
/// swing feet.
///
/// This avoids allocating a dense intermediate matrix (~288KB for typical
/// quadruped MPC) and scanning it for non-zeros. The constraint matrix is
/// ~98% zeros so building CSC directly is significantly faster.
///
/// Equalities (ZeroCone) first, then inequalities (NonnegativeCone).
fn build_constraints_csc(
    config: &MpcConfig,
    contacts: &[Vec<bool>],
    h: usize,
    n_feet: usize,
    n_u_total: usize,
) -> (CscMatrix<f64>, Vec<f64>, usize, usize) {
    let n_u_step = 3 * n_feet;

    // Count constraints
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

    let mu = config.friction_coeff;
    let f_max = config.f_max;

    // Pre-allocate CSC buffers with known capacity.
    // Swing: 1 non-zero per constraint row (3 rows per swing foot).
    // Friction: each foot contributes to fx, fy, fz columns with 1-2 entries each.
    // Max non-zeros: n_swing_eq + n_friction_ineq * 2 (most rows have 2 entries)
    let max_nnz = n_swing_eq + n_friction_ineq * 2;
    let mut colptr = vec![0usize; n_u_total + 1];
    let mut rowval = Vec::with_capacity(max_nnz);
    let mut nzval = Vec::with_capacity(max_nnz);
    let mut b_con = vec![0.0; n_constraints];

    // Build column-by-column.
    // For each decision variable column (force component), determine which
    // constraint rows reference it and add those entries.
    //
    // Row layout: [swing equalities (n_eq)] [friction inequalities (n_ineq)]
    //
    // We need to compute the row index for each constraint. Pre-compute a
    // mapping from (step, foot) to the starting row for that foot's constraints.

    // Swing equality row offsets: (step, foot) → row
    let mut swing_row_map: Vec<Vec<Option<usize>>> = Vec::with_capacity(h);
    let mut eq_row = 0;
    for step_contacts in contacts.iter().take(h) {
        let mut step_map = Vec::with_capacity(n_feet);
        for &in_contact in step_contacts {
            if !in_contact {
                step_map.push(Some(eq_row));
                eq_row += 3;
            } else {
                step_map.push(None);
            }
        }
        swing_row_map.push(step_map);
    }

    // Friction inequality row offsets: (step, foot) → row
    let mut friction_row_map: Vec<Vec<Option<usize>>> = Vec::with_capacity(h);
    let mut ineq_row = n_eq; // friction rows come after swing equalities
    for step_contacts in contacts.iter().take(h) {
        let mut step_map = Vec::with_capacity(n_feet);
        for &in_contact in step_contacts {
            if in_contact {
                step_map.push(Some(ineq_row));
                ineq_row += 6;
            } else {
                step_map.push(None);
            }
        }
        friction_row_map.push(step_map);
    }

    // Fill b_con for friction upper bounds (fz <= f_max)
    for step_map in &friction_row_map {
        for maybe_row in step_map {
            if let Some(base_row) = maybe_row {
                // Row base_row+5 is the fz <= f_max constraint
                b_con[base_row + 5] = f_max;
            }
        }
    }

    // Now build CSC column by column
    for col in 0..n_u_total {
        let step = col / n_u_step;
        let within_step = col % n_u_step;
        let foot = within_step / 3;
        let axis = within_step % 3; // 0=fx, 1=fy, 2=fz

        if step >= h {
            colptr[col + 1] = rowval.len();
            continue;
        }

        let in_contact = contacts[step][foot];

        if !in_contact {
            // Swing equality: this column has a single 1.0 in its row
            if let Some(base_row) = swing_row_map[step][foot] {
                rowval.push(base_row + axis);
                nzval.push(1.0);
            }
        } else {
            // Friction inequality: contribute to this foot's 6 constraint rows
            if let Some(base_row) = friction_row_map[step][foot] {
                match axis {
                    0 => {
                        // fx column: rows base+0 (+1.0) and base+1 (-1.0)
                        rowval.push(base_row);
                        nzval.push(1.0);
                        rowval.push(base_row + 1);
                        nzval.push(-1.0);
                    }
                    1 => {
                        // fy column: rows base+2 (+1.0) and base+3 (-1.0)
                        rowval.push(base_row + 2);
                        nzval.push(1.0);
                        rowval.push(base_row + 3);
                        nzval.push(-1.0);
                    }
                    2 => {
                        // fz column: rows base+0 (-mu), base+1 (-mu),
                        //             base+2 (-mu), base+3 (-mu),
                        //             base+4 (-1.0), base+5 (+1.0)
                        rowval.push(base_row);
                        nzval.push(-mu);
                        rowval.push(base_row + 1);
                        nzval.push(-mu);
                        rowval.push(base_row + 2);
                        nzval.push(-mu);
                        rowval.push(base_row + 3);
                        nzval.push(-mu);
                        rowval.push(base_row + 4);
                        nzval.push(-1.0);
                        rowval.push(base_row + 5);
                        nzval.push(1.0);
                    }
                    _ => unreachable!(),
                }
            }
        }

        colptr[col + 1] = rowval.len();
    }

    let a_csc = CscMatrix::new(n_constraints, n_u_total, colptr, rowval, nzval);
    (a_csc, b_con, n_eq, n_ineq)
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

/// Convert a symmetric `DMatrix<f64>` to upper-triangular `CscMatrix<f64>`.
fn dmatrix_to_csc_upper_tri(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let (nrows, ncols) = m.shape();
    let mut colptr = vec![0usize; ncols + 1];
    // Upper triangle has at most n*(n+1)/2 entries
    let max_nnz = ncols * (ncols + 1) / 2;
    let mut rowval = Vec::with_capacity(max_nnz);
    let mut nzval = Vec::with_capacity(max_nnz);

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
