//! MPC solver: builds and solves the centroidal dynamics QP.
//!
//! Uses Clarabel (pure Rust interior-point solver) to find optimal
//! ground reaction forces that track a desired body trajectory while
//! respecting friction cone and unilateral contact constraints.
//!
//! # QP Formulation
//!
//! Decision variables: z = [x_1, ..., x_H, u_0, ..., u_{H-1}]
//! where x_k is the predicted 13D state and u_k are the 3D foot forces.
//!
//! Cost: sum over k of (x_k - x_ref_k)^T Q (x_k - x_ref_k) + u_k^T R u_k
//!
//! Subject to:
//! - Dynamics: x_{k+1} = A_d x_k + B_d u_k (equality)
//! - Friction cone: |fx| <= mu * fz, |fy| <= mu * fz (inequality)
//! - Unilateral contact: 0 <= fz <= fmax (inequality)
//! - Swing feet: f = 0 (equality)

use std::time::Instant;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, SolverStatus,
    SupportedConeT::{NonnegativeConeT, ZeroConeT},
};
use nalgebra::{DMatrix, DVector, Vector3};

use crate::centroidal::{build_continuous_dynamics, discretize};
use crate::types::{MpcConfig, MpcSolution, ReferenceTrajectory, STATE_DIM};

/// Centroidal MPC solver.
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
    /// # Arguments
    /// * `x0` - Current 13D state vector
    /// * `foot_positions` - World-frame foot positions (one per foot)
    /// * `contacts` - Contact flags: contacts[step][foot], size horizon x n_feet
    /// * `reference` - Desired state trajectory over the horizon
    ///
    /// # Returns
    /// `MpcSolution` with optimal foot forces and convergence info.
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
        let n_x = STATE_DIM * h;
        let n_u = n_u_step * h;
        let n_z = n_x + n_u;

        // 1. Build dynamics matrices at current linearization point
        let yaw = x0[2];
        let com = Vector3::new(x0[3], x0[4], x0[5]);
        let inertia_inv = self
            .config
            .inertia
            .try_inverse()
            .expect("inertia tensor must be invertible");
        let (a_c, b_c) = build_continuous_dynamics(
            yaw,
            foot_positions,
            &com,
            &inertia_inv,
            self.config.mass,
        );
        let (a_d, b_d) = discretize(&a_c, &b_c, self.config.dt);

        // 2. Build cost matrices (P upper triangular, q vector)
        let (p_mat, q_vec) = self.build_cost(h, n_feet, n_z, reference);

        // 3. Build all constraints
        let (a_all, b_all, n_eq, n_ineq) =
            self.build_constraints(&a_d, &b_d, x0, contacts, h, n_feet, n_z);

        // 4. Convert to Clarabel format
        let p_csc = dmatrix_to_csc_upper_tri(&p_mat);
        let a_csc = dmatrix_to_csc(&a_all);

        // 5. Define cones
        let cones = vec![ZeroConeT(n_eq), NonnegativeConeT(n_ineq)];

        // 6. Solve
        let settings = DefaultSettingsBuilder::default()
            .max_iter(self.config.max_solver_iters)
            .verbose(false)
            .tol_gap_abs(1e-5)
            .tol_gap_rel(1e-5)
            .tol_feas(1e-5)
            .build()
            .expect("valid solver settings");

        let q_slice: Vec<f64> = q_vec.iter().copied().collect();
        let b_slice: Vec<f64> = b_all.iter().copied().collect();

        let solver_result = DefaultSolver::new(&p_csc, &q_slice, &a_csc, &b_slice, &cones, settings);

        let converged;
        let mut forces = vec![Vector3::zeros(); n_feet];
        let mut force_trajectory = DVector::zeros(n_u);
        let mut state_trajectory = DVector::zeros(n_x);

        match solver_result {
            Ok(mut solver) => {
                solver.solve();
                let sol = &solver.solution;

                converged = matches!(
                    sol.status,
                    SolverStatus::Solved | SolverStatus::AlmostSolved
                );

                if converged {
                    // Extract state trajectory
                    for i in 0..n_x {
                        state_trajectory[i] = sol.x[i];
                    }

                    // Extract full force trajectory
                    for i in 0..n_u {
                        force_trajectory[i] = sol.x[n_x + i];
                    }

                    // Extract first-step foot forces
                    for (foot, force) in forces.iter_mut().enumerate() {
                        let base = n_x; // u_0 starts after all states
                        *force = Vector3::new(
                            sol.x[base + 3 * foot],
                            sol.x[base + 3 * foot + 1],
                            sol.x[base + 3 * foot + 2],
                        );
                    }
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

    /// Build the cost matrices P (upper triangular) and q.
    fn build_cost(
        &self,
        h: usize,
        n_feet: usize,
        n_z: usize,
        reference: &ReferenceTrajectory,
    ) -> (DMatrix<f64>, DVector<f64>) {
        let n_u_step = 3 * n_feet;
        let n_x = STATE_DIM * h;

        let mut p = DMatrix::zeros(n_z, n_z);
        let mut q = DVector::zeros(n_z);

        // State cost: Q diagonal for each horizon step
        for k in 0..h {
            let x_off = k * STATE_DIM;
            for i in 0..12 {
                p[(x_off + i, x_off + i)] = self.config.q_weights[i];
            }
            // State 12 (gravity constant): zero weight

            // Linear cost term: q_x = -Q * x_ref
            let ref_off = k * STATE_DIM;
            for i in 0..12 {
                q[x_off + i] = -self.config.q_weights[i] * reference.states[ref_off + i];
            }
        }

        // Control cost: R diagonal for each horizon step
        for k in 0..h {
            let u_off = n_x + k * n_u_step;
            for j in 0..n_u_step {
                p[(u_off + j, u_off + j)] = self.config.r_weight;
            }
        }

        (p, q)
    }

    /// Build all constraint matrices (dynamics + contact).
    ///
    /// Returns (A_all, b_all, n_eq, n_ineq) where equalities come first.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn build_constraints(
        &self,
        a_d: &DMatrix<f64>,
        b_d: &DMatrix<f64>,
        x0: &DVector<f64>,
        contacts: &[Vec<bool>],
        h: usize,
        n_feet: usize,
        n_z: usize,
    ) -> (DMatrix<f64>, DVector<f64>, usize, usize) {
        let n_u_step = 3 * n_feet;
        let n_x = STATE_DIM * h;

        // Count constraints
        let n_dyn_eq = STATE_DIM * h;
        let mut n_swing_eq = 0;
        let mut n_friction_ineq = 0;

        for step_contacts in contacts.iter().take(h) {
            for &in_contact in step_contacts {
                if in_contact {
                    n_friction_ineq += 6; // 4 friction + 2 force limits
                } else {
                    n_swing_eq += 3; // fx=0, fy=0, fz=0
                }
            }
        }

        let n_eq = n_dyn_eq + n_swing_eq;
        let n_ineq = n_friction_ineq;
        let n_constraints = n_eq + n_ineq;

        let mut a_all = DMatrix::zeros(n_constraints, n_z);
        let mut b_all = DVector::zeros(n_constraints);

        let mut row = 0;

        // --- Dynamics equality constraints ---
        // k=0: I * x_1 - B_d * u_0 = A_d * x_0
        // k=j (j>=1): -A_d * x_j + I * x_{j+1} - B_d * u_j = 0

        for k in 0..h {
            let x_k1_off = k * STATE_DIM; // x_{k+1} column offset
            let u_k_off = n_x + k * n_u_step; // u_k column offset

            // I * x_{k+1}
            for i in 0..STATE_DIM {
                a_all[(row + i, x_k1_off + i)] = 1.0;
            }

            // -B_d * u_k
            for i in 0..STATE_DIM {
                for j in 0..n_u_step {
                    let val = b_d[(i, j)];
                    if val.abs() > 1e-15 {
                        a_all[(row + i, u_k_off + j)] = -val;
                    }
                }
            }

            if k == 0 {
                // b = A_d * x_0
                let ax0 = a_d * x0;
                for i in 0..STATE_DIM {
                    b_all[row + i] = ax0[i];
                }
            } else {
                // -A_d * x_k
                let x_k_off = (k - 1) * STATE_DIM;
                for i in 0..STATE_DIM {
                    for j in 0..STATE_DIM {
                        let val = a_d[(i, j)];
                        if val.abs() > 1e-15 {
                            a_all[(row + i, x_k_off + j)] = -val;
                        }
                    }
                }
                // b = 0 (already zero)
            }

            row += STATE_DIM;
        }

        // --- Swing foot equality constraints (f = 0) ---
        for (k, step_contacts) in contacts.iter().enumerate().take(h) {
            for (foot, &in_contact) in step_contacts.iter().enumerate() {
                if !in_contact {
                    let u_off = n_x + k * n_u_step + 3 * foot;
                    for j in 0..3 {
                        a_all[(row, u_off + j)] = 1.0;
                        b_all[row] = 0.0;
                        row += 1;
                    }
                }
            }
        }

        assert_eq!(row, n_eq, "Equality constraint count mismatch");

        // --- Friction cone inequality constraints ---
        // For Clarabel NonnegativeCone: A z + s = b with s >= 0 means A z <= b
        let mu = self.config.friction_coeff;
        let f_max = self.config.f_max;

        for (k, step_contacts) in contacts.iter().enumerate().take(h) {
            for (foot, &in_contact) in step_contacts.iter().enumerate() {
                if in_contact {
                    let fx_idx = n_x + k * n_u_step + 3 * foot;
                    let fy_idx = fx_idx + 1;
                    let fz_idx = fx_idx + 2;

                    // 1. fx - mu*fz <= 0
                    a_all[(row, fx_idx)] = 1.0;
                    a_all[(row, fz_idx)] = -mu;
                    b_all[row] = 0.0;
                    row += 1;

                    // 2. -fx - mu*fz <= 0
                    a_all[(row, fx_idx)] = -1.0;
                    a_all[(row, fz_idx)] = -mu;
                    b_all[row] = 0.0;
                    row += 1;

                    // 3. fy - mu*fz <= 0
                    a_all[(row, fy_idx)] = 1.0;
                    a_all[(row, fz_idx)] = -mu;
                    b_all[row] = 0.0;
                    row += 1;

                    // 4. -fy - mu*fz <= 0
                    a_all[(row, fy_idx)] = -1.0;
                    a_all[(row, fz_idx)] = -mu;
                    b_all[row] = 0.0;
                    row += 1;

                    // 5. -fz <= 0  (fz >= 0)
                    a_all[(row, fz_idx)] = -1.0;
                    b_all[row] = 0.0;
                    row += 1;

                    // 6. fz <= f_max
                    a_all[(row, fz_idx)] = 1.0;
                    b_all[row] = f_max;
                    row += 1;
                }
            }
        }

        assert_eq!(row, n_constraints, "Total constraint count mismatch");

        (a_all, b_all, n_eq, n_ineq)
    }
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

/// Convert a symmetric nalgebra `DMatrix<f64>` to upper-triangular `CscMatrix<f64>`.
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
    use nalgebra::Matrix3;

    fn test_config() -> MpcConfig {
        MpcConfig {
            horizon: 5,
            dt: 0.02,
            mass: 8.6,
            inertia: Matrix3::new(0.07, 0.0, 0.0, 0.0, 0.26, 0.0, 0.0, 0.0, 0.28),
            gravity: 9.81,
            friction_coeff: 0.6,
            f_max: 200.0,
            q_weights: [50.0, 50.0, 10.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0, 10.0, 10.0, 1.0],
            r_weight: 1e-4,
            max_solver_iters: 100,
        }
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
    fn standing_balance() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);

        let feet = quadruped_feet_standing();
        // All feet in contact for all horizon steps
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.35,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);

        assert!(solution.converged, "QP must converge for standing balance");

        // Forces should roughly balance gravity: sum of fz ≈ m*g = 8.6*9.81 ≈ 84.4N
        // Tolerance is generous because short horizon + control cost means forces
        // slightly undercompensate gravity (the MPC correctly trades state error vs effort).
        let total_fz: f64 = solution.forces.iter().map(|f| f.z).sum();
        assert_relative_eq!(total_fz, config.mass * config.gravity, epsilon = 15.0);

        // Horizontal forces should be near zero (no desired velocity)
        let total_fx: f64 = solution.forces.iter().map(|f| f.x.abs()).sum();
        let total_fy: f64 = solution.forces.iter().map(|f| f.y.abs()).sum();
        assert!(total_fx < 10.0, "Horizontal force fx should be small: {total_fx}");
        assert!(total_fy < 10.0, "Horizontal force fy should be small: {total_fy}");
    }

    #[test]
    fn forces_satisfy_friction_cone() {
        let config = test_config();
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::new(0.3, 0.0, 0.0),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::new(0.5, 0.0, 0.0),
            0.35,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        let mu = config.friction_coeff;
        for (i, f) in solution.forces.iter().enumerate() {
            // fz >= 0
            assert!(f.z >= -1e-3, "Foot {i}: fz={} must be >= 0", f.z);
            // |fx| <= mu * fz
            assert!(
                f.x.abs() <= mu * f.z + 1e-3,
                "Foot {i}: |fx|={} > mu*fz={}",
                f.x.abs(),
                mu * f.z
            );
            // |fy| <= mu * fz
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
            position: Vector3::new(0.0, 0.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();

        // Trot: FL+RR stance, FR+RL swing
        let contacts: Vec<Vec<bool>> =
            vec![vec![true, false, false, true]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.35,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        // Swing feet (1=FR, 2=RL) should have zero force
        assert!(solution.forces[1].norm() < 1e-3, "FR swing force should be ~0");
        assert!(solution.forces[2].norm() < 1e-3, "RL swing force should be ~0");

        // Stance feet (0=FL, 3=RR) should support the robot
        // 2 feet supporting full weight — tolerance generous for short horizon
        let stance_fz = solution.forces[0].z + solution.forces[3].z;
        assert_relative_eq!(stance_fz, config.mass * config.gravity, epsilon = 30.0);
    }

    #[test]
    fn solve_time_reasonable() {
        let config = MpcConfig {
            horizon: 10,
            ..test_config()
        };
        let solver = MpcSolver::new(config.clone());

        let state = BodyState {
            orientation: Vector3::zeros(),
            position: Vector3::new(0.0, 0.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::zeros(),
        };
        let x0 = state.to_state_vector(config.gravity);
        let feet = quadruped_feet_standing();
        let contacts: Vec<Vec<bool>> = vec![vec![true; 4]; config.horizon];

        let reference = ReferenceTrajectory::constant_velocity(
            &state,
            &Vector3::zeros(),
            0.35,
            0.0,
            config.horizon,
            config.dt,
            config.gravity,
        );

        let solution = solver.solve(&x0, &feet, &contacts, &reference);
        assert!(solution.converged);

        // Should solve in under 50ms (typically < 5ms)
        assert!(
            solution.solve_time_us < 50_000,
            "Solve took {}us, expected < 50000us",
            solution.solve_time_us
        );
    }
}
