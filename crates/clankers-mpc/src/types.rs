//! Core types for the MPC pipeline.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Number of centroidal dynamics states: [Θ(3), p(3), ω(3), v(3), g(1)] = 13.
pub const STATE_DIM: usize = 13;

/// MPC solver configuration.
#[derive(Clone, Debug)]
pub struct MpcConfig {
    /// Prediction horizon (number of future steps).
    pub horizon: usize,
    /// MPC timestep in seconds (e.g., 0.02 for 50 Hz).
    pub dt: f64,
    /// Total robot mass in kg.
    pub mass: f64,
    /// Body-frame inertia tensor (3x3, about CoM).
    pub inertia: Matrix3<f64>,
    /// Gravitational acceleration magnitude (positive, e.g., 9.81).
    pub gravity: f64,
    /// Ground friction coefficient (Coulomb).
    pub friction_coeff: f64,
    /// Maximum normal (vertical) force per foot in N.
    pub f_max: f64,
    /// State error cost weights for the 12 controllable states.
    /// Order: [roll, pitch, yaw, px, py, pz, wx, wy, wz, vx, vy, vz].
    pub q_weights: [f64; 12],
    /// Control effort cost weight (applied uniformly to all force components).
    pub r_weight: f64,
    /// Maximum QP solver iterations.
    pub max_solver_iters: u32,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon: 10,
            dt: 0.02,
            mass: 9.0,
            // Composite inertia: body (5kg) + 4 legs (1kg each at hip offsets).
            // Computed via parallel axis theorem from URDF link inertias.
            inertia: Matrix3::new(
                0.048, 0.0, 0.0, 0.0, 0.122, 0.0, 0.0, 0.0, 0.135,
            ),
            gravity: 9.81,
            friction_coeff: 0.4,
            f_max: 120.0,
            // Tuned via systematic sweep (see .delegate/work/20260227-aggressive-velocity-tuning/).
            // Key changes from initial values:
            //   pz: 50→20 (balance height maintenance vs velocity budget),
            //   vx/vy: 20→150 (prioritize velocity tracking),
            //   r_weight: 1e-6→1e-7 (allow more aggressive forces).
            // Validated 5/5 runs stable: 0.10-0.12 m/s at 0.3 target (was 0.05),
            // max roll 7.1 deg. Config K3 in sweep results.
            q_weights: [
                25.0, 25.0, 10.0,
                5.0, 5.0, 20.0,
                1.0, 1.0, 0.3,
                150.0, 150.0, 5.0,
            ],
            r_weight: 1e-7,
            max_solver_iters: 100,
        }
    }
}

/// Body state in the centroidal dynamics model.
#[derive(Clone, Debug)]
pub struct BodyState {
    /// Euler angles: roll, pitch, yaw (radians).
    pub orientation: Vector3<f64>,
    /// Center of mass position in world frame (meters).
    pub position: Vector3<f64>,
    /// Angular velocity in world frame (rad/s).
    pub angular_velocity: Vector3<f64>,
    /// Linear velocity in world frame (m/s).
    pub linear_velocity: Vector3<f64>,
}

impl BodyState {
    /// Pack into the 13-element state vector: [Θ, p, ω, v, g].
    pub fn to_state_vector(&self, gravity: f64) -> DVector<f64> {
        let mut x = DVector::zeros(STATE_DIM);
        x.fixed_rows_mut::<3>(0).copy_from(&self.orientation);
        x.fixed_rows_mut::<3>(3).copy_from(&self.position);
        x.fixed_rows_mut::<3>(6).copy_from(&self.angular_velocity);
        x.fixed_rows_mut::<3>(9).copy_from(&self.linear_velocity);
        x[12] = gravity;
        x
    }

    /// Unpack from a 13-element state vector.
    pub fn from_state_vector(x: &DVector<f64>) -> Self {
        Self {
            orientation: x.fixed_rows::<3>(0).into(),
            position: x.fixed_rows::<3>(3).into(),
            angular_velocity: x.fixed_rows::<3>(6).into(),
            linear_velocity: x.fixed_rows::<3>(9).into(),
        }
    }
}

/// Contact plan for the MPC horizon.
#[derive(Clone, Debug)]
pub struct ContactPlan {
    /// Contact flags: `contacts[step][foot]` is true if foot is in stance.
    pub contacts: Vec<Vec<bool>>,
    /// Current foot positions in world frame (one per foot).
    pub foot_positions: Vec<Vector3<f64>>,
}

/// Result from the MPC solver.
#[derive(Clone, Debug)]
pub struct MpcSolution {
    /// Optimal foot forces for the current timestep (one `Vector3` per foot).
    pub forces: Vec<Vector3<f64>>,
    /// Full optimal force trajectory over the horizon.
    pub force_trajectory: DVector<f64>,
    /// Predicted state trajectory over the horizon.
    pub state_trajectory: DVector<f64>,
    /// Whether the QP solver converged.
    pub converged: bool,
    /// Solve time in microseconds.
    pub solve_time_us: u64,
}

/// Reference trajectory for the MPC: desired state at each horizon step.
#[derive(Clone, Debug)]
pub struct ReferenceTrajectory {
    /// Stacked reference states: [x_ref_1, ..., x_ref_H] each `STATE_DIM`.
    pub states: DVector<f64>,
}

impl ReferenceTrajectory {
    /// Build a constant-velocity reference from the current state and desired velocity.
    ///
    /// The reference maintains current orientation (zero roll/pitch, current yaw),
    /// desired height, and integrates position at the desired velocity.
    pub fn constant_velocity(
        current: &BodyState,
        desired_velocity: &Vector3<f64>,
        desired_height: f64,
        desired_yaw: f64,
        horizon: usize,
        dt: f64,
        gravity: f64,
    ) -> Self {
        let mut states = DVector::zeros(STATE_DIM * horizon);

        for k in 0..horizon {
            let t = (k + 1) as f64 * dt;
            let offset = k * STATE_DIM;

            // Desired orientation: zero roll/pitch, target yaw
            states[offset] = 0.0; // roll
            states[offset + 1] = 0.0; // pitch
            states[offset + 2] = desired_yaw;

            // Desired position: integrate velocity from current
            states[offset + 3] = current.position.x + desired_velocity.x * t;
            states[offset + 4] = current.position.y + desired_velocity.y * t;
            states[offset + 5] = desired_height;

            // Desired angular velocity: zero
            states[offset + 6] = 0.0;
            states[offset + 7] = 0.0;
            states[offset + 8] = 0.0;

            // Desired linear velocity
            states[offset + 9] = desired_velocity.x;
            states[offset + 10] = desired_velocity.y;
            states[offset + 11] = 0.0; // no vertical velocity

            // Gravity constant
            states[offset + 12] = gravity;
        }

        Self { states }
    }
}

/// Stacked QP matrices for the MPC problem.
#[derive(Clone, Debug)]
pub struct QpProblem {
    /// Cost Hessian (upper triangular).
    pub p_matrix: DMatrix<f64>,
    /// Cost linear term.
    pub q_vector: DVector<f64>,
    /// Constraint matrix (equalities stacked on top of inequalities).
    pub a_matrix: DMatrix<f64>,
    /// Constraint bounds.
    pub b_vector: DVector<f64>,
    /// Number of equality constraints (dynamics).
    pub n_eq: usize,
    /// Number of inequality constraints (friction + limits).
    pub n_ineq: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_state_roundtrip() {
        let state = BodyState {
            orientation: Vector3::new(0.1, -0.05, 0.3),
            position: Vector3::new(1.0, 2.0, 0.35),
            angular_velocity: Vector3::new(0.0, 0.0, 0.5),
            linear_velocity: Vector3::new(0.3, 0.0, 0.0),
        };
        let v = state.to_state_vector(9.81);
        assert_eq!(v.len(), 13);
        assert!((v[12] - 9.81).abs() < 1e-10);

        let recovered = BodyState::from_state_vector(&v);
        assert!((recovered.orientation - state.orientation).norm() < 1e-10);
        assert!((recovered.position - state.position).norm() < 1e-10);
    }

    #[test]
    fn reference_trajectory_constant_velocity() {
        let current = BodyState {
            orientation: Vector3::new(0.0, 0.0, 0.0),
            position: Vector3::new(0.0, 0.0, 0.35),
            angular_velocity: Vector3::zeros(),
            linear_velocity: Vector3::new(0.3, 0.0, 0.0),
        };
        let desired_vel = Vector3::new(0.5, 0.0, 0.0);
        let traj = ReferenceTrajectory::constant_velocity(
            &current, &desired_vel, 0.35, 0.0, 10, 0.02, 9.81,
        );
        assert_eq!(traj.states.len(), 130);

        // Check step 5 position reference: p_x = 0.0 + 0.5 * 5 * 0.02 = 0.05
        let step5_px = traj.states[4 * STATE_DIM + 3];
        assert!((step5_px - 0.05).abs() < 1e-10);
    }
}
