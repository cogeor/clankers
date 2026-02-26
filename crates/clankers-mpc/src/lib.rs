//! Centroidal convex MPC and whole-body controller for legged locomotion.
//!
//! This crate implements the MIT Cheetah convex MPC pipeline (Di Carlo et al.,
//! IROS 2018) for quadruped (and general multi-legged) robots:
//!
//! 1. **Gait Scheduler** — generates contact sequences (trot, walk, stand)
//! 2. **Centroidal MPC** — solves a condensed QP for optimal ground reaction forces
//! 3. **Whole Body Controller** — maps foot forces to joint torques via J^T
//! 4. **Swing Leg Planner** — generates min-jerk foot trajectories
//!
//! # Architecture
//!
//! The MPC treats the robot as a single rigid body ("centroidal dynamics")
//! with massless legs. The inertia is rotated to world frame using yaw-only
//! rotation, and dynamics are discretized via matrix exponential for accuracy.
//!
//! The condensed QP eliminates state variables, solving only for ground
//! reaction forces (12*H variables for 4 feet over H horizon steps).

pub mod centroidal;
pub mod gait;
#[cfg(feature = "bevy")]
pub mod plugin;
pub mod solver;
pub mod swing;
pub mod types;
pub mod wbc;

pub use centroidal::{build_continuous_dynamics, discretize_euler, discretize_matrix_exp};
pub use gait::{GaitScheduler, GaitType};
#[cfg(feature = "bevy")]
pub use plugin::{
    ClankersMpcPlugin, LegConfig, MpcPipelineConfig, MpcPipelineState, build_leg_configs,
    body_state_from_transform,
};
pub use solver::MpcSolver;
pub use swing::{SwingConfig, raibert_foot_target, swing_foot_position, swing_foot_velocity};
pub use types::{BodyState, ContactPlan, MpcConfig, MpcSolution, ReferenceTrajectory};
pub use wbc::{
    LegCommand, compute_leg_jacobian, jacobian_transpose_torques, stance_damping_torques,
    transform_frames_to_world,
};
