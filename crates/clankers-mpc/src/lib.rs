//! Centroidal convex MPC and whole-body controller for legged locomotion.
//!
//! This crate implements a classical Model Predictive Control pipeline for
//! quadruped (and general multi-legged) robots. The pipeline:
//!
//! 1. **Gait Scheduler** — generates contact sequences (trot, walk, stand)
//! 2. **Centroidal MPC** — solves a QP for optimal ground reaction forces
//! 3. **Whole Body Controller** — maps foot forces to joint torques via J^T
//! 4. **Swing Leg Planner** — generates foot trajectories for airborne legs
//!
//! # Architecture
//!
//! The MPC treats the robot as a single rigid body ("centroidal dynamics")
//! with massless legs. This simplification yields a convex QP that can be
//! solved in sub-millisecond time using the Clarabel solver.
//!
//! The WBC then maps the optimal forces back to joint torques using the
//! contact Jacobian transpose, while swing legs track Bezier trajectories
//! using inverse kinematics.

pub mod centroidal;
pub mod gait;
pub mod solver;
pub mod swing;
pub mod types;
pub mod wbc;

pub use centroidal::{build_continuous_dynamics, discretize};
pub use gait::{GaitScheduler, GaitType};
pub use solver::MpcSolver;
pub use swing::{SwingConfig, raibert_foot_target, swing_foot_position};
pub use types::{BodyState, ContactPlan, MpcConfig, MpcSolution, ReferenceTrajectory};
pub use wbc::{LegCommand, compute_leg_jacobian, jacobian_transpose_torques};
