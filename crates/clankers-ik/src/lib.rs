//! Inverse kinematics solver for Clankers robots.
//!
//! Provides forward kinematics, geometric Jacobian computation, and
//! Damped Least Squares (Levenberg-Marquardt) IK solving for arbitrary
//! kinematic chains defined by URDF robot models.
//!
//! # Architecture
//!
//! ```text
//! RobotModel ──► KinematicChain ──► IkSolver ──► joint angles
//! ```
//!
//! The [`KinematicChain`] is extracted from a [`RobotModel`](clankers_urdf::RobotModel)
//! at initialization time. The solver then operates on this chain, taking
//! target poses and current joint positions as input, and producing
//! target joint positions as output.

pub mod chain;
pub mod solver;

pub use chain::KinematicChain;
pub use solver::{DlsConfig, DlsSolver, IkResult, IkTarget};
