//! Shared URDF definitions and helpers for Clankers examples.

/// Simple inverted pendulum: 1 revolute joint.
pub const PENDULUM_URDF: &str = include_str!("../urdf/pendulum.urdf");

/// 2-DOF planar arm: shoulder + elbow revolute joints, fixed end-effector.
pub const TWO_LINK_ARM_URDF: &str = include_str!("../urdf/two_link_arm.urdf");

/// Classic cart-pole: prismatic cart + revolute pole.
pub const CARTPOLE_URDF: &str = include_str!("../urdf/cartpole.urdf");

/// 6-DOF articulated arm: 6 revolute joints with alternating axes.
pub const SIX_DOF_ARM_URDF: &str = include_str!("../urdf/six_dof_arm.urdf");
