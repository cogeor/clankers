//! URDF parsing and robot model representation for Clankers.
//!
//! Provides types for representing a robot's kinematic tree (links, joints,
//! geometry), parsing URDF XML files, and spawning Bevy entities with
//! actuator components.

pub mod error;
pub mod parser;
pub mod spawner;
pub mod types;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use error::UrdfError;
pub use parser::{parse_file, parse_string};
pub use spawner::{JointName, SpawnedRobot, spawn_robot};
pub use types::{
    Collision, Geometry, Inertial, JointData, JointDynamics, JointLimits, JointType, LinkData,
    Material, Origin, RobotModel, Visual,
};
