//! Core data types for in-memory URDF representation.
//!
//! These types are the crate's canonical representation of a robot model,
//! independent of the XML parsing layer. They map closely to URDF concepts
//! but use Rust-native types.

use std::collections::HashMap;

use crate::error::UrdfError;

// ---------------------------------------------------------------------------
// JointType
// ---------------------------------------------------------------------------

/// URDF joint type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JointType {
    /// Rotation about a single axis, with position limits.
    Revolute,
    /// Unlimited rotation about a single axis.
    Continuous,
    /// Translation along an axis, with position limits.
    Prismatic,
    /// No relative motion between parent and child.
    Fixed,
    /// Unconstrained 6-DOF joint (rarely used).
    Floating,
    /// Translation along one axis with no rotation (rarely used).
    Planar,
}

impl JointType {
    /// Whether this joint type has actuatable degrees of freedom.
    pub const fn is_actuated(self) -> bool {
        matches!(self, Self::Revolute | Self::Continuous | Self::Prismatic)
    }
}

// ---------------------------------------------------------------------------
// JointLimits
// ---------------------------------------------------------------------------

/// Limits on a joint's motion, effort, and velocity.
#[derive(Debug, Clone, Default)]
pub struct JointLimits {
    /// Lower position limit (rad or m). `None` means unbounded.
    pub lower: Option<f32>,
    /// Upper position limit (rad or m). `None` means unbounded.
    pub upper: Option<f32>,
    /// Maximum effort (Nm or N).
    pub effort: f32,
    /// Maximum velocity (rad/s or m/s).
    pub velocity: f32,
}

// ---------------------------------------------------------------------------
// JointDynamics
// ---------------------------------------------------------------------------

/// Dynamic properties of a joint (damping and friction).
#[derive(Debug, Clone, Default)]
pub struct JointDynamics {
    /// Viscous damping coefficient (Nm·s/rad).
    pub damping: f32,
    /// Coulomb friction torque (Nm).
    pub friction: f32,
}

// ---------------------------------------------------------------------------
// Origin
// ---------------------------------------------------------------------------

/// A 3D pose specified as position + roll-pitch-yaw.
#[derive(Debug, Clone)]
pub struct Origin {
    /// Translation `[x, y, z]` in meters.
    pub xyz: [f32; 3],
    /// Rotation `[roll, pitch, yaw]` in radians.
    pub rpy: [f32; 3],
}

impl Default for Origin {
    fn default() -> Self {
        Self {
            xyz: [0.0; 3],
            rpy: [0.0; 3],
        }
    }
}

// ---------------------------------------------------------------------------
// Inertial
// ---------------------------------------------------------------------------

/// Inertial properties of a link.
#[derive(Debug, Clone)]
pub struct Inertial {
    /// Origin of the inertial frame relative to the link frame.
    pub origin: Origin,
    /// Mass in kilograms.
    pub mass: f32,
    /// Inertia tensor elements `[ixx, ixy, ixz, iyy, iyz, izz]`.
    pub inertia: [f32; 6],
}

impl Default for Inertial {
    fn default() -> Self {
        Self {
            origin: Origin::default(),
            mass: 0.0,
            inertia: [0.0; 6],
        }
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Geometric shape used for visual or collision elements.
#[derive(Debug, Clone)]
pub enum Geometry {
    Sphere { radius: f32 },
    Box { size: [f32; 3] },
    Cylinder { radius: f32, length: f32 },
    Mesh { filename: String, scale: [f32; 3] },
}

// ---------------------------------------------------------------------------
// Material
// ---------------------------------------------------------------------------

/// Visual material for a link.
#[derive(Debug, Clone)]
pub struct Material {
    /// Material name.
    pub name: String,
    /// RGBA color `[r, g, b, a]`, each in `0.0..=1.0`.
    pub color: Option<[f32; 4]>,
}

// ---------------------------------------------------------------------------
// Visual / Collision
// ---------------------------------------------------------------------------

/// A visual element of a link.
#[derive(Debug, Clone)]
pub struct Visual {
    pub origin: Origin,
    pub geometry: Geometry,
    pub material: Option<Material>,
}

/// A collision element of a link.
#[derive(Debug, Clone)]
pub struct Collision {
    pub origin: Origin,
    pub geometry: Geometry,
}

// ---------------------------------------------------------------------------
// LinkData
// ---------------------------------------------------------------------------

/// In-memory representation of a URDF link.
#[derive(Debug, Clone)]
pub struct LinkData {
    /// Link name.
    pub name: String,
    /// Inertial properties (mass, inertia tensor).
    pub inertial: Option<Inertial>,
    /// Visual geometries.
    pub visuals: Vec<Visual>,
    /// Collision geometries.
    pub collisions: Vec<Collision>,
}

impl LinkData {
    /// Create a link with only a name (no geometry or inertia).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inertial: None,
            visuals: Vec::new(),
            collisions: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// JointData
// ---------------------------------------------------------------------------

/// In-memory representation of a URDF joint.
#[derive(Debug, Clone)]
pub struct JointData {
    /// Joint name.
    pub name: String,
    /// Joint type.
    pub joint_type: JointType,
    /// Parent link name.
    pub parent: String,
    /// Child link name.
    pub child: String,
    /// Joint origin relative to parent link.
    pub origin: Origin,
    /// Joint axis (unit vector, default `[0, 0, 1]`).
    pub axis: [f32; 3],
    /// Motion limits.
    pub limits: JointLimits,
    /// Dynamic properties.
    pub dynamics: JointDynamics,
}

// ---------------------------------------------------------------------------
// RobotModel
// ---------------------------------------------------------------------------

/// Complete in-memory representation of a URDF robot.
///
/// Constructed by the parser and consumed by the spawner. Contains
/// the full kinematic tree: links, joints, and root link name.
#[derive(Debug, Clone)]
pub struct RobotModel {
    /// Robot name.
    pub name: String,
    /// All links, keyed by name.
    pub links: HashMap<String, LinkData>,
    /// All joints, keyed by name.
    pub joints: HashMap<String, JointData>,
    /// Name of the root link (the one never referenced as a child).
    pub root_link: String,
}

impl RobotModel {
    /// Get a link by name.
    pub fn link(&self, name: &str) -> Result<&LinkData, UrdfError> {
        self.links
            .get(name)
            .ok_or_else(|| UrdfError::MissingLink(name.into()))
    }

    /// Get a joint by name.
    pub fn joint(&self, name: &str) -> Result<&JointData, UrdfError> {
        self.joints
            .get(name)
            .ok_or_else(|| UrdfError::MissingJoint(name.into()))
    }

    /// Iterate over actuatable joints (revolute, continuous, prismatic).
    pub fn actuated_joints(&self) -> impl Iterator<Item = &JointData> {
        self.joints.values().filter(|j| j.joint_type.is_actuated())
    }

    /// Number of actuatable degrees of freedom.
    pub fn dof(&self) -> usize {
        self.actuated_joints().count()
    }

    /// Names of all joints, sorted alphabetically.
    pub fn joint_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.joints.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Names of actuated joints, sorted alphabetically.
    pub fn actuated_joint_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.actuated_joints().map(|j| j.name.as_str()).collect();
        names.sort_unstable();
        names
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> RobotModel {
        let mut links = HashMap::new();
        links.insert("base".into(), LinkData::new("base"));
        links.insert("link1".into(), LinkData::new("link1"));
        links.insert("link2".into(), LinkData::new("link2"));

        let mut joints = HashMap::new();
        joints.insert(
            "joint1".into(),
            JointData {
                name: "joint1".into(),
                joint_type: JointType::Revolute,
                parent: "base".into(),
                child: "link1".into(),
                origin: Origin::default(),
                axis: [0.0, 0.0, 1.0],
                limits: JointLimits {
                    lower: Some(-1.57),
                    upper: Some(1.57),
                    effort: 100.0,
                    velocity: 5.0,
                },
                dynamics: JointDynamics::default(),
            },
        );
        joints.insert(
            "joint2".into(),
            JointData {
                name: "joint2".into(),
                joint_type: JointType::Fixed,
                parent: "link1".into(),
                child: "link2".into(),
                origin: Origin::default(),
                axis: [0.0, 0.0, 1.0],
                limits: JointLimits::default(),
                dynamics: JointDynamics::default(),
            },
        );

        RobotModel {
            name: "test_robot".into(),
            links,
            joints,
            root_link: "base".into(),
        }
    }

    // -- JointType --

    #[test]
    fn joint_type_is_actuated() {
        assert!(JointType::Revolute.is_actuated());
        assert!(JointType::Continuous.is_actuated());
        assert!(JointType::Prismatic.is_actuated());
        assert!(!JointType::Fixed.is_actuated());
        assert!(!JointType::Floating.is_actuated());
        assert!(!JointType::Planar.is_actuated());
    }

    // -- Origin --

    #[test]
    fn origin_default_is_zero() {
        let o = Origin::default();
        assert!(o.xyz.iter().all(|v| v.abs() < f32::EPSILON));
        assert!(o.rpy.iter().all(|v| v.abs() < f32::EPSILON));
    }

    // -- Inertial --

    #[test]
    fn inertial_default_is_zero() {
        let i = Inertial::default();
        assert!((i.mass).abs() < f32::EPSILON);
        assert!(i.inertia.iter().all(|v| v.abs() < f32::EPSILON));
    }

    // -- LinkData --

    #[test]
    fn link_data_new() {
        let link = LinkData::new("arm");
        assert_eq!(link.name, "arm");
        assert!(link.inertial.is_none());
        assert!(link.visuals.is_empty());
        assert!(link.collisions.is_empty());
    }

    // -- RobotModel --

    #[test]
    fn model_link_lookup() {
        let model = sample_model();
        assert!(model.link("base").is_ok());
        assert!(model.link("missing").is_err());
    }

    #[test]
    fn model_joint_lookup() {
        let model = sample_model();
        assert!(model.joint("joint1").is_ok());
        assert!(model.joint("missing").is_err());
    }

    #[test]
    fn model_dof() {
        let model = sample_model();
        assert_eq!(model.dof(), 1); // only joint1 is revolute
    }

    #[test]
    fn model_actuated_joint_names() {
        let model = sample_model();
        let names = model.actuated_joint_names();
        assert_eq!(names, vec!["joint1"]);
    }

    #[test]
    fn model_joint_names_sorted() {
        let model = sample_model();
        let names = model.joint_names();
        assert_eq!(names, vec!["joint1", "joint2"]);
    }

    #[test]
    fn joint_limits_default() {
        let lim = JointLimits::default();
        assert!(lim.lower.is_none());
        assert!(lim.upper.is_none());
        assert!((lim.effort).abs() < f32::EPSILON);
        assert!((lim.velocity).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_dynamics_default() {
        let dyn_ = JointDynamics::default();
        assert!((dyn_.damping).abs() < f32::EPSILON);
        assert!((dyn_.friction).abs() < f32::EPSILON);
    }
}
