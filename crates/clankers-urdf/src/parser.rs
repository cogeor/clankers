//! URDF XML parsing using `urdf-rs`.
//!
//! Converts `urdf_rs` types into the crate's canonical [`RobotModel`]
//! representation.

// All conversions from urdf-rs f64 → clankers f32 are intentional truncations.
#![allow(clippy::cast_possible_truncation)]

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::error::UrdfError;
use crate::types::{
    Collision, Geometry, Inertial, JointData, JointDynamics, JointLimits, JointType, LinkData,
    Material, Origin, RobotModel, Visual,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a URDF file from disk into a [`RobotModel`].
pub fn parse_file(path: impl AsRef<Path>) -> Result<RobotModel, UrdfError> {
    let path = path.as_ref();
    let content = std::fs::read_to_string(path).map_err(|e| UrdfError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    parse_string(&content)
}

/// Parse a URDF XML string into a [`RobotModel`].
pub fn parse_string(xml: &str) -> Result<RobotModel, UrdfError> {
    let robot = urdf_rs::read_from_string(xml).map_err(|e| UrdfError::Parse(e.to_string()))?;
    convert_robot(&robot)
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn convert_robot(robot: &urdf_rs::Robot) -> Result<RobotModel, UrdfError> {
    let links: HashMap<String, LinkData> = robot
        .links
        .iter()
        .map(|l| (l.name.clone(), convert_link(l)))
        .collect();

    let joints: HashMap<String, JointData> = robot
        .joints
        .iter()
        .map(|j| convert_joint(j).map(|jd| (jd.name.clone(), jd)))
        .collect::<Result<_, _>>()?;

    // Root link = a link that is never a child of any joint.
    let child_links: HashSet<&str> = joints.values().map(|j| j.child.as_str()).collect();
    let root_link = links
        .keys()
        .find(|name| !child_links.contains(name.as_str()))
        .ok_or(UrdfError::NoRootLink)?
        .clone();

    Ok(RobotModel {
        name: robot.name.clone(),
        links,
        joints,
        root_link,
    })
}

fn convert_link(link: &urdf_rs::Link) -> LinkData {
    LinkData {
        name: link.name.clone(),
        inertial: Some(convert_inertial(&link.inertial)),
        visuals: link.visual.iter().map(convert_visual).collect(),
        collisions: link.collision.iter().map(convert_collision).collect(),
    }
}

fn convert_joint(joint: &urdf_rs::Joint) -> Result<JointData, UrdfError> {
    let joint_type = convert_joint_type(&joint.joint_type)?;

    let dynamics = joint
        .dynamics
        .as_ref()
        .map(convert_dynamics)
        .unwrap_or_default();

    Ok(JointData {
        name: joint.name.clone(),
        joint_type,
        parent: joint.parent.link.clone(),
        child: joint.child.link.clone(),
        origin: convert_pose(&joint.origin),
        axis: vec3_to_f32(&joint.axis.xyz),
        limits: convert_limits(&joint.limit),
        dynamics,
    })
}

fn convert_joint_type(jt: &urdf_rs::JointType) -> Result<JointType, UrdfError> {
    match jt {
        urdf_rs::JointType::Revolute => Ok(JointType::Revolute),
        urdf_rs::JointType::Continuous => Ok(JointType::Continuous),
        urdf_rs::JointType::Prismatic => Ok(JointType::Prismatic),
        urdf_rs::JointType::Fixed => Ok(JointType::Fixed),
        urdf_rs::JointType::Floating => Ok(JointType::Floating),
        urdf_rs::JointType::Planar => Ok(JointType::Planar),
        urdf_rs::JointType::Spherical => Err(UrdfError::UnsupportedJointType("Spherical".into())),
    }
}

fn convert_limits(limit: &urdf_rs::JointLimit) -> JointLimits {
    // urdf-rs defaults lower/upper to 0.0 for joints without limits.
    // We map 0.0 == 0.0 (both zero) as "no position limits".
    let has_limits = (limit.lower - limit.upper).abs() > f64::EPSILON;
    JointLimits {
        lower: if has_limits {
            Some(limit.lower as f32)
        } else {
            None
        },
        upper: if has_limits {
            Some(limit.upper as f32)
        } else {
            None
        },
        effort: limit.effort as f32,
        velocity: limit.velocity as f32,
    }
}

const fn convert_dynamics(dyn_: &urdf_rs::Dynamics) -> JointDynamics {
    JointDynamics {
        damping: dyn_.damping as f32,
        friction: dyn_.friction as f32,
    }
}

fn convert_pose(pose: &urdf_rs::Pose) -> Origin {
    Origin {
        xyz: vec3_to_f32(&pose.xyz),
        rpy: vec3_to_f32(&pose.rpy),
    }
}

fn convert_inertial(inertial: &urdf_rs::Inertial) -> Inertial {
    let i = &inertial.inertia;
    Inertial {
        origin: convert_pose(&inertial.origin),
        mass: inertial.mass.value as f32,
        inertia: [
            i.ixx as f32,
            i.ixy as f32,
            i.ixz as f32,
            i.iyy as f32,
            i.iyz as f32,
            i.izz as f32,
        ],
    }
}

fn convert_visual(visual: &urdf_rs::Visual) -> Visual {
    Visual {
        origin: convert_pose(&visual.origin),
        geometry: convert_geometry(&visual.geometry),
        material: visual.material.as_ref().map(convert_material),
    }
}

fn convert_collision(collision: &urdf_rs::Collision) -> Collision {
    Collision {
        origin: convert_pose(&collision.origin),
        geometry: convert_geometry(&collision.geometry),
    }
}

fn convert_geometry(geom: &urdf_rs::Geometry) -> Geometry {
    match geom {
        urdf_rs::Geometry::Sphere { radius } => Geometry::Sphere {
            radius: *radius as f32,
        },
        urdf_rs::Geometry::Box { size } => Geometry::Box {
            size: vec3_to_f32(size),
        },
        urdf_rs::Geometry::Cylinder { radius, length }
        | urdf_rs::Geometry::Capsule { radius, length } => Geometry::Cylinder {
            radius: *radius as f32,
            length: *length as f32,
        },
        urdf_rs::Geometry::Mesh { filename, scale } => Geometry::Mesh {
            filename: filename.clone(),
            scale: scale.as_ref().map_or([1.0, 1.0, 1.0], |s| vec3_to_f32(s)),
        },
    }
}

fn convert_material(mat: &urdf_rs::Material) -> Material {
    Material {
        name: mat.name.clone(),
        color: mat.color.as_ref().map(|c| {
            [
                c.rgba[0] as f32,
                c.rgba[1] as f32,
                c.rgba[2] as f32,
                c.rgba[3] as f32,
            ]
        }),
    }
}

const fn vec3_to_f32(v: &[f64; 3]) -> [f32; 3] {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_URDF: &str = r#"
        <robot name="test_robot">
            <link name="base_link"/>
        </robot>
    "#;

    const TWO_LINK_URDF: &str = r#"
        <robot name="two_link">
            <link name="base_link">
                <inertial>
                    <mass value="1.0"/>
                    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
                </inertial>
                <visual>
                    <geometry>
                        <cylinder radius="0.05" length="0.5"/>
                    </geometry>
                </visual>
            </link>
            <link name="child_link">
                <visual>
                    <geometry>
                        <sphere radius="0.1"/>
                    </geometry>
                </visual>
                <collision>
                    <geometry>
                        <sphere radius="0.1"/>
                    </geometry>
                </collision>
            </link>
            <joint name="joint1" type="revolute">
                <parent link="base_link"/>
                <child link="child_link"/>
                <origin xyz="0 0 0.5" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="5"/>
                <dynamics damping="0.5" friction="0.1"/>
            </joint>
        </robot>
    "#;

    const MULTI_JOINT_URDF: &str = r#"
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <link name="link2"/>
            <link name="link3"/>
            <joint name="joint1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" effort="50" velocity="2"/>
            </joint>
            <joint name="joint2" type="continuous">
                <parent link="link1"/>
                <child link="link2"/>
                <axis xyz="0 1 0"/>
            </joint>
            <joint name="fixed_end" type="fixed">
                <parent link="link2"/>
                <child link="link3"/>
            </joint>
        </robot>
    "#;

    // -- parse_string --

    #[test]
    fn parse_minimal_urdf() {
        let model = parse_string(MINIMAL_URDF).unwrap();
        assert_eq!(model.name, "test_robot");
        assert_eq!(model.links.len(), 1);
        assert!(model.joints.is_empty());
        assert_eq!(model.root_link, "base_link");
    }

    #[test]
    fn parse_two_link_robot() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        assert_eq!(model.name, "two_link");
        assert_eq!(model.links.len(), 2);
        assert_eq!(model.joints.len(), 1);
        assert_eq!(model.root_link, "base_link");
    }

    #[test]
    fn parse_multi_joint_robot() {
        let model = parse_string(MULTI_JOINT_URDF).unwrap();
        assert_eq!(model.name, "arm");
        assert_eq!(model.links.len(), 4);
        assert_eq!(model.joints.len(), 3);
        assert_eq!(model.dof(), 2); // revolute + continuous, not fixed
        assert_eq!(model.root_link, "base");
    }

    // -- Joint data --

    #[test]
    fn joint_type_parsed_correctly() {
        let model = parse_string(MULTI_JOINT_URDF).unwrap();
        assert_eq!(
            model.joint("joint1").unwrap().joint_type,
            JointType::Revolute
        );
        assert_eq!(
            model.joint("joint2").unwrap().joint_type,
            JointType::Continuous
        );
        assert_eq!(
            model.joint("fixed_end").unwrap().joint_type,
            JointType::Fixed
        );
    }

    #[test]
    fn joint_limits_parsed() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let joint = model.joint("joint1").unwrap();
        let lim = &joint.limits;
        assert!((lim.lower.unwrap() - (-1.57)).abs() < 0.01);
        assert!((lim.upper.unwrap() - 1.57).abs() < 0.01);
        assert!((lim.effort - 100.0).abs() < f32::EPSILON);
        assert!((lim.velocity - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_dynamics_parsed() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let joint = model.joint("joint1").unwrap();
        assert!((joint.dynamics.damping - 0.5).abs() < f32::EPSILON);
        assert!((joint.dynamics.friction - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn joint_axis_parsed() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let joint = model.joint("joint1").unwrap();
        assert!((joint.axis[2] - 1.0).abs() < f32::EPSILON); // z-axis
    }

    #[test]
    fn joint_origin_parsed() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let joint = model.joint("joint1").unwrap();
        assert!((joint.origin.xyz[2] - 0.5).abs() < f32::EPSILON); // z = 0.5
    }

    #[test]
    fn joint_parent_child() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let joint = model.joint("joint1").unwrap();
        assert_eq!(joint.parent, "base_link");
        assert_eq!(joint.child, "child_link");
    }

    // -- Link data --

    #[test]
    fn link_inertial_parsed() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let link = model.link("base_link").unwrap();
        let inertial = link.inertial.as_ref().unwrap();
        assert!((inertial.mass - 1.0).abs() < f32::EPSILON);
        assert!((inertial.inertia[0] - 0.01).abs() < 0.001); // ixx
    }

    #[test]
    fn link_visual_geometry() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let link = model.link("base_link").unwrap();
        assert_eq!(link.visuals.len(), 1);
        match &link.visuals[0].geometry {
            Geometry::Cylinder { radius, length } => {
                assert!((radius - 0.05).abs() < f32::EPSILON);
                assert!((length - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("expected Cylinder, got {other:?}"),
        }
    }

    #[test]
    fn link_collision_geometry() {
        let model = parse_string(TWO_LINK_URDF).unwrap();
        let link = model.link("child_link").unwrap();
        assert_eq!(link.collisions.len(), 1);
        match &link.collisions[0].geometry {
            Geometry::Sphere { radius } => {
                assert!((radius - 0.1).abs() < f32::EPSILON);
            }
            other => panic!("expected Sphere, got {other:?}"),
        }
    }

    // -- Continuous joint (no position limits) --

    #[test]
    fn continuous_joint_has_no_position_limits() {
        let model = parse_string(MULTI_JOINT_URDF).unwrap();
        let joint = model.joint("joint2").unwrap();
        assert!(joint.limits.lower.is_none());
        assert!(joint.limits.upper.is_none());
    }

    // -- Error cases --

    #[test]
    fn parse_invalid_xml() {
        let result = parse_string("<not valid urdf>");
        assert!(result.is_err());
    }

    #[test]
    fn parse_file_not_found() {
        let result = parse_file("/nonexistent/robot.urdf");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UrdfError::Io { .. }));
    }
}
