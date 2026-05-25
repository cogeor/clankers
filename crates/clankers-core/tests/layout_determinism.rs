//! Integration tests for `JointLayout` determinism.
//!
//! Loop 01 / WS1 PR1 — see `docs/plans/WS1-plan.md` § 6.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use clankers_core::layout::{JointKind, JointLayout, JointSpec, JointSpecLimits};
use clankers_core::schema::SchemaMismatch;
use clankers_urdf::parser::parse_string;
use clankers_urdf::types::{JointType, RobotModel};

/// In-test fixture mirroring `crates/clankers-urdf/src/spawner.rs` `ARM_URDF`.
const FIXTURE_URDF: &str = r#"
    <robot name="arm">
        <link name="base"/>
        <link name="link1"/>
        <link name="link2"/>
        <link name="link3"/>
        <joint name="shoulder" type="revolute">
            <parent link="base"/>
            <child link="link1"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1.57" upper="1.57" effort="50" velocity="3"/>
            <dynamics damping="0.5" friction="0.1"/>
        </joint>
        <joint name="elbow" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <axis xyz="0 1 0"/>
            <limit lower="-2.0" upper="2.0" effort="30" velocity="5"/>
        </joint>
        <joint name="wrist_fixed" type="fixed">
            <parent link="link2"/>
            <child link="link3"/>
        </joint>
    </robot>
"#;

/// Convert `clankers_urdf::types::JointType` -> `clankers_core::layout::JointKind`.
const fn map_kind(t: JointType) -> JointKind {
    match t {
        JointType::Revolute => JointKind::Revolute,
        JointType::Continuous => JointKind::Continuous,
        JointType::Prismatic => JointKind::Prismatic,
        JointType::Fixed => JointKind::Fixed,
        JointType::Floating => JointKind::Floating,
        JointType::Planar => JointKind::Planar,
    }
}

/// Build a `JointLayout` from a `RobotModel`, including only actuated joints,
/// in alphabetic name order. This is what PR2 will eventually wire as
/// `RobotModel::to_layout()`.
fn layout_from_model(model: &RobotModel) -> JointLayout {
    let mut names: Vec<&str> = model
        .joints
        .values()
        .filter(|j| j.joint_type.is_actuated())
        .map(|j| j.name.as_str())
        .collect();
    names.sort_unstable();

    let mut specs = Vec::with_capacity(names.len());
    for name in names {
        let j = &model.joints[name];
        specs.push(JointSpec {
            name: j.name.clone(),
            entity: None,
            joint_type: map_kind(j.joint_type),
            limits: JointSpecLimits {
                lower: j.limits.lower,
                upper: j.limits.upper,
                effort: j.limits.effort,
                velocity: j.limits.velocity,
            },
            axis: j.axis,
        });
    }
    JointLayout::new(specs)
}

fn hash_layout(layout: &JointLayout) -> u64 {
    let mut hasher = DefaultHasher::new();
    layout.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn same_urdf_produces_same_layout_hash() {
    let mut digests = Vec::with_capacity(10);
    let mut layouts = Vec::with_capacity(10);
    for _ in 0..10 {
        let model = parse_string(FIXTURE_URDF).expect("parse_string OK");
        let layout = layout_from_model(&model);
        digests.push(hash_layout(&layout));
        layouts.push(layout);
    }
    // All hashes identical.
    let first = digests[0];
    for (i, d) in digests.iter().enumerate() {
        assert_eq!(*d, first, "hash digest at index {i} differs from index 0");
    }
    // All layouts structurally equal.
    let first_layout = &layouts[0];
    for (i, l) in layouts.iter().enumerate() {
        assert_eq!(l, first_layout, "layout at index {i} differs from index 0");
    }
}

#[test]
fn layout_hash_is_order_sensitive() {
    let a = JointSpec {
        name: "alpha".into(),
        entity: None,
        joint_type: JointKind::Revolute,
        limits: JointSpecLimits::default(),
        axis: [0.0, 0.0, 1.0],
    };
    let b = JointSpec {
        name: "beta".into(),
        entity: None,
        joint_type: JointKind::Revolute,
        limits: JointSpecLimits::default(),
        axis: [0.0, 0.0, 1.0],
    };
    let layout_ab = JointLayout::new(vec![a.clone(), b.clone()]);
    let layout_ba = JointLayout::new(vec![b, a]);
    assert_ne!(hash_layout(&layout_ab), hash_layout(&layout_ba));
}

#[test]
fn layout_from_robot_model_orders_alphabetically() {
    let model = parse_string(FIXTURE_URDF).expect("parse_string OK");
    let layout = layout_from_model(&model);
    let names: Vec<&str> = layout.joint_names().collect();
    assert_eq!(names, vec!["elbow", "shoulder"]);
}

#[test]
fn layout_validate_against_self_is_ok() {
    let model = parse_string(FIXTURE_URDF).expect("parse_string OK");
    let layout = layout_from_model(&model);
    assert!(layout.validate_against(&layout).is_ok());
}

#[test]
fn layout_validate_against_version_mismatch_is_err() {
    let model = parse_string(FIXTURE_URDF).expect("parse_string OK");
    let layout_a = layout_from_model(&model);
    let mut layout_b = layout_a.clone();
    layout_b.set_version_for_test(JointLayout::SCHEMA_VERSION + 1);
    let err = layout_a.validate_against(&layout_b).unwrap_err();
    assert!(matches!(err, SchemaMismatch::VersionMismatch { .. }));
}

#[test]
fn layout_validate_against_joint_name_mismatch_is_err() {
    let model = parse_string(FIXTURE_URDF).expect("parse_string OK");
    let layout_a = layout_from_model(&model);
    // Re-build with one renamed joint.
    let mut specs: Vec<JointSpec> = layout_a.joints().to_vec();
    specs[0].name = "renamed".into();
    let layout_b = JointLayout::new(specs);
    let err = layout_a.validate_against(&layout_b).unwrap_err();
    assert!(matches!(err, SchemaMismatch::JointNameMismatch { .. }));
}
