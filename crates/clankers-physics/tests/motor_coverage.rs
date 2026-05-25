//! Conformance test: `validate_motor_coverage` rejects an under-specified
//! `MotorOverrides` and names every missing joint.
//!
//! This pins the runtime invariant described in `MEMORY.md`
//! ("`MotorOverrides` — ALL Joints Must Be Overridden"). Before WS2 PR1,
//! the rule was a prose comment in `crates/clankers-physics/src/rapier/
//! systems.rs` whose violation produced "robot flailing wildly" failures
//! instead of an actionable error. PR1 promotes the rule to a setup-time
//! invariant that the future `clankers validate` CLI (W5) can surface.

use bevy::ecs::entity::Entity;
use clankers_core::layout::{
    JointKind, JointLayout, JointLayoutBuilder, JointSpec, JointSpecLimits,
};
use clankers_core::types::RobotGroup;
use clankers_physics::rapier::systems::{
    MotorOverrideParams, MotorOverrides, validate_motor_coverage,
};

/// Build a synthetic 8-joint layout (6-DOF arm + 2-finger gripper)
/// without parsing URDF. Each slot is bound to a unique stub entity so
/// the override map can address them.
fn synthetic_arm_layout() -> (JointLayout, Vec<Entity>) {
    let mut b = JointLayoutBuilder::default();
    for name in [
        "arm_0",
        "arm_1",
        "arm_2",
        "arm_3",
        "arm_4",
        "arm_5",
        "gripper_left",
        "gripper_right",
    ] {
        b = b.push(JointSpec {
            name: name.into(),
            entity: None,
            joint_type: JointKind::Revolute,
            limits: JointSpecLimits {
                lower: Some(-1.0),
                upper: Some(1.0),
                effort: 1.0,
                velocity: 1.0,
            },
            axis: [0.0, 0.0, 1.0],
        });
    }
    let mut layout = b.build();
    // Bind one stub entity per slot. Entity bits are irrelevant — the
    // validator only checks set membership.
    let entities: Vec<Entity> = (0..layout.len())
        .map(|i| Entity::from_bits(1000 + i as u64))
        .collect();
    layout.bind_entities(&entities);
    (layout, entities)
}

const fn stub_params() -> MotorOverrideParams {
    MotorOverrideParams {
        target_pos: 0.0,
        target_vel: 0.0,
        stiffness: 100.0,
        damping: 10.0,
        max_force: 50.0,
    }
}

#[test]
fn motor_override_missing_joint_is_rejected() {
    let (layout, entities) = synthetic_arm_layout();
    // The `group` argument is part of the signature for future
    // extensibility (per-robot validation); the current implementation
    // checks `overrides` against `layout` directly. An empty group is
    // therefore fine for this test.
    let group = RobotGroup::default();

    // 7-entry overrides — gripper_right (slot 7) intentionally missing.
    let mut overrides = MotorOverrides::default();
    for (i, &e) in entities.iter().enumerate() {
        if i == 7 {
            continue; // skip gripper_right
        }
        overrides.joints.insert(e, stub_params());
    }
    assert_eq!(overrides.joints.len(), 7);

    let err = validate_motor_coverage(&group, &layout, &overrides)
        .expect_err("expected MissingJoints for under-specified overrides");
    assert!(
        err.layout_joint_names.iter().any(|n| n == "gripper_right"),
        "expected gripper_right in MissingJoints; got {:?}",
        err.layout_joint_names
    );
    assert_eq!(
        err.override_joint_count, 7,
        "MissingJoints.override_joint_count should reflect the input"
    );

    // Display message must include the missing joint name for the
    // operator-facing CLI surface (W5).
    let snapshot = format!("{err}");
    assert!(
        snapshot.contains("gripper_right"),
        "Display missing 'gripper_right': {snapshot}"
    );

    // Happy path: 8 entries, every layout joint covered.
    let mut full = MotorOverrides::default();
    for &e in &entities {
        full.joints.insert(e, stub_params());
    }
    assert!(
        validate_motor_coverage(&group, &layout, &full).is_ok(),
        "8-entry overrides must validate cleanly"
    );
}

#[test]
fn motor_override_multiple_missing_joints_are_all_named() {
    let (layout, entities) = synthetic_arm_layout();
    let group = RobotGroup::default();

    // Only arm_0 covered; 7 joints uncovered.
    let mut overrides = MotorOverrides::default();
    overrides.joints.insert(entities[0], stub_params());

    let err = validate_motor_coverage(&group, &layout, &overrides)
        .expect_err("expected MissingJoints for severely under-specified overrides");
    assert_eq!(err.layout_joint_names.len(), 7);
    assert_eq!(err.override_joint_count, 1);
    // Names appear in layout order, alphabetically sorted by the
    // JointLayoutBuilder. arm_0 is the only covered slot.
    let expected_missing = [
        "arm_1",
        "arm_2",
        "arm_3",
        "arm_4",
        "arm_5",
        "gripper_left",
        "gripper_right",
    ];
    assert_eq!(err.layout_joint_names, expected_missing);
}

#[test]
fn motor_override_empty_overrides_lists_every_joint() {
    let (layout, _entities) = synthetic_arm_layout();
    let group = RobotGroup::default();
    let overrides = MotorOverrides::default();

    let err = validate_motor_coverage(&group, &layout, &overrides)
        .expect_err("empty overrides must fail validation");
    assert_eq!(err.layout_joint_names.len(), layout.len());
    assert_eq!(err.override_joint_count, 0);
}
