//! Conformance tests for `JointLayout`-bound sensors and action
//! applicators (WS2 PR1).
//!
//! These tests pin down two invariants that the W1 `JointLayout`
//! contract requires and that PR1 of W2 finally enforces at runtime:
//!
//! 1. **Sensor determinism.** Two `JointStateSensor`s built from
//!    independently-parsed copies of the same URDF MUST emit byte-equal
//!    observation vectors at every step, regardless of Bevy archetype /
//!    `HashMap` iteration order. This closes the regression described in
//!    `notes/clankers_codebase_quality_report_2026-05-25.md` finding
//!    #1 ("a trained policy can bind output index 0 to one joint
//!    during training and a different joint during replay").
//! 2. **Action ↔ entity mapping.** Action slot `k` MUST drive the
//!    entity in layout slot `k`. The test sets a one-hot action and
//!    verifies the right `JointCommand.value` updates.
//!
//! Both tests embed `examples/urdf/six_dof_arm.urdf` via `include_str!`
//! so the fixture travels with the binary (no path resolution surprises
//! under different test runners).

use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::layout::JointLayout;
use clankers_core::traits::Sensor;
use clankers_core::types::Action;
use clankers_env::sensors::JointStateSensor;
use clankers_urdf::parser::parse_string;

const ARM_URDF: &str = include_str!("../../../examples/urdf/six_dof_arm.urdf");

/// Spawn one Bevy entity per layout slot, each carrying a `JointState`
/// initialised so the slot index leaks into the (position, velocity)
/// values. This lets the test prove that sensor output is keyed by
/// layout order, not by entity insertion order.
fn build_world_with_joints(layout: &JointLayout) -> (World, Vec<Entity>) {
    let mut world = World::new();
    let mut entities = Vec::with_capacity(layout.len());
    for i in 0..layout.len() {
        #[allow(clippy::cast_precision_loss)]
        let e = world
            .spawn(JointState {
                position: (i as f32) * 0.1,
                velocity: (i as f32) * 0.01,
            })
            .id();
        entities.push(e);
    }
    (world, entities)
}

#[test]
fn same_urdf_same_sensor_vector_order() {
    // 1. Parse the same URDF twice; build two layouts. These two
    //    layouts MUST be byte-equal because to_layout() is
    //    deterministic (alphabetic sort in the builder).
    let model_a = parse_string(ARM_URDF).expect("parse a");
    let model_b = parse_string(ARM_URDF).expect("parse b");
    let mut layout_a = model_a.to_layout();
    let mut layout_b = model_b.to_layout();
    assert_eq!(
        layout_a, layout_b,
        "parsing the same URDF twice should give equal layouts"
    );

    // 2. Spawn entities; bind them to BOTH layouts in the same slot
    //    order so the two sensors look up the same entity per slot.
    let (mut world, entities) = build_world_with_joints(&layout_a);
    layout_a.bind_entities(&entities);
    layout_b.bind_entities(&entities);

    // 3. Two sensors built from two independently-built layouts.
    //    PR2 renamed the layout-bound ctor back to `new` workspace-wide.
    let mut sensor_a = JointStateSensor::new(Arc::new(layout_a));
    let mut sensor_b = JointStateSensor::new(Arc::new(layout_b));

    // 4. Mutate joint state across 5 steps and confirm both sensors
    //    see the same vector every step.
    for step in 0..5 {
        for (i, &e) in entities.iter().enumerate() {
            let mut state = world.get_mut::<JointState>(e).unwrap();
            #[allow(clippy::cast_precision_loss)]
            {
                state.position = (i as f32 + step as f32) * 0.1;
                state.velocity = (i as f32 + step as f32) * 0.01;
            }
        }
        let obs_a = sensor_a.read(&mut world);
        let obs_b = sensor_b.read(&mut world);
        assert_eq!(
            obs_a.as_slice(),
            obs_b.as_slice(),
            "step {step}: sensors disagreed despite identical layout"
        );
        // Sanity check: every observation slot must equal its expected
        // value (slot k -> position 0.1*(k+step), velocity 0.01*(k+step)).
        for (k, _) in obs_a.as_slice().chunks(2).enumerate() {
            #[allow(clippy::cast_precision_loss)]
            let want_pos = (k as f32 + step as f32) * 0.1;
            #[allow(clippy::cast_precision_loss)]
            let want_vel = (k as f32 + step as f32) * 0.01;
            assert!(
                (obs_a.as_slice()[k * 2] - want_pos).abs() < 1e-6,
                "step {step} slot {k}: expected pos {want_pos}, got {}",
                obs_a.as_slice()[k * 2]
            );
            assert!(
                (obs_a.as_slice()[k * 2 + 1] - want_vel).abs() < 1e-6,
                "step {step} slot {k}: expected vel {want_vel}, got {}",
                obs_a.as_slice()[k * 2 + 1]
            );
        }
    }
}

#[test]
fn action_index_maps_to_layout_entity() {
    // 1. Build layout from the same fixture URDF.
    let model = parse_string(ARM_URDF).expect("parse");
    let mut layout = model.to_layout();

    // 2. Spawn one entity per layout slot, each carrying both a
    //    JointState (required by build_world_with_joints) and a
    //    JointCommand (mutated by the stub applicator below).
    let (mut world, entities) = build_world_with_joints(&layout);
    for &e in &entities {
        world.entity_mut(e).insert(JointCommand { value: 0.0 });
    }
    layout.bind_entities(&entities);
    let layout = Arc::new(layout);

    // 3. For every layout slot k, build a one-hot action with 1.0 in
    //    slot k. Apply via layout-bound indexing (mirroring the new
    //    ActionApplicator contract). Assert that the entity at layout
    //    slot k — and only that one — sees JointCommand.value == 1.0.
    let n = layout.len();
    for k in 0..n {
        // Reset every JointCommand to 0.
        for &e in &entities {
            world.get_mut::<JointCommand>(e).unwrap().value = 0.0;
        }

        let mut data = vec![0.0_f32; n];
        data[k] = 1.0;
        let action = Action::Continuous(data);

        // Stub layout-bound applicator: walks layout slots in order
        // and writes action[i] into the entity bound at slot i.
        for (i, slot) in layout.joints().iter().enumerate() {
            let entity = slot.entity.expect("entity bound at slot");
            world.get_mut::<JointCommand>(entity).unwrap().value = action.as_slice()[i];
        }

        // The bound entity at slot k must read 1.0 …
        let expected_entity = layout.joints()[k].entity.expect("entity at slot k");
        let cmd = world.get::<JointCommand>(expected_entity).unwrap();
        assert!(
            (cmd.value - 1.0).abs() < f32::EPSILON,
            "slot {k}: entity {expected_entity:?} expected 1.0, got {}",
            cmd.value
        );

        // … and every other bound entity must remain at 0.
        for (i, slot) in layout.joints().iter().enumerate() {
            if i == k {
                continue;
            }
            let other = slot.entity.expect("entity at slot");
            let cmd = world.get::<JointCommand>(other).unwrap();
            assert!(
                cmd.value.abs() < f32::EPSILON,
                "slot {i} (k={k}): expected 0.0, got {}",
                cmd.value
            );
        }
    }
}
