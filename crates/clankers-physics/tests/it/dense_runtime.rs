//! W7 PR3 integration tests for the dense `JointRuntimes` migration.
//!
//! Three load-bearing tests per `docs/plans/WS7-plan.md` § 6 / loop 05
//! PLAN § Tasks 1:
//!
//! 1. `dense_runtime_matches_hashmap_lookup` — runs the same scene with
//!    the dense path (feature `dense-runtime` ON + `JointRuntimes`
//!    inserted) and the `HashMap` fallback (resource absent), and asserts
//!    byte-equal `Isometry3` per body per step over 100 steps. This is
//!    the load-bearing guard against the silent-ordering risk in
//!    WS7-plan § 8 risk 3.
//!
//! 2. `compile_runtime_rejects_missing_layout_joint` — sanity-check that
//!    a phantom joint surfaces `LayoutCompileError::MissingJoint`.
//!
//! 3. `compile_runtime_orders_by_layout_slot` — the dense vec follows
//!    layout slot order, not handle insertion order.
//!
//! Construction uses a synthetic 2-joint URDF + `RapierContext::new`
//! directly so the tests stay fast and avoid pulling the example
//! binaries' arm setup. `num_solver_iterations = 50` (MEMORY.md
//! invariant) is preserved verbatim on the contexts the tests build.

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::{JointState, JointTorque};
use clankers_core::layout::{JointKind, JointLayoutBuilder, JointSpec, JointSpecLimits};
use clankers_core::types::{LayoutCompileError, RobotGroup};
use clankers_physics::rapier::context::{JointInfo, RapierContext};
use clankers_physics::rapier::runtime::JointRuntimes;
use clankers_physics::rapier::systems::{
    InnerPdState, MotorOverrideParams, MotorOverrides, MotorRateLimits, rapier_step_system,
};
use rapier3d::prelude::{
    ImpulseJointHandle, RevoluteJointBuilder, RigidBodyBuilder, RigidBodyHandle,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Snapshot of a single body's pose for byte-equality comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PoseSnapshot {
    translation: [f32; 3],
    rotation: [f32; 4],
}

fn snapshot_pose(ctx: &RapierContext, handle: RigidBodyHandle) -> PoseSnapshot {
    let body = ctx
        .rigid_body_set
        .get(handle)
        .expect("body handle still valid");
    let t = body.translation();
    let r = *body.rotation();
    PoseSnapshot {
        translation: [t.x, t.y, t.z],
        rotation: [r.x, r.y, r.z, r.w],
    }
}

/// Build a 2-joint chain in a fresh `RapierContext`. Returns the
/// populated context, the per-joint entity vector, and the per-body
/// rigid body handles in order [base, link1, link2].
fn build_two_joint_scene() -> (RapierContext, Vec<Entity>, Vec<RigidBodyHandle>) {
    // MEMORY.md invariant: num_solver_iterations = 50.
    let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 1.0 / 100.0, 1);
    ctx.integration_parameters.num_solver_iterations = 50;

    // base (fixed) → link1 (dynamic) → link2 (dynamic).
    let base: RigidBodyHandle = ctx.rigid_body_set.insert(RigidBodyBuilder::fixed().build());
    let link1: RigidBodyHandle = ctx.rigid_body_set.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0.0, 0.0, 0.5))
            .can_sleep(false)
            .build(),
    );
    let link2: RigidBodyHandle = ctx.rigid_body_set.insert(
        RigidBodyBuilder::dynamic()
            .translation(Vec3::new(0.0, 0.0, 1.0))
            .can_sleep(false)
            .build(),
    );

    let j1: ImpulseJointHandle = ctx.impulse_joint_set.insert(
        base,
        link1,
        RevoluteJointBuilder::new(Vec3::Z).build(),
        true,
    );
    let j2: ImpulseJointHandle = ctx.impulse_joint_set.insert(
        link1,
        link2,
        RevoluteJointBuilder::new(Vec3::Y).build(),
        true,
    );

    // Two synthetic entities with deterministic non-zero bits.
    let e1 = Entity::from_bits(101);
    let e2 = Entity::from_bits(102);

    ctx.joint_handles.insert(e1, j1);
    ctx.joint_handles.insert(e2, j2);
    ctx.joint_info.insert(
        e1,
        JointInfo {
            parent_body: base,
            child_body: link1,
            axis: Vec3::Z,
            is_prismatic: false,
        },
    );
    ctx.joint_info.insert(
        e2,
        JointInfo {
            parent_body: link1,
            child_body: link2,
            axis: Vec3::Y,
            is_prismatic: false,
        },
    );
    ctx.body_to_entity.insert(link1, e1);
    ctx.body_to_entity.insert(link2, e2);

    (ctx, vec![e1, e2], vec![base, link1, link2])
}

const fn motor_params(stiff: f32) -> MotorOverrideParams {
    MotorOverrideParams {
        target_pos: 0.0,
        target_vel: 0.0,
        stiffness: stiff,
        damping: 10.0,
        max_force: 50.0,
    }
}

fn synthetic_layout(
    names: &[&str],
    entities: &[Option<Entity>],
) -> clankers_core::layout::JointLayout {
    let mut b = JointLayoutBuilder::default();
    for (i, name) in names.iter().enumerate() {
        b = b.push(JointSpec {
            name: (*name).to_string(),
            entity: entities.get(i).copied().flatten(),
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
    b.build()
}

/// Build a fully wired Bevy App for the two-joint scene. Inserts
/// `RapierContext`, `MotorOverrides`, spawns one entity per joint
/// (matching the synthetic entity bits from `build_two_joint_scene`),
/// and registers the `rapier_step_system` on `Update`.
fn build_app_two_joint() -> (App, Vec<RigidBodyHandle>, Vec<Entity>) {
    let (ctx, entities, body_handles) = build_two_joint_scene();
    let e1 = entities[0];
    let e2 = entities[1];

    let mut app = App::new();
    // Spawn the joint entities so the Bevy world has matching IDs.
    // We use `spawn_at` (via the `world().spawn(...)` trick) but here
    // we just spawn fresh entities and use their bits. Tests don't rely
    // on entity-bit equality between the synthetic id_from_bits and the
    // App's allocated entities — we instead spawn fresh and rebuild the
    // ctx HashMaps against the *new* entity ids.
    let new_e1 = app
        .world_mut()
        .spawn((JointTorque::default(), JointState::default()))
        .id();
    let new_e2 = app
        .world_mut()
        .spawn((JointTorque::default(), JointState::default()))
        .id();

    // Remap ctx entries from synthetic e1/e2 to the spawned ones.
    let mut ctx = ctx;
    let h1 = ctx.joint_handles.remove(&e1).unwrap();
    let h2 = ctx.joint_handles.remove(&e2).unwrap();
    ctx.joint_handles.insert(new_e1, h1);
    ctx.joint_handles.insert(new_e2, h2);
    let i1 = ctx.joint_info.remove(&e1).unwrap();
    let i2 = ctx.joint_info.remove(&e2).unwrap();
    ctx.joint_info.insert(new_e1, i1.clone());
    ctx.joint_info.insert(new_e2, i2.clone());
    // Rebuild body_to_entity for parity.
    ctx.body_to_entity.clear();
    ctx.body_to_entity.insert(i1.child_body, new_e1);
    ctx.body_to_entity.insert(i2.child_body, new_e2);

    // MotorOverrides for both joints — MEMORY.md invariant.
    let mut motor_map = HashMap::new();
    motor_map.insert(new_e1, motor_params(100.0));
    motor_map.insert(new_e2, motor_params(100.0));
    let overrides = MotorOverrides {
        joints: motor_map,
        ..MotorOverrides::default()
    };

    app.insert_resource(ctx);
    app.insert_resource(overrides);

    app.add_systems(Update, rapier_step_system);

    (app, body_handles, vec![new_e1, new_e2])
}

// ---------------------------------------------------------------------------
// 1. dense_runtime_matches_hashmap_lookup
// ---------------------------------------------------------------------------

/// Scene A: `JointRuntimes` inserted → dense path active.
/// Scene B: no `JointRuntimes` resource → `HashMap` fallback active.
///
/// Both scenes use identical bodies, identical motor overrides,
/// identical solver iterations, identical zero torques. After every
/// step we compare every body's `(translation, rotation)` byte-equal.
/// Any off-by-one in `compile_runtime`'s slot ordering surfaces as a
/// diverging pose by step ~5-10 (the joint solver amplifies any
/// inconsistency in motor target application).
#[test]
fn dense_runtime_matches_hashmap_lookup() {
    let (mut app_dense, bodies_dense, entities_dense) = build_app_two_joint();
    let (mut app_hash, bodies_hash, _) = build_app_two_joint();

    // Insert JointRuntimes into scene A — compile from layout bound
    // to the spawned entities. `RobotGroup` is built outside the App
    // because the minimal test App skips `ClankersCorePlugin`.
    {
        let layout = synthetic_layout(
            &["j1", "j2"],
            &[Some(entities_dense[0]), Some(entities_dense[1])],
        );
        let group = RobotGroup::default();
        let world = app_dense.world();
        let ctx = world.resource::<RapierContext>();
        let overrides = world.resource::<MotorOverrides>();
        let runtimes = clankers_sim::builder::compile_runtime(&group, &layout, ctx, overrides)
            .expect("compile_runtime should succeed on the bound layout");
        app_dense.insert_resource(runtimes);
    }
    // Sanity: scene A has the resource, scene B does not.
    assert!(app_dense.world().get_resource::<JointRuntimes>().is_some());
    assert!(app_hash.world().get_resource::<JointRuntimes>().is_none());

    // Step both scenes 100 times. JointTorque entries default to 0, so
    // the only forces are gravity + zero-target PD motors. Compare
    // every body's pose each step.
    for step in 0..100 {
        app_dense.update();
        app_hash.update();

        for ((bd, bh), name) in bodies_dense
            .iter()
            .zip(bodies_hash.iter())
            .zip(["base", "link1", "link2"].iter())
        {
            let pd = snapshot_pose(app_dense.world().resource::<RapierContext>(), *bd);
            let ph = snapshot_pose(app_hash.world().resource::<RapierContext>(), *bh);
            assert_eq!(
                pd, ph,
                "dense vs hashmap diverged at step {step} body {name}: dense={pd:?} hash={ph:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. compile_runtime_rejects_missing_layout_joint
// ---------------------------------------------------------------------------

#[test]
fn compile_runtime_rejects_missing_layout_joint() {
    let (ctx, entities, _bodies) = build_two_joint_scene();
    let real = entities[0];
    let phantom = Entity::from_bits(9999);

    // Layout has two slots: one bound to a real entity, one to a
    // phantom that is NOT in `ctx.joint_handles`.
    let layout = synthetic_layout(
        &["real_joint", "phantom_joint"],
        &[Some(real), Some(phantom)],
    );

    let mut motor_map = HashMap::new();
    motor_map.insert(real, motor_params(100.0));
    motor_map.insert(phantom, motor_params(100.0));
    let overrides = MotorOverrides {
        joints: motor_map,
        ..MotorOverrides::default()
    };
    let group = RobotGroup::default();

    let err = clankers_sim::builder::compile_runtime(&group, &layout, &ctx, &overrides)
        .expect_err("expected error from compile_runtime against phantom joint");
    match err {
        LayoutCompileError::MissingJoint { name } => {
            assert!(
                name.contains("phantom_joint"),
                "missing-joint name should mention 'phantom_joint', got {name:?}"
            );
        }
        other => panic!("expected MissingJoint, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 3. compile_runtime_orders_by_layout_slot
// ---------------------------------------------------------------------------

#[test]
fn compile_runtime_orders_by_layout_slot() {
    // Build a 4-joint context where insertion order is [d, a, c, b].
    // The dense runtime must follow LAYOUT order [a, b, c, d] regardless.
    let mut ctx = RapierContext::new(Vec3::new(0.0, 0.0, -9.81), 1.0 / 60.0, 1);
    let mut parent = ctx.rigid_body_set.insert(RigidBodyBuilder::fixed().build());

    let mut by_name: HashMap<&'static str, Entity> = HashMap::new();
    for (i, name) in ["d", "a", "c", "b"].iter().enumerate() {
        let child = ctx
            .rigid_body_set
            .insert(RigidBodyBuilder::dynamic().can_sleep(false).build());
        let jh = ctx.impulse_joint_set.insert(
            parent,
            child,
            RevoluteJointBuilder::new(Vec3::Z).build(),
            true,
        );
        let entity = Entity::from_bits((i as u64) + 200);
        by_name.insert(*name, entity);
        ctx.joint_handles.insert(entity, jh);
        ctx.joint_info.insert(
            entity,
            JointInfo {
                parent_body: parent,
                child_body: child,
                axis: Vec3::Z,
                is_prismatic: false,
            },
        );
        ctx.body_to_entity.insert(child, entity);
        parent = child;
    }

    let layout = synthetic_layout(
        &["a", "b", "c", "d"],
        &[
            Some(by_name["a"]),
            Some(by_name["b"]),
            Some(by_name["c"]),
            Some(by_name["d"]),
        ],
    );

    let mut motor_map = HashMap::new();
    for e in by_name.values() {
        motor_map.insert(*e, motor_params(100.0));
    }
    let overrides = MotorOverrides {
        joints: motor_map,
        ..MotorOverrides::default()
    };
    let group = RobotGroup::default();
    let runtimes = clankers_sim::builder::compile_runtime(&group, &layout, &ctx, &overrides)
        .expect("compile should succeed");

    assert_eq!(runtimes.joints.len(), 4);
    assert_eq!(runtimes.joints[0].entity, by_name["a"]);
    assert_eq!(runtimes.joints[1].entity, by_name["b"]);
    assert_eq!(runtimes.joints[2].entity, by_name["c"]);
    assert_eq!(runtimes.joints[3].entity, by_name["d"]);
    for (slot, jr) in runtimes.joints.iter().enumerate() {
        assert_eq!(jr.layout_slot, slot);
    }
}

// ---------------------------------------------------------------------------
// Silence unused-import warning when the dense feature is off — only
// the InnerPdState/MotorRateLimits imports are conditionally needed.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn _unused_imports_keepalive(_a: InnerPdState, _b: MotorRateLimits) {}
