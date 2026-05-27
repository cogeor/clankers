//! `arm_pick` scenario — headless 6-DOF arm + 2-finger gripper.
//!
//! Spawns the 6-DOF arm URDF (8 actuated joints: 6 arm revolutes +
//! 2 gripper prismatics), wires Rapier physics with 50 solver
//! iterations, adds a fixed table + a dynamic red cube, attaches finger
//! collider geometry, pre-populates [`MotorOverrides`] with `REST_POSE`
//! targets, and registers a layout-bound [`JointStateSensor`].
//!
//! # NOTE — duplicated setup
//!
//! The body of [`ArmPickScenario::build`] is a self-contained
//! near-duplicate of the headless subset of
//! `examples/src/arm_setup.rs::setup_arm` + the table/cube/finger
//! wiring at the top of `examples/examples/arm_pick_gym.rs::main`. **W8
//! PR1 will lift the example bin's setup logic to call this scenario
//! instead**; until then the two implementations may drift.
//!
//! The scenario is authoritative — bugs found here are fixed here;
//! the example may diverge until W8.
//!
//! # `MotorOverrides` applicator gap
//!
//! Per loop 06 PLAN Design choice D, the gym server in
//! [`crate::scenarios`] today uses a generic
//! `JointCommandApplicator`. The arm needs a **`MotorOverrides`-based
//! applicator** to be fully usable via `clankers-app serve --scenario
//! arm_pick`. The W5 PR2 `serve` integration test exercises only
//! `cartpole`, so this gap is intentionally not yet covered. Tracked
//! as W7 follow-up.
//!
//! `clankers-app run --scenario arm_pick` does not call `env.step`
//! with actions — it just runs the physics passively — so the gap is
//! invisible to the run path.
//!
//! # No rendering / no camera
//!
//! Scenario ships zero rendering code (no camera, no light, no
//! `DefaultPlugins`). Runs under the bevy minimal-plugin headless
//! harness. W5 PR2 GPU-off-limits.

use std::collections::HashMap;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::Actuator;
use clankers_actuator_core::prelude::ControlMode;
use clankers_env::prelude::*;
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::{
    MotorOverrideParams, MotorOverrides, RapierBackend, RapierContext, bridge::register_robot,
};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, SharedShape};

use crate::SceneBuilder;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the 6-DOF arm + gripper, included at compile time
/// from `examples/urdf/six_dof_arm.urdf` (4 levels up from this file →
/// workspace root → `examples/urdf/`).
///
/// W8 PR1 will move the URDF assets into a `clankers-assets` crate (or
/// into `clankers-sim` proper) and delete this `include_str!` shim. See
/// loop 06 PLAN Design choice F.
const SIX_DOF_ARM_URDF: &str = include_str!("../../../../examples/urdf/six_dof_arm.urdf");

// ---------------------------------------------------------------------------
// Motor / pose constants — duplicated from examples/src/arm_setup.rs
// ---------------------------------------------------------------------------

/// Default resting pose for the 6-DOF arm.
///
/// j2 tilts shoulder 45 forward, j3 bends elbow 90, j5 pitches wrist
/// down. Duplicated from `examples/src/arm_setup.rs::REST_POSE`.
const REST_POSE: [f32; 6] = [0.0, FRAC_PI_4, FRAC_PI_2, 0.0, FRAC_PI_4, 0.0];

/// Per-joint effort limits for the 6 arm joints (max accelerations,
/// `AccelerationBased` motor model). Duplicated from
/// `examples/src/arm_setup.rs::EFFORT_LIMITS`.
const EFFORT_LIMITS: [f32; 6] = [5000.0, 5000.0, 3000.0, 2000.0, 1000.0, 500.0];

/// Arm joint PD stiffness (acceleration per radian of error).
const ARM_STIFFNESS: f32 = 50_000.0;
/// Arm joint PD damping (acceleration per rad/s of velocity error).
const ARM_DAMPING: f32 = 500.0;

/// Gripper finger travel in meters (prismatic range 0..0.03).
const FINGER_TRAVEL: f32 = 0.03;
/// Gripper PD stiffness.
const GRIPPER_STIFFNESS: f32 = 500.0;
/// Gripper PD damping.
const GRIPPER_DAMPING: f32 = 50.0;
/// Gripper max force.
const GRIPPER_MAX_FORCE: f32 = 100.0;

/// Joint names in chain order (alphabetic, which equals the IK chain
/// order for this arm — verified by inspecting `six_dof_arm.urdf`).
/// The first 6 are arm revolutes; the last 2 are gripper prismatics.
const ARM_JOINT_NAMES: [&str; 8] = [
    "j1_base_yaw",
    "j2_shoulder_pitch",
    "j3_elbow_pitch",
    "j4_forearm_roll",
    "j5_wrist_pitch",
    "j6_wrist_roll",
    "j_finger_left",
    "j_finger_right",
];

// ---------------------------------------------------------------------------
// ArmPickScenario
// ---------------------------------------------------------------------------

/// Builder for the `arm_pick` reference scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the global registry
/// by [`super::register_builtin`] alongside `cartpole`.
pub struct ArmPickScenario;

impl ScenarioBuilder for ArmPickScenario {
    fn name(&self) -> &'static str {
        "arm_pick"
    }

    #[allow(clippy::too_many_lines)] // single linear setup pipeline
    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        // 1. Parse URDF (compile-time include).
        let model = clankers_urdf::parse_string(SIX_DOF_ARM_URDF)
            .expect("failed to parse six_dof_arm URDF (compile-time include)");

        // 2. Build initial positions from REST_POSE (only the 6 arm
        //    joints get rest values; fingers start at 0).
        let initial_positions: HashMap<String, f32> = ARM_JOINT_NAMES
            .iter()
            .take(6)
            .zip(REST_POSE.iter())
            .map(|(name, &pos)| ((*name).to_string(), pos))
            .collect();

        // 3. Build the scene on a fresh App owned by SceneBuilder; we
        //    swap it into the caller's slot at the end. (See the
        //    cartpole.rs module-level comment for why.)
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(cfg.max_steps)
            .with_robot(model.clone(), initial_positions)
            .build();

        // 4. Switch all actuators to position-mode PID — matches
        //    examples/src/arm_setup.rs::setup_arm step 3.
        let spawned = &scene.robots["six_dof_arm"];
        let joint_entity_ids: Vec<Entity> = spawned.joints.values().copied().collect();
        for entity in &joint_entity_ids {
            let mut actuator = scene
                .app
                .world_mut()
                .get_mut::<Actuator>(*entity)
                .expect("Actuator component on spawned joint");
            *actuator = Actuator::new(
                actuator.motor.clone(),
                actuator.transmission.clone(),
                actuator.friction.clone(),
                ControlMode::Position {
                    kp: 100.0,
                    ki: 0.0,
                    kd: 10.0,
                },
            );
        }

        // 5. Add Rapier physics.
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

        // 6. Register robot bodies + set initial motor targets so the
        //    first physics step doesn't jerk from zero to REST_POSE.
        {
            let spawned = &scene.robots["six_dof_arm"];
            let world = scene.app.world_mut();
            let mut ctx = world
                .remove_resource::<RapierContext>()
                .expect("RapierContext present after ClankersPhysicsPlugin");
            register_robot(&mut ctx, &model, spawned, world, true);
            ctx.integration_parameters.num_solver_iterations = 50;

            // Initial motor targets for the 6 arm joints.
            for (i, name) in ARM_JOINT_NAMES.iter().take(6).enumerate() {
                let q0 = REST_POSE[i];
                let Some(&entity) = spawned.joints.get(*name) else {
                    continue;
                };
                let Some(&jh) = ctx.joint_handles.get(&entity) else {
                    continue;
                };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else {
                    continue;
                };
                let axis = if ctx
                    .joint_info
                    .get(&entity)
                    .is_some_and(|info| info.is_prismatic)
                {
                    rapier3d::prelude::JointAxis::LinX
                } else {
                    rapier3d::prelude::JointAxis::AngX
                };
                let max_f = EFFORT_LIMITS.get(i).copied().unwrap_or(500.0);
                joint
                    .data
                    .set_motor(axis, q0, 0.0, ARM_STIFFNESS, ARM_DAMPING);
                joint.data.set_motor_max_force(axis, max_f);
            }

            // 7. Add table + red cube + finger colliders.
            // Table body (fixed, at z=0.4).
            let table_body = ctx.rigid_body_set.insert(
                RigidBodyBuilder::fixed()
                    .translation(Vec3::new(0.35, 0.0, 0.4))
                    .build(),
            );
            let table_collider = ColliderBuilder::cuboid(0.3, 0.2, 0.0125)
                .friction(0.6)
                .build();
            ctx.collider_set.insert_with_parent(
                table_collider,
                table_body,
                &mut ctx.rigid_body_set,
            );
            ctx.body_handles.insert("table".to_string(), table_body);

            // Red cube (dynamic, on the table).
            let cube_pos = Vec3::new(0.3, 0.0, 0.425);
            let cube_body = ctx.rigid_body_set.insert(
                RigidBodyBuilder::dynamic()
                    .translation(cube_pos)
                    .can_sleep(false)
                    .build(),
            );
            let cube_collider = ColliderBuilder::cuboid(0.0125, 0.0125, 0.0125)
                .density(500.0)
                .friction(0.8)
                .build();
            ctx.collider_set
                .insert_with_parent(cube_collider, cube_body, &mut ctx.rigid_body_set);
            ctx.body_handles.insert("red_cube".to_string(), cube_body);

            // Finger collider geometry — `register_robot` doesn't
            // synthesise colliders for URDF link geometry; the arm
            // would have invisible fingers without these.
            for finger_name in ["finger_left", "finger_right"] {
                if let Some(&finger_handle) = ctx.body_handles.get(finger_name) {
                    let finger_collider =
                        ColliderBuilder::new(SharedShape::cuboid(0.005, 0.008, 0.01))
                            .translation(Vec3::new(0.0, 0.0, 0.01))
                            .friction(0.8)
                            .build();
                    ctx.collider_set.insert_with_parent(
                        finger_collider,
                        finger_handle,
                        &mut ctx.rigid_body_set,
                    );
                }
            }

            // Snapshot updated state so `reset_to_initial` covers the
            // table, cube, and finger geometry. PR2 does NOT wire the
            // reset_fn; physics-reset semantics are W7. See module-level
            // "MotorOverrides applicator gap" note.
            ctx.snapshot_initial_state();
            world.insert_resource(ctx);
        }

        // 8. Build the bound 8-joint layout (chain order = alphabetic
        //    order for this URDF — see ARM_JOINT_NAMES note).
        let layout = {
            let bot = &scene.robots["six_dof_arm"];
            let mut layout = model.to_layout();
            let entities: Vec<Entity> = layout
                .joints()
                .iter()
                .map(|spec| {
                    bot.joint_entity(&spec.name)
                        .unwrap_or_else(|| panic!("joint {} not in spawned robot", spec.name))
                })
                .collect();
            layout.bind_entities(&entities);
            Arc::new(layout)
        };

        // 9. Pre-populate MotorOverrides — every joint must be
        //    overridden (MEMORY.md "MotorOverrides — ALL Joints Must
        //    Be Overridden"). Arm joints hold REST_POSE; fingers hold
        //    FINGER_TRAVEL (open).
        let mut overrides = MotorOverrides {
            layout: Some(layout.clone()),
            ..MotorOverrides::default()
        };
        {
            let bot = &scene.robots["six_dof_arm"];
            for (i, name) in ARM_JOINT_NAMES.iter().take(6).enumerate() {
                let Some(entity) = bot.joint_entity(name) else {
                    continue;
                };
                overrides.joints.insert(
                    entity,
                    MotorOverrideParams {
                        target_pos: REST_POSE[i],
                        target_vel: 0.0,
                        stiffness: ARM_STIFFNESS,
                        damping: ARM_DAMPING,
                        max_force: EFFORT_LIMITS[i],
                    },
                );
            }
            for finger_name in ["j_finger_left", "j_finger_right"] {
                let Some(entity) = bot.joint_entity(finger_name) else {
                    continue;
                };
                overrides.joints.insert(
                    entity,
                    MotorOverrideParams {
                        target_pos: FINGER_TRAVEL,
                        target_vel: 0.0,
                        stiffness: GRIPPER_STIFFNESS,
                        damping: GRIPPER_DAMPING,
                        max_force: GRIPPER_MAX_FORCE,
                    },
                );
            }
        }
        scene.app.insert_resource(overrides);

        // 10. Register the joint-state sensor over the 8-DOF layout.
        {
            let world = scene.app.world_mut();
            let mut registry = world
                .remove_resource::<SensorRegistry>()
                .expect("SensorRegistry present after ClankersEnvPlugin");
            let mut buffer = world
                .remove_resource::<ObservationBuffer>()
                .expect("ObservationBuffer present after ClankersEnvPlugin");
            registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        // 11. Swap into the caller's App. See cartpole.rs for the
        //     rationale behind the swap dance.
        std::mem::swap(app, &mut scene.app);

        ScenarioHandle {
            layout: Some(layout),
            max_steps: cfg.max_steps,
        }
    }
}
