//! Arm pick-and-place gym server.
//!
//! 6-DOF arm + 2-finger gripper with table and manipulable objects.
//! Serves a single GymEnv over TCP for the synthetic pipeline.
//!
//! - 8 action DOF: 6 arm joints + 2 gripper fingers
//! - 16 observation DOF: 8 joint positions + 8 joint velocities
//! - Body poses (table, red_cube, end_effector, ...) are populated in `info`
//!
//! Run: `cargo run -p clankers-examples --bin arm_pick_gym`
//! Then connect with: `python python/clankers_synthetic/scripts/run_arm_pick.py`

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_examples::arm_setup::{ArmSetupConfig, setup_arm};
use clankers_gym::prelude::*;
use clankers_physics::rapier::{MotorOverrideParams, MotorOverrides, RapierContext};
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle, SharedShape};

/// Joint entities for the arm + gripper, stored as a resource so the applicator
/// can target specific joints rather than iterating all joints in query order.
#[derive(Resource)]
struct PickJointEntities(Vec<Entity>);

/// Initial positions of dynamic objects for reset.
#[derive(Resource)]
struct ObjectInitialPositions(Vec<(RigidBodyHandle, Vec3)>);

/// Per-joint effort limits (matching arm_ik_viz for stable control).
const EFFORT_LIMITS: [f32; 6] = [80.0, 60.0, 40.0, 20.0, 10.0, 5.0];

/// Maps 8-dim action to `MotorOverrides` on the arm + gripper joint entities.
///
/// Uses Rapier position motors (stiffness/damping/max_force) instead of raw
/// `JointCommand` to avoid oscillation — same approach as `arm_ik_viz`.
struct PickApplicator;

impl clankers_core::traits::ActionApplicator for PickApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action.as_slice();
        let entities = world.resource::<PickJointEntities>().0.clone();
        let mut overrides = world.resource_mut::<MotorOverrides>();
        for (i, &entity) in entities.iter().enumerate() {
            if i >= values.len() {
                break;
            }
            if i < 6 {
                // Arm joints: position motor with per-joint effort limits
                overrides.joints.insert(
                    entity,
                    MotorOverrideParams {
                        target_pos: values[i],
                        target_vel: 0.0,
                        stiffness: 100.0,
                        damping: 10.0,
                        max_force: EFFORT_LIMITS[i],
                    },
                );
            } else {
                // Gripper fingers: softer position motor
                overrides.joints.insert(
                    entity,
                    MotorOverrideParams {
                        target_pos: values[i],
                        target_vel: 0.0,
                        stiffness: 50.0,
                        damping: 5.0,
                        max_force: 10.0,
                    },
                );
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "PickApplicator"
    }
}

fn main() {
    println!("=== Arm Pick-and-Place Gym Server ===\n");

    let max_steps: u32 = 500;
    let num_joints: usize = 8; // 6 arm + 2 gripper
    let address = "127.0.0.1:9880";

    // 1. Setup arm with gripper (8 DOF)
    // Note: use_fixed_update=false because headless gym has no window/frame timing,
    // so FixedUpdate never fires. MotorOverrides work fine on Update schedule.
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: max_steps,
        use_fixed_update: false,
        sensor_dof: num_joints,
    });
    let mut scene = setup.scene;

    // Insert MotorOverrides so the applicator can use Rapier position motors
    scene.app.insert_resource(MotorOverrides::default());

    // 2. Collect joint entities: 6 arm joints + 2 gripper fingers
    let spawned = &scene.robots["six_dof_arm"];
    let finger_left = spawned.joint_entity("j_finger_left");
    let finger_right = spawned.joint_entity("j_finger_right");

    let mut all_joint_entities = setup.joint_entities.clone();
    if let Some(fl) = finger_left {
        all_joint_entities.push(fl);
    }
    if let Some(fr) = finger_right {
        all_joint_entities.push(fr);
    }

    scene
        .app
        .insert_resource(PickJointEntities(all_joint_entities.clone()));

    println!("Robot: six_dof_arm with gripper");
    println!("DOF:   {num_joints} (6 arm + 2 gripper)");
    println!("Arm joints: {:?}", setup.arm_joint_names);

    // 3. Add table and dynamic objects to rapier context
    let mut object_initial_positions = Vec::new();
    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();

        // Increase solver iterations for stable joint constraint enforcement
        ctx.integration_parameters.num_solver_iterations = 50;

        // Table body (fixed, at z=0.4)
        let table_body = ctx.rigid_body_set.insert(
            RigidBodyBuilder::fixed()
                .translation(Vec3::new(0.35, 0.0, 0.4))
                .build(),
        );
        let table_collider = ColliderBuilder::cuboid(0.3, 0.2, 0.0125)
            .friction(0.6)
            .build();
        ctx.collider_set
            .insert_with_parent(table_collider, table_body, &mut ctx.rigid_body_set);
        ctx.body_handles.insert("table".to_string(), table_body);

        // Red cube (dynamic, on the table)
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
        object_initial_positions.push((cube_body, cube_pos));

        // Add colliders to finger links (bridge.rs doesn't create URDF collision geometry).
        // URDF full finger: box 0.01×0.01×0.04 at (0,0,0.02). But fingers point
        // downward at reach config, so the full 4cm length hits the table.
        // Use only the tip portion (2cm) to avoid table collision while still
        // gripping the cube sides.
        for finger_name in ["finger_left", "finger_right"] {
            if let Some(&finger_handle) = ctx.body_handles.get(finger_name) {
                let finger_collider = ColliderBuilder::new(SharedShape::cuboid(0.005, 0.008, 0.01))
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

        // Snapshot updated state for reset
        ctx.snapshot_initial_state();
        world.insert_resource(ctx);
    }

    scene
        .app
        .insert_resource(ObjectInitialPositions(object_initial_positions.clone()));

    // 4. Create gym environment
    let obs_dim = num_joints * 2; // 8 pos + 8 vel = 16
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };

    // Action space: 6 arm joints (±π) + 2 gripper fingers (0..0.03)
    let mut act_low = vec![-std::f32::consts::PI; 6];
    let mut act_high = vec![std::f32::consts::PI; 6];
    act_low.extend_from_slice(&[0.0, 0.0]);
    act_high.extend_from_slice(&[0.03, 0.03]);

    let act_space = ActionSpace::Box {
        low: act_low,
        high: act_high,
    };

    let joint_entities_for_reset = all_joint_entities.clone();

    let mut env = GymEnv::new(scene.app, obs_space, act_space, Box::new(PickApplicator))
        .with_reset_fn(move |world: &mut World| {
            // Reset rapier rigid body positions and velocities
            if let Some(mut ctx) = world.remove_resource::<RapierContext>() {
                ctx.reset_to_initial();

                // Also reset dynamic objects to their initial positions
                if let Some(obj_init) = world.get_resource::<ObjectInitialPositions>() {
                    for &(handle, pos) in &obj_init.0 {
                        if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                            body.set_translation(pos, true);
                            body.set_rotation(Quat::IDENTITY, true);
                            body.set_linvel(Vec3::ZERO, true);
                            body.set_angvel(Vec3::ZERO, true);
                            body.wake_up(true);
                        }
                    }
                }
                world.insert_resource(ctx);
            }
            // Clear motor overrides so stale targets don't persist
            if let Some(mut overrides) = world.get_resource_mut::<MotorOverrides>() {
                overrides.joints.clear();
            }
            // Reset joint states and commands
            for &entity in &joint_entities_for_reset {
                if let Some(mut state) = world.get_mut::<JointState>(entity) {
                    state.position = 0.0;
                    state.velocity = 0.0;
                }
                if let Some(mut cmd) = world.get_mut::<JointCommand>(entity) {
                    cmd.value = 0.0;
                }
            }
        })
        .with_success_fn(|world: &World| {
            // Task success: red_cube z-position >= 0.525m (lifted 0.1m above table)
            let Some(ctx) = world.get_resource::<RapierContext>() else {
                return false;
            };
            let Some(&cube_handle) = ctx.body_handles.get("red_cube") else {
                return false;
            };
            let Some(cube_body) = ctx.rigid_body_set.get(cube_handle) else {
                return false;
            };
            cube_body.translation().z >= 0.525
        });

    // 5. Start server
    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("\nArm pick-and-place gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");
    println!("Scene: table + red_cube");
    println!("Physics: Rapier3D, MotorOverrides (stiffness=100, damping=10)");
    println!("Connect with: python python/clankers_synthetic/scripts/run_arm_pick.py\n");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
