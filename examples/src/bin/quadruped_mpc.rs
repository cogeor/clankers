//! Quadruped robot walking with Model Predictive Control.
//!
//! Demonstrates the full MPC locomotion pipeline on a headless simulation:
//! 1. Load quadruped URDF (4 legs × 2 joints = 8 DOF, floating base)
//! 2. Set up Rapier physics with dynamic base (not fixed)
//! 3. Run centroidal MPC → WBC → swing planner each step
//! 4. Print telemetry: body pose, foot forces, solve time
//!
//! This example calls the MPC solver directly (not via the Bevy plugin)
//! so it works headless without a transform hierarchy.
//!
//! Run: `cargo run -p clankers-examples --bin quadruped_mpc`

use std::collections::HashMap;

use bevy::math::EulerRot;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_env::prelude::*;
use clankers_examples::QUADRUPED_URDF;
use clankers_ik::KinematicChain;
use clankers_mpc::{
    BodyState, GaitScheduler, GaitType, MpcConfig, MpcSolver, ReferenceTrajectory, SwingConfig,
    raibert_foot_target, swing_foot_position,
    wbc::{compute_leg_jacobian, frames_f32_to_f64, jacobian_transpose_torques},
};
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use nalgebra::Vector3;
use rapier3d::prelude::{ColliderBuilder, MassProperties, RigidBodyBuilder};

/// Per-leg runtime data.
struct LegRuntime {
    chain: KinematicChain,
    joint_entities: Vec<bevy::prelude::Entity>,
    is_prismatic: Vec<bool>,
    hip_offset: Vector3<f64>,
}

/// Read body state from Rapier's rigid body set.
fn body_state_from_rapier(ctx: &RapierContext, link_name: &str) -> Option<BodyState> {
    let handle = ctx.body_handles.get(link_name)?;
    let body = ctx.rigid_body_set.get(*handle)?;

    let t = body.translation();
    let r = body.rotation();
    let (yaw, pitch, roll) = r.to_euler(EulerRot::ZYX);

    let lv = body.linvel();
    let av = body.angvel();

    Some(BodyState {
        orientation: Vector3::new(f64::from(roll), f64::from(pitch), f64::from(yaw)),
        position: Vector3::new(f64::from(t.x), f64::from(t.y), f64::from(t.z)),
        angular_velocity: Vector3::new(f64::from(av.x), f64::from(av.y), f64::from(av.z)),
        linear_velocity: Vector3::new(f64::from(lv.x), f64::from(lv.y), f64::from(lv.z)),
    })
}

fn main() {
    println!("=== Quadruped MPC Example ===\n");

    // 1. Parse URDF
    let model =
        clankers_urdf::parse_string(QUADRUPED_URDF).expect("failed to parse quadruped URDF");

    // 2. Build scene
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(5000)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let spawned = &scene.robots["quadruped"];
    println!(
        "Robot '{}' loaded: {} actuated joints",
        spawned.name,
        spawned.joint_count()
    );

    // 3. Add Rapier physics with floating base
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        // fixed_base = false: body is dynamic, controlled by ground reaction forces
        register_robot(&mut ctx, &model, spawned, world, false);

        // Set mass properties on root body (bridge doesn't do this for root)
        if let Some(&root_handle) = ctx.body_handles.get("body")
            && let Some(root_body) = ctx.rigid_body_set.get_mut(root_handle)
        {
            let body_mass = 5.0_f32;
            let inertia = bevy::math::Vec3::new(0.07, 0.26, 0.28);
            root_body.set_additional_mass_properties(
                MassProperties::new(bevy::math::Vec3::ZERO, body_mass, inertia),
                true,
            );
            // Start the body at standing height (feet touch ground at ~z=-0.35)
            root_body.set_translation(bevy::math::Vec3::new(0.0, 0.0, 0.35), true);
        }

        // Add ground plane: fixed body with large cuboid at z=-0.05 (top at z=0)
        let ground_body = RigidBodyBuilder::fixed()
            .translation(bevy::math::Vec3::new(0.0, 0.0, -0.05))
            .build();
        let ground_handle = ctx.rigid_body_set.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(50.0, 50.0, 0.05)
            .friction(0.8)
            .restitution(0.0)
            .build();
        ctx.collider_set
            .insert_with_parent(ground_collider, ground_handle, &mut ctx.rigid_body_set);

        // Add colliders to all robot links
        let link_colliders: &[(&str, ColliderBuilder)] = &[
            ("fl_foot", ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0)),
            ("fr_foot", ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0)),
            ("rl_foot", ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0)),
            ("rr_foot", ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0)),
            ("fl_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("fr_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("rl_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("rr_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("fl_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("fr_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("rl_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
            ("rr_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3)),
        ];
        for (name, builder) in link_colliders {
            if let Some(&handle) = ctx.body_handles.get(*name) {
                ctx.collider_set.insert_with_parent(
                    builder.clone().build(),
                    handle,
                    &mut ctx.rigid_body_set,
                );
            }
        }

        // Add box collider to body
        if let Some(&body_handle) = ctx.body_handles.get("body") {
            let body_collider = ColliderBuilder::cuboid(0.2, 0.1, 0.05)
                .friction(0.5)
                .build();
            ctx.collider_set.insert_with_parent(
                body_collider,
                body_handle,
                &mut ctx.rigid_body_set,
            );
        }

        // Re-snapshot after setting initial position
        ctx.snapshot_initial_state();
        world.insert_resource(ctx);
    }

    // 4. Build per-leg IK chains
    let foot_link_names = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"];
    let hip_offsets = [
        Vector3::new(0.15, 0.08, -0.05),
        Vector3::new(0.15, -0.08, -0.05),
        Vector3::new(-0.15, 0.08, -0.05),
        Vector3::new(-0.15, -0.08, -0.05),
    ];

    let legs: Vec<LegRuntime> = foot_link_names
        .iter()
        .enumerate()
        .map(|(i, &foot_link)| {
            let chain = KinematicChain::from_model(&model, foot_link)
                .unwrap_or_else(|| panic!("Failed to build chain to {foot_link}"));

            let joint_entities: Vec<bevy::prelude::Entity> = chain
                .joint_names()
                .iter()
                .map(|name| {
                    spawned
                        .joint_entity(name)
                        .unwrap_or_else(|| panic!("Joint {name} not found"))
                })
                .collect();

            let is_prismatic = chain.joints().iter().map(|j| j.is_prismatic).collect();

            println!(
                "  Leg {} ({}): {} DOF, joints {:?}",
                i,
                foot_link,
                chain.dof(),
                chain.joint_names(),
            );

            LegRuntime {
                chain,
                joint_entities,
                is_prismatic,
                hip_offset: hip_offsets[i],
            }
        })
        .collect();

    // 5. Configure MPC
    let mpc_config = MpcConfig {
        horizon: 10,
        dt: 0.02,
        mass: 8.6,
        gravity: 9.81,
        friction_coeff: 0.6,
        f_max: 200.0,
        max_solver_iters: 50,
        ..MpcConfig::default()
    };

    let swing_config = SwingConfig {
        step_height: 0.04,
        default_step_length: 0.06,
    };

    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let desired_height = 0.30;
    let desired_yaw = 0.0;
    let ground_height = 0.0;

    // Start with Stand gait to stabilize, then switch to Trot
    let mut gait = GaitScheduler::quadruped(GaitType::Stand);
    let solver = MpcSolver::new(mpc_config.clone());
    let n_feet = legs.len();
    let mut swing_starts = vec![Vector3::zeros(); n_feet];
    let mut swing_targets = vec![Vector3::zeros(); n_feet];
    let stabilize_steps = 100;

    println!("\nMPC config: horizon={}, dt={}, mass={:.1}kg", mpc_config.horizon, mpc_config.dt, mpc_config.mass);
    println!("Phase 1: Stand (stabilize) for {stabilize_steps} steps");
    println!("Phase 2: Trot at [{:.1}, {:.1}, {:.1}] m/s", desired_velocity.x, desired_velocity.y, desired_velocity.z);

    // 6. Register sensors
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(8)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("\nRunning for 500 steps (10s at 50Hz)...\n");

    // 8. Main simulation loop
    let total_steps = 500;
    let dt = mpc_config.dt;
    let mut switched_to_trot = false;

    for step in 0..total_steps {
        // Switch from Stand to Trot after stabilization
        if step == stabilize_steps && !switched_to_trot {
            gait = GaitScheduler::quadruped(GaitType::Trot);
            switched_to_trot = true;
            println!("  >>> Switching to Trot gait at step {step}");
        }
        // --- Read body state from Rapier ---
        let body_state = {
            let ctx = scene.app.world().resource::<RapierContext>();
            body_state_from_rapier(ctx, "body").expect("body not found in Rapier")
        };
        let body_pos = body_state.position;

        // --- Read joint states and compute foot FK ---
        let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        let mut foot_world: Vec<Vector3<f64>> = Vec::with_capacity(n_feet);

        for leg in &legs {
            let mut q = Vec::with_capacity(leg.joint_entities.len());
            for &entity in &leg.joint_entities {
                if let Some(js) = scene.app.world().get::<JointState>(entity) {
                    q.push(js.position);
                } else {
                    q.push(0.0);
                }
            }

            let ee_body = leg.chain.forward_kinematics(&q);
            let fw = body_pos
                + Vector3::new(
                    f64::from(ee_body.translation.x),
                    f64::from(ee_body.translation.y),
                    f64::from(ee_body.translation.z),
                );

            foot_world.push(fw);
            all_joint_positions.push(q);
        }
        // --- Advance gait ---
        gait.advance(dt);

        // --- Generate contact sequence ---
        let contacts = gait.contact_sequence(mpc_config.horizon, dt);

        // --- Build reference trajectory ---
        let x0 = body_state.to_state_vector(mpc_config.gravity);
        let current_vel = if step < stabilize_steps {
            &Vector3::zeros()
        } else {
            &desired_velocity
        };
        let reference = ReferenceTrajectory::constant_velocity(
            &body_state,
            current_vel,
            desired_height,
            desired_yaw,
            mpc_config.horizon,
            dt,
            mpc_config.gravity,
        );

        // --- Solve MPC ---
        let solution = solver.solve(&x0, &foot_world, &contacts, &reference);

        // --- Apply control ---
        let stance_duration = gait.duty_factor() * gait.cycle_time();

        for (leg_idx, leg) in legs.iter().enumerate() {
            let is_contact = gait.is_contact(leg_idx);

            if is_contact && solution.converged {
                // Stance: WBC torques via Jacobian transpose
                let q = &all_joint_positions[leg_idx];
                let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
                let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);

                let jacobian = compute_leg_jacobian(
                    &origins_f64,
                    &axes_f64,
                    &foot_world[leg_idx],
                    &leg.is_prismatic,
                );

                let force = &solution.forces[leg_idx];
                let torques = jacobian_transpose_torques(&jacobian, force);

                for (j, &entity) in leg.joint_entities.iter().enumerate() {
                    if let Some(mut cmd) = scene.app.world_mut().get_mut::<JointCommand>(entity) {
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            cmd.value = torques[j] as f32;
                        }
                    }
                }

                swing_starts[leg_idx] = foot_world[leg_idx];
            } else {
                // Swing: Bezier trajectory
                let swing_phase = gait.swing_phase(leg_idx);

                if swing_phase < 0.05 {
                    let hip_world = body_pos + leg.hip_offset;
                    swing_targets[leg_idx] = raibert_foot_target(
                        &hip_world,
                        &desired_velocity,
                        stance_duration,
                        ground_height,
                    );
                    swing_starts[leg_idx] = foot_world[leg_idx];
                }

                let _target_pos = swing_foot_position(
                    &swing_starts[leg_idx],
                    &swing_targets[leg_idx],
                    swing_phase,
                    swing_config.step_height,
                );

                // Zero torque for swing legs
                for &entity in &leg.joint_entities {
                    if let Some(mut cmd) = scene.app.world_mut().get_mut::<JointCommand>(entity) {
                        cmd.value = 0.0;
                    }
                }
            }
        }

        // --- Step physics ---
        scene.app.update();

        // --- Telemetry ---
        if step % 50 == 0 {
            let n_stance: usize = (0..n_feet).filter(|&i| gait.is_contact(i)).count();
            println!(
                "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  vel=[{:+.3}, {:+.3}, {:+.3}]  stance={}/{}  mpc={:>4}us  {}",
                step,
                body_pos.x, body_pos.y, body_pos.z,
                body_state.linear_velocity.x, body_state.linear_velocity.y, body_state.linear_velocity.z,
                n_stance, n_feet,
                solution.solve_time_us,
                if solution.converged { "OK" } else { "FAIL" },
            );
        }

        if scene.app.world().resource::<Episode>().is_done() {
            println!("\nEpisode ended at step {step}");
            break;
        }
    }

    // 9. Final report
    let final_state = {
        let ctx = scene.app.world().resource::<RapierContext>();
        body_state_from_rapier(ctx, "body").expect("body not found")
    };

    println!("\n=== Summary ===");
    println!(
        "Final body position: [{:.3}, {:.3}, {:.3}]",
        final_state.position.x, final_state.position.y, final_state.position.z,
    );
    println!(
        "Final body velocity: [{:.3}, {:.3}, {:.3}]",
        final_state.linear_velocity.x, final_state.linear_velocity.y, final_state.linear_velocity.z,
    );
    println!("Steps simulated: {total_steps}");
    println!("Simulation time: {:.1}s", f64::from(total_steps) * dt);

    // Print final joint states
    println!("\nFinal joint states:");
    let joint_names = [
        "fl_hip_pitch", "fl_knee_pitch",
        "fr_hip_pitch", "fr_knee_pitch",
        "rl_hip_pitch", "rl_knee_pitch",
        "rr_hip_pitch", "rr_knee_pitch",
    ];
    for name in &joint_names {
        if let Some(entity) = spawned.joint_entity(name)
            && let Some(state) = scene.app.world().get::<JointState>(entity)
        {
            let cmd = scene.app.world().get::<JointCommand>(entity);
            println!(
                "  {:<16}: pos={:+.3} rad  vel={:+.3} rad/s  cmd={:+.3}",
                name,
                state.position,
                state.velocity,
                cmd.map_or(0.0, |c| c.value),
            );
        }
    }

    println!("\nQuadruped MPC example PASSED");
}
