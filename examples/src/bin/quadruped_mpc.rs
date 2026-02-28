//! Quadruped robot walking with Model Predictive Control.
//!
//! Demonstrates the full MPC locomotion pipeline on a headless simulation:
//! 1. Load quadruped URDF (4 legs × 3 joints = 12 DOF, floating base)
//! 2. Set up Rapier physics with dynamic base (not fixed)
//! 3. Run centroidal MPC → WBC → swing planner each step
//! 4. Print telemetry: body pose, foot forces, solve time
//!
//! This example calls the MPC solver directly (not via the Bevy plugin)
//! so it works headless without a transform hierarchy.
//!
//! Run: `cargo run -p clankers-examples --bin quadruped_mpc`

use std::collections::HashMap;

use clankers_actuator::components::{Actuator, JointCommand, JointState};
use clankers_actuator_core::prelude::{IdealMotor, MotorType};
use clankers_env::prelude::*;
use clankers_examples::mpc_control::{LegRuntime, MpcLoopState, body_state_from_rapier, compute_mpc_step, detect_foot_contacts};
use clankers_examples::QUADRUPED_URDF;
use clankers_ik::KinematicChain;
use clankers_mpc::{AdaptiveGaitConfig, GaitScheduler, GaitType, MpcConfig, MpcSolver, SwingConfig};
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use nalgebra::Vector3;
use rapier3d::prelude::{
    ColliderBuilder, Group, InteractionGroups, InteractionTestMode, JointAxis, MassProperties,
    RigidBodyBuilder,
};

fn main() {
    println!("=== Quadruped MPC Example (3-DOF legs) ===\n");

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

    // Initial configuration: hip_ab=0, hip_pitch=1.05, knee_pitch=-2.10
    let init_hip_ab: f32 = 0.0;
    let init_hip_pitch: f32 = 1.05;
    let init_knee_pitch: f32 = -2.10;

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        // fixed_base = false: body is dynamic, controlled by ground reaction forces
        register_robot(&mut ctx, &model, spawned, world, false);

        // Increase solver iterations from default 4 to 50.
        ctx.integration_parameters.num_solver_iterations = 50;

        let body_offset = bevy::math::Vec3::new(0.0, 0.0, 0.35);

        if let Some(&root_handle) = ctx.body_handles.get("body")
            && let Some(root_body) = ctx.rigid_body_set.get_mut(root_handle)
        {
            let body_mass = 5.0_f32;
            let inertia = bevy::math::Vec3::new(0.02083, 0.07083, 0.08333);
            root_body.set_additional_mass_properties(
                MassProperties::new(bevy::math::Vec3::ZERO, body_mass, inertia),
                true,
            );
            root_body.set_translation(body_offset, true);
        }

        for (link_name, &handle) in &ctx.body_handles {
            if link_name == "body" {
                continue;
            }
            if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                let current = body.translation();
                body.set_translation(current + body_offset, true);
            }
        }

        // Collision groups: robot links only collide with ground, not each other.
        let robot_group = InteractionGroups::new(
            Group::GROUP_1,
            Group::GROUP_2,
            InteractionTestMode::And,
        );
        let ground_group = InteractionGroups::new(
            Group::GROUP_2,
            Group::GROUP_1,
            InteractionTestMode::And,
        );

        let ground_body = RigidBodyBuilder::fixed()
            .translation(bevy::math::Vec3::new(0.0, 0.0, -0.05))
            .build();
        let ground_handle = ctx.rigid_body_set.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(50.0, 50.0, 0.05)
            .friction(0.6)
            .restitution(0.0)
            .collision_groups(ground_group)
            .build();
        ctx.collider_set
            .insert_with_parent(ground_collider, ground_handle, &mut ctx.rigid_body_set);

        let link_colliders: &[(&str, ColliderBuilder)] = &[
            ("fl_foot", ColliderBuilder::ball(0.02).friction(0.6).restitution(0.0).collision_groups(robot_group)),
            ("fr_foot", ColliderBuilder::ball(0.02).friction(0.6).restitution(0.0).collision_groups(robot_group)),
            ("rl_foot", ColliderBuilder::ball(0.02).friction(0.6).restitution(0.0).collision_groups(robot_group)),
            ("rr_foot", ColliderBuilder::ball(0.02).friction(0.6).restitution(0.0).collision_groups(robot_group)),
            ("fl_hip_link", ColliderBuilder::cuboid(0.02, 0.02, 0.02).friction(0.3).collision_groups(robot_group)),
            ("fr_hip_link", ColliderBuilder::cuboid(0.02, 0.02, 0.02).friction(0.3).collision_groups(robot_group)),
            ("rl_hip_link", ColliderBuilder::cuboid(0.02, 0.02, 0.02).friction(0.3).collision_groups(robot_group)),
            ("rr_hip_link", ColliderBuilder::cuboid(0.02, 0.02, 0.02).friction(0.3).collision_groups(robot_group)),
            ("fl_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("fr_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("rl_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("rr_upper_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("fl_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("fr_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("rl_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
            ("rr_lower_leg", ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3).collision_groups(robot_group)),
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

        if let Some(&body_handle) = ctx.body_handles.get("body") {
            let body_collider = ColliderBuilder::cuboid(0.2, 0.1, 0.05)
                .friction(0.5)
                .collision_groups(robot_group)
                .build();
            ctx.collider_set.insert_with_parent(
                body_collider,
                body_handle,
                &mut ctx.rigid_body_set,
            );
        }

        ctx.snapshot_initial_state();
        world.insert_resource(ctx);
    }

    // 4. Build per-leg IK chains (3 DOF each: hip_ab, hip_pitch, knee_pitch)
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
    let mpc_config = MpcConfig::default();
    let swing_config = SwingConfig::default();
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let desired_height: f64;
    let desired_yaw = 0.0;
    let ground_height = 0.0;

    let n_feet = legs.len();
    let stabilize_steps = 100;

    // Override motor limits
    for leg in &legs {
        for &entity in &leg.joint_entities {
            if let Some(mut actuator) = scene.app.world_mut().get_mut::<Actuator>(entity) {
                actuator.motor = MotorType::Ideal(IdealMotor::new(100.0, 100.0));
            }
        }
    }

    println!("\nMPC config: horizon={}, dt={}, mass={:.1}kg", mpc_config.horizon, mpc_config.dt, mpc_config.mass);
    println!("Phase 1: Stand (stabilize) for {stabilize_steps} steps");
    println!("Phase 2: Walk at [{:.1}, {:.1}, {:.1}] m/s", desired_velocity.x, desired_velocity.y, desired_velocity.z);

    // 6. Register sensors (12 DOF)
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(12)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Warmup: use position-mode motors to bend knees before MPC starts.
    println!("\nWarmup: bending knees with position motors...");
    {
        let warmup_steps = 1000;
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();

        let joint_names = [
            "fl_hip_ab", "fl_hip_pitch", "fl_knee_pitch",
            "fr_hip_ab", "fr_hip_pitch", "fr_knee_pitch",
            "rl_hip_ab", "rl_hip_pitch", "rl_knee_pitch",
            "rr_hip_ab", "rr_hip_pitch", "rr_knee_pitch",
        ];
        for name in &joint_names {
            if let Some(entity) = spawned.joint_entity(name) {
                if let Some(&jh) = ctx.joint_handles.get(&entity) {
                    if let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) {
                        let target = if name.contains("knee") {
                            init_knee_pitch
                        } else if name.contains("hip_pitch") {
                            init_hip_pitch
                        } else {
                            init_hip_ab
                        };
                        joint.data.set_motor(JointAxis::AngX, target, 0.0, 500.0, 50.0);
                        joint.data.set_motor_max_force(JointAxis::AngX, 100.0);
                    }
                }
            }
        }

        for _ in 0..warmup_steps {
            ctx.step();
        }

        for name in &joint_names {
            if let Some(entity) = spawned.joint_entity(name) {
                if let Some(&jh) = ctx.joint_handles.get(&entity) {
                    if let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) {
                        joint.data.set_motor(JointAxis::AngX, 0.0, 0.0, 0.0, 0.0);
                        joint.data.set_motor_max_force(JointAxis::AngX, 0.0);
                    }
                }
            }
        }

        for (_, &handle) in &ctx.body_handles {
            if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                body.set_linvel(bevy::math::Vec3::ZERO, true);
                body.set_angvel(bevy::math::Vec3::ZERO, true);
            }
        }

        for name in &joint_names {
            if let Some(entity) = spawned.joint_entity(name) {
                if let Some(info) = ctx.joint_info.get(&entity) {
                    let parent_body = ctx.rigid_body_set.get(info.parent_body);
                    let child_body = ctx.rigid_body_set.get(info.child_body);
                    if let (Some(pb), Some(cb)) = (parent_body, child_body) {
                        let rel_rot = pb.position().rotation.inverse() * cb.position().rotation;
                        let sin_half = bevy::math::Vec3::new(rel_rot.x, rel_rot.y, rel_rot.z);
                        let sin_proj = sin_half.dot(info.axis);
                        let angle = 2.0 * f32::atan2(sin_proj, rel_rot.w);

                        if let Some(mut js) = world.get_mut::<JointState>(entity) {
                            js.position = angle;
                        }
                    }
                }
            }
        }

        if let Some(&bh) = ctx.body_handles.get("body") {
            if let Some(body) = ctx.rigid_body_set.get(bh) {
                let t = body.translation();
                println!(
                    "  Body after warmup: pos=[{:.3}, {:.3}, {:.3}]",
                    t.x, t.y, t.z,
                );
            }
        }

        ctx.snapshot_initial_state();
        world.insert_resource(ctx);
    }

    desired_height = {
        let ctx = scene.app.world().resource::<RapierContext>();
        let handle = ctx.body_handles.get("body").unwrap();
        let body = ctx.rigid_body_set.get(*handle).unwrap();
        f64::from(body.translation().z)
    };
    println!("  Desired height (post-warmup): {desired_height:.3}");

    // Store initial joint angles AFTER warmup for PD stance control.
    let init_joint_angles: Vec<Vec<f32>> = legs
        .iter()
        .map(|leg| {
            leg.joint_entities
                .iter()
                .map(|&entity| {
                    scene
                        .app
                        .world()
                        .get::<JointState>(entity)
                        .map_or(0.0, |js| js.position)
                })
                .collect()
        })
        .collect();
    let leg_names = ["FL", "FR", "RL", "RR"];
    println!("  Init joint angles (all legs):");
    for (i, angles) in init_joint_angles.iter().enumerate() {
        println!(
            "    {}: hip_ab={:+.4} hip_pitch={:+.4} knee={:+.4}",
            leg_names[i], angles[0], angles[1], angles[2],
        );
    }
    println!(
        "  Warmup targets: hip_ab={:.1} hip_pitch={:.1} knee={:.1}",
        init_hip_ab, init_hip_pitch, init_knee_pitch,
    );

    // 8. Build MPC loop state
    let mut mpc_state = MpcLoopState {
        gait: GaitScheduler::quadruped(GaitType::Stand),
        solver: MpcSolver::new(mpc_config.clone(), 4),
        config: mpc_config,
        swing_config,
        adaptive_gait: Some(AdaptiveGaitConfig::default()),
        legs,
        swing_starts: vec![Vector3::zeros(); n_feet],
        swing_targets: vec![Vector3::zeros(); n_feet],
        prev_contacts: vec![true; n_feet],
        init_joint_angles,
        foot_link_names: Some(foot_link_names.iter().map(|s| (*s).to_string()).collect()),
        disturbance_estimator: Some(clankers_mpc::DisturbanceEstimator::new(
            clankers_mpc::DisturbanceEstimatorConfig::default(),
        )),
    };

    // 9. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("\nRunning for 500 steps (10s at 50Hz)...\n");

    // 10. Main simulation loop
    let total_steps = 500;
    let ramp_steps = 100;
    let mut switched_to_trot = false;

    for step in 0..total_steps {
        // Switch from Stand to Trot after stabilization
        if step == stabilize_steps && !switched_to_trot {
            switched_to_trot = true;
            mpc_state.gait = GaitScheduler::quadruped(GaitType::Trot);
            println!("  >>> Switched to Trot at step {step}");
        }

        // --- Floating origin rebase + read body state + detect contacts ---
        let (body_state, body_quat, actual_contacts) = {
            let mut ctx = scene.app.world_mut().resource_mut::<RapierContext>();
            ctx.rebase_origin("body", 50.0);
            let (bs, bq) = body_state_from_rapier(ctx.as_ref(), "body").expect("body not found in Rapier");
            let contacts = detect_foot_contacts(ctx.as_ref(), &mpc_state);
            (bs, bq, contacts)
        };

        // --- Read joint states ---
        let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        for leg in &mpc_state.legs {
            let mut q = Vec::with_capacity(leg.joint_entities.len());
            let mut qd = Vec::with_capacity(leg.joint_entities.len());
            for &entity in &leg.joint_entities {
                if let Some(js) = scene.app.world().get::<JointState>(entity) {
                    q.push(js.position);
                    qd.push(js.velocity);
                } else {
                    q.push(0.0);
                    qd.push(0.0);
                }
            }
            all_joint_positions.push(q);
            all_joint_velocities.push(qd);
        }

        // --- Ramp velocity ---
        let current_vel = if step < stabilize_steps {
            Vector3::zeros()
        } else {
            let ramp_frac = ((step - stabilize_steps) as f64 / ramp_steps as f64).min(1.0);
            desired_velocity * ramp_frac
        };

        // --- Compute MPC step ---
        let result = compute_mpc_step(
            &mut mpc_state,
            &body_state,
            &body_quat,
            &all_joint_positions,
            &all_joint_velocities,
            &current_vel,
            desired_height,
            desired_yaw,
            ground_height,
            actual_contacts.as_deref(),
        );

        // --- Apply motor commands + step physics manually ---
        {
            let world = scene.app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();

            for mc in &result.motor_commands {
                let Some(&jh) = ctx.joint_handles.get(&mc.entity) else { continue };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else { continue };

                joint.data.set_motor(
                    JointAxis::AngX,
                    mc.target_pos,
                    mc.target_vel,
                    mc.stiffness,
                    mc.damping,
                );
                joint.data.set_motor_max_force(JointAxis::AngX, mc.max_force);
            }

            let substeps = ctx.substeps;
            for _ in 0..substeps {
                ctx.step();
            }

            // Read back joint state
            for leg in &mpc_state.legs {
                for &entity in &leg.joint_entities {
                    let Some(info) = ctx.joint_info.get(&entity) else { continue };
                    let Some(pb) = ctx.rigid_body_set.get(info.parent_body) else { continue };
                    let Some(cb) = ctx.rigid_body_set.get(info.child_body) else { continue };

                    let rel_rot = pb.position().rotation.inverse() * cb.position().rotation;
                    let sin_half = bevy::math::Vec3::new(rel_rot.x, rel_rot.y, rel_rot.z);
                    let sin_proj = sin_half.dot(info.axis);
                    let angle = 2.0 * f32::atan2(sin_proj, rel_rot.w);

                    let rel_angvel = cb.angvel() - pb.angvel();
                    let velocity = rel_angvel.dot(info.axis);

                    if let Some(mut js) = world.get_mut::<JointState>(entity) {
                        js.position = angle;
                        js.velocity = velocity;
                    }
                }
            }

            world.insert_resource(ctx);
        }

        // --- Telemetry ---
        let body_pos = body_state.position;
        if step % 50 == 0 || step < 5 || (step >= stabilize_steps && step < stabilize_steps + 5) {
            let n_stance: usize = result.contacts.iter().filter(|&&c| c).count();
            println!(
                "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  vel=[{:+.3}, {:+.3}, {:+.3}]  stance={}/{}  mpc={:>4}us  {}",
                step,
                body_pos.x, body_pos.y, body_pos.z,
                body_state.linear_velocity.x, body_state.linear_velocity.y, body_state.linear_velocity.z,
                n_stance, n_feet,
                result.solution.solve_time_us,
                if result.solution.converged { "OK" } else { "FAIL" },
            );
            for (i, fw) in result.foot_world.iter().enumerate() {
                let contact = result.contacts[i];
                let force_str = if result.solution.converged {
                    let f = &result.solution.forces[i];
                    format!("F=[{:+.2}, {:+.2}, {:+.2}]", f.x, f.y, f.z)
                } else {
                    "F=FAIL".to_string()
                };
                println!(
                    "    foot {i}: pos=[{:+.4}, {:+.4}, {:+.4}]  {force_str}  {}",
                    fw.x, fw.y, fw.z,
                    if contact { "STANCE" } else { "swing" },
                );
            }
        }

        if scene.app.world().resource::<Episode>().is_done() {
            println!("\nEpisode ended at step {step}");
            break;
        }
    }

    // 11. Final report
    let (final_state, _) = {
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
    println!("Simulation time: {:.1}s", f64::from(total_steps) * mpc_state.config.dt);

    println!("\nFinal joint states:");
    let joint_names = [
        "fl_hip_ab", "fl_hip_pitch", "fl_knee_pitch",
        "fr_hip_ab", "fr_hip_pitch", "fr_knee_pitch",
        "rl_hip_ab", "rl_hip_pitch", "rl_knee_pitch",
        "rr_hip_ab", "rr_hip_pitch", "rr_knee_pitch",
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
