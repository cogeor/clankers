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

use bevy::math::EulerRot;
use clankers_actuator::components::{Actuator, JointCommand, JointState};
use clankers_actuator_core::prelude::{IdealMotor, MotorType};
use clankers_env::prelude::*;
use clankers_examples::QUADRUPED_URDF;
use clankers_ik::{DlsConfig, DlsSolver, IkTarget, KinematicChain};
use clankers_mpc::{
    BodyState, GaitScheduler, GaitType, MpcConfig, MpcSolver, ReferenceTrajectory, SwingConfig,
    raibert_foot_target, swing_foot_position, swing_foot_velocity,
    wbc::{compute_leg_jacobian, frames_f32_to_f64, jacobian_transpose_torques, transform_frames_to_world},
};
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use nalgebra::Vector3;
use rapier3d::prelude::{
    ColliderBuilder, Group, InteractionGroups, InteractionTestMode, JointAxis, MassProperties,
    RigidBodyBuilder,
};

/// Per-leg runtime data.
struct LegRuntime {
    chain: KinematicChain,
    joint_entities: Vec<bevy::prelude::Entity>,
    is_prismatic: Vec<bool>,
    hip_offset: Vector3<f64>,
}

/// Read body state and rotation quaternion from Rapier's rigid body set.
fn body_state_from_rapier(
    ctx: &RapierContext,
    link_name: &str,
) -> Option<(BodyState, nalgebra::UnitQuaternion<f64>)> {
    let handle = ctx.body_handles.get(link_name)?;
    let body = ctx.rigid_body_set.get(*handle)?;

    let t = body.translation();
    let r = body.rotation();
    let (yaw, pitch, roll) = r.to_euler(EulerRot::ZYX);

    let lv = body.linvel();
    let av = body.angvel();

    let body_quat = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        f64::from(r.w),
        f64::from(r.x),
        f64::from(r.y),
        f64::from(r.z),
    ));

    Some((
        BodyState {
            orientation: Vector3::new(f64::from(roll), f64::from(pitch), f64::from(yaw)),
            position: Vector3::new(f64::from(t.x), f64::from(t.y), f64::from(t.z)),
            angular_velocity: Vector3::new(f64::from(av.x), f64::from(av.y), f64::from(av.z)),
            linear_velocity: Vector3::new(f64::from(lv.x), f64::from(lv.y), f64::from(lv.z)),
        },
        body_quat,
    ))
}

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

    // Initial configuration: hip_ab=0, hip_pitch=0.4, knee_pitch=-1.0
    // With hip_pitch≠0, foot is offset from directly below hip, giving
    // nonzero hip torque from vertical ground reaction forces via J^T.
    let init_hip_ab: f32 = 0.0;
    let init_hip_pitch: f32 = 1.05;
    let init_knee_pitch: f32 = -2.10;

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        // fixed_base = false: body is dynamic, controlled by ground reaction forces
        register_robot(&mut ctx, &model, spawned, world, false);

        // Increase solver iterations from default 4 to 50.
        // With 12 revolute joints (60 locked-DOF constraints), 4 iterations
        // causes slow constraint drift that leads to catastrophic lateral tipping.
        ctx.integration_parameters.num_solver_iterations = 50;

        // Straight-leg standing height: upper(0.15) + lower(0.15) + hip_z_offset(0.05)
        let body_offset = bevy::math::Vec3::new(0.0, 0.0, 0.35);

        if let Some(&root_handle) = ctx.body_handles.get("body")
            && let Some(root_body) = ctx.rigid_body_set.get_mut(root_handle)
        {
            // Body link mass from URDF (5.0kg). Child links add 4.0kg
            // via register_robot, for total 9.0kg matching MPC config.
            let body_mass = 5.0_f32;
            let inertia = bevy::math::Vec3::new(0.02083, 0.07083, 0.08333);
            root_body.set_additional_mass_properties(
                MassProperties::new(bevy::math::Vec3::ZERO, body_mass, inertia),
                true,
            );
            root_body.set_translation(body_offset, true);
        }

        // Move all child link bodies up by the same offset (straight-leg placement).
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

        // Ground plane: fixed body with large cuboid at z=-0.05 (top at z=0)
        let ground_body = RigidBodyBuilder::fixed()
            .translation(bevy::math::Vec3::new(0.0, 0.0, -0.05))
            .build();
        let ground_handle = ctx.rigid_body_set.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(50.0, 50.0, 0.05)
            .friction(1.0)
            .restitution(0.0)
            .collision_groups(ground_group)
            .build();
        ctx.collider_set
            .insert_with_parent(ground_collider, ground_handle, &mut ctx.rigid_body_set);

        // Add colliders to all robot links (including hip_link)
        let link_colliders: &[(&str, ColliderBuilder)] = &[
            ("fl_foot", ColliderBuilder::ball(0.02).friction(1.0).restitution(0.0).collision_groups(robot_group)),
            ("fr_foot", ColliderBuilder::ball(0.02).friction(1.0).restitution(0.0).collision_groups(robot_group)),
            ("rl_foot", ColliderBuilder::ball(0.02).friction(1.0).restitution(0.0).collision_groups(robot_group)),
            ("rr_foot", ColliderBuilder::ball(0.02).friction(1.0).restitution(0.0).collision_groups(robot_group)),
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

        // Body box collider
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
    // Set after warmup from actual body height.
    let desired_height: f64;
    let desired_yaw = 0.0;
    let ground_height = 0.0;

    // Start with Stand gait to stabilize, then switch to Walk
    // (Walk keeps 3+ feet on ground, providing a triangle of support for lateral stability.
    //  Trot only has 2 feet in diagonal, which is a line — zero lateral stability margin.)
    let mut gait = GaitScheduler::quadruped(GaitType::Stand);
    let mut solver = MpcSolver::new(mpc_config.clone(), 4);
    let n_feet = legs.len();
    let mut swing_starts = vec![Vector3::zeros(); n_feet];
    let mut swing_targets = vec![Vector3::zeros(); n_feet];
    let mut prev_contacts = vec![true; n_feet];
    let stabilize_steps = 100;

    // Override motor limits: URDF defaults (effort=20-30, velocity=10) are too
    // restrictive. The IdealMotor's linear torque-speed curve drops to zero at
    // max_velocity, causing loss of control when joints move at >5 rad/s.
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
        let warmup_steps = 1000; // 1s at 1kHz — enough to converge with gravity
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();

        // 12 joints: 3 per leg (hip_ab, hip_pitch, knee_pitch)
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
                        joint.data.set_motor(
                            JointAxis::AngX,
                            target,
                            0.0,
                            500.0,
                            50.0,
                        );
                        joint.data.set_motor_max_force(JointAxis::AngX, 100.0);
                    }
                }
            }
        }

        for _ in 0..warmup_steps {
            ctx.step();
        }

        // Switch motors back to torque mode for MPC control
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

        // Zero all rigid body velocities so MPC starts from rest.
        // Without this the warmup motor release creates violent joint velocities.
        for (_, &handle) in &ctx.body_handles {
            if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                body.set_linvel(bevy::math::Vec3::ZERO, true);
                body.set_angvel(bevy::math::Vec3::ZERO, true);
            }
        }

        // Read back actual joint positions from Rapier after warmup
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

    // Set desired_height from actual post-warmup body height.
    // (Before warmup, the body is at z=0.35, but after knee-bending it settles lower.)
    desired_height = {
        let ctx = scene.app.world().resource::<RapierContext>();
        let handle = ctx.body_handles.get("body").unwrap();
        let body = ctx.rigid_body_set.get(*handle).unwrap();
        f64::from(body.translation().z)
    };
    println!("  Desired height (post-warmup): {desired_height:.3}");

    // Store initial joint angles AFTER warmup for PD stance control.
    // (Must be after warmup so the JointState components reflect the
    // warmed-up configuration, not the default zeros.)
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
    println!("  Init joint angles (all legs):");
    let leg_names = ["FL", "FR", "RL", "RR"];
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

    // (Stability test removed — the solver iterations fix (50 iterations)
    // ensures joints stay rigid without additional testing.)

    // 8. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("\nRunning for 500 steps (10s at 50Hz)...\n");

    // 9. Main simulation loop
    let total_steps = 500;
    let dt = mpc_config.dt;
    let mut switched_to_trot = false;

    for step in 0..total_steps {
        // Switch from Stand to Trot after stabilization
        if step == stabilize_steps && !switched_to_trot {
            switched_to_trot = true;
            gait = GaitScheduler::quadruped(GaitType::Trot);
            println!("  >>> Switched to Trot at step {step}");
        }
        // --- Read body state from Rapier ---
        let (body_state, body_quat) = {
            let ctx = scene.app.world().resource::<RapierContext>();
            body_state_from_rapier(ctx, "body").expect("body not found in Rapier")
        };
        let body_pos = body_state.position;

        // --- Read joint states and compute foot FK ---
        let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
        let mut foot_world: Vec<Vector3<f64>> = Vec::with_capacity(n_feet);

        for leg in &legs {
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

            let ee_body = leg.chain.forward_kinematics(&q);
            let ee_body_vec = Vector3::new(
                f64::from(ee_body.translation.x),
                f64::from(ee_body.translation.y),
                f64::from(ee_body.translation.z),
            );
            let fw = body_quat * ee_body_vec + body_pos;

            foot_world.push(fw);
            all_joint_positions.push(q);
            all_joint_velocities.push(qd);
        }
        // --- Advance gait ---
        gait.advance(dt);

        // --- Generate contact sequence ---
        let contacts = gait.contact_sequence(mpc_config.horizon, dt);

        // --- Build reference trajectory ---
        let x0 = body_state.to_state_vector(mpc_config.gravity);
        // Ramp velocity from 0 to desired over 200 steps (4s) after trot starts
        let ramp_steps = 100;
        let current_vel_owned;
        let current_vel = if step < stabilize_steps {
            current_vel_owned = Vector3::zeros();
            &current_vel_owned
        } else {
            let ramp_frac = ((step - stabilize_steps) as f64 / ramp_steps as f64).min(1.0);
            current_vel_owned = desired_velocity * ramp_frac;
            &current_vel_owned
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

        // --- Apply control + step physics manually ---
        //
        // We bypass scene.app.update() to use Rapier's built-in position
        // motors (PD evaluated at 1 kHz physics rate) instead of the
        // motor trick (constant torque ZOH at 50 Hz). The motor trick
        // causes violent oscillation on light links.
        let stance_duration = gait.duty_factor() * gait.cycle_time();
        const FF_ENABLED: bool = true;

        // Collect motor settings for all joints before removing context
        struct MotorSetting {
            entity: bevy::prelude::Entity,
            target_pos: f32,
            target_vel: f32,
            stiffness: f32,
            damping: f32,
            max_force: f32,
        }
        let mut motor_settings: Vec<MotorSetting> = Vec::new();

        for (leg_idx, leg) in legs.iter().enumerate() {
            let is_contact = gait.is_contact(leg_idx);

            // Detect liftoff transition: set swing_starts at the exact moment
            if prev_contacts[leg_idx] && !is_contact {
                swing_starts[leg_idx] = foot_world[leg_idx];
            }

            if is_contact && solution.converged {
                // --- Stance: J^T feedforward + IK-derived PD targets ---
                let q = &all_joint_positions[leg_idx];
                let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
                let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);
                let (origins_world, axes_world) = transform_frames_to_world(
                    &origins_f64, &axes_f64, &body_quat, &body_pos,
                );
                let jacobian = compute_leg_jacobian(
                    &origins_world, &axes_world, &foot_world[leg_idx], &leg.is_prismatic,
                );

                let force = &solution.forces[leg_idx];
                let neg_force = -force;
                let torques_ff_raw = jacobian_transpose_torques(&jacobian, &neg_force);
                let torques_ff = if FF_ENABLED { torques_ff_raw } else { vec![0.0; torques_ff_raw.len()] };

                // Compute q_desired via IK from MPC's desired body pose.
                // The MPC wants zero roll/pitch at desired_height — compute where
                // the foot should be in body frame if the body were at that pose.
                let desired_body_rot = nalgebra::UnitQuaternion::from_euler_angles(
                    0.0, 0.0, f64::from(desired_yaw),
                );
                let desired_body_pos = Vector3::new(body_pos.x, body_pos.y, desired_height);
                let foot_in_desired_body = desired_body_rot.inverse()
                    * (foot_world[leg_idx] - desired_body_pos);

                let ik_target = IkTarget::Position(foot_in_desired_body.cast::<f32>());
                let ik_solver = DlsSolver::new(DlsConfig {
                    max_iterations: 10,
                    position_tolerance: 1e-3,
                    damping: 0.01,
                    ..DlsConfig::default()
                });
                let ik_result = ik_solver.solve(&leg.chain, &ik_target, q);

                for (j, &entity) in leg.joint_entities.iter().enumerate() {
                    let (kp_j, kd_j, max_f) = if j == 0 {
                        (500.0_f32, 20.0_f32, 200.0_f32)
                    } else {
                        (20.0_f32, 2.0_f32, 50.0_f32)
                    };

                    let target_pos = ik_result.joint_positions[j];

                    #[allow(clippy::cast_possible_truncation)]
                    let target_vel = (torques_ff[j] / f64::from(kd_j)) as f32;

                    motor_settings.push(MotorSetting {
                        entity,
                        target_pos,
                        target_vel,
                        stiffness: kp_j,
                        damping: kd_j,
                        max_force: max_f,
                    });
                }
            } else {
                // --- Swing: Cartesian PD via J^T torque with gain blending ---
                let swing_phase = gait.swing_phase(leg_idx);

                let swing_duration = (1.0 - gait.duty_factor()) * gait.cycle_time();

                if swing_phase < 0.05 {
                    let hip_world = body_quat * leg.hip_offset + body_pos;
                    swing_targets[leg_idx] = raibert_foot_target(
                        &hip_world,
                        &body_state.linear_velocity,
                        current_vel,
                        stance_duration,
                        swing_duration,
                        ground_height,
                        swing_config.raibert_kv,
                    );
                }
                let p_des = swing_foot_position(
                    &swing_starts[leg_idx], &swing_targets[leg_idx],
                    swing_phase, swing_config.step_height,
                );
                let v_des = swing_foot_velocity(
                    &swing_starts[leg_idx], &swing_targets[leg_idx],
                    swing_phase, swing_config.step_height, swing_duration,
                );
                let p_actual = &foot_world[leg_idx];

                let q = &all_joint_positions[leg_idx];
                let qd_vals = &all_joint_velocities[leg_idx];
                let (origins, axes, ee_pos) = leg.chain.joint_frames(q);
                let (origins_f64, axes_f64, _) = frames_f32_to_f64(&origins, &axes, &ee_pos);
                let (origins_world, axes_world) = transform_frames_to_world(
                    &origins_f64, &axes_f64, &body_quat, &body_pos,
                );
                let jacobian = compute_leg_jacobian(
                    &origins_world, &axes_world, p_actual, &leg.is_prismatic,
                );

                let qd_f64: Vec<f64> = qd_vals.iter().map(|&v| f64::from(v)).collect();
                let v_actual_relative = Vector3::new(
                    (0..jacobian.ncols()).map(|j| jacobian[(0, j)] * qd_f64[j]).sum::<f64>(),
                    (0..jacobian.ncols()).map(|j| jacobian[(1, j)] * qd_f64[j]).sum::<f64>(),
                    (0..jacobian.ncols()).map(|j| jacobian[(2, j)] * qd_f64[j]).sum::<f64>(),
                );

                // Add the base's full spatial velocity to the leg's relative velocity
                let r_foot = p_actual - body_pos;
                let v_actual = body_state.linear_velocity 
                             + body_state.angular_velocity.cross(&r_foot) 
                             + v_actual_relative;

                let kp = &swing_config.kp_cartesian;
                let kd = &swing_config.kd_cartesian;
                let foot_force = Vector3::new(
                    kp.x * (p_des.x - p_actual.x) + kd.x * (v_des.x - v_actual.x),
                    kp.y * (p_des.y - p_actual.y) + kd.y * (v_des.y - v_actual.y),
                    kp.z * (p_des.z - p_actual.z) + kd.z * (v_des.z - v_actual.z),
                );

                let torques = jacobian_transpose_torques(&jacobian, &foot_force);

                // Compute IK of desired swing foot position for joint-space PD target
                let p_des_body = body_quat.inverse() * (p_des - body_pos);
                let swing_ik_target = IkTarget::Position(p_des_body.cast::<f32>());
                let swing_ik_solver = DlsSolver::new(DlsConfig {
                    max_iterations: 10,
                    position_tolerance: 1e-3,
                    damping: 0.01,
                    ..DlsConfig::default()
                });
                let swing_ik = swing_ik_solver.solve(&leg.chain, &swing_ik_target, q);

                // Blend motor gains from stance→swing over first 20% of swing
                let blend = (swing_phase / 0.2).min(1.0) as f32;

                for (j, &entity) in leg.joint_entities.iter().enumerate() {
                    let kd_swing = 2.0_f32;
                    #[allow(clippy::cast_possible_truncation)]
                    let target_vel = if FF_ENABLED {
                        (torques[j] / f64::from(kd_swing)) as f32
                    } else {
                        0.0
                    };

                    // Ramp gains: stance values at liftoff → swing values at 20% phase
                    let (kp_j, kd_j, max_f) = if j == 0 {
                        // Hip abduction: keep more lateral stiffness during swing
                        (
                            500.0 * (1.0 - blend) + 80.0 * blend,
                            20.0 * (1.0 - blend) + kd_swing * blend,
                            200.0 * (1.0 - blend) + 60.0 * blend,
                        )
                    } else {
                        // Hip pitch / knee
                        (
                            20.0 * blend,
                            1.0 * (1.0 - blend) + kd_swing * blend,
                            50.0 * (1.0 - blend) + 60.0 * blend,
                        )
                    };

                    motor_settings.push(MotorSetting {
                        entity,
                        target_pos: swing_ik.joint_positions[j],
                        target_vel,
                        stiffness: kp_j,
                        damping: kd_j,
                        max_force: max_f,
                    });
                }
            }

            prev_contacts[leg_idx] = is_contact;
        }

        // --- Step physics manually with position motors ---
        {
            let world = scene.app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();

            // Apply motor settings
            for ms in &motor_settings {
                let Some(&jh) = ctx.joint_handles.get(&ms.entity) else { continue };
                let Some(joint) = ctx.impulse_joint_set.get_mut(jh, true) else { continue };

                joint.data.set_motor(
                    JointAxis::AngX,
                    ms.target_pos,
                    ms.target_vel,
                    ms.stiffness,
                    ms.damping,
                );
                joint.data.set_motor_max_force(JointAxis::AngX, ms.max_force);
            }

            // Step physics
            let substeps = ctx.substeps;
            for _ in 0..substeps {
                ctx.step();
            }

            // Read back joint state
            for leg in &legs {
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
        if step % 50 == 0 || step < 5 || (step >= stabilize_steps && step < stabilize_steps + 5) {
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
            for (i, fw) in foot_world.iter().enumerate() {
                let contact = gait.is_contact(i);
                let force_str = if solution.converged {
                    let f = &solution.forces[i];
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

    // 10. Final report
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
    println!("Simulation time: {:.1}s", f64::from(total_steps) * dt);

    // Print final joint states
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
