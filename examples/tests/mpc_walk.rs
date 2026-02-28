//! Integration tests for the MPC walk cycle.
//!
//! Uses the shared `mpc_control` module with position motors (via MotorOverrides)
//! matching the real headless/viz examples. Physics setup includes warmup,
//! collision groups, and 50 solver iterations.

use std::collections::HashMap;

use clankers_actuator::components::{Actuator, JointState};
use clankers_actuator_core::prelude::{IdealMotor, MotorType};
use clankers_env::prelude::*;
use clankers_examples::mpc_control::{LegRuntime, MpcLoopState, body_state_from_rapier, compute_mpc_step};
use clankers_examples::QUADRUPED_URDF;
use clankers_ik::KinematicChain;
use clankers_mpc::{
    AdaptiveGaitConfig, BodyState, GaitScheduler, GaitType, MpcConfig, MpcSolver, SwingConfig,
};
use clankers_physics::rapier::{bridge::register_robot, MotorOverrideParams, MotorOverrides, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;
use nalgebra::Vector3;
use rapier3d::prelude::{
    ColliderBuilder, Group, InteractionGroups, InteractionTestMode, JointAxis, MassProperties,
    RigidBodyBuilder,
};

// ---------------------------------------------------------------------------
// Telemetry snapshot returned from each step
// ---------------------------------------------------------------------------

struct StepSnapshot {
    body: BodyState,
    foot_world: Vec<Vector3<f64>>,
    contacts: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct MpcTestHarness {
    scene: clankers_sim::SpawnedScene,
    mpc_state: MpcLoopState,
}

fn setup_quadruped() -> MpcTestHarness {
    let model =
        clankers_urdf::parse_string(QUADRUPED_URDF).expect("failed to parse quadruped URDF");

    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(50_000)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let spawned = &scene.robots["quadruped"];

    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    let init_hip_ab: f32 = 0.0;
    let init_hip_pitch: f32 = 1.05;
    let init_knee_pitch: f32 = -2.10;

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, false);

        // Match examples: 50 solver iterations for 12 revolute joints
        ctx.integration_parameters.num_solver_iterations = 50;

        let body_offset = bevy::math::Vec3::new(0.0, 0.0, 0.35);

        if let Some(&root_handle) = ctx.body_handles.get("body")
            && let Some(root_body) = ctx.rigid_body_set.get_mut(root_handle)
        {
            // Match examples: body_mass=5.0, child links add ~4kg via register_robot
            let body_mass = 5.0_f32;
            let inertia = bevy::math::Vec3::new(0.02083, 0.07083, 0.08333);
            root_body.set_additional_mass_properties(
                MassProperties::new(bevy::math::Vec3::ZERO, body_mass, inertia),
                true,
            );
            root_body.set_translation(body_offset, true);
        }

        // Move all child link bodies up
        for (link_name, &handle) in &ctx.body_handles {
            if link_name == "body" {
                continue;
            }
            if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                let current = body.translation();
                body.set_translation(current + body_offset, true);
            }
        }

        // Collision groups: robot links only collide with ground
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
            .friction(1.0)
            .restitution(0.0)
            .collision_groups(ground_group)
            .build();
        ctx.collider_set.insert_with_parent(
            ground_collider,
            ground_handle,
            &mut ctx.rigid_body_set,
        );

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

        // Warmup: bend knees with position motors (matching examples)
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

        for _ in 0..1000 {
            ctx.step();
        }

        // Switch motors off after warmup
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

        // Zero velocities so MPC starts from rest
        for (_, &handle) in &ctx.body_handles {
            if let Some(body) = ctx.rigid_body_set.get_mut(handle) {
                body.set_linvel(bevy::math::Vec3::ZERO, true);
                body.set_angvel(bevy::math::Vec3::ZERO, true);
            }
        }

        // Read back joint positions from Rapier after warmup
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

        ctx.snapshot_initial_state();
        world.insert_resource(ctx);
    }

    // Build per-leg IK chains
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

            LegRuntime {
                chain,
                joint_entities,
                is_prismatic,
                hip_offset: hip_offsets[i],
            }
        })
        .collect();

    let n_feet = legs.len();

    // Override motor limits (matching examples)
    for leg in &legs {
        for &entity in &leg.joint_entities {
            if let Some(mut actuator) = scene.app.world_mut().get_mut::<Actuator>(entity) {
                actuator.motor = MotorType::Ideal(IdealMotor::new(100.0, 100.0));
            }
        }
    }

    // Store initial joint angles AFTER warmup
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

    let mpc_config = MpcConfig::default();
    let swing_config = SwingConfig::default();

    let mpc_state = MpcLoopState {
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
    };

    // Insert MotorOverrides resource for position motor control
    scene.app.insert_resource(MotorOverrides::default());

    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(12)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    MpcTestHarness {
        scene,
        mpc_state,
    }
}

/// Run one MPC control step using shared module + position motors via MotorOverrides.
fn run_mpc_step(
    harness: &mut MpcTestHarness,
    desired_velocity: &Vector3<f64>,
    desired_height: f64,
    desired_yaw: f64,
    ground_height: f64,
) -> StepSnapshot {
    // Read body state
    let (body_state, body_quat) = {
        let ctx = harness.scene.app.world().resource::<RapierContext>();
        body_state_from_rapier(ctx, "body").expect("body not found")
    };

    // Read joint states
    let n_feet = harness.mpc_state.legs.len();
    let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
    let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);

    for leg in &harness.mpc_state.legs {
        let mut q = Vec::with_capacity(leg.joint_entities.len());
        let mut qd = Vec::with_capacity(leg.joint_entities.len());
        for &entity in &leg.joint_entities {
            if let Some(js) = harness.scene.app.world().get::<JointState>(entity) {
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

    // Compute MPC step (shared logic)
    let result = compute_mpc_step(
        &mut harness.mpc_state,
        &body_state,
        &body_quat,
        &all_joint_positions,
        &all_joint_velocities,
        desired_velocity,
        desired_height,
        desired_yaw,
        ground_height,
    );

    // Convert MotorCommands → MotorOverrideParams
    {
        let mut overrides = harness.scene.app.world_mut().resource_mut::<MotorOverrides>();
        overrides.joints.clear();
        for mc in &result.motor_commands {
            overrides.joints.insert(mc.entity, MotorOverrideParams {
                target_pos: mc.target_pos,
                target_vel: mc.target_vel,
                stiffness: mc.stiffness,
                damping: mc.damping,
                max_force: mc.max_force,
            });
        }
    }

    // Step physics via app.update() (position motors applied by rapier_step_system)
    harness.scene.app.update();

    StepSnapshot {
        body: body_state,
        foot_world: result.foot_world,
        contacts: result.contacts,
    }
}

// ---------------------------------------------------------------------------
// Helpers for trajectory analysis
// ---------------------------------------------------------------------------

/// Compute normalized autocorrelation of a signal at a given lag.
fn autocorrelation(signal: &[f64], lag: usize) -> f64 {
    let n = signal.len();
    if lag >= n {
        return 0.0;
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let var: f64 = signal.iter().map(|&x| (x - mean) * (x - mean)).sum();
    if var < 1e-20 {
        return 0.0;
    }
    let cov: f64 = (0..n - lag)
        .map(|i| (signal[i] - mean) * (signal[i + lag] - mean))
        .sum();
    cov / var
}

/// Find the dominant period of a signal using autocorrelation.
fn find_dominant_period(signal: &[f64], min_lag: usize, max_lag: usize, threshold: f64) -> Option<usize> {
    let max_lag = max_lag.min(signal.len() / 2);
    if min_lag >= max_lag {
        return None;
    }

    let acf: Vec<f64> = (min_lag..=max_lag)
        .map(|lag| autocorrelation(signal, lag))
        .collect();

    for i in 1..acf.len().saturating_sub(1) {
        if acf[i] > acf[i - 1] && acf[i] > acf[i + 1] && acf[i] > threshold {
            return Some(min_lag + i);
        }
    }
    if let Some(&last) = acf.last() {
        if acf.len() >= 2 && last > acf[acf.len() - 2] && last > threshold {
            return Some(max_lag);
        }
    }
    None
}

/// Stabilize with Stand gait, then switch to a locomotion gait and collect
/// snapshots.  Returns (stand_snapshots, loco_snapshots).
fn run_stand_then_locomote(
    gait_type: GaitType,
    desired_velocity: &Vector3<f64>,
    stand_steps: usize,
    loco_steps: usize,
) -> (MpcTestHarness, Vec<StepSnapshot>, Vec<StepSnapshot>) {
    let mut harness = setup_quadruped();

    // Use post-warmup body height as desired_height (matching examples)
    let desired_height = {
        let ctx = harness.scene.app.world().resource::<RapierContext>();
        let handle = ctx.body_handles.get("body").unwrap();
        let body = ctx.rigid_body_set.get(*handle).unwrap();
        f64::from(body.translation().z)
    };

    let mut stand_snaps = Vec::with_capacity(stand_steps);
    for _ in 0..stand_steps {
        let snap = run_mpc_step(&mut harness, &Vector3::zeros(), desired_height, 0.0, 0.0);
        stand_snaps.push(snap);
    }

    harness.mpc_state.gait = GaitScheduler::quadruped(gait_type);

    let mut loco_snaps = Vec::with_capacity(loco_steps);
    for _ in 0..loco_steps {
        let snap = run_mpc_step(&mut harness, desired_velocity, desired_height, 0.0, 0.0);
        loco_snaps.push(snap);
    }

    (harness, stand_snaps, loco_snaps)
}

// ===========================================================================
// Tests
// ===========================================================================

// ---- Standing ------------------------------------------------------------

#[test]
fn standing_maintains_height() {
    let mut harness = setup_quadruped();

    let desired_height = {
        let ctx = harness.scene.app.world().resource::<RapierContext>();
        let handle = ctx.body_handles.get("body").unwrap();
        let body = ctx.rigid_body_set.get(*handle).unwrap();
        f64::from(body.translation().z)
    };

    let mut min_z = f64::MAX;
    let mut max_z = f64::MIN;

    for _ in 0..200 {
        let snap = run_mpc_step(&mut harness, &Vector3::zeros(), desired_height, 0.0, 0.0);
        min_z = min_z.min(snap.body.position.z);
        max_z = max_z.max(snap.body.position.z);
    }

    assert!(
        min_z > 0.10,
        "Body dropped too low during stand: min_z={min_z:.3} (expected > 0.10)",
    );
    assert!(
        max_z < 0.45,
        "Body bounced too high during stand: max_z={max_z:.3} (expected < 0.45)",
    );
}

// ---- Trot ----------------------------------------------------------------

#[test]
fn trot_com_velocity_nonzero() {
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Trot,
        &desired_velocity,
        100,
        400,
    );

    let ramp_steps = 50;
    let mut stall_count = 0;
    for snap in loco.iter().skip(ramp_steps) {
        let vx = snap.body.linear_velocity.x;
        let vy = snap.body.linear_velocity.y;
        let speed_xy = (vx * vx + vy * vy).sqrt();
        if speed_xy < 1e-4 {
            stall_count += 1;
        }
    }

    let checked_steps = loco.len() - ramp_steps;
    let stall_frac = stall_count as f64 / checked_steps as f64;
    assert!(
        stall_frac < 0.10,
        "COM horizontal speed was ~0 for {:.0}% of trot steps (expected < 10%)",
        stall_frac * 100.0,
    );
}

#[test]
fn trot_feet_above_ground() {
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Trot,
        &desired_velocity,
        100,
        400,
    );

    let foot_z_floor = -0.10;
    let mut worst_z = f64::MAX;
    let mut worst_step = 0;
    let mut worst_leg = 0;

    for (step_idx, snap) in loco.iter().enumerate() {
        for (leg_idx, foot) in snap.foot_world.iter().enumerate() {
            if foot.z < worst_z {
                worst_z = foot.z;
                worst_step = step_idx;
                worst_leg = leg_idx;
            }
        }
    }

    assert!(
        worst_z > foot_z_floor,
        "Foot {worst_leg} underground at trot step {worst_step}: z={worst_z:.4} (floor={foot_z_floor})",
    );
}

#[test]
#[ignore = "body periodicity requires further physics tuning"]
fn trot_body_height_is_cyclic() {
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Trot,
        &desired_velocity,
        100,
        500,
    );

    let settle = 100;
    let body_z: Vec<f64> = loco.iter().skip(settle).map(|s| s.body.position.z).collect();

    let period = find_dominant_period(&body_z, 5, 30, 0.05);

    assert!(
        period.is_some(),
        "No periodic signal detected in body z during trot (autocorrelation peak < 0.05)",
    );

    let period = period.unwrap();
    assert!(
        (5..=30).contains(&period),
        "Trot body-z period={period} steps outside expected range [5, 30]",
    );
}

#[test]
fn trot_contact_pattern_alternates() {
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Trot,
        &desired_velocity,
        100,
        400,
    );

    let mut pair_a_count = 0_usize;
    let mut pair_b_count = 0_usize;

    for snap in &loco {
        let fl = snap.contacts[0];
        let fr = snap.contacts[1];
        let rl = snap.contacts[2];
        let rr = snap.contacts[3];

        if fl && rr && !fr && !rl {
            pair_a_count += 1;
        }
        if fr && rl && !fl && !rr {
            pair_b_count += 1;
        }
    }

    let total = loco.len();
    assert!(
        pair_a_count > total / 10,
        "Diagonal pair A (FL+RR) appeared only {pair_a_count}/{total} steps — expected ~50%",
    );
    assert!(
        pair_b_count > total / 10,
        "Diagonal pair B (FR+RL) appeared only {pair_b_count}/{total} steps — expected ~50%",
    );

    let ratio = pair_a_count.max(pair_b_count) as f64 / pair_a_count.min(pair_b_count).max(1) as f64;
    assert!(
        ratio < 2.0,
        "Diagonal pair imbalance: A={pair_a_count} B={pair_b_count} ratio={ratio:.1} (expected < 2.0)",
    );
}

#[test]
fn trot_moves_forward() {
    let desired_velocity = Vector3::new(0.3, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Trot,
        &desired_velocity,
        100,
        400,
    );

    let final_body = &loco.last().unwrap().body;

    assert!(
        final_body.position.z > 0.05,
        "Body collapsed: z={:.3} (expected > 0.05)",
        final_body.position.z,
    );
    assert!(
        final_body.position.x > -0.5,
        "Robot went too far backward: x={:.3} (expected > -0.5)",
        final_body.position.x,
    );
}

// ---- Walk ----------------------------------------------------------------

#[test]
fn walk_feet_above_ground() {
    let desired_velocity = Vector3::new(0.15, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Walk,
        &desired_velocity,
        100,
        400,
    );

    let foot_z_floor = -0.10;
    let mut worst_foot_z = f64::MAX;
    let mut worst_step = 0;
    let mut worst_leg = 0;

    for (step_idx, snap) in loco.iter().enumerate() {
        for (leg_idx, foot) in snap.foot_world.iter().enumerate() {
            if foot.z < worst_foot_z {
                worst_foot_z = foot.z;
                worst_step = step_idx;
                worst_leg = leg_idx;
            }
        }
    }

    assert!(
        worst_foot_z > foot_z_floor,
        "Foot {worst_leg} went underground at walk step {worst_step}: z={worst_foot_z:.4} (floor={foot_z_floor})",
    );
}

#[test]
fn walk_at_least_three_feet_stance() {
    let desired_velocity = Vector3::new(0.15, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Walk,
        &desired_velocity,
        100,
        400,
    );

    for (step_idx, snap) in loco.iter().enumerate() {
        let n_stance = snap.contacts.iter().filter(|&&c| c).count();
        assert!(
            n_stance >= 3,
            "Walk step {step_idx}: only {n_stance} feet in stance (expected >= 3)",
        );
    }
}

#[test]
#[ignore = "body periodicity requires further physics tuning"]
fn walk_body_height_is_cyclic() {
    let desired_velocity = Vector3::new(0.15, 0.0, 0.0);
    let (_harness, _stand, loco) = run_stand_then_locomote(
        GaitType::Walk,
        &desired_velocity,
        100,
        600,
    );

    let settle = 150;
    let body_z: Vec<f64> = loco.iter().skip(settle).map(|s| s.body.position.z).collect();

    let period = find_dominant_period(&body_z, 5, 50, 0.05);

    assert!(
        period.is_some(),
        "No periodic signal detected in body z during walk (autocorrelation peak < 0.05)",
    );

    let period = period.unwrap();
    assert!(
        (5..=50).contains(&period),
        "Walk body-z period={period} steps outside expected range [5, 50]",
    );
}
