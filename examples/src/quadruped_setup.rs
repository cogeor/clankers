//! Shared quadruped robot setup: URDF, physics, colliders, warmup, IK chains.
//!
//! All three quadruped binaries (`quadruped_mpc`, `quadruped_mpc_bench`,
//! `quadruped_mpc_viz`) share identical setup code. This module extracts it
//! into a single function so each binary calls `setup_quadruped(config)` and
//! gets back everything it needs to run its simulation/rendering loop.

use std::collections::HashMap;

use clankers_actuator::components::{Actuator, JointState};
use clankers_actuator_core::prelude::{IdealMotor, MotorType};
use clankers_env::prelude::*;
use clankers_ik::KinematicChain;
use clankers_physics::rapier::{bridge::register_robot, InnerPdState, MotorRateLimits, RapierBackend, RapierBackendFixed, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::{SceneBuilder, SpawnedScene};
use clankers_urdf::RobotModel;
use nalgebra::Vector3;
use rapier3d::prelude::{
    ColliderBuilder, Group, InteractionGroups, InteractionTestMode, JointAxis, MassProperties,
    RigidBodyBuilder,
};

use crate::mpc_control::LegRuntime;
use crate::QUADRUPED_URDF;

/// Configuration for quadruped setup — knobs that differ between binaries.
pub struct QuadrupedSetupConfig {
    /// Ground/foot friction coefficient (default 0.6, bench overrides via `--mu-sim`).
    pub sim_friction: f32,
    /// Maximum episode length (default 50_000).
    pub max_episode_steps: u32,
    /// MPC timestep. When `Some` and different from 0.02, sets `control_dt`.
    pub mpc_dt: Option<f64>,
    /// When true, register physics on `FixedUpdate` instead of `Update`.
    pub use_fixed_update: bool,
    /// Motor command rate limit (max position change per step, radians).
    /// `None` disables rate limiting (default).
    pub motor_rate_limit: Option<f32>,
    /// Enable inner PD interpolation across physics substeps (1000Hz effective).
    pub inner_pd: bool,
}

impl Default for QuadrupedSetupConfig {
    fn default() -> Self {
        Self {
            sim_friction: 0.6,
            max_episode_steps: 50_000,
            mpc_dt: None,
            use_fixed_update: false,
            motor_rate_limit: None,
            inner_pd: false,
        }
    }
}

/// Everything produced by `setup_quadruped` that callers need.
pub struct QuadrupedSetup {
    pub scene: SpawnedScene,
    pub model: RobotModel,
    pub legs: Vec<LegRuntime>,
    pub init_joint_angles: Vec<Vec<f32>>,
    pub desired_height: f64,
    pub n_feet: usize,
}

/// Build a fully configured quadruped scene: URDF, physics, colliders, warmup,
/// IK chains, motor limits, sensors, and initial joint readback.
pub fn setup_quadruped(config: QuadrupedSetupConfig) -> QuadrupedSetup {
    // 1. Parse URDF
    let model =
        clankers_urdf::parse_string(QUADRUPED_URDF).expect("failed to parse quadruped URDF");

    // 2. Build scene
    let mut builder = SceneBuilder::new()
        .with_max_episode_steps(config.max_episode_steps)
        .with_robot(model.clone(), HashMap::new());

    if let Some(mpc_dt) = config.mpc_dt {
        if (mpc_dt - 0.02).abs() > 1e-6 {
            builder = builder.with_sim_config(clankers_core::config::SimConfig {
                control_dt: mpc_dt,
                ..clankers_core::config::SimConfig::default()
            });
            println!("  MPC dt={mpc_dt}s ({:.0}Hz), control_dt={mpc_dt}s", 1.0 / mpc_dt);
        }
    }

    let mut scene = builder.build();
    let spawned = &scene.robots["quadruped"];
    println!(
        "Robot '{}' loaded: {} actuated joints",
        spawned.name,
        spawned.joint_count()
    );

    // 3. Add Rapier physics with floating base
    if config.use_fixed_update {
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackendFixed));
    } else {
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));
    }

    // Insert motor rate limits if configured
    if let Some(delta_max) = config.motor_rate_limit {
        scene.app.insert_resource(MotorRateLimits::new(delta_max));
        println!("  Motor rate limit: {delta_max:.3} rad/step");
    }

    // Insert inner PD interpolation state if enabled
    if config.inner_pd {
        scene.app.insert_resource(InnerPdState::default());
        println!("  Inner PD: enabled (interpolating across substeps)");
    }

    let init_hip_ab: f32 = 0.0;
    let init_hip_pitch: f32 = 1.05;
    let init_knee_pitch: f32 = -2.10;

    let sim_friction = config.sim_friction;

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, false);

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
            .friction(sim_friction)
            .restitution(0.0)
            .collision_groups(ground_group)
            .build();
        ctx.collider_set
            .insert_with_parent(ground_collider, ground_handle, &mut ctx.rigid_body_set);

        let link_colliders: &[(&str, ColliderBuilder)] = &[
            ("fl_foot", ColliderBuilder::ball(0.02).friction(sim_friction).restitution(0.0).collision_groups(robot_group)),
            ("fr_foot", ColliderBuilder::ball(0.02).friction(sim_friction).restitution(0.0).collision_groups(robot_group)),
            ("rl_foot", ColliderBuilder::ball(0.02).friction(sim_friction).restitution(0.0).collision_groups(robot_group)),
            ("rr_foot", ColliderBuilder::ball(0.02).friction(sim_friction).restitution(0.0).collision_groups(robot_group)),
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

        // Warmup: bend knees with position motors before MPC starts
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

        if let Some(&bh) = ctx.body_handles.get("body")
            && let Some(body) = ctx.rigid_body_set.get(bh)
        {
            let t = body.translation();
            println!(
                "  Body after warmup: pos=[{:.3}, {:.3}, {:.3}]",
                t.x, t.y, t.z,
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

    let spawned = &scene.robots["quadruped"];
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

    // 5. Override motor limits
    for leg in &legs {
        for &entity in &leg.joint_entities {
            if let Some(mut actuator) = scene.app.world_mut().get_mut::<Actuator>(entity) {
                actuator.motor = MotorType::Ideal(IdealMotor::new(100.0, 100.0));
            }
        }
    }

    // 6. Register sensors (12 DOF)
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(12)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Read desired height from post-warmup body z
    let desired_height = {
        let ctx = scene.app.world().resource::<RapierContext>();
        let handle = ctx.body_handles.get("body").unwrap();
        let body = ctx.rigid_body_set.get(*handle).unwrap();
        f64::from(body.translation().z)
    };
    println!("  Desired height (post-warmup): {desired_height:.3}");

    // 8. Store initial joint angles AFTER warmup for PD stance control
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

    QuadrupedSetup {
        scene,
        model,
        legs,
        init_joint_angles,
        desired_height,
        n_feet,
    }
}
