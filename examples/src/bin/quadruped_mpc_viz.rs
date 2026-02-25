//! Quadruped MPC visualization with windowed Bevy rendering.
//!
//! Extends the headless `quadruped_mpc` example with 3D visualization:
//! colored meshes for body and legs, orbit camera, and egui status panel.
//!
//! The MPC pipeline runs as a Bevy system (same logic as the headless
//! example), reading body state directly from `RapierContext` to avoid
//! the one-frame `GlobalTransform` propagation lag.
//!
//! Physics uses Z-up coordinates (MPC convention). Visual meshes are
//! synced with a Z-up to Y-up coordinate swap so they render correctly
//! under Bevy's default Y-up camera.
//!
//! ## Visual sync strategy
//!
//! Rapier places each link rigid body at the **joint position** (where it
//! connects to its parent), not at the link center. To render correct leg
//! geometry we use "connect the dots": capsules are drawn BETWEEN
//! consecutive rigid body positions (hip-to-knee, knee-to-ankle) so the
//! mesh always spans the full link regardless of joint angle.
//!
//! Run: `cargo run -p clankers-examples --bin quadruped_mpc_viz`

use std::collections::HashMap;

use bevy::math::EulerRot;
use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::ClankersSet;
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
use clankers_teleop::ClankersTeleopPlugin;
use clankers_viz::ClankersVizPlugin;
use nalgebra::Vector3;
use rapier3d::prelude::{ColliderBuilder, MassProperties, RigidBodyBuilder};

// ---------------------------------------------------------------------------
// Visual markers
// ---------------------------------------------------------------------------

/// Body cuboid — synced from the "body" rigid body (position + rotation).
#[derive(Component)]
struct BodyVisual;

/// Leg segment capsule — drawn between two rigid body positions.
/// The capsule midpoint and orientation are recomputed each frame from
/// the start/end link positions in Rapier.
#[derive(Component)]
struct SegmentVisual {
    start_link: &'static str,
    end_link: &'static str,
}

/// Point visual (sphere) — synced to a single rigid body position.
/// Used for joint markers and foot spheres.
#[derive(Component)]
struct PointVisual(&'static str);

// ---------------------------------------------------------------------------
// MPC runtime state
// ---------------------------------------------------------------------------

struct LegRuntime {
    chain: KinematicChain,
    joint_entities: Vec<Entity>,
    is_prismatic: Vec<bool>,
    hip_offset: Vector3<f64>,
}

#[derive(Resource)]
struct QuadMpcState {
    gait: GaitScheduler,
    solver: MpcSolver,
    config: MpcConfig,
    swing_config: SwingConfig,
    legs: Vec<LegRuntime>,
    swing_starts: Vec<Vector3<f64>>,
    swing_targets: Vec<Vector3<f64>>,
    desired_velocity: Vector3<f64>,
    desired_height: f64,
    desired_yaw: f64,
    ground_height: f64,
    step: usize,
    stabilize_steps: usize,
    switched_to_trot: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read body state from Rapier rigid body set (same as headless example).
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

/// Convert a physics position (Z-up) to Bevy visual position (Y-up).
fn phys_to_vis(pos: Vec3) -> Vec3 {
    Vec3::new(pos.x, pos.z, -pos.y)
}

/// Get a Rapier rigid body's world position, converted to Bevy Y-up.
fn link_vis_pos(rapier: &RapierContext, link_name: &str) -> Option<Vec3> {
    let handle = rapier.body_handles.get(link_name)?;
    let body = rapier.rigid_body_set.get(*handle)?;
    Some(phys_to_vis(body.translation()))
}

/// Compute a quaternion that rotates the Y-axis to align with `dir`.
fn rotation_align_y(dir: Vec3) -> Quat {
    let d = dir.normalize_or_zero();
    let dot = Vec3::Y.dot(d);
    if dot > 0.9999 {
        Quat::IDENTITY
    } else if dot < -0.9999 {
        // Antiparallel — rotate 180 deg around X (arbitrary stable axis)
        Quat::from_rotation_x(std::f32::consts::PI)
    } else {
        Quat::from_rotation_arc(Vec3::Y, d)
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn visual meshes
// ---------------------------------------------------------------------------

fn spawn_quadruped_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.4, 0.8),
        ..default()
    });
    let leg_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.6, 0.2),
        ..default()
    });
    let joint_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.85, 0.2),
        ..default()
    });

    // Body: blue cuboid (0.4m long × 0.1m tall × 0.2m wide in Bevy Y-up)
    commands.spawn((
        BodyVisual,
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.1, 0.2))),
        MeshMaterial3d(body_mat),
        Transform::from_xyz(0.0, 0.35, 0.0),
    ));

    // Shared meshes
    let capsule_mesh = meshes.add(Capsule3d::new(0.018, 0.12));
    let joint_sphere = meshes.add(Sphere::new(0.022));
    let foot_sphere = meshes.add(Sphere::new(0.025));

    // Leg segment definitions: (start_link, end_link)
    let leg_segments: &[(&str, &str, &str, &str)] = &[
        // (prefix, upper_link, lower_link, foot_link)
        ("fl", "fl_upper_leg", "fl_lower_leg", "fl_foot"),
        ("fr", "fr_upper_leg", "fr_lower_leg", "fr_foot"),
        ("rl", "rl_upper_leg", "rl_lower_leg", "rl_foot"),
        ("rr", "rr_upper_leg", "rr_lower_leg", "rr_foot"),
    ];

    for &(_prefix, upper, lower, foot) in leg_segments {
        // Hip joint sphere (at upper_leg rigid body = hip joint position)
        commands.spawn((
            PointVisual(upper),
            Mesh3d(joint_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));

        // Upper leg capsule: hip → knee
        commands.spawn((
            SegmentVisual {
                start_link: leak_str(upper),
                end_link: leak_str(lower),
            },
            Mesh3d(capsule_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::default(),
        ));

        // Knee joint sphere (at lower_leg rigid body = knee joint position)
        commands.spawn((
            PointVisual(lower),
            Mesh3d(joint_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));

        // Lower leg capsule: knee → ankle
        commands.spawn((
            SegmentVisual {
                start_link: leak_str(lower),
                end_link: leak_str(foot),
            },
            Mesh3d(capsule_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::default(),
        ));

        // Foot sphere (at foot rigid body = ankle joint position)
        commands.spawn((
            PointVisual(foot),
            Mesh3d(foot_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));
    }
}

/// Convert a known static &str to &'static str for SegmentVisual fields.
/// These are compile-time link name constants so the leak is bounded.
fn leak_str(s: &str) -> &'static str {
    match s {
        "fl_upper_leg" => "fl_upper_leg",
        "fl_lower_leg" => "fl_lower_leg",
        "fl_foot" => "fl_foot",
        "fr_upper_leg" => "fr_upper_leg",
        "fr_lower_leg" => "fr_lower_leg",
        "fr_foot" => "fr_foot",
        "rl_upper_leg" => "rl_upper_leg",
        "rl_lower_leg" => "rl_lower_leg",
        "rl_foot" => "rl_foot",
        "rr_upper_leg" => "rr_upper_leg",
        "rr_lower_leg" => "rr_lower_leg",
        "rr_foot" => "rr_foot",
        _ => unreachable!("unknown link: {s}"),
    }
}

// ---------------------------------------------------------------------------
// MPC control system (Decide phase)
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn mpc_control_system(
    rapier: Res<RapierContext>,
    mut mpc: ResMut<QuadMpcState>,
    states: Query<&JointState>,
    mut commands: Query<&mut JointCommand>,
) {
    let mpc = &mut *mpc;

    // Switch from Stand to Trot after stabilization
    if mpc.step == mpc.stabilize_steps && !mpc.switched_to_trot {
        mpc.gait = GaitScheduler::quadruped(GaitType::Trot);
        mpc.switched_to_trot = true;
        println!("  >>> Switching to Trot gait at step {}", mpc.step);
    }

    let Some(body_state) = body_state_from_rapier(&rapier, "body") else {
        return;
    };
    let body_pos = body_state.position;
    let n_feet = mpc.legs.len();

    // Read joint states and compute foot FK
    let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
    let mut foot_world: Vec<Vector3<f64>> = Vec::with_capacity(n_feet);

    for leg in &mpc.legs {
        let mut q = Vec::with_capacity(leg.joint_entities.len());
        for &entity in &leg.joint_entities {
            if let Ok(js) = states.get(entity) {
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

    // Advance gait and solve MPC
    mpc.gait.advance(mpc.config.dt);

    let contacts = mpc.gait.contact_sequence(mpc.config.horizon, mpc.config.dt);
    let x0 = body_state.to_state_vector(mpc.config.gravity);
    let current_vel = if mpc.step < mpc.stabilize_steps {
        &Vector3::zeros()
    } else {
        &mpc.desired_velocity
    };
    let reference = ReferenceTrajectory::constant_velocity(
        &body_state,
        current_vel,
        mpc.desired_height,
        mpc.desired_yaw,
        mpc.config.horizon,
        mpc.config.dt,
        mpc.config.gravity,
    );

    let solution = mpc.solver.solve(&x0, &foot_world, &contacts, &reference);
    let stance_duration = mpc.gait.duty_factor() * mpc.gait.cycle_time();

    // Apply control per leg
    for (leg_idx, leg) in mpc.legs.iter().enumerate() {
        let is_contact = mpc.gait.is_contact(leg_idx);

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
                if let Ok(mut cmd) = commands.get_mut(entity) {
                    cmd.value = torques[j] as f32;
                }
            }

            mpc.swing_starts[leg_idx] = foot_world[leg_idx];
        } else {
            // Swing: Bezier trajectory (zero torque for now)
            let swing_phase = mpc.gait.swing_phase(leg_idx);

            if swing_phase < 0.05 {
                let hip_world = body_pos + leg.hip_offset;
                mpc.swing_targets[leg_idx] = raibert_foot_target(
                    &hip_world,
                    &mpc.desired_velocity,
                    stance_duration,
                    mpc.ground_height,
                );
                mpc.swing_starts[leg_idx] = foot_world[leg_idx];
            }

            let _target_pos = swing_foot_position(
                &mpc.swing_starts[leg_idx],
                &mpc.swing_targets[leg_idx],
                swing_phase,
                mpc.swing_config.step_height,
            );

            for &entity in &leg.joint_entities {
                if let Ok(mut cmd) = commands.get_mut(entity) {
                    cmd.value = 0.0;
                }
            }
        }
    }

    // Telemetry
    if mpc.step.is_multiple_of(50) {
        let n_stance: usize = (0..n_feet).filter(|&i| mpc.gait.is_contact(i)).count();
        println!(
            "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  stance={}/{}  mpc={:>4}us  {}",
            mpc.step,
            body_pos.x,
            body_pos.y,
            body_pos.z,
            n_stance,
            n_feet,
            solution.solve_time_us,
            if solution.converged { "OK" } else { "FAIL" },
        );
    }

    mpc.step += 1;
}

// ---------------------------------------------------------------------------
// Visual sync systems (after Simulate)
// ---------------------------------------------------------------------------

/// Sync body cuboid from Rapier rigid body position + rotation.
#[allow(clippy::needless_pass_by_value)]
fn sync_body_visual(
    rapier: Res<RapierContext>,
    mut query: Query<&mut Transform, With<BodyVisual>>,
) {
    if let Some(&handle) = rapier.body_handles.get("body")
        && let Some(body) = rapier.rigid_body_set.get(handle)
    {
        let t = body.translation();
        let r = body.rotation();
        for mut transform in &mut query {
            // Z-up -> Y-up position
            transform.translation = phys_to_vis(t);
            // Z-up -> Y-up quaternion: (qx,qy,qz,qw) -> (qx,qz,-qy,qw)
            transform.rotation = Quat::from_xyzw(r.x, r.z, -r.y, r.w);
        }
    }
}

/// Sync leg segment capsules: position at midpoint, orient along the segment.
#[allow(clippy::needless_pass_by_value)]
fn sync_segment_visuals(
    rapier: Res<RapierContext>,
    mut query: Query<(&SegmentVisual, &mut Transform)>,
) {
    for (seg, mut transform) in &mut query {
        if let (Some(start), Some(end)) = (
            link_vis_pos(&rapier, seg.start_link),
            link_vis_pos(&rapier, seg.end_link),
        ) {
            let mid = (start + end) * 0.5;
            let dir = end - start;
            transform.translation = mid;
            transform.rotation = rotation_align_y(dir);
        }
    }
}

/// Sync point visuals (joint spheres, foot spheres) to rigid body position.
#[allow(clippy::needless_pass_by_value)]
fn sync_point_visuals(
    rapier: Res<RapierContext>,
    mut query: Query<(&PointVisual, &mut Transform)>,
) {
    for (point, mut transform) in &mut query {
        if let Some(pos) = link_vis_pos(&rapier, point.0) {
            transform.translation = pos;
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // 1. Parse URDF
    let model =
        clankers_urdf::parse_string(QUADRUPED_URDF).expect("failed to parse quadruped URDF");

    // 2. Build scene
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(50_000)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let spawned = &scene.robots["quadruped"];

    // 3. Add Rapier physics with floating base
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    {
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        // fixed_base = false: body is dynamic, controlled by ground reaction forces
        register_robot(&mut ctx, &model, spawned, world, false);

        // Set mass properties on root body
        if let Some(&root_handle) = ctx.body_handles.get("body")
            && let Some(root_body) = ctx.rigid_body_set.get_mut(root_handle)
        {
            let body_mass = 5.0_f32;
            let inertia = Vec3::new(0.07, 0.26, 0.28);
            root_body.set_additional_mass_properties(
                MassProperties::new(Vec3::ZERO, body_mass, inertia),
                true,
            );
            // Start the body at standing height
            root_body.set_translation(Vec3::new(0.0, 0.0, 0.35), true);
        }

        // Ground plane: fixed body with large cuboid at z=-0.05 (top at z=0)
        let ground_body = RigidBodyBuilder::fixed()
            .translation(Vec3::new(0.0, 0.0, -0.05))
            .build();
        let ground_handle = ctx.rigid_body_set.insert(ground_body);
        let ground_collider = ColliderBuilder::cuboid(50.0, 50.0, 0.05)
            .friction(0.8)
            .restitution(0.0)
            .build();
        ctx.collider_set.insert_with_parent(
            ground_collider,
            ground_handle,
            &mut ctx.rigid_body_set,
        );

        // Add colliders to all robot links
        let link_colliders: &[(&str, ColliderBuilder)] = &[
            (
                "fl_foot",
                ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0),
            ),
            (
                "fr_foot",
                ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0),
            ),
            (
                "rl_foot",
                ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0),
            ),
            (
                "rr_foot",
                ColliderBuilder::ball(0.02).friction(0.8).restitution(0.0),
            ),
            (
                "fl_upper_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "fr_upper_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "rl_upper_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "rr_upper_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "fl_lower_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "fr_lower_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "rl_lower_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
            (
                "rr_lower_leg",
                ColliderBuilder::capsule_z(0.075, 0.015).friction(0.3),
            ),
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

            let joint_entities: Vec<Entity> = chain
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

    // 5. Configure MPC
    let config = MpcConfig {
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

    let gait = GaitScheduler::quadruped(GaitType::Stand);
    let solver = MpcSolver::new(config.clone());

    scene.app.insert_resource(QuadMpcState {
        gait,
        solver,
        config,
        swing_config,
        legs,
        swing_starts: vec![Vector3::zeros(); n_feet],
        swing_targets: vec![Vector3::zeros(); n_feet],
        desired_velocity: Vector3::new(0.3, 0.0, 0.0),
        desired_height: 0.30,
        desired_yaw: 0.0,
        ground_height: 0.0,
        step: 0,
        stabilize_steps: 100,
        switched_to_trot: false,
    });

    // 6. Register sensors
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(8)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 7. Windowed rendering
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers \u{2014} Quadruped MPC".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // 8. Teleop + viz plugins
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene.app.add_plugins(ClankersVizPlugin);

    // 9. Visual meshes
    scene.app.add_systems(Startup, spawn_quadruped_meshes);

    // 10. MPC control (Decide phase, AFTER teleop so MPC commands are final)
    scene.app.add_systems(
        Update,
        mpc_control_system
            .in_set(ClankersSet::Decide)
            .after(clankers_teleop::systems::apply_teleop_commands),
    );

    // 11. Visual sync (after physics — three independent systems)
    scene.app.add_systems(
        Update,
        (sync_body_visual, sync_segment_visuals, sync_point_visuals)
            .after(ClankersSet::Simulate),
    );

    // 12. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Quadruped MPC Visualization");
    println!("  Camera: mouse (orbit/pan/zoom)");
    println!("  Phase 1: Stand (stabilize) -> Phase 2: Trot");
    scene.app.run();
}
