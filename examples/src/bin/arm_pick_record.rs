//! Record arm pick-and-place demonstrations with camera images.
//!
//! Runs a scripted IK-based pick motion and records joint states, actions,
//! body poses, and camera images to MCAP files for offline imitation learning.
//!
//! Usage:
//!   cargo run -p clankers-examples --bin arm_pick_record -- --output output/arm_episodes/ep --episodes 3

use std::path::PathBuf;

use bevy::prelude::*;
use clankers_actuator::components::JointState;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::arm_setup::{
    ArmIkState, ArmSetupConfig, arm_ik_solver, initial_motor_overrides, setup_arm, ARM_DAMPING,
    ARM_STIFFNESS, EFFORT_LIMITS, GRIPPER_DAMPING, GRIPPER_MAX_FORCE, GRIPPER_STIFFNESS,
};
use clankers_ik::IkTarget;
use clankers_physics::rapier::{MotorOverrideParams, MotorOverrides, RapierContext};
use clankers_record::prelude::*;
use clankers_render::camera::spawn_camera_sensor;
use clankers_render::prelude::*;
use clankers_viz::{phys_rot_to_vis, phys_to_vis};
use clap::Parser;
use nalgebra::Vector3;
use rapier3d::prelude::{ColliderBuilder, RigidBodyBuilder, RigidBodyHandle, SharedShape};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Record arm pick-and-place demonstrations with camera images")]
struct Args {
    /// Output path prefix for MCAP files (e.g. output/ep produces ep_001.mcap)
    #[arg(long)]
    output: PathBuf,

    /// Number of episodes to record
    #[arg(long, default_value_t = 3)]
    episodes: u32,

    /// Maximum steps per episode
    #[arg(long, default_value_t = 250)]
    max_steps: u32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Cube position on the table
const CUBE_POS: [f32; 3] = [0.3, 0.0, 0.425];

// IK targets for pick sequence (physics Z-up coords)
const APPROACH_POS: [f32; 3] = [0.3, 0.0, 0.58]; // above cube
const DESCEND_POS: [f32; 3] = [0.3, 0.0, 0.46]; // at cube level
const LIFT_POS: [f32; 3] = [0.3, 0.0, 0.65]; // lifted

const FINGER_OPEN: f32 = 0.03;
const FINGER_CLOSED: f32 = 0.005;

// Phase durations (steps)
const APPROACH_STEPS: u32 = 50;
const DESCEND_STEPS: u32 = 50;
const GRASP_STEPS: u32 = 30;
const LIFT_STEPS: u32 = 50;
const HOLD_STEPS: u32 = 20;

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct GripperEntities([Entity; 2]);

#[derive(Resource)]
struct PickState {
    phase: PickPhase,
    step: u32,
    episode: u32,
    total_episodes: u32,
    done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PickPhase {
    Approach,
    Descend,
    Grasp,
    Lift,
    Hold,
}

impl PickPhase {
    const fn max_steps(self) -> u32 {
        match self {
            Self::Approach => APPROACH_STEPS,
            Self::Descend => DESCEND_STEPS,
            Self::Grasp => GRASP_STEPS,
            Self::Lift => LIFT_STEPS,
            Self::Hold => HOLD_STEPS,
        }
    }

    const fn next(self) -> Option<Self> {
        match self {
            Self::Approach => Some(Self::Descend),
            Self::Descend => Some(Self::Grasp),
            Self::Grasp => Some(Self::Lift),
            Self::Lift => Some(Self::Hold),
            Self::Hold => None,
        }
    }

    const fn ik_target(self) -> Vector3<f32> {
        let pos = match self {
            Self::Approach => APPROACH_POS,
            Self::Descend | Self::Grasp => DESCEND_POS,
            Self::Lift | Self::Hold => LIFT_POS,
        };
        Vector3::new(pos[0], pos[1], pos[2])
    }

    const fn finger_target(self) -> f32 {
        match self {
            Self::Approach | Self::Descend => FINGER_OPEN,
            Self::Grasp | Self::Lift | Self::Hold => FINGER_CLOSED,
        }
    }
}

/// Visual marker for link mesh sync.
#[derive(Component)]
struct LinkVisual(&'static str);

/// Stores initial positions for physics reset (used between episodes).
#[derive(Resource)]
#[allow(dead_code)]
struct ObjectInitialPositions(Vec<(RigidBodyHandle, Vec3)>);

// ---------------------------------------------------------------------------
// Camera config resource
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct ObsCameraSize {
    width: u32,
    height: u32,
}

// ---------------------------------------------------------------------------
// Startup: spawn camera sensor
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn spawn_obs_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut cam_bufs: ResMut<CameraFrameBuffers>,
    size: Res<ObsCameraSize>,
) {
    // Side view camera looking at the workspace
    let cam_config = CameraConfig::new().with_label("image");
    let (entity, _) = spawn_camera_sensor(
        &mut commands,
        &mut images,
        &mut cam_bufs,
        cam_config,
        size.width,
        size.height,
    );
    // Position: side view, looking at table center
    commands
        .entity(entity)
        .insert(Transform::from_xyz(0.6, 0.8, 0.0).looking_at(Vec3::new(0.3, 0.4, 0.0), Vec3::Y));
}

// ---------------------------------------------------------------------------
// Startup: spawn arm meshes + scene objects
// ---------------------------------------------------------------------------

fn spawn_arm_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let base_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.35),
        ..default()
    });
    let link_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.5, 0.8),
        ..default()
    });
    let forearm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.7, 0.3),
        ..default()
    });
    let ee_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.4, 0.1),
        ..default()
    });
    let gripper_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.65),
        ..default()
    });

    for (name, radius, height, y_off, mat) in [
        ("base", 0.08, 0.1, 0.0, &base_mat),
        ("shoulder_link", 0.04, 0.2, 0.1, &link_mat),
        ("upper_arm", 0.035, 0.3, 0.15, &link_mat),
        ("elbow_link", 0.03, 0.1, 0.05, &forearm_mat),
        ("forearm", 0.025, 0.2, 0.1, &forearm_mat),
    ] {
        commands
            .spawn((
                LinkVisual(name),
                Visibility::default(),
                Transform::default(),
            ))
            .with_children(|p| {
                p.spawn((
                    Mesh3d(meshes.add(Cylinder::new(radius, height))),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_xyz(0.0, y_off, 0.0),
                ));
            });
    }

    // End-effector sphere
    commands
        .spawn((
            LinkVisual("end_effector"),
            Visibility::default(),
            Transform::default(),
        ))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(Sphere::new(0.025))),
                MeshMaterial3d(ee_mat),
                Transform::IDENTITY,
            ));
        });

    // Gripper base + fingers
    commands
        .spawn((
            LinkVisual("gripper_base"),
            Visibility::default(),
            Transform::default(),
        ))
        .with_children(|p| {
            p.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.06, 0.02, 0.04))),
                MeshMaterial3d(gripper_mat.clone()),
                Transform::IDENTITY,
            ));
        });

    for finger_name in ["finger_left", "finger_right"] {
        commands
            .spawn((
                LinkVisual(finger_name),
                Visibility::default(),
                Transform::default(),
            ))
            .with_children(|p| {
                p.spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.01, 0.04, 0.01))),
                    MeshMaterial3d(gripper_mat.clone()),
                    Transform::from_xyz(0.0, 0.02, 0.0),
                ));
            });
    }

    // Table surface
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.02, 0.4))),
        MeshMaterial3d(table_mat),
        // physics z=0.4 -> bevy y=0.4, offset by table half-height
        Transform::from_xyz(0.35, 0.4 - 0.01, 0.0),
    ));

    // Red cube visual
    let red_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.15, 0.1),
        ..default()
    });
    commands.spawn((
        LinkVisual("red_cube"),
        Mesh3d(meshes.add(Cuboid::new(0.025, 0.025, 0.025))),
        MeshMaterial3d(red_mat),
        Visibility::default(),
        Transform::default(),
    ));

    // Lighting
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_xyz(1.0, 2.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Main viewport camera (Bevy Y-up: physics (0.3, 0, 0.4) -> bevy (0.3, 0.4, 0))
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.8, 0.8, 0.6).looking_at(Vec3::new(0.3, 0.4, 0.0), Vec3::Y),
    ));
}

// ---------------------------------------------------------------------------
// Runtime systems
// ---------------------------------------------------------------------------

/// Sync link visual transforms from physics bodies.
#[allow(clippy::needless_pass_by_value)]
fn sync_link_visuals(ctx: Res<RapierContext>, mut query: Query<(&LinkVisual, &mut Transform)>) {
    for (link, mut tf) in &mut query {
        let Some(&handle) = ctx.body_handles.get(link.0) else {
            continue;
        };
        let Some(body) = ctx.rigid_body_set.get(handle) else {
            continue;
        };
        tf.translation = phys_to_vis(body.translation());
        tf.rotation = phys_rot_to_vis(body.rotation());
    }
}

/// Populate `PendingBodyPoses` from `RapierContext` each frame.
#[allow(clippy::needless_pass_by_value)]
fn populate_body_poses(ctx: Res<RapierContext>, mut pending: ResMut<PendingBodyPoses>) {
    pending.0.clear();
    for (name, &handle) in &ctx.body_handles {
        let Some(body) = ctx.rigid_body_set.get(handle) else {
            continue;
        };
        let t = body.translation();
        let r = body.rotation();
        pending
            .0
            .insert(name.clone(), [t.x, t.y, t.z, r.x, r.y, r.z, r.w]);
    }
}

/// Capture action: writes IK joint targets + finger targets to `PendingAction`.
#[allow(clippy::needless_pass_by_value)]
fn capture_action_system(
    ik: Res<ArmIkState>,
    gripper: Res<GripperEntities>,
    motor_overrides: Res<MotorOverrides>,
    mut pending: ResMut<PendingAction>,
) {
    let mut values = Vec::with_capacity(8);
    // 6 arm joint targets
    for &entity in &ik.joint_entities {
        let target = motor_overrides
            .joints
            .get(&entity)
            .map_or(0.0, |p| p.target_pos);
        values.push(target);
    }
    // 2 gripper finger targets
    for &entity in &gripper.0 {
        let target = motor_overrides
            .joints
            .get(&entity)
            .map_or(0.0, |p| p.target_pos);
        values.push(target);
    }
    pending.0 = values;
}

/// Scripted pick state machine: IK control + gripper + phase transitions.
#[allow(clippy::needless_pass_by_value)]
fn pick_control_system(
    ik: Res<ArmIkState>,
    gripper: Res<GripperEntities>,
    mut pick_state: ResMut<PickState>,
    episode: Res<Episode>,
    mut motor_overrides: ResMut<MotorOverrides>,
    query: Query<&JointState>,
    mut exit: MessageWriter<AppExit>,
) {
    if pick_state.done || !episode.is_running() {
        return;
    }

    let phase = pick_state.phase;
    pick_state.step += 1;

    // IK solve for current phase target
    let target_pos = phase.ik_target();
    let target = IkTarget::Position(target_pos);

    let mut q_current = Vec::with_capacity(ik.joint_entities.len());
    for &entity in &ik.joint_entities {
        if let Ok(state) = query.get(entity) {
            q_current.push(state.position);
        }
    }

    if q_current.len() == ik.chain.dof() {
        let result = ik.solver.solve(&ik.chain, &target, &q_current);
        for (i, &entity) in ik.joint_entities.iter().enumerate() {
            if i >= 6 {
                break;
            }
            motor_overrides.joints.insert(
                entity,
                MotorOverrideParams {
                    target_pos: result.joint_positions[i],
                    target_vel: 0.0,
                    stiffness: ARM_STIFFNESS,
                    damping: ARM_DAMPING,
                    max_force: EFFORT_LIMITS[i],
                },
            );
        }
    }

    // Gripper control
    let finger_target = phase.finger_target();
    for &entity in &gripper.0 {
        motor_overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos: finger_target,
                target_vel: 0.0,
                stiffness: GRIPPER_STIFFNESS,
                damping: GRIPPER_DAMPING,
                max_force: GRIPPER_MAX_FORCE,
            },
        );
    }

    // Phase transition
    if pick_state.step >= phase.max_steps() {
        if let Some(next) = phase.next() {
            println!("  ep {}: {:?} -> {:?}", pick_state.episode + 1, phase, next);
            pick_state.phase = next;
            pick_state.step = 0;
        } else {
            // Episode done
            println!(
                "  ep {}/{}: pick sequence complete",
                pick_state.episode + 1,
                pick_state.total_episodes
            );
            pick_state.done = true;
            exit.write(AppExit::Success);
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    println!("=== Arm Pick Record ===");
    println!(
        "  episodes={}, max_steps={}, output prefix={}",
        args.episodes,
        args.max_steps,
        args.output.display(),
    );

    // Create output directory
    if let Some(parent) = args.output.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).expect("failed to create output directory");
    }

    for ep in 0..args.episodes {
        let output_path = PathBuf::from(format!("{}_{:03}.mcap", args.output.display(), ep + 1));
        println!(
            "\n--- Episode {}/{} -> {} ---",
            ep + 1,
            args.episodes,
            output_path.display()
        );

        // 1. Setup arm with gripper (8 DOF)
        let setup = setup_arm(ArmSetupConfig {
            max_episode_steps: args.max_steps,
            use_fixed_update: true,
            sensor_dof: 8,
            ..ArmSetupConfig::default()
        });

        let gripper_entities = {
            let spawned = &setup.scene.robots["six_dof_arm"];
            GripperEntities([
                spawned
                    .joint_entity("j_finger_left")
                    .expect("j_finger_left not found"),
                spawned
                    .joint_entity("j_finger_right")
                    .expect("j_finger_right not found"),
            ])
        };

        // Pre-populate motor overrides so motors hold from the first physics step
        let motor_overrides = initial_motor_overrides(&setup, &gripper_entities.0);

        let mut scene = setup.scene;

        // Add Bevy Name components to joint entities so record_joint_states_system
        // can query them (the URDF spawner only adds JointName, not Name).
        {
            let spawned = &scene.robots["six_dof_arm"];
            for (jname, &entity) in &spawned.joints {
                scene
                    .app
                    .world_mut()
                    .entity_mut(entity)
                    .insert(Name::new(jname.clone()));
            }
        }

        // 2. Add table and cube to physics
        let mut object_initial_positions = Vec::new();
        {
            let world = scene.app.world_mut();
            let mut ctx = world.remove_resource::<RapierContext>().unwrap();
            ctx.integration_parameters.num_solver_iterations = 50;

            // Table (fixed)
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

            // Red cube (dynamic)
            let cube_pos = Vec3::new(CUBE_POS[0], CUBE_POS[1], CUBE_POS[2]);
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

            // Finger colliders
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

            ctx.snapshot_initial_state();
            world.insert_resource(ctx);
        }

        scene
            .app
            .insert_resource(ObjectInitialPositions(object_initial_positions));

        // 3. Window (must come before ImageCopyPlugin to avoid duplicate GpuReadbackPlugin)
        scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("Arm Pick Record - Episode {}/{}", ep + 1, args.episodes),
                resolution: (640, 480).into(),
                ..default()
            }),
            ..default()
        }));

        // 4. Rendering plugins (after DefaultPlugins so GpuReadbackPlugin is already registered)
        scene.app.add_plugins(ClankersRenderPlugin);
        scene.app.add_plugins(ImageCopyPlugin);

        // 5. Recorder plugin (with camera)
        scene.app.insert_resource(RecordingConfig {
            output_path: output_path.clone(),
            record_body_poses: true,
            ..RecordingConfig::default()
        });
        scene.app.add_plugins(RecorderPlugin);

        // 6. Resources
        let solver = arm_ik_solver();
        scene.app.insert_resource(ArmIkState {
            chain: setup.chain,
            solver,
            joint_entities: setup.joint_entities.clone(),
            targets: vec![Vector3::new(
                APPROACH_POS[0],
                APPROACH_POS[1],
                APPROACH_POS[2],
            )],
            current_target: 0,
            steps_at_target: 0,
            steps_per_target: 9999,
        });
        scene.app.insert_resource(motor_overrides);
        scene.app.insert_resource(gripper_entities);
        scene.app.insert_resource(PickState {
            phase: PickPhase::Approach,
            step: 0,
            episode: ep,
            total_episodes: args.episodes,
            done: false,
        });
        scene.app.insert_resource(ObsCameraSize {
            width: 64,
            height: 64,
        });

        // 7. Startup systems
        scene
            .app
            .add_systems(Startup, (spawn_arm_meshes, spawn_obs_camera));

        // 8. Runtime systems
        scene
            .app
            .add_systems(FixedUpdate, pick_control_system.in_set(ClankersSet::Decide));
        scene.app.add_systems(
            Update,
            (
                sync_link_visuals.after(ClankersSet::Simulate),
                populate_body_poses.after(ClankersSet::Simulate),
                capture_action_system.after(ClankersSet::Simulate),
            ),
        );

        // 9. Start episode and run
        scene.app.world_mut().resource_mut::<Episode>().reset(None);
        scene.app.run();

        println!("  wrote {}", output_path.display());
    }

    println!(
        "\nArm pick record DONE — {} episodes recorded",
        args.episodes
    );
}
