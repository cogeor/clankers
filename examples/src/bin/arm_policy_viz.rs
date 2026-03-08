//! 6+2 DOF arm driven by a vision-based ONNX policy.
//!
//! Loads a trained vision policy (image + joint_positions -> velocity) and
//! runs it in a Bevy window.  The observation consists of:
//!
//! - End-effector camera image (RGBA, resolution from model metadata)
//! - Joint positions (6 arm + 2 gripper)
//!
//! Policy output is joint velocity, integrated each step as:
//!     target_pos = current_pos + velocity * control_dt
//!
//! **IMPORTANT — Motor control pattern:**
//! All joints (arm AND gripper) MUST use `MotorOverrides` to drive Rapier's
//! built-in PD motor at the physics substep rate.  Joints that fall through
//! to the actuator PID path use ZOH (zero-order hold) torque at the frame
//! rate, which causes oscillation on the arm's light links.  See
//! `arm_ik_viz.rs` and `arm_pick_gym.rs` for the canonical pattern.
//!
//! Run:
//!     cargo run -p clankers-examples --bin arm_policy_viz -- --model vision_bc.onnx

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::path::PathBuf;

use bevy::prelude::*;
use clankers_actuator::components::JointState;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::arm_setup::{
    ArmSetupConfig, initial_motor_overrides, setup_arm, ARM_DAMPING, ARM_STIFFNESS, EFFORT_LIMITS,
    FINGER_TRAVEL, GRIPPER_DAMPING, GRIPPER_MAX_FORCE, GRIPPER_STIFFNESS,
};
use clankers_physics::rapier::{MotorOverrideParams, MotorOverrides, RapierContext};
use clankers_policy::prelude::*;
use clankers_render::camera::spawn_camera_sensor;
use clankers_render::prelude::*;
use clankers_teleop::prelude::*;
use clankers_viz::{ClankersVizPlugin, VizMode, phys_rot_to_vis, phys_to_vis};
use clap::Parser;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Arm vision policy visualization.
#[derive(Parser)]
#[command(name = "arm_policy_viz")]
#[command(about = "Visualize a trained vision ONNX policy on the 6-DOF arm")]
struct Cli {
    /// Path to the ONNX policy model file.
    #[arg(long)]
    model: PathBuf,

    /// Control timestep in seconds for velocity integration.
    #[arg(long, default_value = "0.02")]
    control_dt: f32,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Rest pose for all 8 joints: 6 arm + 2 gripper (fingers open at FINGER_TRAVEL).
const REST_POSE_8: [f32; 8] = [
    0.0, FRAC_PI_4, FRAC_PI_2, 0.0, FRAC_PI_4, 0.0, FINGER_TRAVEL, FINGER_TRAVEL,
];

/// Effort limits for all 8 joints: 6 arm + 2 gripper.
const EFFORT_LIMITS_8: [f32; 8] = [
    EFFORT_LIMITS[0],
    EFFORT_LIMITS[1],
    EFFORT_LIMITS[2],
    EFFORT_LIMITS[3],
    EFFORT_LIMITS[4],
    EFFORT_LIMITS[5],
    GRIPPER_MAX_FORCE,
    GRIPPER_MAX_FORCE,
];

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

/// Joint entities for the 6-DOF arm (excludes gripper).
#[derive(Resource)]
struct ArmJointEntities(Vec<Entity>);

/// The two prismatic finger joint entities (left, right).
#[derive(Resource)]
struct GripperEntities([Entity; 2]);

/// Control timestep for velocity integration.
#[derive(Resource)]
struct ControlDt(f32);

/// Image resolution for the observation camera.
#[derive(Resource)]
struct ObsCameraConfig {
    width: u32,
    height: u32,
}

/// Visual marker for link mesh sync.
#[derive(Component)]
struct LinkVisual(&'static str);

// ---------------------------------------------------------------------------
// Startup: spawn observation camera
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn spawn_obs_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut cam_bufs: ResMut<CameraFrameBuffers>,
    config: Res<ObsCameraConfig>,
) {
    let cam_config = CameraConfig::new().with_label("ee_camera");
    spawn_camera_sensor(
        &mut commands,
        &mut images,
        &mut cam_bufs,
        cam_config,
        config.width,
        config.height,
    );
}

// ---------------------------------------------------------------------------
// Startup: spawn arm meshes
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

    // Simplified link meshes (subset of arm_ik_viz)
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

    // Table surface
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.02, 0.4))),
        MeshMaterial3d(table_mat),
        Transform::from_xyz(0.3, -0.01, 0.0),
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

/// Track the observation camera to the end-effector body.
#[allow(clippy::needless_pass_by_value)]
fn track_ee_camera(ctx: Res<RapierContext>, mut cam_q: Query<&mut Transform, With<SimCamera>>) {
    let Some(&handle) = ctx.body_handles.get("end_effector") else {
        return;
    };
    let Some(body) = ctx.rigid_body_set.get(handle) else {
        return;
    };

    let body_rot = phys_rot_to_vis(body.rotation());
    let ee_pos = phys_to_vis(body.translation());
    let ee_forward = body_rot * Vec3::Y;

    for mut tf in &mut cam_q {
        // Place camera slightly behind the EE, looking forward
        tf.translation = ee_pos - ee_forward * 0.05;
        tf.rotation = body_rot * Quat::from_rotation_x(FRAC_PI_2);
    }
}

/// Apply policy velocity output as motor position targets.
///
/// velocity output -> target_pos = current_pos + vel * dt -> MotorOverrides
///
/// **All joints (arm + gripper) must go through `MotorOverrides`.**
/// Joints that fall through to the actuator PID path use ZOH torque at
/// frame rate, which causes oscillation on light links.
#[allow(clippy::needless_pass_by_value)]
fn apply_velocity_action(
    runner: Res<PolicyRunner>,
    joints_res: Res<ArmJointEntities>,
    gripper: Res<GripperEntities>,
    dt: Res<ControlDt>,
    episode: Res<Episode>,
    joint_q: Query<&JointState>,
    mut motor_overrides: ResMut<MotorOverrides>,
) {
    if !episode.is_running() {
        return;
    }

    let action = runner.action().as_slice();

    // Arm joints (0..6): velocity integration with stiff PD motor
    for (i, &entity) in joints_res.0.iter().enumerate() {
        let velocity = action.get(i).copied().unwrap_or(0.0);
        let current_pos = joint_q.get(entity).map_or(REST_POSE_8[i], |s| s.position);

        let target_pos = current_pos + velocity * dt.0;

        motor_overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos,
                target_vel: 0.0,
                stiffness: ARM_STIFFNESS,
                damping: ARM_DAMPING,
                max_force: EFFORT_LIMITS_8[i],
            },
        );
    }

    // Gripper fingers (6..8): velocity integration with softer PD motor.
    // Uses the same MotorOverrides path to avoid ZOH oscillation.
    let arm_dof = joints_res.0.len();
    for (fi, &entity) in gripper.0.iter().enumerate() {
        let i = arm_dof + fi;
        let velocity = action.get(i).copied().unwrap_or(0.0);
        let current_pos = joint_q.get(entity).map_or(FINGER_TRAVEL, |s| s.position);

        let target_pos = (current_pos + velocity * dt.0).clamp(0.0, FINGER_TRAVEL);

        motor_overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos,
                target_vel: 0.0,
                stiffness: GRIPPER_STIFFNESS,
                damping: GRIPPER_DAMPING,
                max_force: EFFORT_LIMITS_8[i],
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // 1. Load the ONNX policy
    println!("Loading ONNX policy from {}...", cli.model.display());
    let onnx_policy = OnnxPolicy::from_file(&cli.model)
        .unwrap_or_else(|e| panic!("failed to load ONNX model: {e}"));

    let action_dim = onnx_policy.action_dim();
    let is_vision = onnx_policy.is_vision();

    println!(
        "  obs_dim={}, action_dim={}, vision={}",
        onnx_policy.obs_dim(),
        action_dim,
        is_vision,
    );

    if let Some((c, h, w)) = onnx_policy.image_shape() {
        println!("  image: {c}x{h}x{w}");
    }
    if let Some(jd) = onnx_policy.joint_dim() {
        println!("  joint_dim: {jd}");
    }

    // Image resolution from model metadata (default 64x64 RGBA)
    let (img_c, img_h, img_w) = onnx_policy.image_shape().unwrap_or((4, 64, 64));

    // 2. Setup arm — use joint_dim from model if available, else default to 6
    let joint_dim = onnx_policy.joint_dim().unwrap_or(6);
    let sensor_dof = joint_dim;
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: 50_000,
        use_fixed_update: true,
        sensor_dof,
        ..ArmSetupConfig::default()
    });
    let joint_entities = setup.joint_entities.clone();

    // Extract gripper finger entities
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

    // 3. Window + viz (must come before ImageCopyPlugin to avoid duplicate GpuReadbackPlugin)
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers -- Arm Vision Policy".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene
        .app
        .add_plugins(ClankersVizPlugin { fixed_update: true });

    // Start in Policy mode — this binary runs an ONNX policy, not teleop.
    *scene.app.world_mut().resource_mut::<VizMode>() = VizMode::Policy;

    // 4. Add rendering plugins for image observations
    scene.app.add_plugins(ClankersRenderPlugin);
    scene.app.add_plugins(ImageCopyPlugin);

    // 5. Register sensors: image first, then joint state
    //    Observation layout: [image (H*W*C floats)..., pos (J)..., vel (J)...]
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(
            Box::new(ImageSensor::new(
                "ee_camera",
                img_w as u32,
                img_h as u32,
                img_c as u32,
            )),
            &mut buffer,
        );
        registry.register(Box::new(JointStateSensor::new(sensor_dof)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // 6. PolicyRunner + ClankersPolicyPlugin
    let runner = PolicyRunner::new(Box::new(onnx_policy), action_dim);
    scene.app.insert_resource(runner);
    scene.app.add_plugins(ClankersPolicyPlugin);

    // 7. Resources
    scene.app.insert_resource(ArmJointEntities(joint_entities));
    scene.app.insert_resource(gripper_entities);
    scene.app.insert_resource(ControlDt(cli.control_dt));
    scene.app.insert_resource(motor_overrides);
    scene.app.insert_resource(ObsCameraConfig {
        width: img_w as u32,
        height: img_h as u32,
    });

    // 8. Startup systems
    scene
        .app
        .add_systems(Startup, (spawn_arm_meshes, spawn_obs_camera));

    // 9. Runtime: visual sync + camera tracking
    scene.app.add_systems(
        Update,
        (
            sync_link_visuals.after(ClankersSet::Simulate),
            track_ee_camera.after(ClankersSet::Simulate),
        ),
    );

    // 10. Action applicator: velocity -> motor overrides (runs in Decide set)
    scene.app.add_systems(
        FixedUpdate,
        apply_velocity_action.in_set(ClankersSet::Decide),
    );

    // 11. Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Arm Vision Policy Viz");
    println!("  Model: {}", cli.model.display());
    println!("  Image: {img_c}x{img_h}x{img_w}");
    println!("  control_dt: {}s", cli.control_dt);
    scene.app.run();
}
