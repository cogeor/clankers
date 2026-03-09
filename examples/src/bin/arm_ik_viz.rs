//! Arm IK visualization with end-effector camera.
//!
//! Full-screen orbit view with EE camera viewport in the top-right corner.
//! Floating control panel with mode buttons and joint sliders.
//!
//! - **Teleop** (default): control joints via sliders
//! - **Policy**: arm cycles through 6 IK workspace targets
//!
//! Run: `cargo run -p clankers-examples --bin arm_ik_viz`

use std::f32::consts::{FRAC_PI_2, PI};

use bevy::camera::{ClearColorConfig, Viewport};
use bevy::prelude::*;
use bevy_egui::{EguiContexts, PrimaryEguiContext, egui};
use bevy_panorbit_camera::PanOrbitCamera;
use clankers_actuator::components::JointState;
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use nalgebra::Vector3;

use clankers_examples::arm_setup::{
    ArmIkState, ArmSetupConfig, REST_POSE, arm_ik_solver, initial_motor_overrides, setup_arm,
    ARM_DAMPING, ARM_STIFFNESS, EFFORT_LIMITS, FINGER_TRAVEL, GRIPPER_DAMPING, GRIPPER_MAX_FORCE,
    GRIPPER_STIFFNESS,
};
use clankers_examples::arm_visuals::{
    GripperEntities, spawn_arm_link_meshes, sync_link_visuals,
};
use clankers_physics::rapier::{MotorOverrideParams, MotorOverrides, RapierContext};
use clankers_teleop::prelude::*;
use clankers_viz::{ClankersVizPlugin, VizMode, camera, phys_rot_to_vis, phys_to_vis};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const JOINT_LIMITS: [[f32; 2]; 6] = [
    [-PI, PI],           // j1_base_yaw
    [-FRAC_PI_2, 2.356], // j2_shoulder_pitch
    [-2.356, 2.356],     // j3_elbow_pitch
    [-PI, PI],           // j4_forearm_roll
    [-2.094, 2.094],     // j5_wrist_pitch
    [-PI, PI],           // j6_wrist_roll
];

const JOINT_LABELS: [&str; 6] = [
    "Base Yaw",
    "Shoulder",
    "Elbow",
    "FA Roll",
    "Wrist Pitch",
    "Wrist Roll",
];

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct ArmUiState {
    targets: [f32; 6],
    /// Gripper opening: 0.0 = closed, 1.0 = fully open.
    gripper: f32,
}

impl Default for ArmUiState {
    fn default() -> Self {
        Self {
            targets: REST_POSE,
            gripper: 1.0, // start open
        }
    }
}

#[derive(Component)]
struct EeCamera;

#[derive(Component)]
struct CameraVisual;

#[derive(Component)]
struct GoalGizmo;

// ---------------------------------------------------------------------------
// Startup: pin egui to the orbit camera (not the EE overlay)
// ---------------------------------------------------------------------------

fn assign_egui_to_orbit_cam(mut commands: Commands, cam_q: Query<Entity, With<PanOrbitCamera>>) {
    if let Ok(entity) = cam_q.single() {
        commands.entity(entity).insert(PrimaryEguiContext);
    }
}

// ---------------------------------------------------------------------------
// Startup: EE camera with viewport overlay (top-right, 1/4 width)
// ---------------------------------------------------------------------------

fn spawn_ee_camera(mut commands: Commands) {
    commands.spawn((
        EeCamera,
        Camera3d::default(),
        Camera {
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.05, 0.05, 0.1)),
            ..default()
        },
        Projection::Perspective(PerspectiveProjection {
            fov: 70_f32.to_radians(),
            ..default()
        }),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));
}

/// Keep the EE camera viewport pinned to the top-right corner.
fn configure_ee_viewport(windows: Query<&Window>, mut ee_cam: Query<&mut Camera, With<EeCamera>>) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.physical_width();
    let vp_w = w / 4;
    let vp_h = vp_w * 3 / 4; // 4:3 aspect
    for mut cam in &mut ee_cam {
        cam.viewport = Some(Viewport {
            physical_position: UVec2::new(w - vp_w, 0),
            physical_size: UVec2::new(vp_w, vp_h),
            ..default()
        });
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn scene-specific extras (camera visual, workspace objects, gizmo)
// ---------------------------------------------------------------------------

fn spawn_ik_viz_extras(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera frustum visual (small cone at EE)
    let cam_vis_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.2, 0.6, 1.0, 0.5),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands.spawn((
        CameraVisual,
        Mesh3d(meshes.add(Cone::new(0.02, 0.05))),
        MeshMaterial3d(cam_vis_mat),
        Visibility::default(),
        Transform::default(),
    ));

    // Table surface: provides visual grounding for the workspace
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.02, 0.4))),
        MeshMaterial3d(table_mat),
        Transform::from_xyz(0.3, -0.01, 0.0),
    ));

    // Red cube
    let red_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.15, 0.1),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.04, 0.04, 0.04))),
        MeshMaterial3d(red_mat),
        Transform::from_xyz(0.35, 0.02, -0.08),
    ));

    // Green cube
    let green_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.85, 0.15),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.035, 0.035, 0.035))),
        MeshMaterial3d(green_mat),
        Transform::from_xyz(0.25, 0.018, 0.06),
    ));

    // Blue cylinder
    let blue_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.3, 1.0),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.02, 0.05))),
        MeshMaterial3d(blue_mat),
        Transform::from_xyz(0.4, 0.025, 0.05),
    ));

    // Yellow cube
    let yellow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.9, 0.05),
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.03, 0.03, 0.03))),
        MeshMaterial3d(yellow_mat),
        Transform::from_xyz(0.30, 0.015, -0.03),
    ));

    // Goal gizmo: translucent green sphere tracking the current IK target
    let goal_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.1, 1.0, 0.2, 0.4),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    commands.spawn((
        GoalGizmo,
        Mesh3d(meshes.add(Sphere::new(0.03))),
        MeshMaterial3d(goal_mat),
        Visibility::Hidden,
        Transform::default(),
    ));
}

// ---------------------------------------------------------------------------
// Runtime systems
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn sync_ee_camera_system(
    ctx: Res<RapierContext>,
    mut cam_q: Query<&mut Transform, (With<EeCamera>, Without<CameraVisual>)>,
    mut vis_q: Query<&mut Transform, (With<CameraVisual>, Without<EeCamera>)>,
) {
    let Some(&handle) = ctx.body_handles.get("end_effector") else {
        return;
    };
    let Some(body) = ctx.rigid_body_set.get(handle) else {
        return;
    };

    let body_rot = phys_rot_to_vis(body.rotation());
    let cam_orient = Quat::from_rotation_x(FRAC_PI_2);
    let ee_pos = phys_to_vis(body.translation());
    let ee_forward = body_rot * Vec3::Y;

    for mut tf in &mut cam_q {
        tf.translation = ee_pos - ee_forward * 0.05;
        tf.rotation = body_rot * cam_orient;
    }

    for mut tf in &mut vis_q {
        tf.translation = ee_pos + ee_forward * 0.03;
        tf.rotation = body_rot * Quat::from_rotation_x(-FRAC_PI_2);
    }
}

#[allow(clippy::needless_pass_by_value)]
fn update_goal_gizmo_system(
    ik: Res<ArmIkState>,
    mode: Res<VizMode>,
    mut gizmo_q: Query<(&mut Transform, &mut Visibility), With<GoalGizmo>>,
) {
    let Ok((mut tf, mut vis)) = gizmo_q.single_mut() else {
        return;
    };

    if *mode == VizMode::Policy {
        let target = ik.targets[ik.current_target];
        // Physics Z-up -> Bevy Y-up: phys_to_vis(x, y, z) = (x, z, -y)
        tf.translation = Vec3::new(target.x, target.z, -target.y);
        *vis = Visibility::Inherited;
    } else {
        *vis = Visibility::Hidden;
    }
}

#[allow(clippy::needless_pass_by_value)]
fn arm_motor_override_system(
    ik: Res<ArmIkState>,
    mode: Res<VizMode>,
    ui_state: Res<ArmUiState>,
    gripper: Res<GripperEntities>,
    mut motor_overrides: ResMut<MotorOverrides>,
) {
    // Arm joints — always active (no episode gate) so motors hold from frame 1.
    for (i, &entity) in ik.joint_entities.iter().enumerate() {
        if i >= 6 {
            break;
        }
        let target_pos = if *mode == VizMode::Policy {
            motor_overrides
                .joints
                .get(&entity)
                .map_or(ui_state.targets[i], |p| p.target_pos)
        } else {
            ui_state.targets[i]
        };

        motor_overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos,
                target_vel: 0.0,
                stiffness: ARM_STIFFNESS,
                damping: ARM_DAMPING,
                max_force: EFFORT_LIMITS[i],
            },
        );
    }

    // Gripper fingers (always slider-controlled, both modes)
    let finger_pos = ui_state.gripper * FINGER_TRAVEL;
    for &entity in &gripper.0 {
        motor_overrides.joints.insert(
            entity,
            MotorOverrideParams {
                target_pos: finger_pos,
                target_vel: 0.0,
                stiffness: GRIPPER_STIFFNESS,
                damping: GRIPPER_DAMPING,
                max_force: GRIPPER_MAX_FORCE,
            },
        );
    }
}

#[allow(clippy::needless_pass_by_value)]
fn ik_motor_control_system(
    mut ik: ResMut<ArmIkState>,
    mut motor_overrides: ResMut<MotorOverrides>,
    query: Query<&JointState>,
) {
    ik.steps_at_target += 1;
    if ik.steps_at_target >= ik.steps_per_target {
        ik.steps_at_target = 0;
        ik.current_target = (ik.current_target + 1) % ik.targets.len();
    }

    let target_pos = ik.targets[ik.current_target];
    let target = clankers_ik::IkTarget::Position(target_pos);

    let mut q_current = Vec::with_capacity(ik.joint_entities.len());
    for &entity in &ik.joint_entities {
        if let Ok(state) = query.get(entity) {
            q_current.push(state.position);
        }
    }

    if q_current.len() != ik.chain.dof() {
        return;
    }

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

// ---------------------------------------------------------------------------
// Floating egui control panel
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)]
fn arm_panel_system(
    mut contexts: EguiContexts,
    mut mode: ResMut<VizMode>,
    mut ui_state: ResMut<ArmUiState>,
    ik: Res<ArmIkState>,
    ctx: Res<RapierContext>,
    episode: Res<Episode>,
    joint_q: Query<&JointState>,
) {
    let Ok(egui_ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Arm IK Control")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .default_open(true)
        .resizable(true)
        .show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                for candidate in [VizMode::Paused, VizMode::Teleop, VizMode::Policy] {
                    let label = match candidate {
                        VizMode::Paused => "Paused",
                        VizMode::Teleop => "Teleop",
                        VizMode::Policy => "Policy",
                        VizMode::Replay => "Replay",
                    };
                    if ui
                        .add(egui::Button::new(label).selected(*mode == candidate))
                        .clicked()
                    {
                        *mode = candidate;
                    }
                }
            });

            ui.separator();

            let is_teleop = *mode == VizMode::Teleop;
            ui.label(if is_teleop {
                "Joint Targets (drag to move)"
            } else {
                "Joint Positions (read-only)"
            });

            for (i, &entity) in ik.joint_entities.iter().enumerate() {
                if i >= 6 {
                    break;
                }
                let [lo, hi] = JOINT_LIMITS[i];
                if is_teleop {
                    ui.add(
                        egui::Slider::new(&mut ui_state.targets[i], lo..=hi)
                            .text(JOINT_LABELS[i])
                            .step_by(0.01),
                    );
                } else if let Ok(state) = joint_q.get(entity) {
                    let mut val = state.position;
                    ui.horizontal(|ui| {
                        ui.add_enabled(
                            false,
                            egui::Slider::new(&mut val, lo..=hi).text(JOINT_LABELS[i]),
                        );
                        ui.small(format!("{:+.3} rad/s", state.velocity));
                    });
                }
            }

            // Gripper slider (always active)
            ui.separator();
            ui.add(
                egui::Slider::new(&mut ui_state.gripper, 0.0..=1.0)
                    .text("Gripper")
                    .step_by(0.01),
            );

            if is_teleop {
                ui.horizontal(|ui| {
                    if ui.button("Reset to rest").clicked() {
                        ui_state.targets = REST_POSE;
                        ui_state.gripper = 1.0;
                    }
                    if ui.button("Zero all").clicked() {
                        ui_state.targets = [0.0; 6];
                    }
                });
            }

            ui.separator();

            if *mode == VizMode::Policy {
                let t = ik.targets[ik.current_target];
                ui.label(format!("IK target [{:.2}, {:.2}, {:.2}]", t.x, t.y, t.z));
                ui.label(format!(
                    "Target {}/{} (step {}/{})",
                    ik.current_target + 1,
                    ik.targets.len(),
                    ik.steps_at_target,
                    ik.steps_per_target,
                ));
            }

            if let Some(&handle) = ctx.body_handles.get("end_effector")
                && let Some(body) = ctx.rigid_body_set.get(handle)
            {
                let p = body.translation();
                ui.label(format!("EE pos: [{:.3}, {:.3}, {:.3}]", p.x, p.y, p.z));

                // EE orientation as Euler angles in degrees
                let rot = body.rotation();
                let vis_rot = phys_rot_to_vis(rot);
                let (roll, pitch, yaw) = vis_rot.to_euler(EulerRot::XYZ);
                ui.label(format!(
                    "EE orient (r/p/y): [{:.1}, {:.1}, {:.1}] deg",
                    roll.to_degrees(),
                    pitch.to_degrees(),
                    yaw.to_degrees(),
                ));

                // Gripper aperture in mm (two fingers, meters to mm)
                let aperture_mm = ui_state.gripper * FINGER_TRAVEL * 2.0 * 1000.0;
                ui.label(format!("Gripper aperture: {aperture_mm:.1} mm"));
            }

            // Episode info
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Step: {}", episode.step_count));
                ui.label("  Mode:");
                let (mode_text, mode_color) = match *mode {
                    VizMode::Teleop => ("Teleop", egui::Color32::from_rgb(50, 200, 80)),
                    VizMode::Policy => ("Policy", egui::Color32::from_rgb(80, 140, 220)),
                    VizMode::Paused => ("Paused", egui::Color32::from_rgb(160, 160, 160)),
                    VizMode::Replay => ("Replay", egui::Color32::from_rgb(200, 150, 50)),
                };
                ui.colored_label(mode_color, mode_text);
            });

            // Keyboard shortcut help (collapsible)
            ui.separator();
            egui::CollapsingHeader::new("Keyboard Shortcuts")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small("Space: toggle pause");
                    ui.small("T: teleop | P: policy");
                    ui.small("R: reset to rest | G: toggle gripper");
                });
        });
}

// ---------------------------------------------------------------------------
// Keyboard shortcut system
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn keyboard_shortcut_system(
    input: Res<ButtonInput<KeyCode>>,
    mut mode: ResMut<VizMode>,
    mut ui_state: ResMut<ArmUiState>,
    mut last_active_mode: Local<VizMode>,
) {
    if input.just_pressed(KeyCode::Space) {
        if *mode == VizMode::Paused {
            // Restore the previous non-paused mode
            *mode = *last_active_mode;
        } else {
            *last_active_mode = *mode;
            *mode = VizMode::Paused;
        }
    }

    if input.just_pressed(KeyCode::KeyT) {
        *last_active_mode = VizMode::Teleop;
        *mode = VizMode::Teleop;
    }

    if input.just_pressed(KeyCode::KeyP) {
        *last_active_mode = VizMode::Policy;
        *mode = VizMode::Policy;
    }

    if input.just_pressed(KeyCode::KeyR) {
        ui_state.targets = REST_POSE;
        ui_state.gripper = 1.0;
    }

    if input.just_pressed(KeyCode::KeyG) {
        ui_state.gripper = if ui_state.gripper > 0.5 { 0.0 } else { 1.0 };
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let setup = setup_arm(ArmSetupConfig {
        max_episode_steps: 50_000,
        use_fixed_update: true,
        sensor_dof: 8,
        ..ArmSetupConfig::default()
    });

    // Extract gripper finger entities before moving scene
    let gripper_entities = GripperEntities::from_setup(&setup);

    // Pre-populate motor overrides so motors hold from the first physics step.
    let motor_overrides = initial_motor_overrides(
        &setup,
        &gripper_entities.0,
    );

    let mut scene = setup.scene;

    // Table-workspace IK targets (physics Z-up coords).
    // Objects sit at z≈0.02–0.05; we hover 0.15 m above them.
    let targets: Vec<Vector3<f32>> = vec![
        Vector3::new(0.35, -0.08, 0.15), // above red cube
        Vector3::new(0.25, 0.06, 0.15),  // above green cube
        Vector3::new(0.40, 0.05, 0.15),  // above blue cylinder
        Vector3::new(0.30, -0.03, 0.15), // above yellow cube
        Vector3::new(0.20, 0.00, 0.35),  // home — above workspace centre
    ];
    let solver = arm_ik_solver();

    scene.app.insert_resource(ArmIkState {
        chain: setup.chain,
        solver,
        joint_entities: setup.joint_entities,
        targets,
        current_target: 0,
        steps_at_target: 0,
        steps_per_target: 300,
    });
    scene.app.insert_resource(motor_overrides);
    scene.app.insert_resource(ArmUiState::default());
    scene.app.insert_resource(gripper_entities);

    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers -- Arm IK Viz".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    scene.app.add_plugins(ClankersTeleopPlugin);
    scene
        .app
        .add_plugins(ClankersVizPlugin { fixed_update: true });

    // Disable default docked side panel
    scene
        .app
        .world_mut()
        .resource_mut::<clankers_viz::config::VizConfig>()
        .show_panel = false;

    // Motor overrides + IK control
    scene.app.add_systems(
        FixedUpdate,
        (
            arm_motor_override_system.in_set(ClankersSet::Decide),
            ik_motor_control_system
                .in_set(ClankersSet::Decide)
                .run_if(|mode: Res<VizMode>| *mode == VizMode::Policy),
        ),
    );

    // Startup: robot meshes + scene extras + EE camera
    scene
        .app
        .add_systems(Startup, (spawn_arm_link_meshes, spawn_ik_viz_extras));
    scene.app.add_systems(
        Startup,
        (
            assign_egui_to_orbit_cam.after(camera::spawn_camera),
            spawn_ee_camera.after(camera::spawn_camera),
        ),
    );

    // Runtime: visual sync + EE camera viewport + goal gizmo + keyboard shortcuts
    scene.app.add_systems(
        Update,
        (
            sync_link_visuals.after(ClankersSet::Simulate),
            sync_ee_camera_system.after(ClankersSet::Simulate),
            configure_ee_viewport,
            update_goal_gizmo_system,
            keyboard_shortcut_system,
        ),
    );

    // Floating control panel (runs in egui pass so context is ready)
    scene
        .app
        .add_systems(bevy_egui::EguiPrimaryContextPass, arm_panel_system);

    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Arm IK Viz");
    println!("  Teleop: joint sliders | Policy: IK targets");
    println!("  EE camera vignette in top-right corner");
    scene.app.run();
}
