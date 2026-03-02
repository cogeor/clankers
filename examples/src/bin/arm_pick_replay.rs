//! Arm pick trace replay visualization.
//!
//! Loads a dry-run trace JSON and plays back body poses in a Bevy window.
//! No physics simulation — transforms are set directly from recorded data.
//!
//! Run: `cargo run -j 24 -p clankers-examples --bin arm_pick_replay -- <trace.json>`

use std::collections::HashMap;

use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use clankers_viz::{phys_rot_to_vis, phys_to_vis};
use clap::Parser;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Replay an arm pick trace in 3D")]
struct Cli {
    /// Path to the trace JSON file
    trace: String,
}

// ---------------------------------------------------------------------------
// Trace deserialization
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TraceFile {
    plan_id: String,
    steps: Vec<TraceStep>,
}

#[derive(Deserialize)]
struct TraceStep {
    info: TraceStepInfo,
}

#[derive(Deserialize)]
struct TraceStepInfo {
    body_poses: HashMap<String, [f32; 7]>,
    #[serde(default)]
    is_success: bool,
}

// ---------------------------------------------------------------------------
// Resources & Components
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct TracePlayback {
    plan_id: String,
    frames: Vec<HashMap<String, [f32; 7]>>,
    is_success: Vec<bool>,
    cursor: usize,
    playing: bool,
    speed: f32,
    accumulator: f32,
    dt: f32,
}

#[derive(Component)]
struct BodyVisual(String);

// ---------------------------------------------------------------------------
// Startup: camera, ground, lights
// ---------------------------------------------------------------------------

fn spawn_camera_and_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Transform::from_xyz(0.8, 0.8, 0.8).looking_at(Vec3::new(0.2, 0.4, 0.0), Vec3::Y),
        PanOrbitCamera {
            focus: Vec3::new(0.2, 0.4, 0.0),
            radius: Some(1.2),
            ..default()
        },
        Camera3d::default(),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(5.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.38, 0.35),
            ..default()
        })),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
        ..default()
    });
}

// ---------------------------------------------------------------------------
// Startup: arm meshes + table + cube (each tagged with BodyVisual)
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
    let wrist_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.8, 0.2),
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
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    let red_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.15, 0.1),
        ..default()
    });

    // Parent entity with BodyVisual + child mesh at local offset.
    // Parent transform is updated each frame from the trace.
    macro_rules! body {
        ($name:expr, $mesh:expr, $mat:expr) => {
            body!($name, $mesh, $mat, Transform::IDENTITY)
        };
        ($name:expr, $mesh:expr, $mat:expr, $offset:expr) => {
            commands
                .spawn((
                    BodyVisual($name.to_string()),
                    Visibility::default(),
                    Transform::default(),
                ))
                .with_children(|p| {
                    p.spawn((Mesh3d(meshes.add($mesh)), MeshMaterial3d($mat), $offset));
                });
        };
    }

    body!("base", Cylinder::new(0.08, 0.1), base_mat);
    body!(
        "shoulder_link",
        Cylinder::new(0.04, 0.2),
        link_mat.clone(),
        Transform::from_xyz(0.0, 0.1, 0.0)
    );
    body!(
        "upper_arm",
        Cylinder::new(0.035, 0.3),
        link_mat,
        Transform::from_xyz(0.0, 0.15, 0.0)
    );
    body!(
        "elbow_link",
        Cylinder::new(0.03, 0.1),
        forearm_mat.clone(),
        Transform::from_xyz(0.0, 0.05, 0.0)
    );
    body!(
        "forearm",
        Cylinder::new(0.025, 0.2),
        forearm_mat,
        Transform::from_xyz(0.0, 0.1, 0.0)
    );
    body!(
        "wrist_link",
        Cylinder::new(0.02, 0.06),
        wrist_mat,
        Transform::from_xyz(0.0, 0.03, 0.0)
    );
    body!("end_effector", Sphere::new(0.025), ee_mat);
    body!(
        "gripper_base",
        Cuboid::new(0.06, 0.02, 0.04),
        gripper_mat.clone()
    );
    body!(
        "finger_left",
        Cuboid::new(0.01, 0.04, 0.01),
        gripper_mat.clone(),
        Transform::from_xyz(0.0, 0.02, 0.0)
    );
    body!(
        "finger_right",
        Cuboid::new(0.01, 0.04, 0.01),
        gripper_mat,
        Transform::from_xyz(0.0, 0.02, 0.0)
    );

    // Table: physics half-extents (0.3, 0.2, 0.0125) -> visual full (0.6, 0.025, 0.4)
    body!("table", Cuboid::new(0.6, 0.025, 0.4), table_mat);
    // Red cube: physics half-extents (0.0125^3) -> visual full (0.025^3)
    body!("red_cube", Cuboid::new(0.025, 0.025, 0.025), red_mat);
}

// ---------------------------------------------------------------------------
// Update: playback advance
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn playback_advance_system(time: Res<Time>, mut playback: ResMut<TracePlayback>) {
    if !playback.playing {
        return;
    }
    let n = playback.frames.len();
    playback.accumulator += time.delta_secs() * playback.speed;
    #[allow(clippy::while_float)]
    while playback.accumulator >= playback.dt {
        playback.accumulator -= playback.dt;
        playback.cursor += 1;
        if playback.cursor >= n {
            playback.cursor = n - 1;
            playback.playing = false;
            playback.accumulator = 0.0;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Update: apply body poses from trace to BodyVisual entities
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn apply_body_poses_system(
    playback: Res<TracePlayback>,
    mut query: Query<(&BodyVisual, &mut Transform)>,
) {
    let frame = &playback.frames[playback.cursor];
    for (body, mut tf) in &mut query {
        if let Some(pose) = frame.get(&body.0) {
            let pos = Vec3::new(pose[0], pose[1], pose[2]);
            let rot = Quat::from_xyzw(pose[3], pose[4], pose[5], pose[6]);
            tf.translation = phys_to_vis(pos);
            tf.rotation = phys_rot_to_vis(&rot);
        }
    }
}

// ---------------------------------------------------------------------------
// Update: keyboard shortcuts
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn keyboard_system(input: Res<ButtonInput<KeyCode>>, mut playback: ResMut<TracePlayback>) {
    let n = playback.frames.len();

    if input.just_pressed(KeyCode::Space) {
        playback.playing = !playback.playing;
        if playback.playing && playback.cursor >= n - 1 {
            playback.cursor = 0;
            playback.accumulator = 0.0;
        }
    }

    if input.just_pressed(KeyCode::KeyR) {
        playback.cursor = 0;
        playback.playing = false;
        playback.accumulator = 0.0;
    }

    if !playback.playing {
        if input.just_pressed(KeyCode::ArrowRight) && playback.cursor < n - 1 {
            playback.cursor += 1;
        }
        if input.just_pressed(KeyCode::ArrowLeft) && playback.cursor > 0 {
            playback.cursor -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Update: disable orbit camera when egui wants pointer
// ---------------------------------------------------------------------------

fn egui_camera_gate(mut contexts: EguiContexts, mut cameras: Query<&mut PanOrbitCamera>) {
    let egui_wants = contexts
        .ctx_mut()
        .is_ok_and(|ctx| ctx.is_pointer_over_area() || ctx.wants_pointer_input());
    for mut cam in &mut cameras {
        cam.enabled = !egui_wants;
    }
}

// ---------------------------------------------------------------------------
// Egui: replay control panel
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn replay_panel_system(mut contexts: EguiContexts, mut playback: ResMut<TracePlayback>) {
    let Ok(egui_ctx) = contexts.ctx_mut() else {
        return;
    };

    let n = playback.frames.len();

    egui::Window::new("Trace Replay")
        .default_pos([10.0, 10.0])
        .default_width(320.0)
        .resizable(true)
        .show(egui_ctx, |ui| {
            ui.label(format!("Plan: {}", playback.plan_id));
            ui.label(format!("Frame: {} / {n}", playback.cursor + 1));
            ui.label(format!(
                "Time: {:.2}s / {:.2}s",
                playback.cursor as f32 * playback.dt,
                (n - 1) as f32 * playback.dt,
            ));

            ui.separator();

            ui.horizontal(|ui| {
                let label = if playback.playing { "Pause" } else { "Play" };
                if ui.button(label).clicked() {
                    playback.playing = !playback.playing;
                    if playback.playing && playback.cursor >= n - 1 {
                        playback.cursor = 0;
                        playback.accumulator = 0.0;
                    }
                }
                if ui.button("Reset").clicked() {
                    playback.cursor = 0;
                    playback.playing = false;
                    playback.accumulator = 0.0;
                }
            });

            let mut cursor = playback.cursor;
            let resp = ui.add(egui::Slider::new(&mut cursor, 0..=(n - 1)).text("Frame"));
            if resp.changed() {
                playback.cursor = cursor;
                playback.playing = false;
            }

            ui.add(
                egui::Slider::new(&mut playback.speed, 0.1..=5.0)
                    .text("Speed")
                    .step_by(0.1),
            );

            ui.separator();

            let frame = &playback.frames[playback.cursor];
            if let Some(&[_, _, z, ..]) = frame.get("red_cube") {
                ui.label(format!("Cube Z: {z:.4}"));
            }
            if let Some(&[x, y, z, ..]) = frame.get("end_effector") {
                ui.label(format!("EE: [{x:.3}, {y:.3}, {z:.3}]"));
            }

            let success = playback.is_success[playback.cursor];
            let (text, color) = if success {
                ("Success", egui::Color32::from_rgb(50, 200, 80))
            } else {
                ("Not yet", egui::Color32::from_rgb(180, 180, 180))
            };
            ui.colored_label(color, text);

            ui.separator();
            egui::CollapsingHeader::new("Shortcuts")
                .default_open(false)
                .show(ui, |ui| {
                    ui.small("Space: play/pause");
                    ui.small("R: reset");
                    ui.small("Left/Right: step frame (paused)");
                });
        });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let json = std::fs::read_to_string(&cli.trace)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", cli.trace));
    let trace: TraceFile = serde_json::from_str(&json).expect("failed to parse trace JSON");

    let n = trace.steps.len();
    assert!(n > 0, "trace file contains no steps");

    let playback = TracePlayback {
        plan_id: trace.plan_id,
        frames: trace
            .steps
            .iter()
            .map(|s| s.info.body_poses.clone())
            .collect(),
        is_success: trace.steps.iter().map(|s| s.info.is_success).collect(),
        cursor: 0,
        playing: false,
        speed: 1.0,
        accumulator: 0.0,
        dt: 0.02,
    };

    println!(
        "Arm Pick Replay -- {n} frames, {:.1}s",
        (n - 1) as f32 * 0.02
    );

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Clankers -- Arm Pick Replay".to_string(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .insert_resource(playback)
        .add_systems(Startup, (spawn_camera_and_scene, spawn_arm_meshes))
        .add_systems(
            Update,
            (
                playback_advance_system,
                apply_body_poses_system.after(playback_advance_system),
                keyboard_system,
                egui_camera_gate,
            ),
        )
        .add_systems(EguiPrimaryContextPass, replay_panel_system)
        .run();
}
