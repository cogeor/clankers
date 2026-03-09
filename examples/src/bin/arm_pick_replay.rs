//! Arm pick trace replay visualization.
//!
//! Loads a dry-run trace JSON and plays back body poses in a Bevy window.
//! No physics simulation — transforms are set directly from recorded data.
//!
//! ## Recording
//!
//! Use `--record <dir>` to capture each frame as a PNG during auto-playback.
//! Add `--gif <path.gif>` to also assemble frames into an animated GIF.
//! A second GIF (`*_robot.gif`) is automatically produced from the robot
//! observation camera viewport.
//!
//! ```sh
//! cargo run -j 24 -p clankers-examples --bin arm_pick_replay -- trace.json --record frames/ --gif replay.gif
//! ```
//!
//! Run: `cargo run -j 24 -p clankers-examples --bin arm_pick_replay -- <trace.json>`

use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2;
use std::path::PathBuf;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use clankers_examples::arm_visuals::{LinkVisual, spawn_arm_link_meshes};
use clankers_viz::camera::{
    ObsCameraConfig, ObservationCamera, ViewportCorner, spawn_obs_camera,
    sync_obs_camera_viewport,
};
use clankers_viz::{phys_rot_to_vis, phys_to_vis};
use clap::Parser;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(about = "Replay an arm pick trace in 3D")]
struct Cli {
    /// Path to the trace JSON file [default: output/arm_pick_dataset/dry_run_trace.json]
    #[arg(default_value = "output/arm_pick_dataset/dry_run_trace.json")]
    trace: String,

    /// Record frames to this directory as PNGs during auto-playback
    #[arg(long)]
    record: Option<PathBuf>,

    /// Assemble recorded frames into an animated GIF at this path
    #[arg(long)]
    gif: Option<PathBuf>,

    /// Frame skip for recording (capture every Nth frame). Default: 1 (every frame)
    #[arg(long, default_value_t = 1)]
    frame_skip: usize,

    /// After recording, run the augmentation pipeline on captured frames
    /// and produce a second GIF with augmented (sim-to-real) images.
    /// Requires: python -m clankers.augmentation
    #[arg(long)]
    augment: Option<PathBuf>,
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

/// Recording state — present only when `--record` is used.
#[derive(Resource)]
struct RecordingState {
    output_dir: PathBuf,
    frame_count: usize,
    frame_skip: usize,
    last_captured_cursor: Option<usize>,
    gif_path: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Startup: camera + scene
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

    // Ground plane
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

    // Table
    let table_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.6, 0.42),
        ..default()
    });
    commands.spawn((
        LinkVisual("table"),
        Mesh3d(meshes.add(Cuboid::new(0.6, 0.025, 0.4))),
        MeshMaterial3d(table_mat),
        Visibility::default(),
        Transform::default(),
    ));

    // Red cube
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
// Update: apply body poses from trace to LinkVisual entities
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn apply_body_poses_system(
    playback: Res<TracePlayback>,
    mut query: Query<(&LinkVisual, &mut Transform)>,
) {
    let frame = &playback.frames[playback.cursor];
    for (body, mut tf) in &mut query {
        if let Some(pose) = frame.get(body.0) {
            let pos = Vec3::new(pose[0], pose[1], pose[2]);
            let rot = Quat::from_xyzw(pose[3], pose[4], pose[5], pose[6]);
            tf.translation = phys_to_vis(pos);
            tf.rotation = phys_rot_to_vis(&rot);
        }
    }
}

// ---------------------------------------------------------------------------
// Update: observation camera follows end-effector
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn sync_ee_camera_system(
    links: Query<(&LinkVisual, &Transform), Without<ObservationCamera>>,
    mut cam_q: Query<&mut Transform, With<ObservationCamera>>,
) {
    let Some((_, ee_tf)) = links.iter().find(|(lv, _)| lv.0 == "end_effector") else {
        return;
    };
    let ee_pos = ee_tf.translation;
    let ee_rot = ee_tf.rotation;
    let cam_orient = Quat::from_rotation_x(FRAC_PI_2);
    let ee_forward = ee_rot * Vec3::Y;

    for mut tf in &mut cam_q {
        tf.translation = ee_pos - ee_forward * 0.05;
        tf.rotation = ee_rot * cam_orient;
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
// Recording: capture frames as PNGs during playback
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn record_frame_system(
    mut commands: Commands,
    playback: Res<TracePlayback>,
    mut recording: ResMut<RecordingState>,
) {
    // Only record while playing
    if !playback.playing {
        // If playback just stopped and we were recording, signal completion
        if recording.frame_count > 0
            && recording
                .last_captured_cursor
                .is_some_and(|c| c == playback.frames.len() - 1)
        {
            println!(
                "  Recording complete: {} frames saved to {}",
                recording.frame_count,
                recording.output_dir.display()
            );
        }
        return;
    }

    // Skip frames per frame_skip setting
    if recording.frame_skip > 1 && !playback.cursor.is_multiple_of(recording.frame_skip) {
        return;
    }

    // Don't re-capture the same cursor position
    if recording.last_captured_cursor == Some(playback.cursor) {
        return;
    }

    let path = recording
        .output_dir
        .join(format!("frame_{:05}.png", recording.frame_count));
    recording.frame_count += 1;
    recording.last_captured_cursor = Some(playback.cursor);

    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}

/// Auto-start playback and exit when done (only in record mode).
#[allow(clippy::needless_pass_by_value)]
fn record_autoplay_system(
    mut playback: ResMut<TracePlayback>,
    recording: Res<RecordingState>,
    mut exit: MessageWriter<AppExit>,
    mut started: Local<bool>,
) {
    // Auto-start playback on first frame
    if !*started {
        *started = true;
        playback.playing = true;
        playback.cursor = 0;
        playback.accumulator = 0.0;
        println!(
            "  Auto-recording {} frames to {}",
            playback.frames.len(),
            recording.output_dir.display()
        );
        return;
    }

    // Auto-exit when playback finishes
    if !playback.playing && recording.frame_count > 0 {
        println!(
            "  Recording done: {} frames in {}",
            recording.frame_count,
            recording.output_dir.display()
        );
        exit.write(AppExit::Success);
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
// GIF assembly (runs after app exits)
// ---------------------------------------------------------------------------

/// Collect sorted PNG frame paths from a directory.
fn collect_frame_paths(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("failed to read frames directory")
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    paths
}

fn assemble_gif(frames_dir: &std::path::Path, gif_path: &std::path::Path, dt: f32) {
    assemble_gif_with_crop(frames_dir, gif_path, dt, None);
}

/// Assemble PNGs into a GIF, optionally cropping each frame to a sub-region.
fn assemble_gif_with_crop(
    frames_dir: &std::path::Path,
    gif_path: &std::path::Path,
    dt: f32,
    crop: Option<(u32, u32, u32, u32)>, // (x, y, w, h)
) {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::Frame;
    use std::fs::File;

    let frame_paths = collect_frame_paths(frames_dir);

    if frame_paths.is_empty() {
        eprintln!("No PNG frames found in {}", frames_dir.display());
        return;
    }

    println!(
        "Assembling {} frames into {}{}...",
        frame_paths.len(),
        gif_path.display(),
        if crop.is_some() { " (cropped)" } else { "" },
    );

    if let Some(parent) = gif_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).expect("failed to create GIF output directory");
    }

    let file = File::create(gif_path).expect("failed to create GIF file");
    let mut encoder = GifEncoder::new(file);
    encoder.set_repeat(Repeat::Infinite).ok();

    // GIF frame delay is in units of 10ms
    let delay_ms = (dt * 1000.0) as u32;
    let delay = image::Delay::from_numer_denom_ms(delay_ms, 1);

    for path in &frame_paths {
        let img = image::open(path).expect("failed to read frame PNG");
        let rgba = if let Some((x, y, w, h)) = crop {
            image::imageops::crop_imm(&img.to_rgba8(), x, y, w, h).to_image()
        } else {
            img.to_rgba8()
        };
        let frame = Frame::from_parts(rgba, 0, 0, delay);
        encoder.encode_frame(frame).expect("failed to encode frame");
    }

    println!("GIF saved to {}", gif_path.display());
}

/// Derive robot cam GIF path from the main GIF path (e.g. replay.gif → replay_robot.gif).
fn robot_gif_path(main_gif: &std::path::Path) -> PathBuf {
    let stem = main_gif
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = main_gif
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("gif");
    main_gif.with_file_name(format!("{stem}_robot.{ext}"))
}

/// Compute the viewport crop region for the observation camera.
fn obs_cam_crop(window_w: u32, window_h: u32, config: &ObsCameraConfig) -> (u32, u32, u32, u32) {
    let vp_w = (window_w as f32 * config.width_fraction) as u32;
    let vp_h = (vp_w as f32 / config.aspect) as u32;
    let (x, y) = match config.corner {
        ViewportCorner::TopRight => (window_w.saturating_sub(vp_w), 0),
        ViewportCorner::TopLeft => (0, 0),
        ViewportCorner::BottomRight => (window_w.saturating_sub(vp_w), window_h.saturating_sub(vp_h)),
        ViewportCorner::BottomLeft => (0, window_h.saturating_sub(vp_h)),
    };
    (x, y, vp_w, vp_h)
}

// ---------------------------------------------------------------------------
// Augmentation: run Python pipeline on frames, produce augmented GIF
// ---------------------------------------------------------------------------

fn augment_frames(
    frames_dir: &std::path::Path,
    augmented_gif_path: &std::path::Path,
    dt: f32,
) {
    let augmented_dir = frames_dir.join("augmented");
    std::fs::create_dir_all(&augmented_dir).expect("failed to create augmented directory");

    // Collect input frame paths
    let mut frame_paths: Vec<PathBuf> = std::fs::read_dir(frames_dir)
        .expect("failed to read frames directory")
        .filter_map(std::result::Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
        .collect();
    frame_paths.sort();

    if frame_paths.is_empty() {
        eprintln!("No PNG frames found for augmentation in {}", frames_dir.display());
        return;
    }

    println!(
        "Augmenting {} frames (python -m clankers.augmentation)...",
        frame_paths.len()
    );

    // Run augmentation on each frame via Python
    for (i, path) in frame_paths.iter().enumerate() {
        let output_path = augmented_dir.join(format!("frame_{i:05}.png"));
        let status = std::process::Command::new("python")
            .args([
                "-m",
                "clankers.augmentation.single_image",
                &path.to_string_lossy(),
                &output_path.to_string_lossy(),
            ])
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!(
                    "  Warning: augmentation failed for frame {} (exit {}), copying original",
                    i,
                    s.code().unwrap_or(-1)
                );
                std::fs::copy(path, &output_path).ok();
            }
            Err(e) => {
                eprintln!(
                    "  Warning: could not run augmentation for frame {i}: {e}"
                );
                eprintln!("  Falling back: copying frames as-is to augmented GIF");
                // Copy all remaining frames and break
                for (j, p) in frame_paths.iter().enumerate().skip(i) {
                    let out = augmented_dir.join(format!("frame_{j:05}.png"));
                    std::fs::copy(p, out).ok();
                }
                break;
            }
        }

        if (i + 1).is_multiple_of(10) || i == frame_paths.len() - 1 {
            println!("  Augmented {}/{}", i + 1, frame_paths.len());
        }
    }

    // Assemble augmented frames into GIF
    assemble_gif(&augmented_dir, augmented_gif_path, dt);
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

    let window_res = (1280_u32, 720_u32);

    let recording = cli.record.as_ref().map(|dir| {
        std::fs::create_dir_all(dir).expect("failed to create recording directory");
        RecordingState {
            output_dir: dir.clone(),
            frame_count: 0,
            frame_skip: cli.frame_skip,
            last_captured_cursor: None,
            gif_path: cli.gif.clone(),
        }
    });

    println!(
        "Arm Pick Replay -- {n} frames, {:.1}s",
        (n - 1) as f32 * 0.02
    );
    if let Some(ref rec) = recording {
        println!("  Recording to: {}", rec.output_dir.display());
        if let Some(ref gif) = rec.gif_path {
            println!("  GIF output: {}", gif.display());
        }
        if let Some(ref aug) = cli.augment {
            println!("  Augmented GIF: {}", aug.display());
        }
    }

    let obs_config = ObsCameraConfig {
        width_fraction: 0.3,
        aspect: 4.0 / 3.0,
        corner: ViewportCorner::TopRight,
        fov: 70_f32.to_radians(),
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers -- Arm Pick Replay".to_string(),
            resolution: window_res.into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(EguiPlugin::default())
    .add_plugins(PanOrbitCameraPlugin)
    .insert_resource(playback)
    .insert_resource(obs_config.clone())
    .add_systems(Startup, (spawn_camera_and_scene, spawn_arm_link_meshes, spawn_obs_camera))
    .add_systems(
        Update,
        (
            playback_advance_system,
            apply_body_poses_system.after(playback_advance_system),
            sync_ee_camera_system.after(apply_body_poses_system),
            sync_obs_camera_viewport,
            keyboard_system,
            egui_camera_gate,
        ),
    )
    .add_systems(EguiPrimaryContextPass, replay_panel_system);

    // Add recording systems if --record was specified
    if let Some(rec) = recording {
        let gif_path = rec.gif_path.clone();
        let frames_dir = rec.output_dir.clone();
        let dt = 0.02_f32;
        app.insert_resource(rec);
        app.add_systems(
            Update,
            (
                record_frame_system.after(apply_body_poses_system),
                record_autoplay_system.after(record_frame_system),
            ),
        );

        let augment_path = cli.augment;
        let obs_cfg = obs_config.clone();
        let win_size = window_res;
        app.run();

        // After app exits, assemble GIF if requested
        if let Some(gif) = gif_path {
            assemble_gif(&frames_dir, &gif, dt);

            // Robot cam GIF: crop the obs camera viewport from each frame
            let robot_gif = robot_gif_path(&gif);
            let crop = obs_cam_crop(win_size.0, win_size.1, &obs_cfg);
            assemble_gif_with_crop(&frames_dir, &robot_gif, dt, Some(crop));
        }

        // Run augmentation pipeline if requested
        if let Some(aug_gif) = augment_path {
            augment_frames(&frames_dir, &aug_gif, dt);
        }
    } else if cli.gif.is_some() {
        // --gif without --record: look for existing frames
        eprintln!("--gif requires --record <dir> to specify frames directory");
        eprintln!("Usage: arm_pick_replay trace.json --record frames/ --gif output.gif");
        std::process::exit(1);
    } else {
        app.run();
    }
}
