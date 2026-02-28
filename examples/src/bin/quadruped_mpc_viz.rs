//! Quadruped MPC visualization with windowed Bevy rendering.
//!
//! ## Visual sync strategy
//!
//! Rapier places each link rigid body at the **joint position** (where it
//! connects to its parent), not at the link center. To render correct leg
//! geometry we use "connect the dots": capsules are drawn BETWEEN
//! consecutive rigid body positions (hip-to-knee, knee-to-ankle) so the
//! mesh always spans the full link regardless of joint angle.
//!
//! ## 3-DOF legs
//!
//! Each leg has hip_ab (X-axis) + hip_pitch (Y-axis) + knee_pitch (Y-axis).
//! The hip_link and upper_leg are co-located (hip_pitch origin is 0,0,0
//! relative to hip_link), so visuals use upper_leg position for the hip sphere.
//!
//! Run: `cargo run -p clankers-examples --bin quadruped_mpc_viz`

use bevy::prelude::*;
use bevy::time::Fixed;
use bevy_egui::{EguiContexts, egui};
use clap::Parser;
use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::ClankersSet;
use clankers_env::prelude::*;
use clankers_examples::mpc_control::{MpcLoopState, body_state_from_rapier, compute_mpc_step, detect_foot_contacts};
use clankers_examples::quadruped_setup::{QuadrupedSetupConfig, setup_quadruped};
use clankers_mpc::{
    BodyState, GaitScheduler, GaitType, MpcConfig, MpcSolver, SwingConfig,
};
use clankers_physics::rapier::{MotorOverrideParams, MotorOverrides, RapierContext};
use clankers_teleop::ClankersTeleopPlugin;
use clankers_teleop::TeleopConfig;
use clankers_viz::{ClankersVizPlugin, VizMode};
use clankers_viz::systems::VizSimGate;
use nalgebra::Vector3;

#[derive(Parser)]
#[command(about = "Quadruped MPC visualization with windowed Bevy rendering")]
struct Args {
    /// Override MPC timestep in seconds (default 0.02 = 50Hz, use 0.01 for 100Hz)
    #[arg(long, default_value_t = 0.02)]
    mpc_dt: f64,
}

// ---------------------------------------------------------------------------
// Visual markers
// ---------------------------------------------------------------------------

#[derive(Component)]
struct BodyVisual;

#[derive(Component)]
struct SegmentVisual {
    start_link: &'static str,
    end_link: &'static str,
}

#[derive(Component)]
struct PointVisual(&'static str);

// ---------------------------------------------------------------------------
// MPC runtime state
// ---------------------------------------------------------------------------

#[derive(Resource)]
struct MpcUiState {
    mpc_enabled: bool,
    desired_velocity_x: f32,
    desired_velocity_y: f32,
    desired_height: f32,
    desired_yaw: f32,
    gait: GaitType,
    last_converged: bool,
    last_solve_time_us: u64,
    n_stance: usize,
    body_pos: [f32; 3],
    step: usize,
    diag_joint_cmds: Vec<f32>,
    diag_joint_torques: Vec<f32>,
    diag_joint_positions: Vec<f32>,
    diag_joint_velocities: Vec<f32>,
    diag_teleop_mappings: usize,
    diag_teleop_enabled: bool,
    reinit_requested: bool,
    diag_should_step: bool,
    diag_rapier_joint_count: usize,
    diag_frame: u64,
}

impl Default for MpcUiState {
    fn default() -> Self {
        Self {
            mpc_enabled: true,
            desired_velocity_x: 0.0,
            desired_velocity_y: 0.0,
            desired_height: 0.20,
            desired_yaw: 0.0,
            gait: GaitType::Stand,
            last_converged: false,
            last_solve_time_us: 0,
            n_stance: 0,
            body_pos: [0.0; 3],
            step: 0,
            diag_joint_cmds: Vec::new(),
            diag_joint_torques: Vec::new(),
            diag_joint_positions: Vec::new(),
            diag_joint_velocities: Vec::new(),
            diag_teleop_mappings: 0,
            diag_teleop_enabled: false,
            reinit_requested: false,
            diag_should_step: false,
            diag_rapier_joint_count: 0,
            diag_frame: 0,
        }
    }
}

#[derive(Resource)]
struct QuadMpcState {
    inner: MpcLoopState,
    step: usize,
    stabilize_steps: usize,
    current_gait_type: GaitType,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn body_state_from_rapier_viz(ctx: &RapierContext, link_name: &str) -> Option<(BodyState, Quat, nalgebra::UnitQuaternion<f64>)> {
    let handle = ctx.body_handles.get(link_name)?;
    let body = ctx.rigid_body_set.get(*handle)?;
    let bevy_quat = *body.rotation();
    let (bs, na_quat) = body_state_from_rapier(ctx, link_name)?;
    Some((bs, bevy_quat, na_quat))
}

fn phys_to_vis(pos: Vec3) -> Vec3 {
    Vec3::new(pos.x, pos.z, -pos.y)
}

fn link_vis_pos(rapier: &RapierContext, link_name: &str) -> Option<Vec3> {
    let handle = rapier.body_handles.get(link_name)?;
    let body = rapier.rigid_body_set.get(*handle)?;
    Some(phys_to_vis(body.translation()))
}

fn rotation_align_y(dir: Vec3) -> Quat {
    let d = dir.normalize_or_zero();
    let dot = Vec3::Y.dot(d);
    if dot > 0.9999 {
        Quat::IDENTITY
    } else if dot < -0.9999 {
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

    // Body cuboid
    commands.spawn((
        BodyVisual,
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.1, 0.2))),
        MeshMaterial3d(body_mat),
        Transform::from_xyz(0.0, 0.35, 0.0),
    ));

    let capsule_mesh = meshes.add(Capsule3d::new(0.018, 0.12));
    let joint_sphere = meshes.add(Sphere::new(0.022));
    let foot_sphere = meshes.add(Sphere::new(0.025));

    let leg_segments: &[(&str, &str, &str, &str)] = &[
        ("fl", "fl_upper_leg", "fl_lower_leg", "fl_foot"),
        ("fr", "fr_upper_leg", "fr_lower_leg", "fr_foot"),
        ("rl", "rl_upper_leg", "rl_lower_leg", "rl_foot"),
        ("rr", "rr_upper_leg", "rr_lower_leg", "rr_foot"),
    ];

    for &(_prefix, upper, lower, foot) in leg_segments {
        commands.spawn((
            PointVisual(upper),
            Mesh3d(joint_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));
        commands.spawn((
            SegmentVisual {
                start_link: leak_str(upper),
                end_link: leak_str(lower),
            },
            Mesh3d(capsule_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::default(),
        ));
        commands.spawn((
            PointVisual(lower),
            Mesh3d(joint_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));
        commands.spawn((
            SegmentVisual {
                start_link: leak_str(lower),
                end_link: leak_str(foot),
            },
            Mesh3d(capsule_mesh.clone()),
            MeshMaterial3d(leg_mat.clone()),
            Transform::default(),
        ));
        commands.spawn((
            PointVisual(foot),
            Mesh3d(foot_sphere.clone()),
            MeshMaterial3d(joint_mat.clone()),
            Transform::default(),
        ));
    }
}

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
    mut rapier: ResMut<RapierContext>,
    mut mpc: ResMut<QuadMpcState>,
    mut mpc_ui: ResMut<MpcUiState>,
    mut motor_overrides: ResMut<MotorOverrides>,
    mut states: Query<&mut JointState>,
    mut commands: Query<&mut JointCommand>,
) {
    // --- Reinitialize: reset physics and MPC state to post-warmup config ---
    if mpc_ui.reinit_requested {
        mpc_ui.reinit_requested = false;
        // SAFETY: we need mutable access to rapier for reset
        let rapier = rapier.into_inner();
        rapier.reset_to_initial();

        for leg in &mpc.inner.legs {
            for &entity in &leg.joint_entities {
                if let Some(info) = rapier.joint_info.get(&entity)
                    && let Some(pb) = rapier.rigid_body_set.get(info.parent_body)
                    && let Some(cb) = rapier.rigid_body_set.get(info.child_body)
                {
                    let rel_rot = pb.position().rotation.inverse() * cb.position().rotation;
                    let sin_half = Vec3::new(rel_rot.x, rel_rot.y, rel_rot.z);
                    let sin_proj = sin_half.dot(info.axis);
                    let angle = 2.0 * f32::atan2(sin_proj, rel_rot.w);

                    if let Ok(mut js) = states.get_mut(entity) {
                        js.position = angle;
                        js.velocity = 0.0;
                    }
                }
            }
        }

        let mpc = &mut *mpc;
        mpc.step = 0;
        mpc.inner.gait = GaitScheduler::quadruped(GaitType::Stand);
        mpc.current_gait_type = GaitType::Stand;
        for v in &mut mpc.inner.swing_starts {
            *v = Vector3::zeros();
        }
        for v in &mut mpc.inner.swing_targets {
            *v = Vector3::zeros();
        }

        mpc_ui.gait = GaitType::Stand;
        mpc_ui.desired_velocity_x = 0.0;
        mpc_ui.desired_velocity_y = 0.0;
        mpc_ui.desired_yaw = 0.0;
        mpc_ui.step = 0;

        motor_overrides.joints.clear();

        println!("  >>> Reinitialized to post-warmup state");
        return;
    }

    if !mpc_ui.mpc_enabled {
        return;
    }

    let mpc = &mut *mpc;

    let desired_velocity = Vector3::new(
        f64::from(mpc_ui.desired_velocity_x).clamp(-1.0, 1.0),
        f64::from(mpc_ui.desired_velocity_y).clamp(-0.5, 0.5),
        0.0,
    );
    let desired_height = f64::from(mpc_ui.desired_height);
    let desired_yaw = f64::from(mpc_ui.desired_yaw);
    let ground_height = 0.0;

    // Auto-switch to Trot at stabilize_steps (matching headless)
    if mpc.step == mpc.stabilize_steps && mpc.current_gait_type == GaitType::Stand {
        mpc_ui.gait = GaitType::Trot;
        mpc_ui.desired_velocity_x = 0.3;
    }

    if mpc_ui.gait != mpc.current_gait_type {
        mpc.inner.gait = GaitScheduler::quadruped(mpc_ui.gait);
        mpc.current_gait_type = mpc_ui.gait;
        println!("  >>> Switched to {:?} at step {}", mpc_ui.gait, mpc.step);
    }

    // Floating origin: keep Rapier coords near zero for f32 precision
    rapier.rebase_origin("body", 50.0);

    let Some((body_state, _body_rot_bevy, body_quat_na)) = body_state_from_rapier_viz(&rapier, "body") else {
        return;
    };
    let body_pos = body_state.position;
    let n_feet = mpc.inner.legs.len();

    // --- Read joint states ---
    let mut all_joint_positions: Vec<Vec<f32>> = Vec::with_capacity(n_feet);
    let mut all_joint_velocities: Vec<Vec<f32>> = Vec::with_capacity(n_feet);

    for leg in &mpc.inner.legs {
        let mut q = Vec::with_capacity(leg.joint_entities.len());
        let mut qd = Vec::with_capacity(leg.joint_entities.len());
        for &entity in &leg.joint_entities {
            if let Ok(js) = states.get(entity) {
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

    // Velocity ramp
    let ramp_steps = 100;
    let current_vel = if mpc.step < mpc.stabilize_steps {
        Vector3::zeros()
    } else {
        let ramp_frac = ((mpc.step - mpc.stabilize_steps) as f64 / ramp_steps as f64).min(1.0);
        desired_velocity * ramp_frac
    };

    // --- Detect contacts + compute MPC step ---
    let actual_contacts = detect_foot_contacts(&rapier, &mpc.inner);
    let result = compute_mpc_step(
        &mut mpc.inner,
        &body_state,
        &body_quat_na,
        &all_joint_positions,
        &all_joint_velocities,
        &current_vel,
        desired_height,
        desired_yaw,
        ground_height,
        actual_contacts.as_deref(),
    );

    // --- Convert MotorCommands → MotorOverrideParams ---
    for mc in &result.motor_commands {
        motor_overrides.joints.insert(mc.entity, MotorOverrideParams {
            target_pos: mc.target_pos,
            target_vel: mc.target_vel,
            stiffness: mc.stiffness,
            damping: mc.damping,
            max_force: mc.max_force,
        });

        if let Ok(mut cmd) = commands.get_mut(mc.entity) {
            cmd.value = 0.0;
        }
    }

    // --- Update UI display state ---
    let n_stance: usize = result.contacts.iter().filter(|&&c| c).count();
    mpc_ui.last_converged = result.solution.converged;
    mpc_ui.last_solve_time_us = result.solution.solve_time_us;
    mpc_ui.n_stance = n_stance;
    mpc_ui.body_pos = [body_pos.x as f32, body_pos.y as f32, body_pos.z as f32];
    mpc_ui.step = mpc.step;

    if mpc.step.is_multiple_of(50) {
        println!(
            "  step {:4}: pos=[{:+.3}, {:+.3}, {:+.3}]  vel=[{:+.3}, {:+.3}, {:+.3}]  stance={}/{}  mpc={:>4}us  {}  cmd_vel=[{:.2}, {:.2}]",
            mpc.step,
            body_pos.x,
            body_pos.y,
            body_pos.z,
            body_state.linear_velocity.x,
            body_state.linear_velocity.y,
            body_state.linear_velocity.z,
            n_stance,
            n_feet,
            result.solution.solve_time_us,
            if result.solution.converged { "OK" } else { "FAIL" },
            current_vel.x,
            current_vel.y,
        );
        if result.solution.converged {
            for (i, f) in result.solution.forces.iter().enumerate() {
                let contact = result.contacts.get(i).copied().unwrap_or(false);
                println!(
                    "    foot {i}: F=[{:+.2}, {:+.2}, {:+.2}]  {}",
                    f.x, f.y, f.z,
                    if contact { "STANCE" } else { "swing" },
                );
            }
        }
    }

    mpc.step += 1;
}

// ---------------------------------------------------------------------------
// MPC egui panel
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn mpc_panel_system(mut contexts: EguiContexts, mut mpc_ui: ResMut<MpcUiState>, mode: Res<VizMode>) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("MPC Control")
        .default_pos([320.0, 10.0])
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            let toggle_text = if mpc_ui.mpc_enabled {
                "MPC: ON (click to disable)"
            } else {
                "MPC: OFF (click to enable)"
            };
            let toggle_color = if mpc_ui.mpc_enabled {
                egui::Color32::from_rgb(50, 200, 50)
            } else {
                egui::Color32::from_rgb(200, 50, 50)
            };
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new(toggle_text).color(toggle_color).strong())
                            .min_size(egui::vec2(190.0, 28.0)),
                    )
                    .clicked()
                {
                    mpc_ui.mpc_enabled = !mpc_ui.mpc_enabled;
                }
                if ui
                    .add(
                        egui::Button::new(egui::RichText::new("Reset").strong())
                            .min_size(egui::vec2(60.0, 28.0)),
                    )
                    .clicked()
                {
                    mpc_ui.reinit_requested = true;
                }
            });
            ui.separator();

            ui.label("Gait");
            ui.horizontal(|ui| {
                for gait in [GaitType::Stand, GaitType::Trot, GaitType::Walk, GaitType::Bound] {
                    let label = match gait {
                        GaitType::Stand => "Stand",
                        GaitType::Trot => "Trot",
                        GaitType::Walk => "Walk",
                        GaitType::Bound => "Bound",
                    };
                    if ui
                        .add(egui::Button::new(label).selected(mpc_ui.gait == gait))
                        .clicked()
                    {
                        mpc_ui.gait = gait;
                    }
                }
            });
            ui.separator();

            ui.label("Body Velocity");
            ui.add(
                egui::Slider::new(&mut mpc_ui.desired_velocity_x, -1.0..=1.0)
                    .text("Vx (m/s)")
                    .step_by(0.05),
            );
            ui.add(
                egui::Slider::new(&mut mpc_ui.desired_velocity_y, -0.5..=0.5)
                    .text("Vy (m/s)")
                    .step_by(0.05),
            );
            ui.separator();

            ui.add(
                egui::Slider::new(&mut mpc_ui.desired_height, 0.15..=0.35)
                    .text("Height (m)")
                    .step_by(0.01),
            );

            ui.add(
                egui::Slider::new(&mut mpc_ui.desired_yaw, -1.0..=1.0)
                    .text("Yaw (rad)")
                    .step_by(0.05),
            );
            ui.separator();

            ui.label("Status");
            egui::Grid::new("mpc_status")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Mode:");
                    ui.label(format!("{:?}", *mode));
                    ui.end_row();

                    ui.label("Step:");
                    ui.label(format!("{}", mpc_ui.step));
                    ui.end_row();

                    ui.label("MPC:");
                    ui.label(if mpc_ui.last_converged { "OK" } else { "FAIL" });
                    ui.end_row();

                    ui.label("Solve:");
                    ui.label(format!("{} us", mpc_ui.last_solve_time_us));
                    ui.end_row();

                    ui.label("Stance:");
                    ui.label(format!("{}/4", mpc_ui.n_stance));
                    ui.end_row();

                    ui.label("Body pos:");
                    ui.label(format!(
                        "[{:+.3}, {:+.3}, {:+.3}]",
                        mpc_ui.body_pos[0], mpc_ui.body_pos[1], mpc_ui.body_pos[2],
                    ));
                    ui.end_row();
                });

            if !mpc_ui.mpc_enabled {
                ui.separator();
                ui.colored_label(egui::Color32::YELLOW, "MPC disabled — sliders control joints");
            }

            ui.separator();
            egui::CollapsingHeader::new("Pipeline Diagnostics")
                .default_open(true)
                .show(ui, |ui| {
                    egui::Grid::new("diag_grid")
                        .num_columns(2)
                        .spacing([12.0, 2.0])
                        .show(ui, |ui| {
                            ui.label("SimGate:");
                            ui.label(if mpc_ui.diag_should_step { "STEPPING" } else { "BLOCKED" });
                            ui.end_row();

                            ui.label("Teleop:");
                            ui.label(format!(
                                "{} ({} mappings)",
                                if mpc_ui.diag_teleop_enabled { "ON" } else { "OFF" },
                                mpc_ui.diag_teleop_mappings,
                            ));
                            ui.end_row();

                            ui.label("Rapier joints:");
                            ui.label(format!("{}", mpc_ui.diag_rapier_joint_count));
                            ui.end_row();

                            ui.label("Frame:");
                            ui.label(format!("{}", mpc_ui.diag_frame));
                            ui.end_row();
                        });

                    if !mpc_ui.diag_joint_cmds.is_empty() {
                        ui.separator();
                        ui.label("Per-joint pipeline:");
                        egui::Grid::new("joint_diag_grid")
                            .num_columns(5)
                            .spacing([6.0, 1.0])
                            .show(ui, |ui| {
                                ui.strong("J#");
                                ui.strong("Cmd");
                                ui.strong("Trq");
                                ui.strong("Pos");
                                ui.strong("Vel");
                                ui.end_row();

                                let n = mpc_ui.diag_joint_cmds.len();
                                for i in 0..n {
                                    ui.label(format!("{i}"));
                                    ui.label(format!("{:+.3}", mpc_ui.diag_joint_cmds[i]));
                                    ui.label(format!(
                                        "{:+.3}",
                                        mpc_ui.diag_joint_torques.get(i).unwrap_or(&0.0)
                                    ));
                                    ui.label(format!(
                                        "{:+.3}",
                                        mpc_ui.diag_joint_positions.get(i).unwrap_or(&0.0)
                                    ));
                                    ui.label(format!(
                                        "{:+.3}",
                                        mpc_ui.diag_joint_velocities.get(i).unwrap_or(&0.0)
                                    ));
                                    ui.end_row();
                                }
                            });
                    }
                });
        });
}

// ---------------------------------------------------------------------------
// Visual sync systems
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn sync_body_visual(rapier: Res<RapierContext>, mut query: Query<&mut Transform, With<BodyVisual>>) {
    if let Some(&handle) = rapier.body_handles.get("body")
        && let Some(body) = rapier.rigid_body_set.get(handle)
    {
        let t = body.translation();
        let r = body.rotation();
        for mut transform in &mut query {
            transform.translation = phys_to_vis(t);
            transform.rotation = Quat::from_xyzw(r.x, r.z, -r.y, r.w);
        }
    }
}

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
// Pipeline diagnostic system
// ---------------------------------------------------------------------------

#[allow(clippy::needless_pass_by_value)]
fn diagnostic_readback_system(
    mut mpc_ui: ResMut<MpcUiState>,
    gate: Res<VizSimGate>,
    teleop_config: Res<TeleopConfig>,
    rapier: Res<RapierContext>,
    joints: Query<(&JointCommand, &JointState, &JointTorque)>,
    mpc_state: Option<Res<QuadMpcState>>,
) {
    mpc_ui.diag_should_step = gate.should_step;
    mpc_ui.diag_teleop_enabled = teleop_config.enabled;
    mpc_ui.diag_teleop_mappings = teleop_config.mappings.len();
    mpc_ui.diag_rapier_joint_count = rapier.joint_handles.len();
    mpc_ui.diag_frame += 1;

    let mut cmds = Vec::new();
    let mut trqs = Vec::new();
    let mut poss = Vec::new();
    let mut vels = Vec::new();

    if let Some(ref mpc) = mpc_state {
        for leg in &mpc.inner.legs {
            for &entity in &leg.joint_entities {
                if let Ok((cmd, state, torque)) = joints.get(entity) {
                    cmds.push(cmd.value);
                    trqs.push(torque.value);
                    poss.push(state.position);
                    vels.push(state.velocity);
                } else {
                    cmds.push(f32::NAN);
                    trqs.push(f32::NAN);
                    poss.push(f32::NAN);
                    vels.push(f32::NAN);
                }
            }
        }
    } else {
        for (cmd, state, torque) in &joints {
            cmds.push(cmd.value);
            trqs.push(torque.value);
            poss.push(state.position);
            vels.push(state.velocity);
        }
    }

    let frame = mpc_ui.diag_frame;
    if frame <= 5 || frame.is_multiple_of(120) {
        println!("=== DIAG frame {frame} ===");
        println!(
            "  gate.should_step={} teleop.enabled={} teleop.mappings={} rapier.joints={}",
            gate.should_step,
            teleop_config.enabled,
            teleop_config.mappings.len(),
            rapier.joint_handles.len(),
        );
        println!("  mpc_enabled={}", mpc_ui.mpc_enabled);
        for i in 0..cmds.len() {
            println!(
                "  J{i}: cmd={:+.4} trq={:+.4} pos={:+.4} vel={:+.4}",
                cmds[i], trqs[i], poss[i], vels[i],
            );
        }

        for (ch, mapping) in &teleop_config.mappings {
            let in_rapier = rapier.joint_handles.contains_key(&mapping.entity);
            let has_components = joints.get(mapping.entity).is_ok();
            if !in_rapier || !has_components {
                println!(
                    "  WARNING: teleop {ch} entity {:?} rapier={in_rapier} components={has_components}",
                    mapping.entity,
                );
            }
        }
    }

    mpc_ui.diag_joint_cmds = cmds;
    mpc_ui.diag_joint_torques = trqs;
    mpc_ui.diag_joint_positions = poss;
    mpc_ui.diag_joint_velocities = vels;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();
    let mpc_dt = args.mpc_dt;

    // --- Shared setup (FixedUpdate for frame-rate decoupled physics) ---
    let setup = setup_quadruped(QuadrupedSetupConfig {
        mpc_dt: Some(mpc_dt),
        use_fixed_update: true,
        ..QuadrupedSetupConfig::default()
    });
    let mut scene = setup.scene;
    let desired_height = setup.desired_height as f32;
    let n_feet = setup.n_feet;

    let all_joint_entities: Vec<Entity> = setup.legs.iter().flat_map(|leg| leg.joint_entities.clone()).collect();

    // --- MPC config ---
    let mut config = MpcConfig::default();
    config.dt = mpc_dt;
    let swing_config = SwingConfig::default();

    scene.app.insert_resource(QuadMpcState {
        inner: MpcLoopState {
            gait: GaitScheduler::quadruped(GaitType::Stand),
            solver: MpcSolver::new(config.clone(), 4),
            config,
            swing_config,
            adaptive_gait: None,
            legs: setup.legs,
            swing_starts: vec![Vector3::zeros(); n_feet],
            swing_targets: vec![Vector3::zeros(); n_feet],
            prev_contacts: vec![true; n_feet],
            init_joint_angles: setup.init_joint_angles,
            foot_link_names: None,
            disturbance_estimator: None,
        },
        step: 0,
        stabilize_steps: 100,
        current_gait_type: GaitType::Stand,
    });

    scene.app.insert_resource(MotorOverrides::default());

    scene.app.insert_resource(MpcUiState {
        desired_height,
        ..Default::default()
    });

    // RobotGroup + sensors
    {
        let world = scene.app.world_mut();
        world.resource_mut::<clankers_core::types::RobotGroup>().allocate("quadruped".to_string(), all_joint_entities);
    }

    // Windowed rendering
    scene.app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers \u{2014} Quadruped MPC (3-DOF)".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // Teleop + viz plugins (fixed_update decouples sim from render rate)
    scene.app.add_plugins(ClankersTeleopPlugin);
    scene.app.add_plugins(ClankersVizPlugin { fixed_update: true });

    scene.app.world_mut().resource_mut::<clankers_viz::config::VizConfig>().show_panel = false;

    // Set FixedUpdate timestep to match MPC control rate (50Hz default)
    scene.app.insert_resource(Time::<Fixed>::from_seconds(mpc_dt));

    // Visual meshes
    scene.app.add_systems(Startup, spawn_quadruped_meshes);

    // MPC control (Decide phase, on FixedUpdate for frame-rate decoupling)
    scene.app.add_systems(
        FixedUpdate,
        mpc_control_system
            .in_set(ClankersSet::Decide)
            .after(clankers_teleop::systems::apply_teleop_commands),
    );

    // MPC egui panel
    scene
        .app
        .add_systems(bevy_egui::EguiPrimaryContextPass, mpc_panel_system);

    // Visual sync + diagnostics
    scene.app.add_systems(
        Update,
        (
            sync_body_visual,
            sync_segment_visuals,
            sync_point_visuals,
            diagnostic_readback_system,
        )
            .after(ClankersSet::Simulate),
    );

    // Start episode
    scene.app.world_mut().resource_mut::<Episode>().reset(None);

    println!("Quadruped MPC Visualization (3-DOF legs, 12 DOF)");
    println!("  Camera: mouse (orbit/pan/zoom)");
    println!("  MPC toggle: ON by default (MPC controls legs)");
    println!("  MPC OFF + Teleop mode: sliders control joints directly");
    println!("  MPC panel: toggle MPC, adjust velocity/height/yaw/gait");
    scene.app.run();
}
