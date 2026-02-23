//! egui side panel for visualization.
//!
//! Displays mode controls, simulation controls, episode info, joint states
//! with interactive sliders, observation buffer, and action vector.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_env::buffer::ObservationBuffer;
use clankers_env::episode::Episode;
use clankers_policy::runner::PolicyRunner;
use clankers_sim::EpisodeStats;
use clankers_teleop::TeleopCommander;

use crate::config::VizConfig;
use crate::mode::VizMode;

/// System that renders the egui side panel each frame.
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
pub fn side_panel_system(
    mut contexts: EguiContexts,
    mut viz_config: ResMut<VizConfig>,
    mut mode: ResMut<VizMode>,
    mut episode: ResMut<Episode>,
    mut commander: ResMut<TeleopCommander>,
    joints: Query<(Entity, &JointCommand, &JointState, &JointTorque)>,
    obs_buffer: Res<ObservationBuffer>,
    policy_runner: Option<Res<PolicyRunner>>,
    stats: Option<Res<EpisodeStats>>,
) {
    if !viz_config.show_panel {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::SidePanel::left("viz_panel")
        .default_width(300.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Clankers Viz");
            ui.separator();

            mode_section(ui, &mut mode, policy_runner.is_some());
            ui.separator();

            controls_section(ui, &mut viz_config, &mut mode, &mut episode, &mut commander);
            ui.separator();

            episode_section(ui, &episode, stats.as_deref());
            ui.separator();

            joints_section(ui, &joints, &mut commander, *mode);
            ui.separator();

            observation_section(ui, &obs_buffer);

            action_section(ui, policy_runner.as_deref(), &commander, *mode);
        });
}

fn mode_section(ui: &mut egui::Ui, mode: &mut ResMut<VizMode>, has_policy: bool) {
    ui.label("Mode");
    ui.horizontal(|ui| {
        for candidate in [VizMode::Paused, VizMode::Teleop, VizMode::Policy] {
            let enabled = candidate != VizMode::Policy || has_policy;
            let button = egui::Button::new(candidate.label()).selected(**mode == candidate);
            let response = ui.add_enabled(enabled, button);
            if !enabled {
                response.on_disabled_hover_text("No policy loaded");
            } else if response.clicked() {
                **mode = candidate;
            }
        }
    });
}

fn controls_section(
    ui: &mut egui::Ui,
    viz_config: &mut ResMut<VizConfig>,
    mode: &mut ResMut<VizMode>,
    episode: &mut ResMut<Episode>,
    commander: &mut ResMut<TeleopCommander>,
) {
    ui.label("Controls");

    ui.horizontal(|ui| {
        if ui.button("Reset").clicked() {
            episode.reset(None);
            commander.clear();
        }

        if mode.is_simulating() {
            if ui.button("Pause").clicked() {
                **mode = VizMode::Paused;
            }
        } else {
            if ui.button("Resume").clicked() {
                **mode = VizMode::Teleop;
            }
            if ui.button("Step").clicked() {
                viz_config.step_once = true;
            }
        }
    });

    ui.add(
        egui::Slider::new(&mut viz_config.sim_speed, 0.1..=10.0)
            .text("Speed")
            .logarithmic(true),
    );
}

fn episode_section(ui: &mut egui::Ui, episode: &Episode, stats: Option<&EpisodeStats>) {
    ui.label("Episode");

    egui::Grid::new("episode_grid")
        .num_columns(2)
        .spacing([20.0, 4.0])
        .show(ui, |ui| {
            ui.label("State:");
            ui.label(format!("{:?}", episode.state));
            ui.end_row();

            ui.label("Step:");
            ui.label(format!("{}", episode.step_count));
            ui.end_row();

            ui.label("Episode #:");
            ui.label(format!("{}", episode.episode_number));
            ui.end_row();

            if let Some(seed) = episode.seed {
                ui.label("Seed:");
                ui.label(format!("{seed}"));
                ui.end_row();
            }

            if let Some(stats) = stats {
                ui.label("Total episodes:");
                ui.label(format!("{}", stats.episodes_completed));
                ui.end_row();

                ui.label("Total steps:");
                ui.label(format!("{}", stats.total_steps));
                ui.end_row();

                if let Some(mean) = stats.mean_episode_length() {
                    ui.label("Mean length:");
                    ui.label(format!("{mean:.1}"));
                    ui.end_row();
                }
            }
        });
}

fn joints_section(
    ui: &mut egui::Ui,
    joints: &Query<(Entity, &JointCommand, &JointState, &JointTorque)>,
    commander: &mut ResMut<TeleopCommander>,
    mode: VizMode,
) {
    ui.label("Joints");

    if joints.is_empty() {
        ui.label("No joints spawned.");
        return;
    }

    let is_teleop = mode == VizMode::Teleop;

    egui::Grid::new("joints_grid")
        .num_columns(5)
        .spacing([8.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Joint");
            ui.strong("Command");
            ui.strong("Pos (rad)");
            ui.strong("Vel (rad/s)");
            ui.strong("Torque (Nm)");
            ui.end_row();

            for (i, (_entity, cmd, state, torque)) in joints.iter().enumerate() {
                ui.label(format!("J{i}"));

                if is_teleop {
                    let channel = format!("joint_{i}");
                    let mut value = commander.get(&channel);
                    if ui
                        .add(egui::Slider::new(&mut value, -1.0..=1.0).show_value(true))
                        .changed()
                    {
                        commander.set(channel, value);
                    }
                } else {
                    ui.label(format!("{:.3}", cmd.value));
                }

                ui.label(format!("{:.3}", state.position));
                ui.label(format!("{:.3}", state.velocity));
                ui.label(format!("{:.3}", torque.value));
                ui.end_row();
            }
        });
}

fn observation_section(ui: &mut egui::Ui, buffer: &ObservationBuffer) {
    if buffer.dim() == 0 {
        return;
    }

    ui.separator();
    egui::CollapsingHeader::new("Observation")
        .default_open(false)
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    let slots = buffer.slots();
                    if slots.is_empty() {
                        for (i, &v) in buffer.as_slice().iter().enumerate() {
                            ui.label(format!("[{i}] {v:.4}"));
                        }
                    } else {
                        for (si, slot) in slots.iter().enumerate() {
                            let data = buffer.read(si);
                            ui.label(format!("{} [{}..{}]:", slot.name, slot.offset, slot.offset + slot.dim));
                            ui.indent(slot.name.as_str(), |ui| {
                                for (j, &v) in data.iter().enumerate() {
                                    ui.label(format!("[{}] {v:.4}", slot.offset + j));
                                }
                            });
                        }
                    }
                });
        });
}

fn action_section(
    ui: &mut egui::Ui,
    policy_runner: Option<&PolicyRunner>,
    commander: &TeleopCommander,
    mode: VizMode,
) {
    egui::CollapsingHeader::new("Action")
        .default_open(false)
        .show(ui, |ui| {
            match mode {
                VizMode::Policy => {
                    if let Some(runner) = policy_runner {
                        ui.label(format!("Policy: {}", runner.policy_name()));
                        for (i, &v) in runner.action().as_slice().iter().enumerate() {
                            ui.label(format!("[{i}] {v:.4}"));
                        }
                    } else {
                        ui.label("No policy loaded.");
                    }
                }
                VizMode::Teleop => {
                    ui.label("Source: Teleop");
                    for (ch, v) in commander.iter() {
                        ui.label(format!("{ch}: {v:.4}"));
                    }
                }
                VizMode::Paused => {
                    ui.label("Simulation paused.");
                }
            }
        });
}
