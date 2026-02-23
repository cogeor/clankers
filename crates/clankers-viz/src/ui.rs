//! egui side panel for visualization.
//!
//! Displays joint states, episode info, simulation controls, and mode switching.

use bevy::prelude::*;
use bevy_egui::{EguiContexts, egui};

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_env::episode::Episode;

use crate::config::VizConfig;
use crate::mode::VizMode;

/// System that renders the egui side panel each frame.
#[allow(clippy::too_many_arguments)]
pub fn side_panel_system(
    mut contexts: EguiContexts,
    mut viz_config: ResMut<VizConfig>,
    mut mode: ResMut<VizMode>,
    mut episode: ResMut<Episode>,
    joints: Query<(Entity, &JointCommand, &JointState, &JointTorque)>,
) {
    if !viz_config.show_panel {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    egui::SidePanel::left("viz_panel")
        .default_width(280.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.heading("Clankers Viz");
            ui.separator();

            // -- Mode selector --
            mode_section(ui, &mut mode);
            ui.separator();

            // -- Simulation controls --
            controls_section(ui, &mut viz_config, &mut episode);
            ui.separator();

            // -- Episode info --
            episode_section(ui, &episode);
            ui.separator();

            // -- Joint states --
            joints_section(ui, &joints);
        });
}

fn mode_section(ui: &mut egui::Ui, mode: &mut ResMut<VizMode>) {
    ui.label("Mode");
    ui.horizontal(|ui| {
        for candidate in [VizMode::Paused, VizMode::Teleop, VizMode::Policy] {
            if ui
                .selectable_label(**mode == candidate, candidate.label())
                .clicked()
            {
                **mode = candidate;
            }
        }
    });
}

fn controls_section(
    ui: &mut egui::Ui,
    viz_config: &mut ResMut<VizConfig>,
    episode: &mut ResMut<Episode>,
) {
    ui.label("Controls");

    ui.horizontal(|ui| {
        if ui.button("Reset").clicked() {
            episode.reset(None);
        }
        if ui
            .button(if viz_config.start_paused {
                "Resume"
            } else {
                "Pause"
            })
            .clicked()
        {
            viz_config.start_paused = !viz_config.start_paused;
        }
    });

    ui.add(
        egui::Slider::new(&mut viz_config.sim_speed, 0.1..=10.0)
            .text("Speed")
            .logarithmic(true),
    );
}

fn episode_section(ui: &mut egui::Ui, episode: &Episode) {
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
        });
}

fn joints_section(
    ui: &mut egui::Ui,
    joints: &Query<(Entity, &JointCommand, &JointState, &JointTorque)>,
) {
    ui.label("Joints");

    if joints.is_empty() {
        ui.label("No joints spawned.");
        return;
    }

    egui::Grid::new("joints_grid")
        .num_columns(4)
        .spacing([8.0, 4.0])
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Joint");
            ui.strong("Pos (rad)");
            ui.strong("Vel (rad/s)");
            ui.strong("Torque (Nm)");
            ui.end_row();

            for (i, (_entity, cmd, state, torque)) in joints.iter().enumerate() {
                ui.label(format!("J{i}"));
                ui.label(format!("{:.3}", state.position));
                ui.label(format!("{:.3}", state.velocity));
                ui.label(format!("{:.3} (cmd: {:.2})", torque.value, cmd.value));
                ui.end_row();
            }
        });
}
