//! `clankers-app viz` — interactive Bevy visualisation.
//!
//! Preserves the pre-PR1 `run_viz` body verbatim. Not exercised by any
//! test in W5 PR1 (GPU is temporarily off-limits in this orchestration).
//! Compile-only.

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_env::prelude::*;
use clankers_sim::ClankersSimPlugin;

/// Interactive 3D visualisation with teleop and policy controls.
pub fn execute(num_joints: usize, max_steps: u32) {
    use clankers_teleop::ClankersTeleopPlugin;
    use clankers_viz::ClankersVizPlugin;

    let mut app = App::new();

    // Windowed Bevy with full rendering.
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Clankers Viz".to_string(),
            resolution: (1280, 720).into(),
            ..default()
        }),
        ..default()
    }));

    // Simulation stack.
    app.add_plugins(ClankersSimPlugin);
    app.add_plugins(ClankersTeleopPlugin);

    // Visualisation layer.
    app.add_plugins(ClankersVizPlugin::default());

    // Spawn demo joints (cylinders as visual stand-ins).
    for i in 0..num_joints {
        app.world_mut().spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
            Name::new(format!("joint_{i}")),
        ));
    }

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    println!("starting viz with {num_joints} joints, max_steps={max_steps}");
    app.run();
}
