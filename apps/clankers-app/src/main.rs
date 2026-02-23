//! Minimal headless Clankers simulation.
//!
//! Runs a short episode with the full Clankers stack (core + actuator + env)
//! and prints episode statistics.

use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_env::prelude::*;
use clankers_sim::{ClankersSimPlugin, EpisodeStats};

fn main() {
    let mut app = bevy::prelude::App::new();
    app.add_plugins(ClankersSimPlugin);

    // Spawn a single joint.
    app.world_mut().spawn((
        Actuator::default(),
        JointCommand::default(),
        JointState::default(),
        JointTorque::default(),
    ));

    app.finish();
    app.cleanup();

    // Configure a short episode.
    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = 10;

    app.world_mut().resource_mut::<Episode>().reset(None);

    // Run the episode.
    for _ in 0..10 {
        app.update();
    }

    let stats = app.world().resource::<EpisodeStats>();
    println!(
        "episodes: {}, steps: {}",
        stats.episodes_completed, stats.total_steps
    );
}
