//! `clankers-app run` — scenario-driven local execution.
//!
//! Stub in W5 PR1. The legacy `Headless` variant's body is reached via
//! [`execute_default`] (PR1 back-compat shim). The scenario-driven
//! `--scenario` path lands in W5 PR2 (loop 6 of the W3/W4/W5
//! orchestration).

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_env::prelude::*;
use clankers_sim::{ClankersSimPlugin, EpisodeStats};

/// Scenario-driven local execution (`clankers-app run --scenario <name>`).
///
/// Implemented in W5 PR2 (loop 6 of `20260526-013019-w3-w4-w5-impl`).
#[allow(dead_code)]
pub fn execute() -> ! {
    unimplemented!("WS5 PR2 — see .delegate/work/20260526-013019-w3-w4-w5-impl/06");
}

/// Legacy `Headless` mode body — kept here so `main.rs` stays under
/// the 120-line cap. Behaviour preserved verbatim from the pre-PR1
/// `apps/clankers-app/src/main.rs::run_headless`.
pub fn execute_default(episodes: u32, max_steps: u32, seed: Option<u64>) {
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);

    app.world_mut().spawn((
        Actuator::default(),
        JointCommand::default(),
        JointState::default(),
        JointTorque::default(),
    ));

    app.finish();
    app.cleanup();

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    for ep in 0..episodes {
        app.world_mut().resource_mut::<Episode>().reset(seed);

        for _ in 0..max_steps {
            app.update();
            if app.world().resource::<Episode>().is_done() {
                break;
            }
        }

        let episode = app.world().resource::<Episode>();
        println!("episode {}: steps={}", ep + 1, episode.step_count);
    }

    let stats = app.world().resource::<EpisodeStats>();
    println!(
        "\ntotal: episodes={}, steps={}",
        stats.episodes_completed, stats.total_steps
    );
}
