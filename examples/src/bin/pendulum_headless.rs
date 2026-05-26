//! Headless pendulum simulation.
//!
//! Thin wrapper around [`clankers_sim::scenarios::pendulum::PendulumScenario`].
//! Drives sinusoidal torque on the pivot joint and reports episode
//! statistics.
//!
//! Run: `cargo run -p clankers-examples --bin pendulum_headless`

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::traits::Sensor;
use clankers_env::prelude::{Episode, JointStateSensor, ObservationBuffer};
use clankers_sim::scenarios::pendulum::{PendulumConfig, PendulumScenario};
use clankers_sim::{EpisodeStats, ScenarioConfig};

fn main() {
    println!("=== Pendulum Headless Example ===\n");
    let cfg = ScenarioConfig::default();
    let pend_cfg = PendulumConfig::default();
    let mut artefacts = PendulumScenario::build_with(&cfg, &pend_cfg);
    let pivot = artefacts.pivot;
    let obs_dim = artefacts
        .scene
        .app
        .world()
        .resource::<ObservationBuffer>()
        .dim();
    println!("Pendulum loaded; obs dim = {obs_dim}");
    for ep in 0..3 {
        println!("\n--- Episode {} ---", ep + 1);
        artefacts
            .scene
            .app
            .world_mut()
            .resource_mut::<Episode>()
            .reset(Some(ep));
        for step in 0..50 {
            let t = step as f32 * 0.02;
            let command = 5.0 * (t * 3.0).sin();
            artefacts
                .scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(pivot)
                .unwrap()
                .value = command;
            artefacts.scene.app.update();
            if step % 10 == 0 {
                let state = artefacts
                    .scene
                    .app
                    .world()
                    .get::<JointState>(pivot)
                    .unwrap();
                let torque = artefacts
                    .scene
                    .app
                    .world()
                    .get::<JointTorque>(pivot)
                    .unwrap();
                println!(
                    "  step {step:3}: cmd={command:+6.2} pos={:+6.3} rad  vel={:+7.3} rad/s  torque={:+6.2} Nm",
                    state.position, state.velocity, torque.value
                );
            }
            if artefacts.scene.app.world().resource::<Episode>().is_done() {
                println!("  -> episode terminated at step {step}");
                break;
            }
        }
        let mut sensor = JointStateSensor::new(artefacts.layout.clone());
        let obs = sensor.read(artefacts.scene.app.world_mut());
        println!("  Sensor read: [pos={:.3}, vel={:.3}]", obs[0], obs[1]);
    }
    let stats = artefacts.scene.app.world().resource::<EpisodeStats>();
    println!(
        "\n=== Summary ===\nEpisodes completed: {}\nTotal steps:        {}",
        stats.episodes_completed, stats.total_steps
    );
    println!("\nPendulum headless example PASSED");
}
