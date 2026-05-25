//! Headless pendulum simulation.
//!
//! Tests the full pipeline: URDF parsing -> SceneBuilder -> episode lifecycle
//! -> actuator dynamics -> sensor reading -> episode stats.
//!
//! Run: `cargo run -p clankers-examples --bin pendulum_headless`

use std::collections::HashMap;
use std::sync::Arc;

use clankers_actuator::components::{JointCommand, JointState, JointTorque};
use clankers_core::traits::Sensor;
use clankers_env::prelude::*;
use clankers_examples::PENDULUM_URDF;
use clankers_sim::{EpisodeStats, SceneBuilder};

fn main() {
    println!("=== Pendulum Headless Example ===\n");

    // ---------------------------------------------------------------
    // 1. Parse URDF and build scene
    // ---------------------------------------------------------------
    let model = clankers_urdf::parse_string(PENDULUM_URDF).expect("failed to parse pendulum URDF");
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(50)
        .with_robot(model.clone(), HashMap::new())
        .build();

    let bot = &scene.robots["pendulum"];
    let pivot = bot.joint_entity("pivot").expect("missing pivot joint");
    println!(
        "Robot '{}' loaded: {} actuated joint(s)",
        bot.name,
        bot.joint_count()
    );

    // ---------------------------------------------------------------
    // 2. Build layout (bound to the pivot entity) and register sensors
    // ---------------------------------------------------------------
    let layout = {
        let mut layout = model.to_layout();
        let entities: Vec<bevy::prelude::Entity> = layout
            .joints()
            .iter()
            .map(|spec| {
                bot.joint_entity(&spec.name)
                    .unwrap_or_else(|| panic!("joint {} not in spawned robot", spec.name))
            })
            .collect();
        layout.bind_entities(&entities);
        Arc::new(layout)
    };

    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
        registry.register(
            Box::new(JointTorqueSensor::new(layout.clone())),
            &mut buffer,
        );
        println!("Observation dimension: {}", buffer.dim());
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // ---------------------------------------------------------------
    // 3. Run episodes
    // ---------------------------------------------------------------
    let num_episodes = 3;

    for ep in 0..num_episodes {
        println!("\n--- Episode {} ---", ep + 1);

        // Reset with a seed for reproducibility
        scene
            .app
            .world_mut()
            .resource_mut::<Episode>()
            .reset(Some(ep as u64));

        for step in 0..50 {
            // Apply a sinusoidal torque command
            let t = step as f32 * 0.02;
            let command = 5.0 * (t * 3.0).sin();
            scene
                .app
                .world_mut()
                .get_mut::<JointCommand>(pivot)
                .unwrap()
                .value = command;

            scene.app.update();

            // Read joint state
            let state = scene.app.world().get::<JointState>(pivot).unwrap();
            let torque = scene.app.world().get::<JointTorque>(pivot).unwrap();

            if step % 10 == 0 {
                println!(
                    "  step {:3}: cmd={:+6.2} pos={:+6.3} rad  vel={:+7.3} rad/s  torque={:+6.2} Nm",
                    step, command, state.position, state.velocity, torque.value
                );
            }

            if scene.app.world().resource::<Episode>().is_done() {
                println!("  -> episode terminated at step {step}");
                break;
            }
        }

        // Also test the sensor read path
        let mut sensor = JointStateSensor::new(layout.clone());
        let obs = sensor.read(scene.app.world_mut());
        println!("  Sensor read: [pos={:.3}, vel={:.3}]", obs[0], obs[1]);
    }

    // ---------------------------------------------------------------
    // 4. Report stats
    // ---------------------------------------------------------------
    let stats = scene.app.world().resource::<EpisodeStats>();
    println!("\n=== Summary ===");
    println!("Episodes completed: {}", stats.episodes_completed);
    println!("Total steps:        {}", stats.total_steps);
    if let Some(mean) = stats.mean_episode_length() {
        println!("Mean episode len:   {mean:.1}");
    }
    println!("Step history:       {:?}", stats.step_history);
    println!("\nPendulum headless example PASSED");
}
