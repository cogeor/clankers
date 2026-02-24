//! Headless multi-environment cart-pole benchmark.
//!
//! Creates N independent cart-pole environments (each with its own Bevy App +
//! Rapier physics) and measures throughput + memory.
//!
//! Run: `cargo run -p clankers-examples --bin cartpole_vec_benchmark --release`

use std::collections::HashMap;
use std::time::Instant;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::prelude::*;
use clankers_core::types::{Action, ActionSpace, ObservationSpace};
use clankers_env::prelude::*;
use clankers_env::vec_env::VecEnvConfig;
use clankers_env::vec_runner::VecEnvInstance;
use clankers_examples::CARTPOLE_URDF;
use clankers_gym::prelude::*;
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;

/// Writes action values to joint commands in spawn order.
struct CartPoleApplicator;

impl ActionApplicator for CartPoleApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action.as_slice();
        let mut query = world.query::<&mut JointCommand>();
        for (i, mut cmd) in query.iter_mut(world).enumerate() {
            if i < values.len() {
                cmd.value = values[i];
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "CartPoleApplicator"
    }
}

/// Create one headless cart-pole GymEnv with Rapier physics.
fn make_cartpole_env() -> GymEnv {
    let max_steps: u32 = 500;
    let num_joints: usize = 2;

    let model = clankers_urdf::parse_string(CARTPOLE_URDF).expect("parse URDF");

    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(max_steps)
        .with_robot(model.clone(), HashMap::new())
        .build();

    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    {
        let spawned = &scene.robots["cartpole"];
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        world.insert_resource(ctx);
    }

    // Register sensors
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(
            Box::new(JointStateSensor::new(num_joints)),
            &mut buffer,
        );
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    let obs_dim = num_joints * 2;
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    GymEnv::new(scene.app, obs_space, act_space, Box::new(CartPoleApplicator))
        .with_reset_fn(|world: &mut World| {
            if let Some(mut ctx) = world.remove_resource::<RapierContext>() {
                ctx.reset_to_initial();
                world.insert_resource(ctx);
            }
            let mut query = world.query::<(&mut JointState, &mut JointCommand)>();
            for (mut state, mut cmd) in query.iter_mut(world) {
                state.position = 0.0;
                state.velocity = 0.0;
                cmd.value = 0.0;
            }
        })
}

fn main() {
    let env_counts = [1, 2, 4, 8, 16, 32];
    let steps_per_env = 500;

    println!("=== Cart-Pole Multi-Environment Benchmark ===\n");
    println!("Each environment: Bevy App + Rapier3D physics, 4-dim obs, 2-dim action");
    println!("Steps per environment: {steps_per_env}\n");

    for &num_envs in &env_counts {
        // Measure creation time
        let create_start = Instant::now();
        let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
            .map(|_| Box::new(make_cartpole_env()) as Box<dyn VecEnvInstance>)
            .collect();
        let create_elapsed = create_start.elapsed();

        let obs_dim = 4;
        let config = VecEnvConfig::new(num_envs as u16);
        let obs_space = ObservationSpace::Box {
            low: vec![-10.0; obs_dim],
            high: vec![10.0; obs_dim],
        };
        let act_space = ActionSpace::Box {
            low: vec![-1.0; 2],
            high: vec![1.0; 2],
        };

        let mut vec_env = GymVecEnv::new(envs, config, obs_space, act_space);

        // Reset all
        let reset_start = Instant::now();
        vec_env.reset_all(Some(42));
        let reset_elapsed = reset_start.elapsed();

        // Step all with random actions
        let step_start = Instant::now();
        let mut total_steps = 0u64;
        for _ in 0..steps_per_env {
            let actions: Vec<Action> = (0..num_envs)
                .map(|_| Action::Continuous(vec![0.5, 0.0])) // constant push right
                .collect();
            vec_env.step_all(&actions);
            total_steps += num_envs as u64;
        }
        let step_elapsed = step_start.elapsed();

        let steps_per_sec = total_steps as f64 / step_elapsed.as_secs_f64();
        let ms_per_step = step_elapsed.as_secs_f64() * 1000.0 / steps_per_env as f64;

        // Read final observations to verify physics worked
        let final_obs = vec_env.runner().get_obs(0);
        let cart_pos = final_obs.as_slice()[0];

        println!(
            "envs={num_envs:3} | create={:.1}s | reset={:.0}ms | \
             step={:.2}s ({:.0} steps/s, {:.2}ms/batch) | \
             final_cart_pos={cart_pos:+.2}",
            create_elapsed.as_secs_f64(),
            reset_elapsed.as_secs_f64() * 1000.0,
            step_elapsed.as_secs_f64(),
            steps_per_sec,
            ms_per_step,
        );
    }

    println!("\nDone.");
}
