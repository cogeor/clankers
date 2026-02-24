//! Vectorized cart-pole gym server with Rapier physics.
//!
//! Creates N headless cart-pole environments and serves them via VecGymServer.
//!
//! Run: `cargo run -p clankers-examples --bin cartpole_vec_gym -- 8`
//! Then: `python python/examples/cartpole_vec_benchmark.py`

use std::collections::HashMap;
use std::env;

use bevy::prelude::*;
use clankers_actuator::components::JointCommand;
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
}

fn main() {
    let num_envs: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let address = "127.0.0.1:9878";

    println!("=== Vectorized Cart-Pole Gym Server ===\n");
    println!("Creating {num_envs} cart-pole environments with Rapier physics...");

    let envs: Vec<Box<dyn VecEnvInstance>> = (0..num_envs)
        .map(|i| {
            let env = make_cartpole_env();
            println!("  env {i}: ready");
            Box::new(env) as Box<dyn VecEnvInstance>
        })
        .collect();

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

    let server = VecGymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("\nVec cart-pole gym server listening on {addr}");
    println!("num_envs={num_envs}, obs_dim={obs_dim}, act_dim=2, max_steps=500");
    println!("Connect with: python python/examples/cartpole_vec_benchmark.py\n");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut vec_env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
