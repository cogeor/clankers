//! Cart-pole gym server.
//!
//! Tests: URDF with prismatic + revolute joints, GymEnv, GymServer,
//! sensor registration, action applicator, TCP protocol.
//!
//! Run: `cargo run -p clankers-examples --bin cartpole_gym`
//! Then connect with: `python examples/python/gym_client.py`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::JointCommand;
use clankers_core::prelude::*;
use clankers_env::prelude::*;
use clankers_examples::CARTPOLE_URDF;
use clankers_gym::prelude::*;
use clankers_sim::ClankersSimPlugin;
use clankers_urdf::spawn_robot;

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

fn main() {
    println!("=== Cart-Pole Gym Server Example ===\n");

    let max_steps: u32 = 500;
    let num_joints: usize = 2; // cart_slide (prismatic) + pole_hinge (revolute)
    let address = "127.0.0.1:9877";

    // ---------------------------------------------------------------
    // 1. Parse URDF and verify structure
    // ---------------------------------------------------------------
    let model = clankers_urdf::parse_string(CARTPOLE_URDF).expect("failed to parse cartpole URDF");
    println!("Robot: {}", model.name);
    println!("DOF:   {}", model.dof());
    println!("Joints: {:?}", model.actuated_joint_names());

    // ---------------------------------------------------------------
    // 2. Build Bevy app with sim plugin
    // ---------------------------------------------------------------
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);

    // Spawn robot into the world
    let spawned = spawn_robot(app.world_mut(), &model, &HashMap::new());
    println!(
        "Spawned '{}' with {} joints",
        spawned.name,
        spawned.joint_count()
    );

    // ---------------------------------------------------------------
    // 3. Register sensors
    // ---------------------------------------------------------------
    {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        // Joint state: 2 joints * 2 (pos + vel) = 4 values
        registry.register(
            Box::new(JointStateSensor::new(num_joints)),
            &mut buffer,
        );
        println!("Observation dim: {}", buffer.dim());
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    // ---------------------------------------------------------------
    // 4. Create gym environment
    // ---------------------------------------------------------------
    let obs_dim = num_joints * 2;
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    let mut env = GymEnv::new(app, obs_space, act_space, Box::new(CartPoleApplicator));

    // ---------------------------------------------------------------
    // 5. Start server
    // ---------------------------------------------------------------
    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("\nCart-pole gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");
    println!("Connect with: python examples/python/gym_client.py\n");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
