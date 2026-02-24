//! Cart-pole gym server with Rapier physics.
//!
//! Headless cart-pole environment matching OpenAI Gym CartPole-v1 parameters.
//! Serves a single GymEnv over TCP for Python RL training.
//!
//! Run: `cargo run -p clankers-examples --bin cartpole_gym`
//! Then connect with: `python python/examples/cartpole_read_state.py`

use std::collections::HashMap;

use bevy::prelude::*;
use clankers_actuator::components::JointCommand;
use clankers_core::prelude::*;
use clankers_env::prelude::*;
use clankers_examples::CARTPOLE_URDF;
use clankers_gym::prelude::*;
use clankers_physics::rapier::{bridge::register_robot, RapierBackend, RapierContext};
use clankers_physics::ClankersPhysicsPlugin;
use clankers_sim::SceneBuilder;

/// Writes action values to joint commands in spawn order.
///
/// Action layout: [cart_force, pole_torque]
/// For standard CartPole, only cart_force (index 0) is used;
/// pole_torque (index 1) should be 0 (passive joint).
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
    println!("=== Cart-Pole Gym Server (with Rapier Physics) ===\n");

    let max_steps: u32 = 500;
    let num_joints: usize = 2; // cart_slide (prismatic) + pole_hinge (continuous)
    let address = "127.0.0.1:9877";

    // ---------------------------------------------------------------
    // 1. Parse URDF
    // ---------------------------------------------------------------
    let model =
        clankers_urdf::parse_string(CARTPOLE_URDF).expect("failed to parse cartpole URDF");
    println!("Robot: {}", model.name);
    println!("DOF:   {}", model.dof());
    println!("Joints: {:?}", model.actuated_joint_names());

    // ---------------------------------------------------------------
    // 2. Build scene with SceneBuilder + physics
    // ---------------------------------------------------------------
    let mut scene = SceneBuilder::new()
        .with_max_episode_steps(max_steps)
        .with_robot(model.clone(), HashMap::new())
        .build();

    // Add Rapier physics backend
    scene
        .app
        .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

    // Register robot bodies/joints with the rapier context
    {
        let spawned = &scene.robots["cartpole"];
        let world = scene.app.world_mut();
        let mut ctx = world.remove_resource::<RapierContext>().unwrap();
        register_robot(&mut ctx, &model, spawned, world, true);
        world.insert_resource(ctx);
    }

    println!(
        "Spawned '{}' with {} joints + Rapier physics",
        scene.robots["cartpole"].name,
        scene.robots["cartpole"].joint_count()
    );

    // ---------------------------------------------------------------
    // 3. Register sensors
    // ---------------------------------------------------------------
    {
        let world = scene.app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        // Joint state: 2 joints × 2 (pos + vel) = 4 obs values
        // Layout: [cart_pos, cart_vel, pole_angle, pole_vel]
        registry.register(
            Box::new(JointStateSensor::new(num_joints)),
            &mut buffer,
        );
        println!("Observation dim: {}", buffer.dim());
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    // ---------------------------------------------------------------
    // 4. Create gym environment
    // ---------------------------------------------------------------
    let obs_dim = num_joints * 2; // 4: [cart_pos, cart_vel, pole_angle, pole_vel]
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    // Action: [cart_force, pole_torque] normalized to [-1, 1]
    // The actuator scales by effort limit (10N for cart, 0N for pole)
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    let mut env = GymEnv::new(
        scene.app,
        obs_space,
        act_space,
        Box::new(CartPoleApplicator),
    );

    // ---------------------------------------------------------------
    // 5. Start server
    // ---------------------------------------------------------------
    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("\nCart-pole gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");
    println!("Physics: Rapier3D, gravity [0,0,-9.81], 20 substeps/frame");
    println!("Connect with: python python/examples/cartpole_read_state.py\n");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}
