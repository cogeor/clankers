//! Clankers robotics simulation CLI.
//!
//! Provides three modes of operation:
//! - `headless`: Run N episodes locally and print statistics
//! - `serve`: Start a TCP gym server for remote training clients
//! - `info`: Print workspace crate versions and configuration

use bevy::prelude::*;
use clap::{Parser, Subcommand};

use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_core::prelude::*;
use clankers_env::prelude::*;
use clankers_sim::{ClankersSimPlugin, EpisodeStats};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Clankers robotics simulation framework.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run episodes locally and print statistics.
    Headless {
        /// Number of episodes to run.
        #[arg(short = 'n', long, default_value_t = 1)]
        episodes: u32,

        /// Maximum steps per episode.
        #[arg(short, long, default_value_t = 100)]
        max_steps: u32,

        /// Random seed.
        #[arg(short, long)]
        seed: Option<u64>,
    },

    /// Start a TCP gym server for remote training clients.
    Serve {
        /// Address to bind (e.g. 127.0.0.1:9876).
        #[arg(short, long, default_value = "127.0.0.1:9876")]
        address: String,

        /// Number of joints in the default environment.
        #[arg(short, long, default_value_t = 2)]
        joints: usize,

        /// Maximum steps per episode.
        #[arg(short, long, default_value_t = 1000)]
        max_steps: u32,
    },

    /// Print crate information.
    Info,
}

// ---------------------------------------------------------------------------
// NoopApplicator
// ---------------------------------------------------------------------------

/// Default action applicator that writes action values to joint commands.
struct JointCommandApplicator;

impl ActionApplicator for JointCommandApplicator {
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
        "JointCommandApplicator"
    }
}

// ---------------------------------------------------------------------------
// Mode implementations
// ---------------------------------------------------------------------------

fn run_headless(episodes: u32, max_steps: u32, seed: Option<u64>) {
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
        println!(
            "episode {}: steps={}, reward={:.3}",
            ep + 1,
            episode.step_count,
            episode.total_reward
        );
    }

    let stats = app.world().resource::<EpisodeStats>();
    println!(
        "\ntotal: episodes={}, steps={}",
        stats.episodes_completed, stats.total_steps
    );
}

fn run_serve(address: &str, num_joints: usize, max_steps: u32) {
    use clankers_gym::prelude::*;

    // Build the Bevy app for the gym environment
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);

    for _ in 0..num_joints {
        app.world_mut().spawn((
            Actuator::default(),
            JointCommand::default(),
            JointState::default(),
            JointTorque::default(),
        ));
    }

    // Register a sensor
    {
        let world = app.world_mut();
        let mut registry = world.remove_resource::<SensorRegistry>().unwrap();
        let mut buffer = world.remove_resource::<ObservationBuffer>().unwrap();
        registry.register(Box::new(JointStateSensor::new(num_joints)), &mut buffer);
        world.insert_resource(buffer);
        world.insert_resource(registry);
    }

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = max_steps;

    let obs_dim = num_joints * 2; // position + velocity
    let obs_space = ObservationSpace::Box {
        low: vec![-10.0; obs_dim],
        high: vec![10.0; obs_dim],
    };
    let act_space = ActionSpace::Box {
        low: vec![-1.0; num_joints],
        high: vec![1.0; num_joints],
    };

    let mut env = GymEnv::new(app, obs_space, act_space, Box::new(JointCommandApplicator));

    let server = GymServer::bind(address).expect("failed to bind server");
    let addr = server.local_addr().expect("failed to get address");
    println!("clankers gym server listening on {addr}");
    println!("joints={num_joints}, obs_dim={obs_dim}, act_dim={num_joints}, max_steps={max_steps}");

    loop {
        println!("waiting for client...");
        match server.serve_one(&mut env) {
            Ok(()) => println!("client disconnected cleanly"),
            Err(e) => eprintln!("client error: {e}"),
        }
    }
}

fn run_info() {
    println!("clankers v{}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("crates:");
    println!("  clankers-core        {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-actuator    {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-env         {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-gym         {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-sim         {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-urdf        {}", env!("CARGO_PKG_VERSION"));
    println!("  clankers-domain-rand {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("edition: 2024");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Headless {
            episodes,
            max_steps,
            seed,
        }) => run_headless(episodes, max_steps, seed),
        Some(Commands::Serve {
            address,
            joints,
            max_steps,
        }) => run_serve(&address, joints, max_steps),
        Some(Commands::Info) => run_info(),
        None => {
            // Default: run headless with defaults
            run_headless(1, 100, None);
        }
    }
}
