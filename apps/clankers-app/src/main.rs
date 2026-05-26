//! Clankers robotics simulation CLI.
//!
//! Subcommands ship in W5 across PR1–PR4. Read-only (`info`, `validate`,
//! `inspect`) ship in PR1; write/run (`run`, `serve`, `record`, `replay`,
//! `bench`) ship in PR2–PR4. The legacy `headless`, `serve`, `viz`,
//! `info` variants remain available during the transition.

use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod commands;

use commands::inspect::InspectTarget;
use commands::validate::ValidateArgs;

/// Clankers robotics simulation framework.
#[derive(Parser)]
#[command(version, about, bin_name = "clankers-app")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    // ---- W5 PR1: read-only ---------------------------------------------
    /// Print workspace metadata.
    Info {
        /// Emit structured JSON to stdout.
        #[arg(long)]
        json: bool,
    },

    /// Validate a URDF / scenario / scene / policy artefact.
    Validate(ValidateArgs),

    /// Inspect an artefact (URDF, MCAP, ONNX, scene).
    Inspect {
        #[command(subcommand)]
        target: InspectTarget,
    },

    // ---- Legacy (preserved during W5 PR1) ------------------------------
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

    /// Interactive 3D visualization with teleop and policy controls.
    Viz {
        /// Number of joints in the demo robot.
        #[arg(short, long, default_value_t = 4)]
        joints: usize,
        /// Maximum steps per episode.
        #[arg(short, long, default_value_t = 1000)]
        max_steps: u32,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info { json }) => commands::info::execute(json),
        Some(Commands::Validate(args)) => commands::validate::execute(&args),
        Some(Commands::Inspect { target }) => commands::inspect::execute(target),
        Some(Commands::Headless {
            episodes,
            max_steps,
            seed,
        }) => {
            commands::run::execute_default(episodes, max_steps, seed);
            ExitCode::SUCCESS
        }
        Some(Commands::Serve {
            address,
            joints,
            max_steps,
        }) => {
            commands::serve::execute_default(&address, joints, max_steps);
            ExitCode::SUCCESS
        }
        Some(Commands::Viz { joints, max_steps }) => {
            commands::viz::execute(joints, max_steps);
            ExitCode::SUCCESS
        }
        None => {
            commands::run::execute_default(1, 100, None);
            ExitCode::SUCCESS
        }
    }
}
