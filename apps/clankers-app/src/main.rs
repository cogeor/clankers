//! Clankers robotics simulation CLI. Subcommands ship in W5 PR1–PR4.

use std::process::ExitCode;

use clap::{Parser, Subcommand};

mod commands;

use commands::bench::BenchArgs;
use commands::info::InfoArgs;
use commands::inspect::InspectTarget;
use commands::record::RecordArgs;
use commands::replay::ReplayArgs;
use commands::run::RunArgs;
use commands::serve::ServeArgs;
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
    /// Print workspace metadata.
    Info(InfoArgs),

    /// Validate a URDF / scenario / scene / policy artefact.
    Validate(ValidateArgs),

    /// Inspect an artefact (URDF, MCAP, ONNX, scene).
    Inspect {
        #[command(subcommand)]
        target: InspectTarget,
    },

    /// Run a scenario locally and print per-episode summaries.
    Run(RunArgs),

    /// Serve a scenario (or the legacy synthetic env) over TCP for gym
    /// clients.
    Serve(ServeArgs),

    /// Capture a scenario run to MCAP.
    Record(RecordArgs),

    /// Replay an MCAP recording.
    Replay(ReplayArgs),

    /// Measure scenario throughput headlessly; emit CSV/JSON row.
    Bench(BenchArgs),

    /// Deprecated: use `run --scenario <name>`. Hidden from `--help`.
    #[command(hide = true)]
    Headless {
        #[arg(short = 'n', long, default_value_t = 1)]
        episodes: u32,
        #[arg(short, long, default_value_t = 100)]
        max_steps: u32,
        #[arg(short, long)]
        seed: Option<u64>,
    },

    /// Interactive 3D visualization with teleop and policy controls.
    Viz {
        /// W8 scope — accepted but ignored in PR2.
        #[arg(long)]
        scenario: Option<String>,
        #[arg(short, long, default_value_t = 4)]
        joints: usize,
        #[arg(short, long, default_value_t = 1000)]
        max_steps: u32,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Info(args)) => commands::info::execute(&args),
        Some(Commands::Validate(args)) => commands::validate::execute(&args),
        Some(Commands::Inspect { target }) => commands::inspect::execute(target),
        Some(Commands::Run(args)) => commands::run::execute(&args),
        Some(Commands::Serve(args)) => commands::serve::execute(&args),
        Some(Commands::Record(args)) => commands::record::execute(&args),
        Some(Commands::Replay(args)) => commands::replay::execute(&args),
        Some(Commands::Bench(args)) => commands::bench::execute(&args),
        Some(Commands::Headless {
            episodes,
            max_steps,
            seed,
        }) => {
            eprintln!("clankers-app headless is deprecated; use `run --scenario <name>` instead.");
            commands::run::execute_default(episodes, max_steps, seed);
            ExitCode::SUCCESS
        }
        Some(Commands::Viz {
            scenario,
            joints,
            max_steps,
        }) => {
            if scenario.is_some() {
                eprintln!("--scenario for viz is W8 scope; running default");
            }
            commands::viz::execute(joints, max_steps);
            ExitCode::SUCCESS
        }
        None => {
            commands::run::execute_default(1, 100, None);
            ExitCode::SUCCESS
        }
    }
}
