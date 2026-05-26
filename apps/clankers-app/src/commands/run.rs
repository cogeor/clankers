//! `clankers-app run` — scenario-driven local execution.
//!
//! W5 PR2 — scenarios via `--scenario <name>`. When `--scenario` is
//! omitted, the body falls through to [`execute_default`] which
//! preserves the pre-PR1 `Headless` behaviour verbatim (back-compat).

use std::path::PathBuf;
use std::process::ExitCode;

use bevy::prelude::*;
use clankers_actuator::components::{Actuator, JointCommand, JointState, JointTorque};
use clankers_env::prelude::*;
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{ClankersSimPlugin, EpisodeStats, ScenarioConfig, ScenarioRegistry};
use clap::Args;
use serde::Serialize;

/// CLI flags for `clankers-app run`.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Built-in scenario name. Without this flag the legacy headless
    /// empty-app path runs via [`execute_default`] (back-compat with
    /// the `Headless` subcommand).
    #[arg(long)]
    pub scenario: Option<String>,

    /// Number of episodes.
    #[arg(short = 'n', long, default_value_t = 1)]
    pub episodes: u32,

    /// Maximum steps per episode.
    #[arg(long, default_value_t = 1000)]
    pub max_steps: u32,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Path to an ONNX policy (W7). Returns "not yet implemented" if
    /// supplied this loop.
    #[arg(long)]
    pub policy: Option<PathBuf>,

    /// Path to an MCAP recording sink (W5 PR3). Same fallback.
    #[arg(long)]
    pub record: Option<PathBuf>,

    /// Emit NDJSON per-episode summaries to stdout instead of human
    /// text.
    #[arg(long)]
    pub json: bool,
}

/// One row in the NDJSON / per-episode text output stream.
#[derive(Serialize)]
struct EpisodeSummary {
    episode: u32,
    steps: u32,
    success: bool,
    truncated: bool,
}

/// Scenario-driven local execution (`clankers-app run --scenario
/// <name>`). When `--scenario` is `None`, delegates to
/// [`execute_default`].
pub fn execute(args: &RunArgs) -> ExitCode {
    let Some(name) = args.scenario.as_deref() else {
        // Back-compat: no --scenario means the legacy empty-app path.
        execute_default(args.episodes, args.max_steps, args.seed);
        return ExitCode::SUCCESS;
    };

    if args.policy.is_some() {
        eprintln!("--policy not yet supported (W7)");
        return ExitCode::from(2);
    }
    if args.record.is_some() {
        eprintln!("--record not yet supported (W5 PR3)");
        return ExitCode::from(2);
    }

    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(name) else {
        eprintln!("unknown scenario: {name}");
        return ExitCode::from(1);
    };

    let cfg = ScenarioConfig {
        seed: args.seed,
        max_steps: args.max_steps,
        headless: true,
        record_path: None,
    };

    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle = builder.build(&mut app, &cfg);
    app.finish();
    app.cleanup();

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = handle.max_steps;

    for ep in 1..=args.episodes {
        app.world_mut().resource_mut::<Episode>().reset(args.seed);

        for _ in 0..handle.max_steps {
            app.update();
            if app.world().resource::<Episode>().is_done() {
                break;
            }
        }

        // Clone before re-borrowing the world on the next iteration
        // (same trick used in execute_default below).
        let episode = app.world().resource::<Episode>().clone();
        let summary = EpisodeSummary {
            episode: ep,
            steps: episode.step_count,
            success: matches!(episode.state, EpisodeState::Done),
            truncated: matches!(episode.state, EpisodeState::Truncated),
        };
        if args.json {
            println!(
                "{}",
                serde_json::to_string(&summary).expect("EpisodeSummary is Serialize")
            );
        } else {
            println!(
                "episode {}: steps={}, success={}, truncated={}",
                summary.episode, summary.steps, summary.success, summary.truncated
            );
        }
    }

    ExitCode::SUCCESS
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
