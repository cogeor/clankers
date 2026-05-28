//! Legacy single-scenario benchmark body (W5 PR4 V1 schema).
//!
//! Drives a registered scenario through `--runs` measurement passes and
//! emits a [`super::csv::CSV_HEADER`]-shaped row. Also exposes
//! [`run_once`] which `bench mpc` reuses.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use clankers_env::prelude::*;
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{
    ClankersSimPlugin, ScenarioBuilder, ScenarioConfig, ScenarioHandle, ScenarioRegistry,
};

use super::args::BenchArgs;
use super::csv::{print_human_v1, write_csv_row_v1};
use super::stats::aggregate_v1;

pub(super) fn execute(args: &BenchArgs) -> ExitCode {
    let Some(scenario_name) = args.scenario.as_deref() else {
        eprintln!(
            "bench: --scenario is required for the legacy single-scenario surface (or use a subcommand: vec / protocol / record / mpc)"
        );
        return ExitCode::from(2);
    };

    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(scenario_name) else {
        eprintln!("unknown scenario: {scenario_name}");
        return ExitCode::from(1);
    };

    let cfg = ScenarioConfig {
        seed: Some(args.seed.unwrap_or(0)),
        max_steps: args.max_steps,
        headless: true,
        record_path: None,
    };
    let seed = args.seed.unwrap_or(0);

    for _ in 0..args.warmup_runs {
        let _ = run_once(builder, &cfg, seed);
    }

    let mut per_run_wall_ms: Vec<f64> = Vec::with_capacity(args.runs as usize);
    let mut per_run_steps_per_sec: Vec<f64> = Vec::with_capacity(args.runs as usize);
    let mut all_step_durations: Vec<Duration> = Vec::new();
    let mut total_steps: u32 = 0;

    for _ in 0..args.runs {
        let (wall, step_samples) = run_once(builder, &cfg, seed);
        let run_steps = u32::try_from(step_samples.len()).unwrap_or(u32::MAX);
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        let run_sps = if wall_ms > 0.0 {
            f64::from(run_steps) / wall_ms * 1000.0
        } else {
            0.0
        };
        per_run_steps_per_sec.push(run_sps);
        all_step_durations.extend(step_samples);
        total_steps = total_steps.saturating_add(run_steps);
    }

    let row = aggregate_v1(
        scenario_name,
        args,
        total_steps,
        &per_run_wall_ms,
        &per_run_steps_per_sec,
        &mut all_step_durations,
    );

    if let Some(path) = args.csv.as_ref()
        && let Err(err) = write_csv_row_v1(path, &row)
    {
        eprintln!("failed to write CSV: {err}");
        return ExitCode::from(1);
    }

    if args.json {
        match serde_json::to_string(&row) {
            Ok(json) => println!("{json}"),
            Err(err) => {
                eprintln!("failed to serialise JSON: {err}");
                return ExitCode::from(1);
            }
        }
    } else if args.csv.is_none() {
        print_human_v1(&row);
    }

    ExitCode::SUCCESS
}

/// Build a fresh `App`, run one episode, and return `(wall, per-step
/// durations)`.
pub(super) fn run_once(
    builder: &dyn ScenarioBuilder,
    cfg: &ScenarioConfig,
    seed: u64,
) -> (Duration, Vec<Duration>) {
    let mut app = App::new();
    app.add_plugins(ClankersSimPlugin);
    let handle: ScenarioHandle = builder.build(&mut app, cfg);
    app.finish();
    app.cleanup();

    app.world_mut()
        .resource_mut::<EpisodeConfig>()
        .max_episode_steps = handle.max_steps;
    app.world_mut().resource_mut::<Episode>().reset(Some(seed));

    let wall_start = Instant::now();
    let mut per_step: Vec<Duration> = Vec::with_capacity(handle.max_steps as usize);
    for _ in 0..handle.max_steps {
        let step_start = Instant::now();
        app.update();
        per_step.push(step_start.elapsed());
        if app.world().resource::<Episode>().is_done() {
            break;
        }
    }
    (wall_start.elapsed(), per_step)
}
