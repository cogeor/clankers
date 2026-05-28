//! `bench mpc` body — MPC scenario throughput.
//!
//! Reuses [`super::scenario::run_once`] (the legacy single-scenario
//! driver) but emits the V2 CSV schema so its rows live in the same
//! file as `bench vec`/`protocol`/`record` cells.

use std::process::ExitCode;
use std::time::Duration;

use clankers_sim::scenarios::register_builtin;
use clankers_sim::{ScenarioConfig, ScenarioRegistry};

use super::args::{BenchArgs, MpcArgs};
use super::csv::{print_human_v2, write_csv_row_v2};
use super::scenario::run_once;
use super::stats::aggregate_v2;

pub(super) fn execute(args: &BenchArgs, _m_args: &MpcArgs) -> ExitCode {
    // PR4 plan-deviation: default scenario is `arm_pick` (W5 PR2 has
    // it). Loop 08 (W8 PR2) will lift `quadruped_trot` into the
    // scenario registry and this default flips to it.
    let scenario_name = args
        .scenario
        .clone()
        .unwrap_or_else(|| "arm_pick".to_owned());

    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(&scenario_name) else {
        eprintln!("bench mpc: unknown scenario: {scenario_name}");
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

    let mut per_run_wall_ms = Vec::with_capacity(args.runs as usize);
    let mut per_run_sps = Vec::with_capacity(args.runs as usize);
    let mut step_durs: Vec<Duration> = Vec::new();
    let mut total_steps: u32 = 0;

    for _ in 0..args.runs {
        let (wall, samples) = run_once(builder, &cfg, seed);
        let run_steps = u32::try_from(samples.len()).unwrap_or(u32::MAX);
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        per_run_sps.push(if wall_ms > 0.0 {
            f64::from(run_steps) / wall_ms * 1000.0
        } else {
            0.0
        });
        step_durs.extend(samples);
        total_steps = total_steps.saturating_add(run_steps);
    }

    let row = aggregate_v2(
        &format!("mpc_{scenario_name}"),
        args,
        total_steps,
        &per_run_wall_ms,
        &per_run_sps,
        &mut step_durs,
        0,
        0,
        0.0,
        &format!("kind=mpc;scenario={scenario_name}"),
    );

    if let Some(path) = args.csv.as_ref()
        && let Err(e) = write_csv_row_v2(path, &row)
    {
        eprintln!("bench mpc: failed to write CSV: {e}");
        return ExitCode::from(1);
    }
    if args.json {
        if let Ok(s) = serde_json::to_string(&row) {
            println!("{s}");
        }
    } else if args.csv.is_none() {
        print_human_v2(&row);
    }
    ExitCode::SUCCESS
}
