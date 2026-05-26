//! `clankers-app bench` — headless throughput micro-benchmark.
//!
//! Single-subcommand surface: `bench --scenario <name>` runs the
//! scenario `runs` times (default 5) after `warmup-runs` discarded
//! warmup runs (default 3), measures wall-clock + per-step latency, and
//! emits a row to stdout (human / JSON) or appends one row to a CSV file.
//!
//! # CSV schema (LOCKED — v1)
//!
//! The CSV header is the verbatim string:
//!
//! ```text
//! scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes
//! ```
//!
//! Column semantics:
//!
//! | # | Column | Type | Units | Meaning |
//! |---|---|---|---|---|
//! | 1 | `scenario` | string | — | Scenario name passed to `--scenario`. |
//! | 2 | `steps` | u32 | — | Total measured simulation steps (`runs × max_steps`). |
//! | 3 | `runs` | u32 | — | Number of measurement runs (excludes warmup). |
//! | 4 | `max_steps` | u32 | — | Per-run `--max-steps`. |
//! | 5 | `wall_ms_total` | f64 | ms | Sum of per-run wall-clock durations. |
//! | 6 | `wall_ms_per_run_mean` | f64 | ms | Mean per-run wall-clock duration. |
//! | 7 | `wall_ms_per_run_std` | f64 | ms | Bessel-corrected sample std (n-1) of per-run wall-clock. |
//! | 8 | `steps_per_sec_mean` | f64 | steps/s | Mean of per-run `(steps / wall_ms × 1000)`. |
//! | 9 | `step_us_p50` | f64 | µs | Median per-step latency across all measurement runs. |
//! | 10 | `step_us_p99` | f64 | µs | 99th percentile per-step latency. |
//! | 11 | `notes` | string | — | Free-form annotation (`hostname=...;commit=...;unix_ts=...;warmup_runs=...`). |
//!
//! **Additive-only contract.** Future loops (W7 perf, W8 CI) may add columns
//! ONLY at the END of each row. Existing columns must not be reordered,
//! renamed, or removed — CI parsers (W8 PR3) read by column name AND
//! position, so an append at column 12+ is safe but a reshuffle is not.
//! Any column-semantics change (e.g. units change from ms to µs) must
//! introduce a NEW column name with the unit suffix.
//!
//! See `.delegate/work/20260526-013019-w3-w4-w5-impl/08/PLAN.md` § Design
//! choice A for the full lock rationale.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use clankers_env::prelude::*;
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{
    ClankersSimPlugin, ScenarioBuilder, ScenarioConfig, ScenarioHandle, ScenarioRegistry,
};
use clap::Args;
use serde::Serialize;

/// CSV header — keep in lockstep with [`BenchRow`] field order.
const CSV_HEADER: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes";

/// CLI flags for `clankers-app bench`.
#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Built-in scenario name (`cartpole`, `arm_pick`).
    #[arg(long)]
    pub scenario: String,

    /// Per-run `max_steps`.
    #[arg(long, default_value_t = 1000)]
    pub max_steps: u32,

    /// Random seed forwarded to `Episode::reset`. Defaults to `0` inside
    /// `execute` for maximum reproducibility across runs.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of measurement runs (default 5; LOCKED at v1 schema —
    /// `runs × max_steps` is the `steps` column).
    #[arg(long, default_value_t = 5)]
    pub runs: u32,

    /// Number of warmup runs (default 3; samples discarded).
    #[arg(long, default_value_t = 3)]
    pub warmup_runs: u32,

    /// Append one row to this CSV file. Creates the file with the locked
    /// header if it does not exist or is empty.
    #[arg(long)]
    pub csv: Option<PathBuf>,

    /// Emit a single JSON object to stdout instead of the human-readable
    /// table. Mutually informative with `--csv`: both may be set; `--csv`
    /// writes the file and `--json` writes stdout.
    #[arg(long)]
    pub json: bool,
}

/// One row of the v1 CSV schema. Field order MUST match [`CSV_HEADER`]
/// — `csv::Writer::serialize` writes fields in declaration order.
#[derive(Serialize, Debug, Clone)]
struct BenchRow {
    scenario: String,
    steps: u32,
    runs: u32,
    max_steps: u32,
    wall_ms_total: f64,
    wall_ms_per_run_mean: f64,
    wall_ms_per_run_std: f64,
    steps_per_sec_mean: f64,
    step_us_p50: f64,
    step_us_p99: f64,
    notes: String,
}

/// Execute `clankers-app bench --scenario <name>`.
pub fn execute(args: &BenchArgs) -> ExitCode {
    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let Some(builder) = registry.get(&args.scenario) else {
        eprintln!("unknown scenario: {}", args.scenario);
        return ExitCode::from(1);
    };

    let cfg = ScenarioConfig {
        seed: Some(args.seed.unwrap_or(0)),
        max_steps: args.max_steps,
        headless: true,
        record_path: None,
    };
    let seed = args.seed.unwrap_or(0);

    // ---- warmup runs (samples discarded) ----------------------------------
    for _ in 0..args.warmup_runs {
        let _ = run_once(builder, &cfg, seed);
    }

    // ---- measurement runs -------------------------------------------------
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

    let wall_ms_total: f64 = per_run_wall_ms.iter().sum();
    let wall_ms_per_run_mean = if args.runs > 0 {
        wall_ms_total / f64::from(args.runs)
    } else {
        0.0
    };
    let wall_ms_per_run_std = stddev(&per_run_wall_ms);
    #[allow(clippy::cast_precision_loss)]
    let steps_per_sec_mean = if per_run_steps_per_sec.is_empty() {
        0.0
    } else {
        per_run_steps_per_sec.iter().sum::<f64>() / per_run_steps_per_sec.len() as f64
    };
    let step_us_p50 = percentile_us(&mut all_step_durations.clone(), 0.50);
    let step_us_p99 = percentile_us(&mut all_step_durations, 0.99);

    let row = BenchRow {
        scenario: args.scenario.clone(),
        steps: total_steps,
        runs: args.runs,
        max_steps: args.max_steps,
        wall_ms_total,
        wall_ms_per_run_mean,
        wall_ms_per_run_std,
        steps_per_sec_mean,
        step_us_p50,
        step_us_p99,
        notes: build_notes(args),
    };

    // ---- emit --------------------------------------------------------------
    if let Some(path) = args.csv.as_ref()
        && let Err(err) = write_csv_row(path, &row)
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
        print_human(&row);
    }

    ExitCode::SUCCESS
}

/// Build a fresh `App`, run one episode, and return `(wall, per-step
/// durations)`. Each invocation tears down the previous `App` so physics
/// state never leaks across runs.
fn run_once(
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

/// Flat quantile over sorted `Duration` samples, returned in microseconds.
fn percentile_us(samples: &mut [Duration], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_unstable();
    let n = samples.len();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut idx = (n as f64 * q).floor() as usize;
    if idx >= n {
        idx = n - 1;
    }
    samples[idx].as_secs_f64() * 1_000_000.0
}

/// Bessel-corrected sample standard deviation (n-1). Returns 0 for n < 2.
fn stddev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let sq_sum: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum();
    (sq_sum / (n - 1.0)).sqrt()
}

/// Self-describing `notes` field. Format:
/// `hostname=<x>;commit=<x>;unix_ts=<x>;profile=<x>;warmup_runs=<x>`.
///
/// `profile` is `debug` when `cfg!(debug_assertions)` (i.e. `cargo run`
/// without `--release` / `--profile bench`), else `release-like`. Baseline
/// CSVs should be generated under release builds for representative
/// numbers; the `profile` field lets W8 CI parsers reject debug-mode
/// rows without scraping the host.
fn build_notes(args: &BenchArgs) -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned());
    let commit = std::env::var("GITHUB_SHA")
        .or_else(|_| std::env::var("GIT_COMMIT"))
        .unwrap_or_else(|_| "local".to_owned());
    let unix_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release-like"
    };
    format!(
        "hostname={host};commit={commit};unix_ts={unix_ts};profile={profile};warmup_runs={}",
        args.warmup_runs
    )
}

/// Append the row to `path`, writing the locked CSV header if the file
/// is missing or empty.
fn write_csv_row(path: &Path, row: &BenchRow) -> std::io::Result<()> {
    let file_existed_with_content = path.exists()
        && std::fs::metadata(path)
            .map(|m| m.len() > 0)
            .unwrap_or(false);

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }

    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);

    if !file_existed_with_content {
        // Write the verbatim locked header exactly once. We avoid
        // `has_headers(true)` so the header bytes are byte-equal to the
        // documented constant, independent of csv-crate quoting.
        wtr.write_record(CSV_HEADER.split(','))?;
    }
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

/// Two-column key→value pretty print for terminal use.
fn print_human(row: &BenchRow) {
    println!("bench: {}", row.scenario);
    println!("  steps:                {}", row.steps);
    println!("  runs:                 {}", row.runs);
    println!("  max_steps:            {}", row.max_steps);
    println!("  wall_ms_total:        {:.3}", row.wall_ms_total);
    println!("  wall_ms_per_run_mean: {:.3}", row.wall_ms_per_run_mean);
    println!("  wall_ms_per_run_std:  {:.3}", row.wall_ms_per_run_std);
    println!("  steps_per_sec_mean:   {:.1}", row.steps_per_sec_mean);
    println!("  step_us_p50:          {:.2}", row.step_us_p50);
    println!("  step_us_p99:          {:.2}", row.step_us_p99);
    println!("  notes:                {}", row.notes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_field_count_matches_bench_row() {
        // 11 columns LOCKED. If a future loop adds a column at the end
        // of `BenchRow`, this assertion must be bumped IN THE SAME PR
        // that adds the column.
        let cols = CSV_HEADER.split(',').count();
        assert_eq!(cols, 11, "schema lock: 11 columns");
    }

    #[test]
    fn csv_header_column_names_locked() {
        let expected = [
            "scenario",
            "steps",
            "runs",
            "max_steps",
            "wall_ms_total",
            "wall_ms_per_run_mean",
            "wall_ms_per_run_std",
            "steps_per_sec_mean",
            "step_us_p50",
            "step_us_p99",
            "notes",
        ];
        let actual: Vec<&str> = CSV_HEADER.split(',').collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn percentile_us_returns_zero_for_empty() {
        let mut samples: Vec<Duration> = Vec::new();
        assert!(percentile_us(&mut samples, 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_us_median_of_known_samples() {
        let mut samples: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        // q=0.5 → idx=floor(100*0.5)=50 → samples[50] = 51us.
        assert!((percentile_us(&mut samples, 0.5) - 51.0).abs() < 0.01);
        // q=0.99 → idx=floor(100*0.99)=99 → samples[99] = 100us.
        let mut samples2: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        assert!((percentile_us(&mut samples2, 0.99) - 100.0).abs() < 0.01);
    }

    #[test]
    fn stddev_zero_for_one_sample() {
        assert!(stddev(&[42.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn stddev_bessel_corrected() {
        // [2, 4, 4, 4, 5, 5, 7, 9]: Bessel std = 2.138...
        let xs = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = stddev(&xs);
        assert!((s - 2.138_089_935_2).abs() < 1e-6, "got {s}");
    }
}
