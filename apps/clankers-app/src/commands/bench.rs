//! `clankers-app bench` — headless throughput micro-benchmark suite.
//!
//! W7 PR4 extends the W5 PR4 single-scenario surface into a subcommand
//! tree:
//!
//! ```text
//! clankers bench [LEGACY-FLAGS]                # W5 PR4 path
//! clankers bench scenario [LEGACY-FLAGS]        # explicit alias
//! clankers bench vec --envs N1,N2,...           # NEW (W7 PR4)
//! clankers bench protocol --envs N1,N2,...      # NEW (W7 PR4)
//! clankers bench record [--async] [--frames N]  # NEW (W7 PR4)
//! clankers bench mpc [--scenario <name>]        # NEW (W7 PR4)
//! ```
//!
//! # CSV schema lock
//!
//! Two CSV headers exist:
//!
//! - [`CSV_HEADER`] — the W5 PR4 11-column lock. Used by the legacy
//!   single-scenario path AND by `bench scenario` (the explicit alias).
//!   Existing baselines (`cartpole_baseline.csv`, `arm_pick_baseline.csv`)
//!   retain byte-equal headers.
//! - [`CSV_HEADER_V2`] — additive 15-column schema (11 W5 cols + 4 new
//!   trailing cols: `num_envs`, `p95_us`, `dropped_frames`,
//!   `throughput_x`). Used by `bench vec`, `bench protocol`,
//!   `bench record`, `bench mpc`. Comparator
//!   `scripts/compare_baseline.py` reads by column name so additive
//!   schema changes don't break it.
//!
//! Both headers share the first 11 column names byte-for-byte so the
//! V2 schema is a strict superset of V1.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::MinimalPlugins;
use bevy::prelude::*;
use clankers_core::time::SimTime;
use clankers_core::types::{Action, Observation, ResetInfo, ResetResult, StepInfo, StepResult};
use clankers_env::prelude::*;
use clankers_env::vec_runner::{VecEnvInstance, runner_for};
use clankers_gym::binary_frame;
use clankers_record::prelude::*;
use clankers_sim::scenarios::register_builtin;
use clankers_sim::{
    ClankersSimPlugin, ScenarioBuilder, ScenarioConfig, ScenarioHandle, ScenarioRegistry,
};
use clap::{Args, Subcommand};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CSV schema constants
// ---------------------------------------------------------------------------

/// W5 PR4 11-column header — LOCKED. The legacy single-scenario path
/// (`bench [--scenario ...]`) writes this exact byte sequence.
pub const CSV_HEADER: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes";

/// W7 PR4 additive 15-column header. First 11 columns are byte-equal to
/// [`CSV_HEADER`] so existing tooling keyed on `(scenario, num_envs)`
/// via header-name lookup keeps working.
pub const CSV_HEADER_V2: &str = "scenario,steps,runs,max_steps,wall_ms_total,wall_ms_per_run_mean,wall_ms_per_run_std,steps_per_sec_mean,step_us_p50,step_us_p99,notes,num_envs,p95_us,dropped_frames,throughput_x";

// ---------------------------------------------------------------------------
// BenchArgs — subcommand tree with legacy flag-path back-compat
// ---------------------------------------------------------------------------

/// `clankers-app bench` flags. The subcommand (`kind`) is optional —
/// when absent, the legacy W5 PR4 single-scenario surface runs.
#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Subcommand (`scenario`, `vec`, `protocol`, `record`, `mpc`). When
    /// absent, the legacy `--scenario` flag surface is used and the
    /// CSV header stays at the W5 PR4 lock.
    #[command(subcommand)]
    pub kind: Option<BenchKind>,

    /// Built-in scenario name. Used by the legacy flag surface and by
    /// `bench mpc` as the default scenario selector.
    #[arg(long, global = true)]
    pub scenario: Option<String>,

    /// Per-run `max_steps`.
    #[arg(long, global = true, default_value_t = 1000)]
    pub max_steps: u32,

    /// Random seed forwarded to `Episode::reset`.
    #[arg(long, global = true)]
    pub seed: Option<u64>,

    /// Number of measurement runs.
    #[arg(long, global = true, default_value_t = 5)]
    pub runs: u32,

    /// Number of warmup runs.
    #[arg(long, global = true, default_value_t = 3)]
    pub warmup_runs: u32,

    /// Append one row to this CSV file.
    #[arg(long, global = true)]
    pub csv: Option<PathBuf>,

    /// Emit a single JSON object to stdout.
    #[arg(long, global = true)]
    pub json: bool,
}

/// Subcommand variants for `clankers-app bench`.
#[derive(Subcommand, Debug)]
pub enum BenchKind {
    /// Explicit alias for the legacy `--scenario <name>` surface
    /// (W5 PR4). Uses the V1 CSV header.
    Scenario,
    /// Throughput sweep over parallel vec-env runner sizes (W7 PR1).
    Vec(VecArgs),
    /// Encoding throughput sweep: binary frame vs JSON (W7 PR2).
    Protocol(ProtocolArgs),
    /// Recorder write rate: sync vs async at multiple buffer
    /// capacities (W7 PR4).
    Record(RecordBenchArgs),
    /// MPC scenario throughput (uses W7 PR3 dense joint runtime when
    /// the scenario built it).
    Mpc(MpcArgs),
}

/// `bench vec` subcommand-specific flags.
#[derive(Args, Debug)]
pub struct VecArgs {
    /// Comma-separated env counts to sweep. One CSV row emitted per
    /// entry. Default: `1,2,4,8`.
    #[arg(long, default_value = "1,2,4,8")]
    pub envs: String,

    /// Per-step busy-loop duration in microseconds. `0` (default)
    /// preserves the overhead-floor baseline shape (scenario
    /// `vec_parallel`). When `>0`, the synthetic env busy-loops for
    /// the given duration per `step()` call and the emitted scenario
    /// renames to `vec_throughput_{work_us}us` so throughput baselines
    /// don't collide with the overhead-floor baseline. Recommended
    /// realistic-work value: `100`.
    #[arg(long, default_value_t = 0)]
    pub work_us: u32,

    /// Optional parallel/sequential throughput ratio floor. When `>0.0`,
    /// the bench exits non-zero if the gated row's `throughput_x < K`.
    /// Default `0.0` = no gate (dev runs unaffected). CI passes `2.0`
    /// (conservative for a 4-core GHA runner; dev hardware sees ~4.5).
    /// Gates on the row matching `--envs` value `8` when present, else
    /// the highest-N row available (with a warning).
    /// Requires `--work-us > 0` — otherwise exits 2 (misconfigured)
    /// because the overhead-floor scenario can't satisfy a >1.0 ratio.
    #[arg(long, default_value_t = 0.0)]
    pub ratio_gate: f64,
}

/// `bench protocol` subcommand-specific flags.
#[derive(Args, Debug)]
pub struct ProtocolArgs {
    /// Comma-separated env counts to sweep.
    #[arg(long, default_value = "1,2,4,8")]
    pub envs: String,
    /// Observation dimension per env (default 16 per WS7-plan § 7).
    #[arg(long, default_value_t = 16)]
    pub obs_dim: u32,
    /// Total batch encodings per run.
    #[arg(long, default_value_t = 10_000)]
    pub batches: u32,
}

/// `bench record` subcommand-specific flags.
#[derive(Args, Debug)]
pub struct RecordBenchArgs {
    /// Number of synthetic joint frames per cell.
    #[arg(long, default_value_t = 10_000)]
    pub frames: u32,
    /// Comma-separated buffer capacities to sweep in async mode
    /// (default: `256,1024,4096`). The first cell is always sync mode.
    #[arg(long, default_value = "256,1024,4096")]
    pub buffers: String,
    /// Number of joints in each synthetic frame.
    #[arg(long, default_value_t = 8)]
    pub joints: usize,
}

/// `bench mpc` subcommand-specific flags.
#[derive(Args, Debug)]
pub struct MpcArgs {
    // PR4 deferral: `quadruped_trot` scenario is not yet registered
    // (W8 loop 8 lifts it from the standalone example). Default to
    // `arm_pick` (W5 PR2 ships this scenario). Loop 08 will swap the
    // default to `quadruped_trot` once registered.
}

// ---------------------------------------------------------------------------
// BenchRow / BenchRowV2
// ---------------------------------------------------------------------------

/// One row of the W5 PR4 v1 CSV schema. Field order matches
/// [`CSV_HEADER`] exactly.
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

/// One row of the W7 PR4 v2 additive CSV schema. The first 11 fields
/// match [`BenchRow`] for byte equality on the shared header prefix.
#[derive(Serialize, Debug, Clone)]
struct BenchRowV2 {
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
    /// Number of parallel envs (vec/protocol); 0 elsewhere.
    num_envs: u32,
    /// 95th percentile per-step latency in microseconds.
    p95_us: f64,
    /// Dropped frames during the run (record); 0 elsewhere.
    dropped_frames: u64,
    /// Binary-vs-JSON encoding throughput ratio (protocol); 0 elsewhere.
    throughput_x: f64,
}

// ---------------------------------------------------------------------------
// execute — dispatch
// ---------------------------------------------------------------------------

/// Execute `clankers-app bench`. Dispatches on `args.kind`.
pub fn execute(args: &BenchArgs) -> ExitCode {
    match &args.kind {
        None | Some(BenchKind::Scenario) => execute_scenario(args),
        Some(BenchKind::Vec(a)) => bench_vec(args, a),
        Some(BenchKind::Protocol(a)) => bench_protocol(args, a),
        Some(BenchKind::Record(a)) => bench_record(args, a),
        Some(BenchKind::Mpc(a)) => bench_mpc(args, a),
    }
}

// ---------------------------------------------------------------------------
// Legacy scenario body (V1 schema)
// ---------------------------------------------------------------------------

fn execute_scenario(args: &BenchArgs) -> ExitCode {
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

fn aggregate_v1(
    scenario_name: &str,
    args: &BenchArgs,
    total_steps: u32,
    per_run_wall_ms: &[f64],
    per_run_steps_per_sec: &[f64],
    all_step_durations: &mut [Duration],
) -> BenchRow {
    let wall_ms_total: f64 = per_run_wall_ms.iter().sum();
    let wall_ms_per_run_mean = if args.runs > 0 {
        wall_ms_total / f64::from(args.runs)
    } else {
        0.0
    };
    let wall_ms_per_run_std = stddev(per_run_wall_ms);
    #[allow(clippy::cast_precision_loss)]
    let steps_per_sec_mean = if per_run_steps_per_sec.is_empty() {
        0.0
    } else {
        per_run_steps_per_sec.iter().sum::<f64>() / per_run_steps_per_sec.len() as f64
    };
    let step_us_p50 = percentile_us(&mut all_step_durations.to_vec(), 0.50);
    let step_us_p99 = percentile_us(all_step_durations, 0.99);

    BenchRow {
        scenario: scenario_name.to_owned(),
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
    }
}

/// Build a fresh `App`, run one episode, and return `(wall, per-step
/// durations)`.
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

// ---------------------------------------------------------------------------
// bench vec
// ---------------------------------------------------------------------------

/// Synthetic constant env used by `bench vec` / `bench protocol`.
///
/// Cheap to construct + step so the measurement isolates the runner
/// overhead. Uses an `obs_dim`-wide constant observation and ignores
/// the action.
///
/// `work_us` is an opt-in per-`step()` busy-loop knob. When zero (the
/// default) `step()` is a handful of nanoseconds — useful for measuring
/// runner overhead floors. When `>0`, `step()` busy-spins on
/// `Instant::now()` until the requested microsecond budget has elapsed,
/// with `std::hint::black_box(())` inside the loop body to prevent the
/// optimiser from eliding the wait. This makes the env representative
/// of realistic per-step work so the parallel runner can actually amortise
/// rayon's dispatch overhead. The busy-loop is sensitive to CPU frequency
/// scaling, but `bench vec` reports a *ratio* (parallel / sequential) so
/// both arms see the same scaling.
struct ConstBenchEnv {
    obs_dim: usize,
    work_us: u32,
}

impl VecEnvInstance for ConstBenchEnv {
    fn reset(&mut self, _seed: Option<u64>) -> ResetResult {
        ResetResult {
            observation: Observation::zeros(self.obs_dim),
            info: ResetInfo::default(),
        }
    }

    fn step(&mut self, _action: &Action) -> StepResult {
        if self.work_us > 0 {
            let target = Duration::from_micros(u64::from(self.work_us));
            let start = Instant::now();
            while start.elapsed() < target {
                std::hint::black_box(());
            }
        }
        StepResult {
            observation: Observation::zeros(self.obs_dim),
            reward: 0.0,
            terminated: false,
            truncated: false,
            info: StepInfo::default(),
        }
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }
}

fn parse_env_list(s: &str) -> Result<Vec<u16>, String> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<u16>()
                .map_err(|e| format!("invalid env count '{t}': {e}"))
        })
        .collect()
}

fn bench_vec(args: &BenchArgs, vec_args: &VecArgs) -> ExitCode {
    let env_counts = match parse_env_list(&vec_args.envs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench vec: {e}");
            return ExitCode::from(1);
        }
    };

    if env_counts.is_empty() {
        eprintln!("bench vec: --envs must contain at least one value");
        return ExitCode::from(1);
    }

    let obs_dim: usize = 4; // cartpole-like dims for representative numbers
    let seed = args.seed.unwrap_or(0);
    let work_us = vec_args.work_us;

    if vec_args.ratio_gate > 0.0 && work_us == 0 {
        eprintln!(
            "bench vec: --ratio-gate {:.2} requires --work-us > 0 (the overhead-floor scenario `vec_parallel` cannot satisfy a >1.0 ratio gate). Re-run with --work-us 100 or drop --ratio-gate.",
            vec_args.ratio_gate
        );
        return ExitCode::from(2);
    }

    let mut gate_rows: Vec<(u16, f64)> = Vec::with_capacity(env_counts.len());

    for &n in &env_counts {
        // Warmup: parallel and sequential each warmed once.
        let _ = run_vec_cell(
            n,
            obs_dim,
            args.warmup_runs,
            args.max_steps,
            seed,
            true,
            work_us,
        );
        let _ = run_vec_cell(
            n,
            obs_dim,
            args.warmup_runs,
            args.max_steps,
            seed,
            false,
            work_us,
        );

        let (par_wall_ms, par_steps_per_sec, par_step_durs) =
            run_vec_cell(n, obs_dim, args.runs, args.max_steps, seed, true, work_us);
        let (seq_wall_ms, seq_steps_per_sec, _) =
            run_vec_cell(n, obs_dim, args.runs, args.max_steps, seed, false, work_us);

        let throughput_ratio = if seq_steps_per_sec_mean(&seq_steps_per_sec) > 0.0 {
            seq_steps_per_sec_mean(&par_steps_per_sec) / seq_steps_per_sec_mean(&seq_steps_per_sec)
        } else {
            0.0
        };

        gate_rows.push((n, throughput_ratio));

        if n == 8 && throughput_ratio < 3.0 && throughput_ratio > 0.0 {
            eprintln!(
                "bench vec: WARNING — N=8 parallel/sequential ratio = {throughput_ratio:.2}, below the WS7 § 7 gate of 3.0 (informational; CI baseline gate decides PASS/FAIL)"
            );
        }

        let total_steps = u32::try_from(par_wall_ms.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(args.max_steps);
        let mut step_durs_owned = par_step_durs.clone();
        let scenario = if work_us == 0 {
            "vec_parallel".to_string()
        } else {
            format!("vec_throughput_{work_us}us")
        };
        let row = aggregate_v2(
            &scenario,
            args,
            total_steps,
            &par_wall_ms,
            &par_steps_per_sec,
            &mut step_durs_owned,
            u32::from(n),
            0,
            throughput_ratio,
            &format!(
                "kind=vec;parallel=true;seq_wall_ms_mean={:.3};work_us={work_us}",
                seq_mean(&seq_wall_ms)
            ),
        );

        if let Some(path) = args.csv.as_ref()
            && let Err(e) = write_csv_row_v2(path, &row)
        {
            eprintln!("bench vec: failed to write CSV: {e}");
            return ExitCode::from(1);
        }

        if args.json {
            if let Ok(s) = serde_json::to_string(&row) {
                println!("{s}");
            }
        } else if args.csv.is_none() {
            print_human_v2(&row);
        }
    }

    evaluate_gate(&gate_rows, vec_args.ratio_gate)
}

/// Apply the ratio gate (if enabled) and convert to the canonical
/// `ExitCode` (0 pass / no gate, 1 fail). Extracted from `bench_vec` to
/// keep that function under the workspace clippy `too_many_lines` cap and
/// to keep the gate's exit-code mapping in one auditable spot.
fn evaluate_gate(gate_rows: &[(u16, f64)], gate: f64) -> ExitCode {
    if gate <= 0.0 {
        return ExitCode::SUCCESS;
    }
    match check_ratio_gate(gate_rows, gate) {
        Ok((n, x)) => {
            eprintln!("RATIO GATE: PASS N={n} throughput_x={x:.3} >= {gate:.3}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(1)
        }
    }
}

fn run_vec_cell(
    n: u16,
    obs_dim: usize,
    runs: u32,
    max_steps: u32,
    seed: u64,
    parallel: bool,
    work_us: u32,
) -> (Vec<f64>, Vec<f64>, Vec<Duration>) {
    let mut per_run_wall_ms = Vec::with_capacity(runs as usize);
    let mut per_run_sps = Vec::with_capacity(runs as usize);
    let mut all_step_durs: Vec<Duration> = Vec::new();

    for _ in 0..runs {
        // Fresh runner per measurement run — matches scenario path
        // behaviour where each run gets a clean App.
        let envs: Vec<Box<dyn VecEnvInstance + Send>> = (0..n)
            .map(|_| Box::new(ConstBenchEnv { obs_dim, work_us }) as Box<dyn VecEnvInstance + Send>)
            .collect();
        let cfg = VecEnvConfig::new(n).with_parallel(parallel);
        let mut runner = runner_for(envs, cfg);
        runner.reset_all(Some(seed));
        let action_template = Action::zeros(1);
        let actions: Vec<Action> = (0..n).map(|_| action_template.clone()).collect();

        let wall_start = Instant::now();
        let mut step_durs = Vec::with_capacity(max_steps as usize);
        for _ in 0..max_steps {
            let s = Instant::now();
            runner.step_all(&actions);
            step_durs.push(s.elapsed());
        }
        let wall = wall_start.elapsed();
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        let total_env_steps = f64::from(max_steps) * f64::from(n);
        let sps = if wall_ms > 0.0 {
            total_env_steps / wall_ms * 1000.0
        } else {
            0.0
        };
        per_run_sps.push(sps);
        all_step_durs.extend(step_durs);
    }

    (per_run_wall_ms, per_run_sps, all_step_durs)
}

fn seq_mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let n = samples.len() as f64;
        samples.iter().sum::<f64>() / n
    }
}

fn seq_steps_per_sec_mean(samples: &[f64]) -> f64 {
    seq_mean(samples)
}

// ---------------------------------------------------------------------------
// bench protocol
// ---------------------------------------------------------------------------

fn bench_protocol(args: &BenchArgs, p_args: &ProtocolArgs) -> ExitCode {
    let env_counts = match parse_env_list(&p_args.envs) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench protocol: {e}");
            return ExitCode::from(1);
        }
    };

    for &n in &env_counts {
        let total_floats = usize::from(n) * p_args.obs_dim as usize;
        // Synthetic batch payload (deterministic via the splitmix
        // hash of (seed, index)).
        let seed = args.seed.unwrap_or(0xDEAD_BEEF);
        let data: Vec<f32> = (0..total_floats)
            .map(|i| {
                let h = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u64);
                f32::from_bits((h & 0x7FFF_FFFF) as u32 | 0x3F00_0000) - 1.0
            })
            .collect();

        // Warmup
        for _ in 0..args.warmup_runs {
            let _ = binary_frame::encode_batch_f32(u32::from(n), p_args.obs_dim, &data);
            let _ = serde_json::to_vec(&data).expect("serialise");
        }

        let mut per_run_wall_ms = Vec::with_capacity(args.runs as usize);
        let mut per_run_bin_sps = Vec::with_capacity(args.runs as usize);
        let mut per_run_json_sps = Vec::with_capacity(args.runs as usize);
        let mut step_durs: Vec<Duration> = Vec::new();

        for _ in 0..args.runs {
            // Binary
            let bin_start = Instant::now();
            for _ in 0..p_args.batches {
                let s = Instant::now();
                let _ = binary_frame::encode_batch_f32(u32::from(n), p_args.obs_dim, &data);
                step_durs.push(s.elapsed());
            }
            let bin_wall = bin_start.elapsed();

            // JSON
            let json_start = Instant::now();
            for _ in 0..p_args.batches {
                let _ = serde_json::to_vec(&data).expect("serialise");
            }
            let json_wall = json_start.elapsed();

            let bin_wall_ms = bin_wall.as_secs_f64() * 1000.0;
            let json_wall_ms = json_wall.as_secs_f64() * 1000.0;
            per_run_wall_ms.push(bin_wall_ms + json_wall_ms);
            per_run_bin_sps.push(if bin_wall_ms > 0.0 {
                f64::from(p_args.batches) / bin_wall_ms * 1000.0
            } else {
                0.0
            });
            per_run_json_sps.push(if json_wall_ms > 0.0 {
                f64::from(p_args.batches) / json_wall_ms * 1000.0
            } else {
                0.0
            });
        }

        let bin_sps = seq_mean(&per_run_bin_sps);
        let json_sps = seq_mean(&per_run_json_sps);
        let throughput_ratio = if json_sps > 0.0 {
            bin_sps / json_sps
        } else {
            0.0
        };
        let total_steps = args.runs * p_args.batches;

        let row = aggregate_v2(
            "protocol_binary_vs_json",
            args,
            total_steps,
            &per_run_wall_ms,
            &per_run_bin_sps,
            &mut step_durs,
            u32::from(n),
            0,
            throughput_ratio,
            &format!(
                "kind=protocol;obs_dim={};binary_sps={:.0};json_sps={:.0}",
                p_args.obs_dim, bin_sps, json_sps
            ),
        );

        if let Some(path) = args.csv.as_ref()
            && let Err(e) = write_csv_row_v2(path, &row)
        {
            eprintln!("bench protocol: failed to write CSV: {e}");
            return ExitCode::from(1);
        }

        if args.json {
            if let Ok(s) = serde_json::to_string(&row) {
                println!("{s}");
            }
        } else if args.csv.is_none() {
            print_human_v2(&row);
        }
    }

    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// bench record
// ---------------------------------------------------------------------------

fn parse_buffer_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid buffer capacity '{t}': {e}"))
        })
        .collect()
}

fn bench_record(args: &BenchArgs, r_args: &RecordBenchArgs) -> ExitCode {
    let buffers = match parse_buffer_list(&r_args.buffers) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench record: {e}");
            return ExitCode::from(1);
        }
    };

    // Cell 1: sync mode.
    if let Err(code) = run_record_cell(args, r_args, None) {
        return code;
    }

    // Cells 2..N: async with each requested capacity.
    for &cap in &buffers {
        if let Err(code) = run_record_cell(args, r_args, Some(cap)) {
            return code;
        }
    }

    ExitCode::SUCCESS
}

fn run_record_cell(
    args: &BenchArgs,
    r_args: &RecordBenchArgs,
    async_capacity: Option<usize>,
) -> Result<(), ExitCode> {
    let mode_label = async_capacity.map_or_else(|| "sync".to_string(), |c| format!("async@{c}"));

    let mut per_run_wall_ms = Vec::with_capacity(args.runs as usize);
    let mut per_run_sps = Vec::with_capacity(args.runs as usize);
    let mut step_durs: Vec<Duration> = Vec::new();
    let mut last_dropped: u64 = 0;

    // Warmup
    for _ in 0..args.warmup_runs {
        let _ = drive_record_run(r_args, async_capacity);
    }

    for _ in 0..args.runs {
        let (wall, per_frame_durs, dropped) = drive_record_run(r_args, async_capacity);
        last_dropped = dropped;
        let wall_ms = wall.as_secs_f64() * 1000.0;
        per_run_wall_ms.push(wall_ms);
        let sps = if wall_ms > 0.0 {
            f64::from(r_args.frames) / wall_ms * 1000.0
        } else {
            0.0
        };
        per_run_sps.push(sps);
        step_durs.extend(per_frame_durs);
    }

    let total_steps = args.runs * r_args.frames;
    let row = aggregate_v2(
        &format!("record_{mode_label}"),
        args,
        total_steps,
        &per_run_wall_ms,
        &per_run_sps,
        &mut step_durs,
        0,
        last_dropped,
        0.0,
        &format!(
            "kind=record;mode={mode_label};joints={};frames={}",
            r_args.joints, r_args.frames
        ),
    );

    if let Some(path) = args.csv.as_ref()
        && let Err(e) = write_csv_row_v2(path, &row)
    {
        eprintln!("bench record: failed to write CSV: {e}");
        return Err(ExitCode::from(1));
    }
    if args.json {
        if let Ok(s) = serde_json::to_string(&row) {
            println!("{s}");
        }
    } else if args.csv.is_none() {
        print_human_v2(&row);
    }
    Ok(())
}

fn drive_record_run(
    r_args: &RecordBenchArgs,
    async_capacity: Option<usize>,
) -> (Duration, Vec<Duration>, u64) {
    // Synthetic frame payload — 8 joints by default. We bypass the
    // Bevy app and exercise the recorder hot path directly to keep
    // the measurement deterministic and free of scheduler noise.
    let tmp = std::env::temp_dir().join(format!(
        "clankers_bench_record_{}_{}.mcap",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    let cfg = RecordingConfig {
        output_path: tmp.clone(),
        record_joints: true,
        record_actions: false,
        record_rewards: false,
        record_body_poses: false,
        async_mode: async_capacity.is_some(),
        async_buffer_capacity: async_capacity.unwrap_or(256),
    };

    // Spin up a minimal Bevy app to drive setup_channels (which
    // registers the manifest + joint_states channels and installs
    // async if requested). After the first update the channels are
    // live; we then bypass the per-system writers and call
    // `Recorder::write_joint_frame` directly per frame.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(SimTime::new());
    app.insert_resource(cfg);
    app.add_plugins(RecorderPlugin);
    app.finish();
    app.cleanup();
    app.update();

    let channel_id = app
        .world()
        .resource::<clankers_record::recorder::ChannelIds>()
        .joints
        .expect("/joint_states channel registered");

    let names: Vec<String> = (0..r_args.joints).map(|i| format!("j{i}")).collect();
    let positions = vec![0.0_f32; r_args.joints];
    let velocities = vec![0.0_f32; r_args.joints];
    let torques = vec![0.0_f32; r_args.joints];
    let frames = r_args.frames;

    let wall_start = Instant::now();
    let mut per_frame_durs: Vec<Duration> = Vec::with_capacity(frames as usize);

    {
        // NonSend: we have to scope our access so we drop the borrow
        // before reading `DroppedFrames`.
        let world = app.world_mut();
        for i in 0..frames {
            let frame = JointFrame {
                timestamp_ns: u64::from(i) * 1_000_000,
                names: names.clone(),
                positions: positions.clone(),
                velocities: velocities.clone(),
                torques: torques.clone(),
            };
            let s = Instant::now();
            if let Some(mut rec) =
                world.get_non_send_resource_mut::<clankers_record::recorder::Recorder>()
            {
                let _ = rec.write_joint_frame(channel_id, &frame);
            }
            per_frame_durs.push(s.elapsed());
        }
    }
    let wall = wall_start.elapsed();

    let dropped = app.world().resource::<DroppedFrames>().get();

    // Drop the app (and the recorder) so the MCAP file finalises and
    // the worker thread joins before we measure.
    drop(app);
    // Best-effort cleanup; ignore failure (Windows may still hold a
    // handle briefly).
    let _ = std::fs::remove_file(&tmp);

    (wall, per_frame_durs, dropped)
}

// ---------------------------------------------------------------------------
// bench mpc
// ---------------------------------------------------------------------------

fn bench_mpc(args: &BenchArgs, _m_args: &MpcArgs) -> ExitCode {
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

// ---------------------------------------------------------------------------
// V2 aggregation + IO + pretty-print
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn aggregate_v2(
    scenario_name: &str,
    args: &BenchArgs,
    total_steps: u32,
    per_run_wall_ms: &[f64],
    per_run_sps: &[f64],
    step_durs: &mut [Duration],
    num_envs: u32,
    dropped_frames: u64,
    throughput_x: f64,
    notes_suffix: &str,
) -> BenchRowV2 {
    let wall_ms_total: f64 = per_run_wall_ms.iter().sum();
    let wall_ms_per_run_mean = if args.runs > 0 {
        wall_ms_total / f64::from(args.runs)
    } else {
        0.0
    };
    let wall_ms_per_run_std = stddev(per_run_wall_ms);
    let steps_per_sec_mean = seq_mean(per_run_sps);
    let step_us_p50 = percentile_us(&mut step_durs.to_vec(), 0.50);
    let step_us_p95 = percentile_us(&mut step_durs.to_vec(), 0.95);
    let step_us_p99 = percentile_us(step_durs, 0.99);

    let base_notes = build_notes(args);
    let notes = if notes_suffix.is_empty() {
        base_notes
    } else {
        format!("{base_notes};{notes_suffix}")
    };

    BenchRowV2 {
        scenario: scenario_name.to_owned(),
        steps: total_steps,
        runs: args.runs,
        max_steps: args.max_steps,
        wall_ms_total,
        wall_ms_per_run_mean,
        wall_ms_per_run_std,
        steps_per_sec_mean,
        step_us_p50,
        step_us_p99,
        notes,
        num_envs,
        p95_us: step_us_p95,
        dropped_frames,
        throughput_x,
    }
}

fn write_csv_row_v1(path: &Path, row: &BenchRow) -> std::io::Result<()> {
    write_csv_row_inner(path, row, CSV_HEADER)
}

fn write_csv_row_v2(path: &Path, row: &BenchRowV2) -> std::io::Result<()> {
    write_csv_row_inner(path, row, CSV_HEADER_V2)
}

fn write_csv_row_inner<T: Serialize>(path: &Path, row: &T, header: &str) -> std::io::Result<()> {
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
        wtr.write_record(header.split(','))?;
    }
    wtr.serialize(row)?;
    wtr.flush()?;
    Ok(())
}

fn print_human_v1(row: &BenchRow) {
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

fn print_human_v2(row: &BenchRowV2) {
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
    println!("  num_envs:             {}", row.num_envs);
    println!("  p95_us:               {:.2}", row.p95_us);
    println!("  dropped_frames:       {}", row.dropped_frames);
    println!("  throughput_x:         {:.3}", row.throughput_x);
    println!("  notes:                {}", row.notes);
}

// ---------------------------------------------------------------------------
// Shared helpers (V1 + V2)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Public re-exports for benches/vec.rs Criterion target
// ---------------------------------------------------------------------------

/// Public entry point used by the Criterion target `benches/vec.rs` so
/// `cargo bench --bench vec` numbers match `clankers bench vec`.
///
/// Runs `runs` measurement runs of `max_steps` `step_all` calls on
/// `n_envs` `ConstBenchEnv` instances, returning the mean steps-per-sec.
///
/// `work_us` is the per-`step()` busy-loop duration in microseconds. Pass
/// `0` for the overhead-floor measurement (matches the
/// `vec_step_all` Criterion group + `vec_parallel` CLI scenario). Pass a
/// positive value (e.g. `100`) to mirror the `vec_throughput_{work_us}us`
/// scenario where parallelism can actually amortise rayon's dispatch
/// overhead.
///
/// `#[allow(dead_code)]` because clankers-app is a binary crate; the
/// bench target is the only external consumer and Cargo doesn't see
/// the cross-target use as "live" from the main.rs perspective.
#[must_use]
#[allow(dead_code)]
pub fn bench_vec_cell(n: u16, runs: u32, max_steps: u32, parallel: bool, work_us: u32) -> f64 {
    let (_, sps, _) = run_vec_cell(n, 4, runs, max_steps, 0, parallel, work_us);
    seq_mean(&sps)
}

// ---------------------------------------------------------------------------
// Ratio-gate helper
// ---------------------------------------------------------------------------

/// Outcome of a ratio-gate check.
///
/// `Ok((n, x))` — gate passed (or no rows to gate on; returns sentinel
/// `(0, 0.0)`). `Err(msg)` — gate failed; `msg` is the human-readable
/// failure reason already formatted for stderr (starts with `RATIO GATE: FAIL`).
/// The caller is responsible for emitting the `RATIO GATE: PASS ...` line
/// on success and choosing the exit code.
///
/// `rows` is a list of `(num_envs, throughput_x)` pairs in the order the
/// bench loop produced them. The gate prefers `N=8` and falls back to the
/// highest-N row available, logging a warning to stderr in the fallback case.
fn check_ratio_gate(rows: &[(u16, f64)], gate: f64) -> Result<(u16, f64), String> {
    if rows.is_empty() {
        eprintln!("RATIO GATE: no rows; gate skipped");
        return Ok((0, 0.0));
    }

    let (n, x) = if let Some(&row) = rows.iter().find(|(n, _)| *n == 8) {
        row
    } else {
        let &row = rows
            .iter()
            .max_by_key(|(n, _)| *n)
            .expect("rows non-empty (checked above)");
        eprintln!(
            "RATIO GATE: --envs did not include 8; gating on N={} instead",
            row.0
        );
        row
    };

    if x >= gate {
        Ok((n, x))
    } else {
        Err(format!(
            "RATIO GATE: FAIL N={n} throughput_x={x:.3} < {gate:.3}"
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_bench_env_work_us_zero_is_fast() {
        let mut env = ConstBenchEnv {
            obs_dim: 4,
            work_us: 0,
        };
        let action = Action::zeros(1);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = env.step(&action);
        }
        // 100 steps × ~50ns ≈ 5µs. Allow 5ms for CI noise.
        assert!(start.elapsed() < std::time::Duration::from_millis(5));
    }

    #[test]
    fn const_bench_env_work_us_busy_loops_for_at_least_target() {
        let mut env = ConstBenchEnv {
            obs_dim: 4,
            work_us: 500,
        };
        let action = Action::zeros(1);
        let start = std::time::Instant::now();
        let _ = env.step(&action);
        // 500µs target. Allow some slop downward (timer granularity on
        // Windows) but require >= 250µs to confirm the loop ran.
        assert!(
            start.elapsed() >= std::time::Duration::from_micros(250),
            "busy-loop returned too fast: {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn csv_header_v1_field_count_unchanged() {
        // 11 columns LOCKED. If this fails we broke the W5 PR4 contract.
        let cols = CSV_HEADER.split(',').count();
        assert_eq!(cols, 11, "schema lock: 11 columns");
    }

    #[test]
    fn csv_header_v1_column_names_locked() {
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
    fn csv_header_v2_extends_v1_by_four_columns() {
        let v1: Vec<&str> = CSV_HEADER.split(',').collect();
        let v2: Vec<&str> = CSV_HEADER_V2.split(',').collect();
        assert_eq!(v2.len(), v1.len() + 4);
        assert_eq!(&v2[..v1.len()], v1.as_slice());
        assert_eq!(
            &v2[v1.len()..],
            ["num_envs", "p95_us", "dropped_frames", "throughput_x"].as_slice(),
        );
    }

    #[test]
    fn percentile_us_returns_zero_for_empty() {
        let mut samples: Vec<Duration> = Vec::new();
        assert!(percentile_us(&mut samples, 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_us_median_of_known_samples() {
        let mut samples: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        assert!((percentile_us(&mut samples, 0.5) - 51.0).abs() < 0.01);
        let mut samples2: Vec<Duration> = (1_u64..=100).map(Duration::from_micros).collect();
        assert!((percentile_us(&mut samples2, 0.99) - 100.0).abs() < 0.01);
    }

    #[test]
    fn stddev_zero_for_one_sample() {
        assert!(stddev(&[42.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn stddev_bessel_corrected() {
        let xs = [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = stddev(&xs);
        assert!((s - 2.138_089_935_2).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn parse_env_list_basic() {
        assert_eq!(parse_env_list("1,2,4,8").unwrap(), vec![1, 2, 4, 8]);
        assert_eq!(parse_env_list(" 16 ").unwrap(), vec![16]);
        assert!(parse_env_list("1,abc,2").is_err());
    }

    #[test]
    fn bench_vec_cell_produces_positive_throughput() {
        let sps = bench_vec_cell(2, 1, 100, false, 0);
        assert!(sps > 0.0, "expected positive sps, got {sps}");
    }

    #[test]
    fn check_ratio_gate_empty_rows_passes_with_warning() {
        // Opt-in gate; if the bench produced no rows (e.g. degenerate
        // --envs), don't fail CI — that's a configuration problem,
        // not a regression. The helper returns Ok with sentinel (0, 0.0).
        let result = check_ratio_gate(&[], 2.0);
        assert!(result.is_ok(), "empty rows should not fail the gate");
    }

    #[test]
    fn check_ratio_gate_passes_when_ratio_meets_floor() {
        // N=8 row with throughput_x = 4.5 vs gate K=2.0 -> pass.
        let rows = vec![(1u16, 1.0_f64), (8u16, 4.5_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_ok());
        let (n, x) = result.unwrap();
        assert_eq!(n, 8);
        assert!((x - 4.5).abs() < 1e-9);
    }

    #[test]
    fn check_ratio_gate_fails_with_specific_message_when_below_floor() {
        // N=8 row with throughput_x = 0.5 vs gate K=2.0 -> fail.
        let rows = vec![(1u16, 1.0_f64), (8u16, 0.5_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        // The message MUST mention both the observed value and the
        // gate, so the CI log tells the on-call engineer why it failed.
        assert!(msg.contains("0.5") || msg.contains("0.500"), "msg={msg}");
        assert!(msg.contains("2.0") || msg.contains("2.000"), "msg={msg}");
        assert!(msg.to_uppercase().contains("FAIL"), "msg={msg}");
    }

    #[test]
    fn check_ratio_gate_falls_back_to_highest_n_when_8_missing() {
        // --envs 1,2,4 -> no N=8 row. Gate on N=4 instead, passing if
        // that row meets the floor.
        let rows = vec![(1u16, 1.0_f64), (2u16, 1.8_f64), (4u16, 3.2_f64)];
        let result = check_ratio_gate(&rows, 2.0);
        assert!(result.is_ok());
        let (n, x) = result.unwrap();
        assert_eq!(n, 4, "fallback should pick highest-N row");
        assert!((x - 3.2).abs() < 1e-9);
    }
}
