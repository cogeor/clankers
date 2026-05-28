//! Clap CLI types for `clankers-app bench`.
//!
//! The subcommand tree (`kind`) is optional — when absent, the legacy
//! W5 PR4 single-scenario surface runs (see [`super::scenario`]).

use std::path::PathBuf;

use clap::{Args, Subcommand};

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
