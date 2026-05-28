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
//!
//! # Module layout
//!
//! This module is split by concern:
//!
//! - [`args`] — Clap CLI types (`BenchArgs`, `BenchKind`, per-subcommand
//!   arg structs).
//! - [`csv`] — CSV schema constants, row types, header-locked writers,
//!   and human-readable pretty-printers.
//! - [`stats`] — percentile/stddev/aggregation helpers + `build_notes`.
//! - [`scenario`] — legacy single-scenario benchmark body (V1 schema).
//! - [`mod@vec`] — `bench vec` body, `ConstBenchEnv`, `run_vec_cell`, and
//!   the public [`bench_vec_cell`] entry point used by the Criterion
//!   target `benches/vec.rs`.
//! - [`protocol`] — `bench protocol` body.
//! - [`record`] — `bench record` body.
//! - [`mpc`] — `bench mpc` body.
//! - [`gate`] — ratio-gate logic shared between `bench vec` and CI.

use std::process::ExitCode;

mod args;
mod csv;
mod gate;
mod mpc;
mod protocol;
mod record;
mod scenario;
mod stats;
mod vec;

pub use args::{BenchArgs, BenchKind};
// `clankers-app` is a binary crate so the lint can't see external consumers
// of these re-exports (the Criterion target at `benches/vec.rs` mirrors the
// `bench_vec_cell` shape but doesn't `extern crate` it). Keep the re-exports
// `pub` so the module path stays stable for any future direct consumer.
#[allow(unused_imports)]
pub use args::{MpcArgs, ProtocolArgs, RecordBenchArgs, VecArgs};
#[allow(unused_imports)]
pub use csv::{CSV_HEADER, CSV_HEADER_V2};
#[allow(unused_imports)]
pub use vec::bench_vec_cell;

/// Execute `clankers-app bench`. Dispatches on `args.kind`.
pub fn execute(args: &BenchArgs) -> ExitCode {
    match &args.kind {
        None | Some(BenchKind::Scenario) => scenario::execute(args),
        Some(BenchKind::Vec(a)) => vec::execute(args, a),
        Some(BenchKind::Protocol(a)) => protocol::execute(args, a),
        Some(BenchKind::Record(a)) => record::execute(args, a),
        Some(BenchKind::Mpc(a)) => mpc::execute(args, a),
    }
}
