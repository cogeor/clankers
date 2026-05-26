//! `clankers-app info` — workspace metadata.
//!
//! Emits a fixed-shape JSON document or a human-readable summary
//! describing the binary version, protocol version, build profile,
//! workspace crates, and the set of registered scenarios.
//!
//! # STABLE CONTRACT
//!
//! The JSON schema below is a stable contract for the duration of the
//! W5 workstream (PR1–PR4). Adding keys is allowed; removing or
//! renaming keys requires a minor-version bump on `clankers-app` and a
//! CHANGELOG entry. Downstream consumers (`python/` clients, CI checks,
//! `jq` pipelines) may rely on this shape.
//!
//! ```json
//! {
//!   "version": "0.1.0",
//!   "protocol_version": "1.1.0",
//!   "build_profile": "debug",
//!   "edition": "2024",
//!   "crates": [{"name": "...", "version": "..."}, ...],
//!   "scenarios": []
//! }
//! ```

use std::process::ExitCode;

use clankers_sim::{ScenarioRegistry, scenarios::register_builtin};
use clap::Args;
use serde::Serialize;

const CRATES_HAND_CURATED: &[&str] = &[
    "clankers-core",
    "clankers-sim",
    "clankers-env",
    "clankers-gym",
    "clankers-urdf",
    "clankers-record",
    "clankers-physics",
    "clankers-viz",
];

/// Flags for `clankers-app info`.
#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Emit structured JSON to stdout.
    #[arg(long)]
    pub json: bool,
    /// W7 PR4 diagnostic: print a JSON object describing the recorder
    /// dropped-frame state.
    ///
    /// **Loop-6 plan-deviation:** the printed value is always 0 because
    /// reading a live `DroppedFrames` counter requires cross-process
    /// IPC out of scope for this loop. The flag plumbing exists so the
    /// end-to-end shape is stable; W8 will tighten the value semantics
    /// when the recorder grows a status socket. Tests assert only the
    /// JSON shape, not the numeric value.
    #[arg(long)]
    pub record_stats: bool,
}

#[derive(Serialize)]
struct CrateEntry {
    name: &'static str,
    version: &'static str,
}

#[derive(Serialize)]
struct InfoOutput {
    version: &'static str,
    protocol_version: &'static str,
    build_profile: &'static str,
    edition: &'static str,
    crates: Vec<CrateEntry>,
    scenarios: Vec<&'static str>,
}

#[derive(Serialize)]
struct RecordStatsOutput {
    dropped_frames: u64,
}

/// Print workspace metadata. Returns `SUCCESS` unconditionally.
pub fn execute(args: &InfoArgs) -> ExitCode {
    if args.record_stats {
        // Plan-deviation: reading a live recorder counter requires
        // IPC. For W7 PR4 we surface the JSON shape with a static 0.
        let stats = RecordStatsOutput { dropped_frames: 0 };
        let s = serde_json::to_string(&stats).expect("RecordStatsOutput serialisable");
        println!("{s}");
        return ExitCode::SUCCESS;
    }

    let mut registry = ScenarioRegistry::new();
    register_builtin(&mut registry);
    let scenarios = registry.list_builtin();

    let out = InfoOutput {
        version: env!("CARGO_PKG_VERSION"),
        protocol_version: clankers_gym::protocol::PROTOCOL_VERSION,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        edition: "2024",
        // All workspace crates share `version.workspace = true`, so the
        // app's `CARGO_PKG_VERSION` is correct for each entry. List is
        // hand-curated — see CHANGELOG for the maintenance contract.
        crates: CRATES_HAND_CURATED
            .iter()
            .map(|name| CrateEntry {
                name,
                version: env!("CARGO_PKG_VERSION"),
            })
            .collect(),
        scenarios,
    };

    if args.json {
        let s = serde_json::to_string_pretty(&out).expect("InfoOutput is always serializable");
        println!("{s}");
    } else {
        println!("clankers v{}", out.version);
        println!("protocol: {}", out.protocol_version);
        println!("build profile: {}", out.build_profile);
        println!("edition: {}", out.edition);
        println!();
        println!("crates:");
        for c in &out.crates {
            println!("  {:<20} {}", c.name, c.version);
        }
        println!();
        if out.scenarios.is_empty() {
            println!("scenarios: (none registered — see W5 PR2)");
        } else {
            println!("scenarios:");
            for name in &out.scenarios {
                println!("  {name}");
            }
        }
    }

    ExitCode::SUCCESS
}
