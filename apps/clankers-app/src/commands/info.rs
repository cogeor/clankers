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

/// Print workspace metadata. Returns `SUCCESS` unconditionally.
pub fn execute(json: bool) -> ExitCode {
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

    if json {
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
