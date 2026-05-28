//! Packaging / release boundaries.
//!
//! The workspace is published as `clankers-core`, `clankers-physics`,
//! `clankers-gym`, ... + the Python `clankers` and `clankers_synthetic`
//! packages, but there's no committed list of which crates / packages
//! are released, which are internal-only, and what the release
//! checklist actually is.
//!
//! This module documents the packaging contract and provides a typed
//! release-checklist data structure so CI / the release-bot can fail
//! loudly when a step is missed.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ReleaseChannel
// ---------------------------------------------------------------------------

/// Where a workspace artifact is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseChannel {
    /// crates.io.
    CratesIo,
    /// `PyPI`.
    PyPi,
    /// Internal-only — not published, but referenced by other
    /// workspace crates.
    Internal,
}

// ---------------------------------------------------------------------------
// ReleaseArtifact
// ---------------------------------------------------------------------------

/// One published artifact (crate or Python package).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseArtifact {
    /// Package name as published.
    pub name: String,
    /// Channel.
    pub channel: ReleaseChannel,
    /// One-line description of what the package contains.
    pub purpose: String,
}

/// The canonical list of release artifacts.
///
/// CI compares the workspace's actual package list against this; new
/// crates / Python packages must append here before release. Removals
/// require a major bump on every dependent.
#[must_use]
pub fn release_artifact_table() -> Vec<ReleaseArtifact> {
    use ReleaseChannel::{CratesIo, Internal, PyPi};
    let a = |name: &str, channel, purpose: &str| ReleaseArtifact {
        name: name.to_string(),
        channel,
        purpose: purpose.to_string(),
    };
    vec![
        // Rust crates published to crates.io.
        a(
            "clankers-core",
            CratesIo,
            "Types, traits, contracts, manifests. The 'spine' crate.",
        ),
        a(
            "clankers-physics",
            CratesIo,
            "Rapier-backed physics behind engine-neutral APIs.",
        ),
        a(
            "clankers-actuator",
            CratesIo,
            "Joint motor / PID actuator components.",
        ),
        a("clankers-urdf", CratesIo, "URDF parser + spawner."),
        a("clankers-env", CratesIo, "Episode + observation buffer."),
        a("clankers-gym", CratesIo, "Gym TCP server + protocol."),
        a(
            "clankers-record",
            CratesIo,
            "MCAP recorders (sync + async).",
        ),
        a(
            "clankers-sim",
            CratesIo,
            "Scenarios + scene builder. Re-exports the meta-plugin.",
        ),
        a(
            "clankers-domain-rand",
            CratesIo,
            "Domain randomisation primitives.",
        ),
        a(
            "clankers-render",
            CratesIo,
            "Bevy render plugin + cosmos pipeline.",
        ),
        a("clankers-teleop", CratesIo, "Teleop input handling."),
        a(
            "clankers-policy",
            CratesIo,
            "Policy applicator + ONNX loader.",
        ),
        a(
            "clankers-mpc",
            CratesIo,
            "Quadruped MPC reference controller.",
        ),
        // Internal-only crates (used inside the workspace; not published).
        a(
            "clankers-test-utils",
            Internal,
            "Test fixtures and helpers. Not published to avoid pulling \
             test infra into downstream graphs.",
        ),
        a(
            "clankers-gym-fixtures",
            Internal,
            "Gym integration test fixtures.",
        ),
        a("clankers-app", Internal, "CLI binary; ships as `clankers`."),
        a("xtask", Internal, "Workspace dev / release tooling."),
        // Python packages.
        a(
            "clankers",
            PyPi,
            "Gymnasium-compatible client + recorder + dataset tooling.",
        ),
        a(
            "clankers_synthetic",
            PyPi,
            "Synthetic data pipeline (planner + compiler + parser).",
        ),
    ]
}

// ---------------------------------------------------------------------------
// ReleaseChecklist
// ---------------------------------------------------------------------------

/// One step in the release checklist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChecklistStep {
    /// Short id used as the CI step key.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this step must pass for the release to proceed
    /// (vs. advisory).
    pub required: bool,
}

/// The canonical release checklist.
#[must_use]
pub fn release_checklist() -> Vec<ChecklistStep> {
    let s = |id: &str, description: &str, required| ChecklistStep {
        id: id.to_string(),
        description: description.to_string(),
        required,
    };
    vec![
        s("fmt", "cargo fmt --all + ruff format passed.", true),
        s(
            "clippy",
            "cargo clippy -- -D warnings on the full workspace.",
            true,
        ),
        s(
            "test",
            "cargo test -j 8 --workspace AND pytest -q passed.",
            true,
        ),
        s(
            "manifest_schema",
            "RunManifest::is_compatible holds for every checked-in sample.",
            true,
        ),
        s(
            "baseline_schema",
            "Every baselines/<task>/expected_metrics.json parses with the \
             current ExpectedMetrics::from_json.",
            true,
        ),
        s(
            "stability_table",
            "canonical_tier_table() covers every public module + has no \
             duplicates.",
            true,
        ),
        s(
            "release_artifact_table",
            "release_artifact_table() matches the workspace's actual \
             publishable crates.",
            true,
        ),
        s(
            "changelog",
            "CHANGELOG.md updated for every published crate.",
            false,
        ),
        s(
            "version_bump",
            "Workspace version bumped, Cargo.lock regenerated.",
            true,
        ),
        s(
            "python_pyproject",
            "python/pyproject.toml + python/clankers_synthetic/pyproject.toml \
             versions aligned with the Rust workspace.",
            true,
        ),
        s(
            "smoke_runs",
            "examples/quadruped_mpc + cartpole smoke runs pass under \
             release-mode builds.",
            true,
        ),
        s(
            "manifest_stamped",
            "CI captures a RunManifest sidecar for each smoke run; failures \
             surface RecorderHealth.error_count > 0.",
            false,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn artifact_names_are_unique() {
        let table = release_artifact_table();
        let mut seen: HashSet<&str> = HashSet::new();
        for entry in &table {
            assert!(
                seen.insert(entry.name.as_str()),
                "duplicate artifact entry for {}",
                entry.name
            );
        }
    }

    #[test]
    fn each_channel_has_at_least_one_entry() {
        let table = release_artifact_table();
        let mut crates_io = 0usize;
        let mut pypi = 0usize;
        let mut internal = 0usize;
        for entry in &table {
            match entry.channel {
                ReleaseChannel::CratesIo => crates_io += 1,
                ReleaseChannel::PyPi => pypi += 1,
                ReleaseChannel::Internal => internal += 1,
            }
        }
        assert!(crates_io > 0);
        assert!(pypi > 0);
        assert!(internal > 0);
    }

    #[test]
    fn checklist_step_ids_are_unique() {
        let list = release_checklist();
        let ids: HashSet<&str> = list.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids.len(), list.len());
    }

    #[test]
    fn at_least_one_required_step() {
        let list = release_checklist();
        let required = list.iter().filter(|s| s.required).count();
        assert!(required > 5, "release checklist should be substantive");
    }
}
