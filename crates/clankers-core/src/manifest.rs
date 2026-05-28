//! Run manifest — stamped metadata for every recorded artifact (G3).
//!
//! CODE_QUALITY_REVIEW § "Gap 3: No Manifest / Provenance Layer".
//! Currently MCAP traces and CSV baselines have ad-hoc, partial
//! metadata. Without a single canonical schema, downstream consumers
//! (Python evaluators, baseline comparators, training reproducibility
//! tooling) can't validate that the artifact came from the contract
//! they think it did.
//!
//! [`RunManifest`] is the schema. Every recorded artifact — MCAP trace,
//! CSV benchmark row, baseline blob — should carry a serialisable
//! `RunManifest` next to it. CLI surfaces (`clankers inspect`,
//! `clankers validate`, `clankers compare`) operate on this schema
//! so callers never reach for the raw protobuf / row format.
//!
//! ## Stability
//!
//! Bumped via [`MANIFEST_SCHEMA_VERSION`]. Decoders refuse manifests
//! whose major version doesn't match — silently coercing across versions
//! is exactly the contract-erosion the review called out.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::env_spec::EnvId;

/// Major schema version. Bump on breaking field changes; never reuse
/// an old value. Minor additions / additive enum variants don't bump
/// this — they live behind `serde(default)` fields.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// SoftwareVersions
// ---------------------------------------------------------------------------

/// Versions of the producing software stack.
///
/// Includes the clankers workspace version, the rapier3d version, and
/// any Python / training stack identifiers the producer cares to stamp.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SoftwareVersions {
    /// Workspace cargo version (`CARGO_PKG_VERSION`) of the producer.
    pub clankers: String,
    /// Concrete physics backend name + version.
    pub physics_backend: String,
    /// Free-form additional component versions (e.g. python client,
    /// `stable-baselines3`, custom model id).
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// SeedInfo
// ---------------------------------------------------------------------------

/// Seed material that fixed the run.
///
/// `master_seed` is the user-supplied seed; per-env seeds are derived
/// from it via `SeedHierarchy`. We record both so reproducibility checks
/// don't have to re-derive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedInfo {
    /// User-visible seed for the run.
    pub master_seed: u64,
    /// Per-env seeds in env-index order, if the run was vectorised.
    #[serde(default)]
    pub per_env_seeds: Vec<u64>,
}

// ---------------------------------------------------------------------------
// RunManifest
// ---------------------------------------------------------------------------

/// Canonical metadata schema for every recorded artifact.
///
/// Stored alongside MCAP traces (as a sidecar JSON), embedded in CSV
/// baseline blobs (as a header row), and stamped into the run summary
/// the CLI prints. The schema is the single source of truth for what
/// produced a given dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// Manifest schema version. Always [`MANIFEST_SCHEMA_VERSION`] at
    /// produce time; verified by [`Self::is_compatible`] at read time.
    pub schema_version: u32,
    /// Stable id of the env / task that produced the run.
    pub env_id: EnvId,
    /// ISO-8601 UTC timestamp the run started.
    pub started_at: String,
    /// Wall-clock duration in milliseconds.
    pub wall_duration_ms: u64,
    /// Number of environments stepped in parallel.
    pub num_envs: u32,
    /// Total step count across all envs.
    pub total_steps: u64,
    /// Episodes completed across all envs.
    pub episodes_completed: u64,
    /// Software versions.
    pub software: SoftwareVersions,
    /// Seed information.
    pub seeds: SeedInfo,
    /// Hash of the bound `EnvSpec`. Stamped so consumers can detect
    /// silent task changes between artifacts ("looks like cartpole but
    /// the reward function changed").
    pub env_spec_hash: String,
    /// Free-form scalar metrics the producer wants to surface (e.g.
    /// mean reward, mean episode length).
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    /// Free-form tags (e.g. "baseline", "ci", "smoke").
    #[serde(default)]
    pub tags: Vec<String>,
}

impl RunManifest {
    /// Whether `self` is compatible with the current
    /// [`MANIFEST_SCHEMA_VERSION`]. We allow exact-major matches only.
    #[must_use]
    pub const fn is_compatible(&self) -> bool {
        self.schema_version == MANIFEST_SCHEMA_VERSION
    }

    /// Convenience: deserialise from JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError::Parse`] on JSON syntax errors or
    /// [`ManifestError::UnsupportedSchemaVersion`] when the schema
    /// version disagrees.
    pub fn from_json(s: &str) -> Result<Self, ManifestError> {
        let m: Self = serde_json::from_str(s).map_err(|e| ManifestError::Parse(e.to_string()))?;
        if !m.is_compatible() {
            return Err(ManifestError::UnsupportedSchemaVersion {
                got: m.schema_version,
                expected: MANIFEST_SCHEMA_VERSION,
            });
        }
        Ok(m)
    }

    /// Serialise to JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError::Parse`] if `serde_json::to_string` fails.
    /// This is effectively impossible for the current schema (no
    /// non-`Serialize` types), but the typed-error contract stays.
    pub fn to_json(&self) -> Result<String, ManifestError> {
        serde_json::to_string_pretty(self).map_err(|e| ManifestError::Parse(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// ManifestError
// ---------------------------------------------------------------------------

/// Failure modes for manifest decoding.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ManifestError {
    /// JSON parsing failure.
    #[error("manifest parse error: {0}")]
    Parse(String),
    /// Schema-version mismatch with the running stack.
    #[error("unsupported manifest schema version {got} (expected {expected})")]
    UnsupportedSchemaVersion {
        /// Wire value.
        got: u32,
        /// Compiled-in [`MANIFEST_SCHEMA_VERSION`].
        expected: u32,
    },
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

/// Result of comparing two manifests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestComparison {
    /// Whether the env id matched.
    pub env_match: bool,
    /// Whether the env spec hash matched (false ⇒ silent task drift).
    pub env_spec_match: bool,
    /// Whether the software clankers version matched.
    pub clankers_version_match: bool,
    /// Per-metric delta (right - left). Includes only metrics present
    /// on both sides.
    pub metric_deltas: BTreeMap<String, f64>,
    /// Metrics present only on one side.
    pub asymmetric_metrics: Vec<String>,
}

/// Produce a [`ManifestComparison`] between two manifests.
#[must_use]
pub fn compare_manifests(left: &RunManifest, right: &RunManifest) -> ManifestComparison {
    let env_match = left.env_id == right.env_id;
    let env_spec_match = left.env_spec_hash == right.env_spec_hash;
    let clankers_version_match = left.software.clankers == right.software.clankers;

    let mut metric_deltas = BTreeMap::new();
    let mut asymmetric_metrics = Vec::new();
    for (k, lv) in &left.metrics {
        match right.metrics.get(k) {
            Some(rv) => {
                metric_deltas.insert(k.clone(), rv - lv);
            }
            None => asymmetric_metrics.push(k.clone()),
        }
    }
    for k in right.metrics.keys() {
        if !left.metrics.contains_key(k) {
            asymmetric_metrics.push(k.clone());
        }
    }
    asymmetric_metrics.sort();
    asymmetric_metrics.dedup();

    ManifestComparison {
        env_match,
        env_spec_match,
        clankers_version_match,
        metric_deltas,
        asymmetric_metrics,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RunManifest {
        let mut metrics = BTreeMap::new();
        metrics.insert("mean_reward".to_string(), 12.5);
        metrics.insert("mean_episode_length".to_string(), 480.0);
        RunManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            env_id: EnvId::new("clankers", "cartpole_v1"),
            started_at: "2026-05-28T12:00:00Z".to_string(),
            wall_duration_ms: 60_000,
            num_envs: 8,
            total_steps: 100_000,
            episodes_completed: 207,
            software: SoftwareVersions {
                clankers: "0.1.0".to_string(),
                physics_backend: "rapier3d-0.27".to_string(),
                extra: BTreeMap::new(),
            },
            seeds: SeedInfo {
                master_seed: 42,
                per_env_seeds: (0..8).collect(),
            },
            env_spec_hash: "deadbeef".to_string(),
            metrics,
            tags: vec!["ci".to_string(), "smoke".to_string()],
        }
    }

    #[test]
    fn manifest_roundtrips_through_json() {
        let m = sample();
        let s = m.to_json().unwrap();
        let back = RunManifest::from_json(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn from_json_rejects_old_schema_version() {
        let mut bad = sample();
        bad.schema_version = 0;
        let s = serde_json::to_string(&bad).unwrap();
        let err = RunManifest::from_json(&s).unwrap_err();
        assert_eq!(
            err,
            ManifestError::UnsupportedSchemaVersion {
                got: 0,
                expected: MANIFEST_SCHEMA_VERSION,
            }
        );
    }

    #[test]
    fn compare_detects_env_drift() {
        let mut left = sample();
        let mut right = sample();
        right.env_id = EnvId::new("clankers", "cartpole_v2");
        right.env_spec_hash = "beadbeef".to_string();
        let cmp = compare_manifests(&left, &right);
        assert!(!cmp.env_match);
        assert!(!cmp.env_spec_match);
        // Same metrics on both sides; deltas should be zero.
        assert!(cmp.metric_deltas.values().all(|d| d.abs() < 1e-9));
        // No asymmetric metrics.
        assert!(cmp.asymmetric_metrics.is_empty());
        // Sanity: equal manifests compare equal.
        right = left.clone();
        let cmp_eq = compare_manifests(&left, &right);
        assert!(cmp_eq.env_match);
        assert!(cmp_eq.env_spec_match);

        // Asymmetric metric: only on right.
        left.metrics.remove("mean_reward");
        let cmp_asym = compare_manifests(&left, &right);
        assert!(
            cmp_asym
                .asymmetric_metrics
                .contains(&"mean_reward".to_string())
        );
    }

    #[test]
    fn parse_error_propagates() {
        let err = RunManifest::from_json("not valid json").unwrap_err();
        assert!(matches!(err, ManifestError::Parse(_)));
    }
}
