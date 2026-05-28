//! Unified config schema.
//!
//! The workspace has `SimConfig`, `EpisodeConfig`, training-side YAML,
//! and per-binary ad-hoc CLI flags. There's no single document that
//! captures "the full configuration of a run", which makes
//! reproducibility (manifest stamping) hard.
//!
//! [`UnifiedConfig`] is the schema. The CLI's `--config <file>` flag
//! reads this; the resolved config gets stamped into the
//! [`crate::manifest::RunManifest`] so future-you knows exactly which
//! knobs produced the artifact.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::config::SimConfig;
use crate::env_spec::TaskId;

/// Schema-version pin so config files don't silently drift across
/// stack revisions.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Sub-sections
// ---------------------------------------------------------------------------

/// Recorder section of the unified config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecorderConfig {
    /// Enable recording.
    #[serde(default)]
    pub enabled: bool,
    /// Output path (MCAP). `None` lets the consumer pick a default.
    #[serde(default)]
    pub output: Option<String>,
    /// Per-kind subscription filter; empty means "record every kind".
    #[serde(default)]
    pub kind_filter: Vec<String>,
    /// Async-writer queue capacity. Drops past this are surfaced via
    /// `RecorderHealth`.
    #[serde(default = "default_recorder_queue")]
    pub queue_capacity: u32,
}

const fn default_recorder_queue() -> u32 {
    1024
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output: None,
            kind_filter: Vec::new(),
            queue_capacity: default_recorder_queue(),
        }
    }
}

/// Training-side hyperparams.
///
/// The training stack is Python-side; the fields here cover the bits
/// the Rust side needs to know (seed, `num_envs`, parallelism mode).
/// Detailed RL hyperparams live in the `extra` map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Master seed for the run (per-env seeds derived via [`crate::seed::SeedHierarchy`]).
    #[serde(default)]
    pub master_seed: u64,
    /// Number of envs to run in parallel.
    #[serde(default = "default_num_envs")]
    pub num_envs: u32,
    /// Vec-env mode ("auto", "sequential", "parallel").
    #[serde(default = "default_vec_mode")]
    pub vec_mode: String,
    /// Free-form extra hyperparams (e.g. PPO lr, gamma).
    #[serde(default)]
    pub extra: BTreeMap<String, f64>,
}

const fn default_num_envs() -> u32 {
    1
}

fn default_vec_mode() -> String {
    "auto".to_string()
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            master_seed: 0,
            num_envs: default_num_envs(),
            vec_mode: default_vec_mode(),
            extra: BTreeMap::new(),
        }
    }
}

/// Server section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfigSection {
    /// Bind address (e.g. "127.0.0.1:51234").
    #[serde(default = "default_bind")]
    pub bind: String,
    /// Read timeout in milliseconds. `None` means none.
    #[serde(default)]
    pub read_timeout_ms: Option<u32>,
    /// Write timeout in milliseconds.
    #[serde(default)]
    pub write_timeout_ms: Option<u32>,
}

fn default_bind() -> String {
    "127.0.0.1:0".to_string()
}

impl Default for ServerConfigSection {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            read_timeout_ms: Some(30_000),
            write_timeout_ms: Some(30_000),
        }
    }
}

// ---------------------------------------------------------------------------
// UnifiedConfig
// ---------------------------------------------------------------------------

/// The canonical unified config file shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Pinned schema version. Must equal [`CONFIG_SCHEMA_VERSION`] at
    /// load time.
    pub schema_version: u32,
    /// Task id (must resolve in the registry).
    pub task: TaskId,
    /// Physics / simulation timing.
    #[serde(default)]
    pub sim: SimConfig,
    /// Training section.
    #[serde(default)]
    pub training: TrainingConfig,
    /// Recorder section.
    #[serde(default)]
    pub recorder: RecorderConfig,
    /// Server section.
    #[serde(default)]
    pub server: ServerConfigSection,
}

/// Failure modes when loading [`UnifiedConfig`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum UnifiedConfigError {
    /// Parse failure (JSON or TOML).
    #[error("unified config parse error: {0}")]
    Parse(String),
    /// Schema-version mismatch with the running stack.
    #[error("unified config schema version {got} not supported (expected {expected})")]
    UnsupportedSchemaVersion {
        /// Wire value.
        got: u32,
        /// Compiled-in [`CONFIG_SCHEMA_VERSION`].
        expected: u32,
    },
}

impl UnifiedConfig {
    /// Load from JSON.
    ///
    /// # Errors
    ///
    /// [`UnifiedConfigError::Parse`] on syntax errors;
    /// [`UnifiedConfigError::UnsupportedSchemaVersion`] when the file's
    /// schema version disagrees with [`CONFIG_SCHEMA_VERSION`].
    pub fn from_json(s: &str) -> Result<Self, UnifiedConfigError> {
        let cfg: Self =
            serde_json::from_str(s).map_err(|e| UnifiedConfigError::Parse(e.to_string()))?;
        if cfg.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(UnifiedConfigError::UnsupportedSchemaVersion {
                got: cfg.schema_version,
                expected: CONFIG_SCHEMA_VERSION,
            });
        }
        Ok(cfg)
    }

    /// Load from TOML.
    ///
    /// # Errors
    ///
    /// As [`Self::from_json`] but parses TOML.
    pub fn from_toml(s: &str) -> Result<Self, UnifiedConfigError> {
        let cfg: Self = toml::from_str(s).map_err(|e| UnifiedConfigError::Parse(e.to_string()))?;
        if cfg.schema_version != CONFIG_SCHEMA_VERSION {
            return Err(UnifiedConfigError::UnsupportedSchemaVersion {
                got: cfg.schema_version,
                expected: CONFIG_SCHEMA_VERSION,
            });
        }
        Ok(cfg)
    }

    /// Serialise to JSON (pretty). Used by `clankers inspect
    /// --resolved-config` and by manifest stamping.
    ///
    /// # Errors
    ///
    /// Returns [`UnifiedConfigError::Parse`] only if `serde_json::to_string`
    /// fails — effectively unreachable for the current schema.
    pub fn to_json(&self) -> Result<String, UnifiedConfigError> {
        serde_json::to_string_pretty(self).map_err(|e| UnifiedConfigError::Parse(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> UnifiedConfig {
        UnifiedConfig {
            schema_version: CONFIG_SCHEMA_VERSION,
            task: TaskId::new("clankers", "cartpole_v1"),
            sim: SimConfig::default(),
            training: TrainingConfig {
                master_seed: 42,
                num_envs: 8,
                vec_mode: "parallel".to_string(),
                ..TrainingConfig::default()
            },
            recorder: RecorderConfig::default(),
            server: ServerConfigSection::default(),
        }
    }

    #[test]
    fn roundtrips_through_json() {
        let cfg = sample();
        let s = cfg.to_json().unwrap();
        let back = UnifiedConfig::from_json(&s).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn rejects_old_schema_version() {
        let mut bad = sample();
        bad.schema_version = 0;
        let s = serde_json::to_string(&bad).unwrap();
        let err = UnifiedConfig::from_json(&s).unwrap_err();
        assert_eq!(
            err,
            UnifiedConfigError::UnsupportedSchemaVersion {
                got: 0,
                expected: CONFIG_SCHEMA_VERSION
            }
        );
    }

    #[test]
    fn from_toml_loads_minimal_doc() {
        let toml = r#"
schema_version = 1
task = { namespace = "clankers", name = "cartpole_v1" }

[training]
master_seed = 7
num_envs = 4
vec_mode = "sequential"
"#;
        let cfg = UnifiedConfig::from_toml(toml).unwrap();
        assert_eq!(cfg.task.name, "cartpole_v1");
        assert_eq!(cfg.training.num_envs, 4);
        // Recorder defaulted because not present.
        assert!(!cfg.recorder.enabled);
    }

    #[test]
    fn defaults_are_sane() {
        let r = RecorderConfig::default();
        assert!(!r.enabled);
        let t = TrainingConfig::default();
        assert_eq!(t.num_envs, 1);
        assert_eq!(t.vec_mode, "auto");
        let s = ServerConfigSection::default();
        assert_eq!(s.read_timeout_ms, Some(30_000));
    }
}
