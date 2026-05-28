//! Baselines registry schema (G4).
//!
//! CODE_QUALITY_REVIEW § "Gap 4: No Baselines Registry". The repo has
//! benchmark CSVs but no canonical "what good looks like" for any task
//! — every comparison is recomputed locally. The `baselines/` directory
//! is the registry; each subdirectory carries a `config.toml`, an
//! `expected_metrics.json`, and (on recapture) a `RunManifest`.
//!
//! This module defines the typed deserialiser for `expected_metrics.json`
//! so CI / `clankers compare` work against a single schema rather than
//! ad-hoc JSON parsing.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Schema version pin so future field changes don't silently load.
pub const BASELINE_SCHEMA_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// MetricTolerance / MetricTarget
// ---------------------------------------------------------------------------

/// Whether the target is a floor (higher is better) or ceiling
/// (lower is better).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricDirection {
    /// Higher value is better (reward, throughput).
    HigherIsBetter,
    /// Lower value is better (mean step time, error).
    LowerIsBetter,
}

/// Tolerance for a single metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricTarget {
    /// Target value (the centre of the expected band).
    pub target: f64,
    /// Absolute tolerance. `None` if only relative tolerance applies.
    #[serde(default)]
    pub tolerance_abs: Option<f64>,
    /// Relative tolerance as a percentage of `target`. `None` if only
    /// absolute tolerance applies.
    #[serde(default)]
    pub tolerance_pct: Option<f64>,
    /// Direction of the tolerance band.
    pub direction: MetricDirection,
}

impl MetricTarget {
    /// Whether `observed` falls within the tolerance band of this target.
    ///
    /// - `higher_is_better`: pass if `observed >= target - tol`.
    /// - `lower_is_better`:  pass if `observed <= target + tol`.
    ///
    /// `tol` is the larger of `tolerance_abs` and
    /// `target * tolerance_pct / 100`. At least one of the two must be
    /// supplied; otherwise the band is zero-width and any deviation
    /// fails (intentional — `None`-both is a config bug).
    #[must_use]
    pub fn passes(&self, observed: f64) -> bool {
        let abs = self.tolerance_abs.unwrap_or(0.0);
        let pct = self
            .tolerance_pct
            .map_or(0.0, |p| (self.target.abs()) * (p / 100.0));
        let tol = abs.max(pct);
        match self.direction {
            MetricDirection::HigherIsBetter => observed >= self.target - tol,
            MetricDirection::LowerIsBetter => observed <= self.target + tol,
        }
    }
}

// ---------------------------------------------------------------------------
// RecaptureMetadata
// ---------------------------------------------------------------------------

/// How the baseline is recaptured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecaptureMetadata {
    /// Recapture trigger (e.g. "workflow_dispatch:baseline").
    pub trigger: String,
    /// Recapture frequency ("manual", "nightly", "weekly", ...).
    pub frequency: String,
    /// ISO-8601 timestamp of the last recapture. `None` if never.
    #[serde(default)]
    pub last_recaptured_at: Option<String>,
}

// ---------------------------------------------------------------------------
// ExpectedMetrics
// ---------------------------------------------------------------------------

/// Schema for `baselines/<task>/expected_metrics.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedMetrics {
    /// Schema version. Must equal [`BASELINE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Task id ("namespace/name" form — same as `EnvId::as_path`).
    pub task: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// Per-metric targets.
    pub metrics: BTreeMap<String, MetricTarget>,
    /// Recapture metadata.
    pub recapture: RecaptureMetadata,
}

/// Failure modes for [`ExpectedMetrics::from_json`].
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BaselineError {
    /// JSON parse failure.
    #[error("baseline parse error: {0}")]
    Parse(String),
    /// Schema-version mismatch.
    #[error("baseline schema version {got} not supported (expected {expected})")]
    UnsupportedSchemaVersion {
        /// Wire value.
        got: u32,
        /// Compiled-in [`BASELINE_SCHEMA_VERSION`].
        expected: u32,
    },
}

impl ExpectedMetrics {
    /// Deserialise from JSON, validating the schema version.
    ///
    /// # Errors
    ///
    /// [`BaselineError::Parse`] on JSON parse failure;
    /// [`BaselineError::UnsupportedSchemaVersion`] when the file's
    /// schema version doesn't match [`BASELINE_SCHEMA_VERSION`].
    pub fn from_json(s: &str) -> Result<Self, BaselineError> {
        let m: Self = serde_json::from_str(s).map_err(|e| BaselineError::Parse(e.to_string()))?;
        if m.schema_version != BASELINE_SCHEMA_VERSION {
            return Err(BaselineError::UnsupportedSchemaVersion {
                got: m.schema_version,
                expected: BASELINE_SCHEMA_VERSION,
            });
        }
        Ok(m)
    }

    /// Check `observed` (a metric name → value map) against the expected
    /// targets. Returns the list of metric names that failed the
    /// tolerance check; an empty list means every observed metric is
    /// within band.
    ///
    /// Metrics declared in the baseline but missing from `observed` are
    /// considered failures (the producer should always emit them).
    /// Metrics present in `observed` but not declared are ignored.
    #[must_use]
    pub fn check<'a>(&'a self, observed: &BTreeMap<String, f64>) -> Vec<&'a str> {
        let mut failures = Vec::new();
        for (k, target) in &self.metrics {
            match observed.get(k) {
                Some(v) if target.passes(*v) => {}
                _ => failures.push(k.as_str()),
            }
        }
        failures
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn target(value: f64, tol_abs: f64, dir: MetricDirection) -> MetricTarget {
        MetricTarget {
            target: value,
            tolerance_abs: Some(tol_abs),
            tolerance_pct: None,
            direction: dir,
        }
    }

    fn sample() -> ExpectedMetrics {
        let mut metrics = BTreeMap::new();
        metrics.insert(
            "mean_reward".to_string(),
            target(480.0, 50.0, MetricDirection::HigherIsBetter),
        );
        ExpectedMetrics {
            schema_version: BASELINE_SCHEMA_VERSION,
            task: "clankers/cartpole_v1".to_string(),
            description: "smoke".to_string(),
            metrics,
            recapture: RecaptureMetadata {
                trigger: "workflow_dispatch:baseline".to_string(),
                frequency: "manual".to_string(),
                last_recaptured_at: None,
            },
        }
    }

    #[test]
    fn passes_higher_is_better_inside_band() {
        let t = target(100.0, 10.0, MetricDirection::HigherIsBetter);
        assert!(t.passes(95.0));
        assert!(t.passes(110.0));
        assert!(!t.passes(80.0));
    }

    #[test]
    fn passes_lower_is_better_inside_band() {
        let t = target(0.5, 0.1, MetricDirection::LowerIsBetter);
        assert!(t.passes(0.55));
        assert!(!t.passes(0.8));
    }

    #[test]
    fn relative_tolerance_widens_band() {
        let t = MetricTarget {
            target: 100.0,
            tolerance_abs: None,
            tolerance_pct: Some(20.0),
            direction: MetricDirection::HigherIsBetter,
        };
        assert!(t.passes(80.0)); // 20% off
        assert!(!t.passes(70.0));
    }

    #[test]
    fn check_collects_failures() {
        let m = sample();
        let mut observed = BTreeMap::new();
        observed.insert("mean_reward".to_string(), 100.0);
        let failures = m.check(&observed);
        assert_eq!(failures, vec!["mean_reward"]);

        observed.insert("mean_reward".to_string(), 470.0);
        assert!(m.check(&observed).is_empty());
    }

    #[test]
    fn check_missing_metric_is_failure() {
        let m = sample();
        let observed: BTreeMap<String, f64> = BTreeMap::new();
        let failures = m.check(&observed);
        assert_eq!(failures, vec!["mean_reward"]);
    }

    #[test]
    fn from_json_rejects_old_schema() {
        let mut bad = sample();
        bad.schema_version = 0;
        let s = serde_json::to_string(&bad).unwrap();
        let err = ExpectedMetrics::from_json(&s).unwrap_err();
        assert_eq!(
            err,
            BaselineError::UnsupportedSchemaVersion {
                got: 0,
                expected: BASELINE_SCHEMA_VERSION
            }
        );
    }

    #[test]
    fn checked_in_baselines_load_with_current_schema() {
        // Two baselines committed at G4 landing time. If either schema
        // drifts away from the current parser, this test breaks loudly.
        let cartpole = include_str!("../../../baselines/cartpole_v1/expected_metrics.json");
        let parsed = ExpectedMetrics::from_json(cartpole).expect("cartpole baseline parses");
        assert_eq!(parsed.task, "clankers/cartpole_v1");

        let vec_throughput =
            include_str!("../../../baselines/vec_throughput/expected_metrics.json");
        let parsed = ExpectedMetrics::from_json(vec_throughput).expect("vec baseline parses");
        assert_eq!(parsed.task, "clankers/vec_throughput");
    }
}
