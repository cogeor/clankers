//! Termination reason taxonomy (G12).
//!
//! CODE_QUALITY_REVIEW § "Gap 12: Reward & Termination Ownership Is
//! Unclear". `StepResult` carries `terminated: bool` + `truncated:
//! bool` today, but no signal about *why* — a fallen-pole termination
//! and a "off the workspace" termination both flatten to `terminated:
//! true`, and downstream evaluators can't distinguish them in a
//! recorded trace.
//!
//! [`TerminationReason`] adds the missing channel. The Bevy
//! `Episode` resource takes one of these when ending; the recorder
//! writes it into the trace; evaluators read it back to compute
//! per-reason success rates.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// TerminationReason
// ---------------------------------------------------------------------------

/// Why an episode ended.
///
/// Stays mutually exclusive with the `truncated` flag — a `Truncated`
/// reason here would be redundant. `terminated && reason == None` is
/// allowed for legacy producers but flagged by the recorder as a
/// soft warning.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerminationReason {
    /// Task succeeded (e.g. cube placed on target, distance < eps).
    Success {
        /// Optional task-specific success metric name.
        #[serde(default)]
        metric: Option<String>,
    },
    /// Task failed by an explicit predicate (e.g. pole fell, robot
    /// tipped over, contact force exceeded).
    Failure {
        /// Which predicate fired.
        predicate: String,
    },
    /// Physics blew up (NaN in body state, contact impulse > sanity
    /// threshold). The recorder treats these specially because they
    /// usually indicate a sim-tuning bug, not real task behaviour.
    PhysicsFault {
        /// Short diagnostic of the fault.
        diagnostic: String,
    },
    /// Reset condition outside the task definition (e.g. external
    /// scripted abort). Useful for synthetic-data pipelines that
    /// reset on schedule rather than on outcome.
    External {
        /// Who triggered the reset.
        source: String,
    },
}

impl TerminationReason {
    /// Whether this reason represents a successful episode for
    /// success-rate metrics.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Whether this reason indicates an environment-level fault that
    /// should not count as legitimate task data.
    #[must_use]
    pub const fn is_fault(&self) -> bool {
        matches!(self, Self::PhysicsFault { .. })
    }

    /// Short slug for the variant (CLI flags, prometheus labels).
    #[must_use]
    pub const fn slug(&self) -> &'static str {
        match self {
            Self::Success { .. } => "success",
            Self::Failure { .. } => "failure",
            Self::PhysicsFault { .. } => "physics_fault",
            Self::External { .. } => "external",
        }
    }
}

// ---------------------------------------------------------------------------
// RewardOwnership
// ---------------------------------------------------------------------------

/// Where the reward signal for a given trace step originated.
///
/// Stamped per-step so evaluators reading a recorded trace can verify
/// the producer matches the manifest's declared
/// [`crate::env_spec::RewardProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardOwnership {
    /// Reward came from a Rust-side `RewardFunction` impl.
    Rust,
    /// Reward came from the Python-side `RewardFunction`.
    Python,
    /// Reward was zero / pass-through (no reward provider attached).
    None,
}

impl RewardOwnership {
    /// Short slug.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::None => "none",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn termination_reason_classification() {
        let s = TerminationReason::Success { metric: None };
        assert!(s.is_success());
        assert!(!s.is_fault());

        let f = TerminationReason::Failure {
            predicate: "pole_fallen".to_string(),
        };
        assert!(!f.is_success());
        assert!(!f.is_fault());

        let p = TerminationReason::PhysicsFault {
            diagnostic: "NaN pose".to_string(),
        };
        assert!(!p.is_success());
        assert!(p.is_fault());

        let e = TerminationReason::External {
            source: "scheduler".to_string(),
        };
        assert!(!e.is_success());
        assert!(!e.is_fault());
    }

    #[test]
    fn termination_reason_serialises_tagged() {
        let r = TerminationReason::Failure {
            predicate: "pole_fallen".to_string(),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"failure\""));
        let back: TerminationReason = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn reward_ownership_serialises_snake_case() {
        let json = serde_json::to_string(&RewardOwnership::Python).unwrap();
        assert_eq!(json, "\"python\"");
    }

    #[test]
    fn termination_reason_slugs_unique() {
        let slugs = [
            TerminationReason::Success { metric: None }.slug(),
            TerminationReason::Failure {
                predicate: "x".to_string(),
            }
            .slug(),
            TerminationReason::PhysicsFault {
                diagnostic: "x".to_string(),
            }
            .slug(),
            TerminationReason::External {
                source: "x".to_string(),
            }
            .slug(),
        ];
        let unique: std::collections::HashSet<&&str> = slugs.iter().collect();
        assert_eq!(unique.len(), slugs.len());
    }
}
