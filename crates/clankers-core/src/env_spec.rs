//! First-class task / environment specification (G1).
//!
//! CODE_QUALITY_REVIEW § "Gap 1: Task / Environment Authoring Is
//! Ad-Hoc". Today new tasks land as bespoke example binaries that each
//! reinvent reward, termination, observation, and scene wiring. The
//! review's prescribed shape:
//!
//! ```text
//! EnvSpec { id, scene, action, observation, reset, reward, termination }
//! ```
//!
//! This module defines the data shapes. Wiring (`SceneBuilder::from_env_spec`,
//! `RewardProvider` resolution, registry integration) is follow-up
//! work — those changes touch examples and the recorder and are best
//! handled in dedicated loops. The foundation here is what every
//! follow-up loop can attach to without re-arguing field names.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::types::{ActionSpace, ObservationSpace};

// ---------------------------------------------------------------------------
// EnvId
// ---------------------------------------------------------------------------

/// Stable identifier for a task / environment.
///
/// Two-part: registry namespace + task name. Examples:
/// `EnvId::new("clankers", "quadruped_mpc")`,
/// `EnvId::new("user", "my_pick_task")`. Stored in
/// [`RunManifest`](crate::manifest::RunManifest) so reruns can resolve
/// the exact task they were trained against (G3).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvId {
    /// Namespace (e.g. "clankers", "user", "examples").
    pub namespace: String,
    /// Task name within the namespace.
    pub name: String,
}

impl EnvId {
    /// Construct an `EnvId` from owned strings.
    #[must_use]
    pub fn new(namespace: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            name: name.into(),
        }
    }

    /// Render as `"namespace/name"`. Use for registry keys, CLI flags,
    /// MCAP manifest entries.
    #[must_use]
    pub fn as_path(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }
}

// ---------------------------------------------------------------------------
// SceneSpec
// ---------------------------------------------------------------------------

/// Scene shape declarable in an `EnvSpec`.
///
/// References by path / id keep the spec serialisable. Concrete scene
/// loading is performed by [`SceneBuilder`](https://docs.rs/clankers-sim)
/// at task instantiation time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneSpec {
    /// URDF / scene file path or registered id.
    pub robot_id: String,
    /// Optional list of object names to spawn.
    #[serde(default)]
    pub object_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// ActionContract / ObservationContract
// ---------------------------------------------------------------------------

/// Action contract — what the task expects the policy to emit.
///
/// References [`ActionSpace`] so the contract stays a single source of
/// truth for the dimensionality, bounds, and discrete-vs-continuous
/// distinction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionContract {
    /// Action space the policy must produce.
    pub space: ActionSpace,
    /// Human-readable description for docs / inspection.
    #[serde(default)]
    pub description: String,
}

/// Observation contract — what the task hands back to the policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationContract {
    /// Observation space the env produces.
    pub space: ObservationSpace,
    /// Human-readable description for docs / inspection.
    #[serde(default)]
    pub description: String,
}

// ---------------------------------------------------------------------------
// ResetDistribution
// ---------------------------------------------------------------------------

/// Initial-state distribution for episode reset.
///
/// Stored as a tagged enum so future randomisation modes (Gaussian,
/// uniform, latent-conditioned) can extend the contract without
/// breaking existing tasks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResetDistribution {
    /// Always reset to a fixed initial state.
    Fixed,
    /// Sample reset states uniformly over each joint's bounds.
    UniformJointBounds,
    /// Add Gaussian noise per joint with a per-joint std.
    GaussianAroundDefault {
        /// Std deviation per joint slot (radians or metres).
        std_per_joint: Vec<f32>,
    },
}

// ---------------------------------------------------------------------------
// RewardContract / TerminationContract
// ---------------------------------------------------------------------------

/// Where the reward signal is computed.
///
/// CODE_QUALITY_REVIEW § Gap 12 — make ownership explicit so policies
/// and evaluators agree on which side of the wire produced the reward.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewardProvider {
    /// Rust-side `RewardFunction` impl (registered by the task plugin).
    Rust,
    /// Python-side `RewardFunction` registered against the task id.
    Python,
}

/// Reward contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewardContract {
    /// Provider side.
    pub provider: RewardProvider,
    /// Symbolic id of the reward function (registry key).
    pub function_id: String,
    /// Free-form parameters bound to the reward function.
    #[serde(default)]
    pub params: BTreeMap<String, f64>,
}

/// Termination conditions for the task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerminationContract {
    /// Hard step limit. `None` means rely on physical terminations only.
    #[serde(default)]
    pub max_episode_steps: Option<u32>,
    /// Ordered list of termination predicates evaluated each step.
    /// Predicates are looked up by id in the task registry.
    #[serde(default)]
    pub predicates: Vec<String>,
}

// ---------------------------------------------------------------------------
// EnvSpec
// ---------------------------------------------------------------------------

/// The canonical task / environment specification.
///
/// A complete `EnvSpec` is enough to:
///
/// - Look up the task in the registry by [`EnvId`].
/// - Build the scene via `SceneBuilder` (G1 follow-up).
/// - Validate observations / actions on the wire (G2).
/// - Stamp the run manifest (G3).
/// - Render docs (G5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvSpec {
    /// Stable identifier.
    pub id: EnvId,
    /// Scene to build.
    pub scene: SceneSpec,
    /// Action contract.
    pub action: ActionContract,
    /// Observation contract.
    pub observation: ObservationContract,
    /// Initial-state distribution.
    pub reset: ResetDistribution,
    /// Reward contract.
    pub reward: RewardContract,
    /// Termination contract.
    pub termination: TerminationContract,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> EnvSpec {
        EnvSpec {
            id: EnvId::new("clankers", "cartpole_v1"),
            scene: SceneSpec {
                robot_id: "cartpole".to_string(),
                object_ids: vec![],
            },
            action: ActionContract {
                space: ActionSpace::Box {
                    low: vec![-1.0],
                    high: vec![1.0],
                },
                description: "scalar lateral force".to_string(),
            },
            observation: ObservationContract {
                space: ObservationSpace::Box {
                    low: vec![-10.0; 4],
                    high: vec![10.0; 4],
                },
                description: "(x, x_dot, theta, theta_dot)".to_string(),
            },
            reset: ResetDistribution::Fixed,
            reward: RewardContract {
                provider: RewardProvider::Rust,
                function_id: "cartpole_upright".to_string(),
                params: BTreeMap::new(),
            },
            termination: TerminationContract {
                max_episode_steps: Some(500),
                predicates: vec!["pole_fallen".to_string()],
            },
        }
    }

    #[test]
    fn env_id_as_path_formats_namespace_slash_name() {
        let id = EnvId::new("clankers", "quadruped_mpc");
        assert_eq!(id.as_path(), "clankers/quadruped_mpc");
    }

    #[test]
    fn env_spec_roundtrips_through_json() {
        let spec = sample();
        let json = serde_json::to_string(&spec).unwrap();
        let back: EnvSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn reset_distribution_tagged_serde() {
        let r = ResetDistribution::GaussianAroundDefault {
            std_per_joint: vec![0.1, 0.2],
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"kind\":\"gaussian_around_default\""));
        let back: ResetDistribution = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn reward_provider_serializes_as_snake_case() {
        let json = serde_json::to_string(&RewardProvider::Python).unwrap();
        assert_eq!(json, "\"python\"");
    }
}
