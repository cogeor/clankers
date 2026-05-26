//! Scenario registry — named, reproducible simulation setups.
//!
//! A "scenario" is a self-contained recipe for spawning a robot, sensors,
//! and any auxiliary world geometry into a Bevy [`App`]. Each scenario is
//! identified by a `&'static str` name and constructed via a
//! [`ScenarioBuilder`] trait implementation. Registered scenarios are
//! discovered through a [`ScenarioRegistry`].
//!
//! This module is the W5 PR1 skeleton — it defines the trait, the
//! registry, and the public types; the two reference scenarios
//! (`arm_pick`, `cartpole`) ship in W5 PR2 (loop 6 of the
//! `20260526-013019-w3-w4-w5-impl` orchestration).
//!
//! # Stability
//!
//! [`ScenarioConfig`] is **field-locked across W5 PR1–PR4**. The
//! `clankers-app` CLI's `--seed`, `--max-steps`, `--headless`,
//! `--record` flags all map directly to fields here; downstream tools
//! (Python clients, future schedulers) rely on the schema. Adding a new
//! field is additive and requires a CHANGELOG note on `clankers-sim`;
//! removing or renaming a field is a breaking change.
//!
//! [`ScenarioHandle`] is **not** locked — PR2 may add fields (e.g.
//! `observation_dim`). Builders should construct it via a struct literal
//! so additive changes do not silently break callers.
//!
//! # Limitations
//!
//! `ScenarioRegistry` is keyed on `&'static str`, which forces every
//! scenario name to be a string literal known at compile time.
//! Runtime-loaded scenarios (e.g. loaded from a `.scenario.toml`
//! manifest) are explicitly out of scope for the entire W5 workstream;
//! they will require a different code path (e.g. a parallel
//! `DynamicScenarioRegistry` keyed on `String`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy::prelude::App;
use clankers_core::layout::JointLayout;

pub mod arm_pick;
pub mod cartpole;

// ---------------------------------------------------------------------------
// ScenarioConfig
// ---------------------------------------------------------------------------

/// Runtime configuration shared by every scenario builder.
///
/// The field set is **stable across W5 PR1–PR4**. See the module-level
/// docs for the stability contract.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Optional RNG seed. `None` means "non-deterministic" (use a
    /// thread-local entropy source). Builders that randomise initial
    /// conditions must honour this.
    pub seed: Option<u64>,
    /// Hard cap on simulation steps before the episode terminates.
    pub max_steps: u32,
    /// `true` for batch / CI / training runs (no rendering plugins).
    /// `false` for visualisation runs.
    pub headless: bool,
    /// If `Some`, the scenario is wired to record into the given path
    /// via `clankers-record`. `None` disables recording.
    pub record_path: Option<PathBuf>,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        Self {
            seed: None,
            max_steps: 1000,
            headless: true,
            record_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ScenarioHandle
// ---------------------------------------------------------------------------

/// Per-build artefact returned by [`ScenarioBuilder::build`].
///
/// PR1 ships a minimal shape. PR2 may add fields additively — builders
/// should construct via struct literal so additive changes do not
/// silently break callers.
#[derive(Debug, Clone)]
pub struct ScenarioHandle {
    /// The robot's joint layout, if the scenario spawned a URDF-backed
    /// robot. `None` for scenarios that do not have a single canonical
    /// robot (e.g. multi-robot demos).
    pub layout: Option<Arc<JointLayout>>,
    /// The maximum number of simulation steps for this scenario
    /// instance, mirroring [`ScenarioConfig::max_steps`] but resolved
    /// against any scenario-specific clamp.
    pub max_steps: u32,
}

// ---------------------------------------------------------------------------
// ScenarioBuilder trait
// ---------------------------------------------------------------------------

/// A factory for one named scenario.
///
/// Implementors are typically zero-sized unit structs registered as
/// trait objects in [`ScenarioRegistry`]. The trait is `Send + Sync +
/// 'static` so registries can live in `Arc`s shared across threads
/// (e.g. multi-tenant gym servers).
pub trait ScenarioBuilder: Send + Sync + 'static {
    /// Stable identifier — must match the registry key.
    fn name(&self) -> &'static str;

    /// Build the scenario into `app`, returning a [`ScenarioHandle`].
    ///
    /// The builder is responsible for adding any required plugins,
    /// spawning entities, and wiring sensors. The caller has already
    /// added [`crate::ClankersSimPlugin`] when this is invoked from
    /// the CLI; scenarios should not re-add core plugins.
    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle;
}

// ---------------------------------------------------------------------------
// ScenarioRegistry
// ---------------------------------------------------------------------------

/// A name → builder map.
///
/// Use [`Self::default`] to obtain an empty registry, then
/// [`register_builtin`] to install the shipped scenarios. In PR1 the
/// builtin list is empty; PR2 fills it with `arm_pick` and `cartpole`.
#[derive(Default)]
pub struct ScenarioRegistry {
    builders: HashMap<&'static str, Box<dyn ScenarioBuilder>>,
}

impl ScenarioRegistry {
    /// Create an empty registry. Equivalent to [`Self::default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a builder. Overwrites any existing entry with the same
    /// `name()`.
    pub fn register(&mut self, builder: Box<dyn ScenarioBuilder>) {
        self.builders.insert(builder.name(), builder);
    }

    /// Look up a builder by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&dyn ScenarioBuilder> {
        self.builders.get(name).map(std::convert::AsRef::as_ref)
    }

    /// All registered names, sorted alphabetically.
    ///
    /// Used by `clankers-app info --json` to populate the `scenarios`
    /// array. Sorting is required so two invocations of the same
    /// binary produce byte-equal output regardless of `HashMap`
    /// iteration order.
    #[must_use]
    pub fn list_builtin(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.builders.keys().copied().collect();
        names.sort_unstable();
        names
    }
}

/// Install the shipped built-in scenarios into `registry`.
///
/// Registers `arm_pick` and `cartpole` (W5 PR2). The list is
/// alphabetically-sorted by [`ScenarioRegistry::list_builtin`] before
/// being surfaced to callers.
pub fn register_builtin(registry: &mut ScenarioRegistry) {
    registry.register(Box::new(arm_pick::ArmPickScenario));
    registry.register(Box::new(cartpole::CartpoleScenario));
}

// ---------------------------------------------------------------------------
// ScenarioError
// ---------------------------------------------------------------------------

/// Errors surfaced when looking up or building a scenario.
#[derive(Debug, thiserror::Error)]
pub enum ScenarioError {
    /// No builder is registered under the given name.
    #[error("unknown scenario: {name}")]
    UnknownScenario {
        /// The requested scenario name.
        name: String,
    },

    /// The named scenario is reserved for a future PR and not yet
    /// installed in the registry.
    #[error("scenario '{name}' is not yet implemented (planned for {migrated_in})")]
    NotImplementedYet {
        /// The requested scenario name.
        name: String,
        /// Human-readable note pointing at the PR / milestone that
        /// installs the builder.
        migrated_in: &'static str,
    },
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyScenario;

    impl ScenarioBuilder for DummyScenario {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn build(&self, _app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
            ScenarioHandle {
                layout: None,
                max_steps: cfg.max_steps,
            }
        }
    }

    struct OtherScenario;

    impl ScenarioBuilder for OtherScenario {
        fn name(&self) -> &'static str {
            "alpha"
        }

        fn build(&self, _app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
            ScenarioHandle {
                layout: None,
                max_steps: cfg.max_steps,
            }
        }
    }

    #[test]
    fn config_default_field_set() {
        let cfg = ScenarioConfig::default();
        assert!(cfg.seed.is_none());
        assert_eq!(cfg.max_steps, 1000);
        assert!(cfg.headless);
        assert!(cfg.record_path.is_none());
    }

    #[test]
    fn registry_register_and_get_round_trip() {
        let mut registry = ScenarioRegistry::new();
        registry.register(Box::new(DummyScenario));

        let builder = registry.get("dummy").expect("registered scenario");
        assert_eq!(builder.name(), "dummy");

        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn list_builtin_is_alphabetic() {
        let mut registry = ScenarioRegistry::new();
        registry.register(Box::new(DummyScenario)); // "dummy"
        registry.register(Box::new(OtherScenario)); // "alpha"

        assert_eq!(registry.list_builtin(), vec!["alpha", "dummy"]);
    }

    #[test]
    fn pr2_register_builtin_has_arm_pick_and_cartpole() {
        let mut registry = ScenarioRegistry::new();
        register_builtin(&mut registry);
        assert_eq!(registry.list_builtin(), vec!["arm_pick", "cartpole"]);
    }

    #[test]
    fn error_messages() {
        let e = ScenarioError::UnknownScenario { name: "foo".into() };
        assert_eq!(e.to_string(), "unknown scenario: foo");

        let e = ScenarioError::NotImplementedYet {
            name: "arm_pick".into(),
            migrated_in: "W5 PR2",
        };
        assert!(e.to_string().contains("arm_pick"));
        assert!(e.to_string().contains("W5 PR2"));
    }
}
