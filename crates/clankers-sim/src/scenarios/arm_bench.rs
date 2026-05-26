//! `arm_bench` scenario — headless 6-DOF arm for offline benchmark /
//! demonstration recording.
//!
//! Identical scene to [`arm_ik`](super::arm_ik) (the headless 6-DOF
//! arm + position-mode PID + Rapier with `num_solver_iterations = 50`
//! + bound `JointLayout` + sensors).
//!
//! [`ArmBenchConfig`] exposes a few benchmark-specific knobs:
//!
//! - Recording is OFF by default (callers attach `RecorderPlugin`
//!   themselves; `clankers-sim` does not depend on `clankers-record`).
//! - The default `max_episode_steps` is higher (matches the bench
//!   defaults — 500 vs. 20 for two-link).
//! - The fixed-update variant of the physics backend may be selected.
//!
//! IK logic + record glue stay in the example bin
//! (`examples/src/bin/arm_bench.rs`); `clankers-sim` does NOT depend on
//! `clankers-ik` or `clankers-record`.

use bevy::prelude::App;

use crate::scenarios::arm_ik::{ArmIkArtifacts, ArmIkConfig, ArmIkScenario};
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// Per-scenario knobs for the `arm_bench` scenario.
///
/// Per W8 PR1 Design choice B, scenario-specific tuning rides on a
/// private config struct consumed by [`ArmBenchScenario::build_with`]
/// rather than expanding the field-locked
/// [`ScenarioConfig`](crate::scenarios::ScenarioConfig).
#[derive(Debug, Clone)]
pub struct ArmBenchConfig {
    /// Hard cap on episode steps. Defaults to 500.
    pub max_episode_steps: u32,
    /// Whether to install the fixed-update physics backend
    /// (`RapierBackendFixed`). Defaults to `false` (matches the bench
    /// today which runs on `Update`).
    pub use_fixed_update: bool,
}

impl Default for ArmBenchConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 500,
            use_fixed_update: false,
        }
    }
}

/// Builder for the `arm_bench` scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the registry by
/// [`super::register_builtin`] alongside the rest of the arm family.
pub struct ArmBenchScenario;

impl ScenarioBuilder for ArmBenchScenario {
    fn name(&self) -> &'static str {
        "arm_bench"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let bench_cfg = ArmBenchConfig {
            max_episode_steps: cfg.max_steps,
            ..ArmBenchConfig::default()
        };
        let artefacts = Self::build_with_artifacts(cfg, &bench_cfg);
        let handle = ScenarioHandle {
            layout: Some(artefacts.joint_layout.clone()),
            max_steps: cfg.max_steps,
        };
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl ArmBenchScenario {
    /// Parametrised build path. Returns the same artefact bundle
    /// [`ArmIkScenario::build_with_artifacts`] uses so the bin's IK
    /// state machine + record loop can drive the scene without
    /// re-parsing the URDF.
    #[must_use]
    pub fn build_with_artifacts(
        cfg: &ScenarioConfig,
        bench_cfg: &ArmBenchConfig,
    ) -> ArmIkArtifacts {
        let ik_cfg = ArmIkConfig {
            max_episode_steps: bench_cfg.max_episode_steps,
            use_fixed_update: bench_cfg.use_fixed_update,
            // 6-DOF chain-order sensor matches the pre-PR1 bench layout.
            sensor_dof: 6,
            initial_positions: crate::scenarios::arm_ik::REST_POSE,
        };
        ArmIkScenario::build_with_artifacts(cfg, &ik_cfg)
    }
}
