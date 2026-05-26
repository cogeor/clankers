//! `domain_rand_pendulum` scenario — pendulum with motor + friction randomisation.
//!
//! Lifts the scene-setup body of `examples/src/bin/domain_rand.rs`
//! lines 22-72: pendulum URDF → `ClankersDomainRandPlugin` →
//! `DomainRandConfig` with `MotorRandomizer` + `FrictionRandomizer`.
//!
//! Per W8 PR2 PLAN: only ONE scenario is needed. The variance and
//! determinism phase loops in the bin are orchestration on top of the
//! same scene — they stay in the bin.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_domain_rand::prelude::*;

use crate::SceneBuilder;
use crate::SpawnedScene;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the pendulum.
const PENDULUM_URDF: &str = include_str!("../../../../examples/urdf/pendulum.urdf");

/// Per-scenario knobs for the `domain_rand_pendulum` scenario.
#[derive(Debug, Clone)]
pub struct DomainRandPendulumConfig {
    /// Seed for the domain-rand RNG. Default 42.
    pub seed: u64,
    /// `(min, max)` for motor `max_torque`. Default `(5.0, 20.0)`.
    pub max_torque_range: (f32, f32),
    /// `(min, max)` for motor `max_velocity`. Default `(2.0, 10.0)`.
    pub max_vel_range: (f32, f32),
    /// `(min, max)` for friction coulomb coefficient. Default `(0.01, 0.2)`.
    pub coulomb_range: (f32, f32),
    /// `(min, max)` for friction viscous coefficient. Default `(0.01, 0.5)`.
    pub viscous_range: (f32, f32),
    /// Max episode length. Default 20.
    pub max_episode_steps: u32,
}

impl Default for DomainRandPendulumConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            max_torque_range: (5.0, 20.0),
            max_vel_range: (2.0, 10.0),
            coulomb_range: (0.01, 0.2),
            viscous_range: (0.01, 0.5),
            max_episode_steps: 20,
        }
    }
}

/// Artifacts returned by [`DomainRandPendulumScenario::build_with`].
pub struct DomainRandPendulumArtifacts {
    /// The spawned-scene wrapper.
    pub scene: SpawnedScene,
    /// The pivot joint entity (callers drive a command on this).
    pub pivot: Entity,
    /// Layout bound to the pivot.
    pub layout: Arc<clankers_core::layout::JointLayout>,
}

/// Builder for the `domain_rand_pendulum` scenario.
pub struct DomainRandPendulumScenario;

impl ScenarioBuilder for DomainRandPendulumScenario {
    fn name(&self) -> &'static str {
        "domain_rand_pendulum"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let dr_cfg = DomainRandPendulumConfig {
            max_episode_steps: cfg.max_steps,
            seed: cfg.seed.unwrap_or(42),
            ..DomainRandPendulumConfig::default()
        };
        let artefacts = Self::build_with(cfg, &dr_cfg);
        let handle = ScenarioHandle {
            layout: Some(artefacts.layout),
            max_steps: cfg.max_steps,
        };
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl DomainRandPendulumScenario {
    /// Parametrised build path.
    #[must_use]
    pub fn build_with(
        cfg: &ScenarioConfig,
        dr_cfg: &DomainRandPendulumConfig,
    ) -> DomainRandPendulumArtifacts {
        let _ = cfg.seed;

        let model = clankers_urdf::parse_string(PENDULUM_URDF)
            .expect("failed to parse pendulum URDF (compile-time include)");
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(dr_cfg.max_episode_steps.max(1))
            .with_robot(model.clone(), HashMap::new())
            .build();

        let pivot = scene.robots["pendulum"]
            .joint_entity("pivot")
            .expect("missing pivot joint");

        let layout = {
            let bot = &scene.robots["pendulum"];
            let mut layout = model.to_layout();
            let entities: Vec<Entity> = layout
                .joints()
                .iter()
                .map(|spec| {
                    bot.joint_entity(&spec.name)
                        .unwrap_or_else(|| panic!("joint {} not in pendulum", spec.name))
                })
                .collect();
            layout.bind_entities(&entities);
            Arc::new(layout)
        };

        // Wire domain randomisation.
        scene.app.add_plugins(ClankersDomainRandPlugin);
        let motor_rand = MotorRandomizer {
            max_torque: Some(
                RandomizationRange::uniform(dr_cfg.max_torque_range.0, dr_cfg.max_torque_range.1)
                    .expect("motor max_torque range"),
            ),
            max_velocity: Some(
                RandomizationRange::uniform(dr_cfg.max_vel_range.0, dr_cfg.max_vel_range.1)
                    .expect("motor max_velocity range"),
            ),
            ..Default::default()
        };
        let friction_rand = FrictionRandomizer {
            coulomb: Some(
                RandomizationRange::uniform(dr_cfg.coulomb_range.0, dr_cfg.coulomb_range.1)
                    .expect("friction coulomb range"),
            ),
            viscous: Some(
                RandomizationRange::uniform(dr_cfg.viscous_range.0, dr_cfg.viscous_range.1)
                    .expect("friction viscous range"),
            ),
            ..Default::default()
        };
        let dr_config = DomainRandConfig::default()
            .with_seed(dr_cfg.seed)
            .with_actuator(ActuatorRandomizer {
                motor: motor_rand,
                friction: friction_rand,
                ..Default::default()
            });
        scene.app.insert_resource(dr_config);

        DomainRandPendulumArtifacts {
            scene,
            pivot,
            layout,
        }
    }
}
