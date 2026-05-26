//! `pendulum` scenario — headless single-DOF pendulum.
//!
//! Lifts the scene-setup body of `examples/src/bin/pendulum_headless.rs`:
//! URDF parse → [`crate::SceneBuilder`] → bound [`JointLayout`] →
//! optional [`JointStateSensor`] + [`JointTorqueSensor`].
//!
//! Episode driving (sinusoidal torque, sensor reads, stats reporting)
//! stays in the example bin — the scenario only owns scene setup.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_env::prelude::*;

use crate::SceneBuilder;
use crate::SpawnedScene;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the pendulum (single revolute joint).
const PENDULUM_URDF: &str = include_str!("../../../../examples/urdf/pendulum.urdf");

/// Per-scenario knobs for the `pendulum` scenario.
#[derive(Debug, Clone)]
pub struct PendulumConfig {
    /// Max steps per episode (consumed by the bin's episode loop).
    pub max_episode_steps: u32,
    /// When `true`, register a [`JointTorqueSensor`] in addition to the
    /// default [`JointStateSensor`]. Default `true` — matches the
    /// pre-PR2 bin behaviour.
    pub register_torque_sensor: bool,
}

impl Default for PendulumConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 50,
            register_torque_sensor: true,
        }
    }
}

/// Artifacts returned by [`PendulumScenario::build_with`].
pub struct PendulumArtifacts {
    /// The spawned-scene wrapper (owns the App + robots map).
    pub scene: SpawnedScene,
    /// The parsed URDF model.
    pub model: clankers_urdf::RobotModel,
    /// The single `pivot` joint entity.
    pub pivot: Entity,
    /// Layout bound to the pivot entity.
    pub layout: Arc<clankers_core::layout::JointLayout>,
}

/// Builder for the `pendulum` scenario.
pub struct PendulumScenario;

impl ScenarioBuilder for PendulumScenario {
    fn name(&self) -> &'static str {
        "pendulum"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let pend_cfg = PendulumConfig {
            max_episode_steps: cfg.max_steps,
            ..PendulumConfig::default()
        };
        let artefacts = Self::build_with(cfg, &pend_cfg);
        let handle = ScenarioHandle {
            layout: Some(artefacts.layout),
            max_steps: cfg.max_steps,
        };
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl PendulumScenario {
    /// Parametrised build path.
    #[must_use]
    pub fn build_with(cfg: &ScenarioConfig, pend_cfg: &PendulumConfig) -> PendulumArtifacts {
        let _ = cfg.seed; // honoured by callers seeding `Episode` directly
        let model = clankers_urdf::parse_string(PENDULUM_URDF)
            .expect("failed to parse pendulum URDF (compile-time include)");

        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(pend_cfg.max_episode_steps.max(1))
            .with_robot(model.clone(), HashMap::new())
            .build();

        let pivot = scene.robots["pendulum"]
            .joint_entity("pivot")
            .expect("missing pivot joint");

        // Build layout bound to the spawned joint entities.
        let layout = {
            let bot = &scene.robots["pendulum"];
            let mut layout = model.to_layout();
            let entities: Vec<Entity> = layout
                .joints()
                .iter()
                .map(|spec| {
                    bot.joint_entity(&spec.name)
                        .unwrap_or_else(|| panic!("joint {} not in spawned robot", spec.name))
                })
                .collect();
            layout.bind_entities(&entities);
            Arc::new(layout)
        };

        // Register sensors.
        {
            let world = scene.app.world_mut();
            let mut registry = world
                .remove_resource::<SensorRegistry>()
                .expect("SensorRegistry present after ClankersEnvPlugin");
            let mut buffer = world
                .remove_resource::<ObservationBuffer>()
                .expect("ObservationBuffer present after ClankersEnvPlugin");
            registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
            if pend_cfg.register_torque_sensor {
                registry.register(
                    Box::new(JointTorqueSensor::new(layout.clone())),
                    &mut buffer,
                );
            }
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        PendulumArtifacts {
            scene,
            model,
            pivot,
            layout,
        }
    }
}
