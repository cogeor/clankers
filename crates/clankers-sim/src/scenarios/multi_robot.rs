//! `multi_robot` scenario — pendulum + 2-link arm + 6-DOF arm in one scene.
//!
//! Lifts the scene-setup body of `examples/src/bin/multi_robot.rs`:
//! three `SceneBuilder::with_robot` calls + `RobotGroup` verification.
//! Per-robot command driving + sensor reads stay in the bin (orchestration).

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_urdf::RobotModel;

use crate::SceneBuilder;
use crate::SpawnedScene;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the pendulum.
const PENDULUM_URDF: &str = include_str!("../../../../examples/urdf/pendulum.urdf");
/// URDF source for the 2-link planar arm.
const TWO_LINK_ARM_URDF: &str = include_str!("../../../../examples/urdf/two_link_arm.urdf");
/// URDF source for the 6-DOF arm.
const SIX_DOF_ARM_URDF: &str = include_str!("../../../../examples/urdf/six_dof_arm.urdf");

/// Per-scenario knobs for the `multi_robot` scenario.
#[derive(Debug, Clone)]
pub struct MultiRobotConfig {
    /// Include the pendulum (`RobotId` 0). Default `true`.
    pub include_pendulum: bool,
    /// Include the 2-link arm. Default `true`.
    pub include_two_link: bool,
    /// Include the 6-DOF arm. Default `true`.
    pub include_six_dof: bool,
    /// Max episode steps. Default 30 (matches the pre-PR2 bin).
    pub max_episode_steps: u32,
}

impl Default for MultiRobotConfig {
    fn default() -> Self {
        Self {
            include_pendulum: true,
            include_two_link: true,
            include_six_dof: true,
            max_episode_steps: 30,
        }
    }
}

/// Artifacts returned by [`MultiRobotScenario::build_with`].
pub struct MultiRobotArtifacts {
    /// The spawned-scene wrapper.
    pub scene: SpawnedScene,
    /// Parsed URDF models, keyed by robot name (`"pendulum"`,
    /// `"two_link_arm"`, `"six_dof_arm"`) — for robots that were
    /// included.
    pub models: HashMap<String, RobotModel>,
    /// Per-robot bound layouts.
    pub layouts: HashMap<String, Arc<clankers_core::layout::JointLayout>>,
}

/// Builder for the `multi_robot` scenario.
pub struct MultiRobotScenario;

impl ScenarioBuilder for MultiRobotScenario {
    fn name(&self) -> &'static str {
        "multi_robot"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let mr_cfg = MultiRobotConfig {
            max_episode_steps: cfg.max_steps,
            ..MultiRobotConfig::default()
        };
        let artefacts = Self::build_with(cfg, &mr_cfg);
        // Pick whichever layout exists first as the "primary".
        let layout = artefacts
            .layouts
            .get("pendulum")
            .or_else(|| artefacts.layouts.get("two_link_arm"))
            .or_else(|| artefacts.layouts.get("six_dof_arm"))
            .cloned();
        let handle = ScenarioHandle {
            layout,
            max_steps: cfg.max_steps,
        };
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl MultiRobotScenario {
    /// Parametrised build path.
    #[must_use]
    pub fn build_with(cfg: &ScenarioConfig, mr_cfg: &MultiRobotConfig) -> MultiRobotArtifacts {
        let _ = cfg.seed;

        let mut models: HashMap<String, RobotModel> = HashMap::new();
        if mr_cfg.include_pendulum {
            models.insert(
                "pendulum".to_string(),
                clankers_urdf::parse_string(PENDULUM_URDF).expect("parse pendulum URDF"),
            );
        }
        if mr_cfg.include_two_link {
            models.insert(
                "two_link_arm".to_string(),
                clankers_urdf::parse_string(TWO_LINK_ARM_URDF).expect("parse two_link_arm URDF"),
            );
        }
        if mr_cfg.include_six_dof {
            models.insert(
                "six_dof_arm".to_string(),
                clankers_urdf::parse_string(SIX_DOF_ARM_URDF).expect("parse six_dof_arm URDF"),
            );
        }

        let mut builder = SceneBuilder::new().with_max_episode_steps(mr_cfg.max_episode_steps);
        // SceneBuilder::with_robot is consumed sequentially; we apply in
        // a fixed order so RobotId assignment is reproducible:
        // pendulum (0), two_link_arm (1), six_dof_arm (2) — matching the
        // pre-PR2 bin order.
        if let Some(m) = models.get("pendulum") {
            builder = builder.with_robot(m.clone(), HashMap::new());
        }
        if let Some(m) = models.get("two_link_arm") {
            builder = builder.with_robot(m.clone(), HashMap::new());
        }
        if let Some(m) = models.get("six_dof_arm") {
            builder = builder.with_robot(m.clone(), HashMap::new());
        }
        let scene = builder.build();

        let mut layouts: HashMap<String, Arc<clankers_core::layout::JointLayout>> = HashMap::new();
        for (name, model) in &models {
            let Some(bot) = scene.robots.get(name) else {
                continue;
            };
            let mut layout = model.to_layout();
            let entities: Vec<Entity> = layout
                .joints()
                .iter()
                .map(|spec| {
                    bot.joint_entity(&spec.name)
                        .unwrap_or_else(|| panic!("joint {} not in {}", spec.name, name))
                })
                .collect();
            layout.bind_entities(&entities);
            layouts.insert(name.clone(), Arc::new(layout));
        }

        MultiRobotArtifacts {
            scene,
            models,
            layouts,
        }
    }
}
