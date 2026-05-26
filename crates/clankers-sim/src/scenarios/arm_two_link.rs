//! `arm_two_link` scenario — minimal 2-link planar arm (shoulder + elbow).
//!
//! Spawns the 2-DOF planar arm URDF (used by `arm_with_policy.rs` to
//! exercise the [`crate::SceneBuilder`] + sensor pipeline against the
//! smallest meaningful articulated robot). No physics backend is
//! required; the scenario installs only the actuator + env plugins via
//! [`crate::SceneBuilder`]. Callers wanting Rapier can layer
//! [`clankers_physics::ClankersPhysicsPlugin`] on themselves.
//!
//! # Deviation
//!
//! Per W8 PR1 PLAN deviation 1, this file is NOT in WS8-plan § 4 NEW
//! table. It exists because `arm_with_policy.rs` uses a 2-DOF URDF that
//! cannot reuse the 6-DOF `arm_ik` / `arm_pick` / `arm_bench` family.
//! Listed as deviation 1 in the loop 7 PLAN.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_env::prelude::*;

use crate::SceneBuilder;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the 2-link planar arm, included at compile time from
/// `examples/urdf/two_link_arm.urdf`.
const TWO_LINK_ARM_URDF: &str = include_str!("../../../../examples/urdf/two_link_arm.urdf");

/// Per-scenario knobs that do NOT belong on
/// [`ScenarioConfig`](crate::scenarios::ScenarioConfig).
///
/// Per W8 PR1 Design choice B, the W5 PR1 `ScenarioConfig` field set is
/// locked; scenario-specific tuning rides on private config structs
/// consumed by [`ArmTwoLinkScenario::build_with`].
#[derive(Debug, Clone)]
pub struct ArmTwoLinkConfig {
    /// Hard cap on episode steps. Defaults to
    /// `ScenarioConfig::max_steps` if not overridden.
    pub max_episode_steps: u32,
    /// Whether to register a
    /// [`JointCommandSensor`](clankers_env::prelude::JointCommandSensor)
    /// alongside the standard
    /// [`JointStateSensor`](clankers_env::prelude::JointStateSensor).
    /// Defaults to `true` (matches `arm_with_policy.rs` today).
    pub register_command_sensor: bool,
}

impl Default for ArmTwoLinkConfig {
    fn default() -> Self {
        Self {
            max_episode_steps: 20,
            register_command_sensor: true,
        }
    }
}

/// Builder for the `arm_two_link` scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the registry by
/// [`super::register_builtin`] alongside the rest of the arm family.
pub struct ArmTwoLinkScenario;

impl ScenarioBuilder for ArmTwoLinkScenario {
    fn name(&self) -> &'static str {
        "arm_two_link"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let two_link_cfg = ArmTwoLinkConfig {
            max_episode_steps: cfg.max_steps,
            ..ArmTwoLinkConfig::default()
        };
        Self::build_with(app, cfg, &two_link_cfg)
    }
}

impl ArmTwoLinkScenario {
    /// Parametrised build path consumed by `arm_with_policy.rs` and any
    /// other caller that wants to tweak the per-scenario knobs without
    /// touching the field-locked [`ScenarioConfig`].
    ///
    /// Returns a [`ScenarioHandle`] carrying the bound 2-joint layout.
    pub fn build_with(
        app: &mut App,
        cfg: &ScenarioConfig,
        two_link_cfg: &ArmTwoLinkConfig,
    ) -> ScenarioHandle {
        let arm_model = clankers_urdf::parse_string(TWO_LINK_ARM_URDF)
            .expect("failed to parse two_link_arm URDF (compile-time include)");

        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(two_link_cfg.max_episode_steps.min(cfg.max_steps).max(1))
            .with_robot(arm_model.clone(), HashMap::new())
            .build();

        // Build a layout bound to the spawned 2-DOF arm joints.
        let layout = {
            let bot = &scene.robots["two_link_arm"];
            let mut layout = arm_model.to_layout();
            let entities: Vec<Entity> = layout
                .joints()
                .iter()
                .map(|spec| {
                    bot.joint_entity(&spec.name)
                        .unwrap_or_else(|| panic!("joint {} not in arm", spec.name))
                })
                .collect();
            layout.bind_entities(&entities);
            Arc::new(layout)
        };

        // Register the joint-state sensor (and optionally the command
        // sensor) on the bound layout.
        {
            let world = scene.app.world_mut();
            let mut registry = world
                .remove_resource::<SensorRegistry>()
                .expect("SensorRegistry present after ClankersEnvPlugin");
            let mut buffer = world
                .remove_resource::<ObservationBuffer>()
                .expect("ObservationBuffer present after ClankersEnvPlugin");
            registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
            if two_link_cfg.register_command_sensor {
                registry.register(
                    Box::new(JointCommandSensor::new(layout.clone())),
                    &mut buffer,
                );
            }
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        // Swap into the caller's App; see arm_pick.rs for the rationale.
        std::mem::swap(app, &mut scene.app);

        ScenarioHandle {
            layout: Some(layout),
            max_steps: cfg.max_steps,
        }
    }
}
