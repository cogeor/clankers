//! `cartpole` scenario — headless OpenAI-Gym-style cart-pole.
//!
//! Spawns the `CARTPOLE_URDF` (2 joints: `cart_slide` prismatic +
//! `pole_hinge` continuous) with Rapier physics, registers a
//! [`JointStateSensor`] over the layout, and returns a
//! [`ScenarioHandle`] carrying the post-`bind_entities` `JointLayout`.
//!
//! # Loop 8 extension (W8 PR2)
//!
//! Loop 6 shipped the headless trait impl; loop 8 extends with:
//! - [`CartpoleConfig`] private knobs.
//! - [`CartpoleScenario::build_with`] parametrised entry point.
//! - [`CartPoleApplicator`] (the action-application glue moved out of
//!   the example bins).
//!
//! # App-swap convention
//!
//! The trait signature is `fn build(&self, app: &mut App, ...)`. The
//! caller hands in a fresh `App` already wired with
//! [`crate::ClankersSimPlugin`]; this scenario internally uses
//! [`crate::SceneBuilder`] which constructs its own `App`. To honour
//! the trait contract we `std::mem::swap` the SceneBuilder-built App
//! into the caller's slot at the end of `build`.
//!
//! # No rendering
//!
//! This scenario ships zero rendering code (no camera, no light, no
//! `DefaultPlugins`). It runs under the bevy minimal-plugin headless
//! harness. Visualisation bins add their own `DefaultPlugins` BEFORE
//! calling [`CartpoleScenario::build_with`] with `with_viz = true`.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_actuator::components::{JointCommand, JointState};
use clankers_core::layout::JointLayout;
use clankers_core::traits::ActionApplicator;
use clankers_core::types::Action;
use clankers_env::prelude::*;
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::{RapierBackend, RapierContext, bridge::register_robot};

use crate::SceneBuilder;
use crate::SpawnedScene;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the cart-pole robot, included at compile time from
/// `examples/urdf/cartpole.urdf`.
const CARTPOLE_URDF: &str = include_str!("../../../../examples/urdf/cartpole.urdf");

/// Default cart-pole observation dimension: `[cart_pos, cart_vel,
/// pole_angle, pole_vel]`.
pub const OBS_DIM: usize = 4;

/// Default cart-pole action dimension: `[cart_force, pole_torque]`.
/// The `pole_torque` is always 0 (passive joint) — but the layout has 2
/// slots so we must accept 2-element actions.
pub const ACT_DIM: usize = 2;

/// Per-scenario knobs for the `cartpole` scenario.
///
/// Per W8 PR1 Design B + W8 PR2 Design D: per-scenario config rides on a
/// private struct, not on [`ScenarioConfig`].
#[derive(Debug, Clone)]
pub struct CartpoleConfig {
    /// Observation dimension. Default `[OBS_DIM]`; bins may shrink to
    /// observe only `[cart_pos, pole_angle]` etc.
    pub obs_dim: usize,
    /// When `true`, additionally register a [`JointCommandSensor`]
    /// (rare — only `cartpole_gym.rs` currently uses this).
    pub register_command_sensor: bool,
    /// When `true`, the bin will wire `DefaultPlugins` separately; the
    /// scenario does not install Rapier rendering. Has no effect on
    /// scene setup — informational only.
    pub with_viz: bool,
    /// Max episode steps. Default 500 (matches `OpenAI` Gym `CartPole-v1`).
    pub max_episode_steps: u32,
}

impl Default for CartpoleConfig {
    fn default() -> Self {
        Self {
            obs_dim: OBS_DIM,
            register_command_sensor: false,
            with_viz: false,
            max_episode_steps: 500,
        }
    }
}

/// Artifacts returned by [`CartpoleScenario::build_with`].
pub struct CartpoleArtifacts {
    /// The spawned-scene wrapper (owns the App + robots map).
    pub scene: SpawnedScene,
    /// The parsed URDF model.
    pub model: clankers_urdf::RobotModel,
    /// Layout bound to the 2 cartpole joint entities.
    pub layout: Arc<JointLayout>,
}

/// Writes action values to joint commands in layout slot order.
///
/// Action layout: `[cart_force, pole_torque]`.
/// For standard `CartPole`, only `cart_force` (index 0) is used;
/// `pole_torque` (index 1) should be 0 (passive joint).
pub struct CartPoleApplicator {
    layout: Arc<JointLayout>,
}

impl CartPoleApplicator {
    /// Construct a new applicator over the given (bound) layout.
    #[must_use]
    pub const fn new(layout: Arc<JointLayout>) -> Self {
        Self { layout }
    }
}

impl ActionApplicator for CartPoleApplicator {
    fn apply(&self, world: &mut World, action: &Action) {
        let values = action
            .as_continuous()
            .expect("ActionApplicator contract: continuous action expected");
        for (i, entity) in self.layout.bound_entities().enumerate() {
            if i >= values.len() {
                break;
            }
            if let Some(mut cmd) = world.get_mut::<JointCommand>(entity) {
                cmd.value = values[i];
            }
        }
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "CartPoleApplicator"
    }

    fn layout(&self) -> &JointLayout {
        &self.layout
    }
}

// ---------------------------------------------------------------------------
// CartpoleScenario
// ---------------------------------------------------------------------------

/// Builder for the `cartpole` reference scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the global registry
/// by [`super::register_builtin`] alongside the arm family.
pub struct CartpoleScenario;

impl ScenarioBuilder for CartpoleScenario {
    fn name(&self) -> &'static str {
        "cartpole"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        let cart_cfg = CartpoleConfig {
            max_episode_steps: cfg.max_steps,
            ..CartpoleConfig::default()
        };
        let artefacts = Self::build_with(cfg, &cart_cfg);
        let handle = ScenarioHandle {
            layout: Some(artefacts.layout),
            max_steps: cfg.max_steps,
        };
        let mut scene = artefacts.scene;
        std::mem::swap(app, &mut scene.app);
        handle
    }
}

impl CartpoleScenario {
    /// Parametrised build path — used by `cartpole_gym.rs`,
    /// `cartpole_vec_gym.rs`, `cartpole_vec_benchmark.rs`, and
    /// `cartpole_policy_viz.rs`.
    #[must_use]
    pub fn build_with(cfg: &ScenarioConfig, cart_cfg: &CartpoleConfig) -> CartpoleArtifacts {
        let _ = cfg.seed;
        let model = clankers_urdf::parse_string(CARTPOLE_URDF)
            .expect("failed to parse cartpole URDF (compile-time include)");

        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(cart_cfg.max_episode_steps.max(1))
            .with_robot(model.clone(), HashMap::new())
            .build();

        // Add the Rapier physics backend.
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

        // Register robot bodies / joints with the rapier context.
        {
            let spawned = &scene.robots["cartpole"];
            let world = scene.app.world_mut();
            let mut ctx = world
                .remove_resource::<RapierContext>()
                .expect("RapierContext present after ClankersPhysicsPlugin");
            register_robot(&mut ctx, &model, spawned, world, true);
            world.insert_resource(ctx);
        }

        // Build the bound layout and register the joint-state sensor.
        let layout = {
            let bot = &scene.robots["cartpole"];
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
        {
            let world = scene.app.world_mut();
            let mut registry = world
                .remove_resource::<SensorRegistry>()
                .expect("SensorRegistry present after ClankersEnvPlugin");
            let mut buffer = world
                .remove_resource::<ObservationBuffer>()
                .expect("ObservationBuffer present after ClankersEnvPlugin");
            registry.register(Box::new(JointStateSensor::new(layout.clone())), &mut buffer);
            if cart_cfg.register_command_sensor {
                registry.register(
                    Box::new(JointCommandSensor::new(layout.clone())),
                    &mut buffer,
                );
            }
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        CartpoleArtifacts {
            scene,
            model,
            layout,
        }
    }

    /// Reset-fn callback baked into the cartpole reset path.
    ///
    /// Resets `RapierContext` to its initial snapshot, then zeros all
    /// joint state + command components. Used by the cartpole gym /
    /// vec-gym bins.
    pub fn reset_world(world: &mut World) {
        if let Some(mut ctx) = world.remove_resource::<RapierContext>() {
            ctx.reset_to_initial();
            world.insert_resource(ctx);
        }
        let mut query = world.query::<(&mut JointState, &mut JointCommand)>();
        for (mut state, mut cmd) in query.iter_mut(world) {
            state.position = 0.0;
            state.velocity = 0.0;
            cmd.value = 0.0;
        }
    }
}
