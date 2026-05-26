//! `cartpole` scenario — headless OpenAI-Gym-style cart-pole.
//!
//! Spawns the `CARTPOLE_URDF` (2 joints: `cart_slide` prismatic +
//! `pole_hinge` continuous) with Rapier physics, registers a
//! [`JointStateSensor`] over the layout, and returns a
//! [`ScenarioHandle`] carrying the post-`bind_entities` `JointLayout`.
//!
//! # NOTE — duplicated setup
//!
//! The setup body is a self-contained near-duplicate of the headless
//! subset of `examples/src/bin/cartpole_gym.rs`. **W8 PR1 will lift the
//! example bin's setup logic to call this scenario instead**; until
//! then the two implementations may drift. The scenario is the
//! authoritative source — do **not** edit the example bin to fix bugs
//! that originated here.
//!
//! # App-swap convention
//!
//! The trait signature is `fn build(&self, app: &mut App, ...)`. The
//! caller hands in a fresh `App` already wired with
//! [`crate::ClankersSimPlugin`]; this scenario internally uses
//! [`crate::SceneBuilder`] which constructs its own `App`. To honour
//! the trait contract we `std::mem::swap` the SceneBuilder-built App
//! into the caller's slot at the end of `build`. The original
//! caller-supplied App (which only had `ClankersSimPlugin` and no
//! robot) is dropped.
//!
//! W8 PR1 will introduce `SceneBuilder::build_into(&mut App)` so this
//! swap dance can disappear.
//!
//! # No rendering
//!
//! This scenario ships zero rendering code (no camera, no light, no
//! `DefaultPlugins`). It runs under the bevy minimal-plugin headless
//! harness, matching the W5 PR2 GPU-off-limits constraint.

use std::collections::HashMap;
use std::sync::Arc;

use bevy::prelude::*;
use clankers_env::prelude::*;
use clankers_physics::ClankersPhysicsPlugin;
use clankers_physics::rapier::{RapierBackend, RapierContext, bridge::register_robot};

use crate::SceneBuilder;
use crate::scenarios::{ScenarioBuilder, ScenarioConfig, ScenarioHandle};

/// URDF source for the cart-pole robot, included at compile time from
/// `examples/urdf/cartpole.urdf` (4 levels up from this file → workspace
/// root → `examples/urdf/`).
///
/// W8 PR1 will move the URDF assets into a `clankers-assets` crate (or
/// into `clankers-sim` proper) and delete this `include_str!` shim. See
/// loop 06 PLAN Design choice F.
const CARTPOLE_URDF: &str = include_str!("../../../../examples/urdf/cartpole.urdf");

// ---------------------------------------------------------------------------
// CartpoleScenario
// ---------------------------------------------------------------------------

/// Builder for the `cartpole` reference scenario.
///
/// Implements [`ScenarioBuilder`]; registered into the global registry
/// by [`super::register_builtin`] alongside `arm_pick`.
pub struct CartpoleScenario;

impl ScenarioBuilder for CartpoleScenario {
    fn name(&self) -> &'static str {
        "cartpole"
    }

    fn build(&self, app: &mut App, cfg: &ScenarioConfig) -> ScenarioHandle {
        // 1. Parse URDF (compile-time include — failure here is a build
        //    bug, not a runtime one).
        let model = clankers_urdf::parse_string(CARTPOLE_URDF)
            .expect("failed to parse cartpole URDF (compile-time include)");

        // 2. Build the scene on a fresh App owned by SceneBuilder; we
        //    swap it into the caller's slot at the end. See the
        //    module-level "App-swap convention" note.
        let mut scene = SceneBuilder::new()
            .with_max_episode_steps(cfg.max_steps)
            .with_robot(model.clone(), HashMap::new())
            .build();

        // 3. Add the Rapier physics backend.
        scene
            .app
            .add_plugins(ClankersPhysicsPlugin::new(RapierBackend));

        // 4. Register robot bodies / joints with the rapier context.
        {
            let spawned = &scene.robots["cartpole"];
            let world = scene.app.world_mut();
            let mut ctx = world
                .remove_resource::<RapierContext>()
                .expect("RapierContext present after ClankersPhysicsPlugin");
            register_robot(&mut ctx, &model, spawned, world, true);
            world.insert_resource(ctx);
        }

        // 5. Build the bound layout and register the joint-state sensor.
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
            world.insert_resource(buffer);
            world.insert_resource(registry);
        }

        // 6. Move the SceneBuilder-owned App into the caller's slot. The
        //    pre-swap App (which the caller wired with ClankersSimPlugin
        //    only) gets dropped at the end of this block.
        std::mem::swap(app, &mut scene.app);

        ScenarioHandle {
            layout: Some(layout),
            max_steps: cfg.max_steps,
        }
    }
}
