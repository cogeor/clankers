//! [`RapierBackend`] — concrete physics backend using raw `rapier3d`.

use bevy::prelude::*;

use clankers_core::config::SimConfig;
use clankers_core::ClankersSet;

use crate::backend::PhysicsBackend;

use super::context::RapierContext;
use super::systems::rapier_step_system;

/// Shared setup: insert RapierContext resource from SimConfig.
fn insert_rapier_context(app: &mut App) {
    let sim_config = app.world().resource::<SimConfig>();

    let gravity = Vec3::new(
        sim_config.gravity[0],
        sim_config.gravity[1],
        sim_config.gravity[2],
    );
    #[allow(clippy::cast_possible_truncation)]
    let dt = sim_config.physics_dt as f32;
    let substeps = sim_config.substeps();

    let context = RapierContext::new(gravity, dt, substeps);
    app.insert_resource(context);
}

/// Raw rapier3d physics backend.
///
/// Inserts a [`RapierContext`] resource and registers the physics step system
/// in [`ClankersSet::Simulate`] on the `Update` schedule.
/// Reads gravity and timestep from [`SimConfig`].
pub struct RapierBackend;

impl PhysicsBackend for RapierBackend {
    fn build(&self, app: &mut App) {
        insert_rapier_context(app);
        app.add_systems(Update, rapier_step_system.in_set(ClankersSet::Simulate));
    }

    fn name(&self) -> &str {
        "rapier3d"
    }
}

/// Rapier3d physics backend that runs on `FixedUpdate` instead of `Update`.
///
/// Use this when the simulation should run at a fixed rate decoupled from the
/// render frame rate (e.g. visualization binaries). Set the `FixedUpdate`
/// timestep via `Time<Fixed>` before calling `app.run()`.
pub struct RapierBackendFixed;

impl PhysicsBackend for RapierBackendFixed {
    fn build(&self, app: &mut App) {
        insert_rapier_context(app);
        app.add_systems(FixedUpdate, rapier_step_system.in_set(ClankersSet::Simulate));
    }

    fn name(&self) -> &str {
        "rapier3d-fixed"
    }
}
