//! [`RapierBackend`] — concrete physics backend using raw `rapier3d`.

use bevy::prelude::*;

use clankers_core::config::SimConfig;
use clankers_core::ClankersSet;

use crate::backend::PhysicsBackend;

use super::context::RapierContext;
use super::systems::rapier_step_system;

/// Raw rapier3d physics backend.
///
/// Inserts a [`RapierContext`] resource and registers the physics step system
/// in [`ClankersSet::Simulate`]. Reads gravity and timestep from [`SimConfig`].
pub struct RapierBackend;

impl PhysicsBackend for RapierBackend {
    fn build(&self, app: &mut App) {
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

        app.add_systems(Update, rapier_step_system.in_set(ClankersSet::Simulate));
    }

    fn name(&self) -> &str {
        "rapier3d"
    }
}
