// clankers-core: Types, traits, config, time, errors for Clankers robotics simulation.

pub mod baselines;
pub mod config;
pub mod env_spec;
pub mod error;
pub mod layout;
pub mod manifest;
pub mod physics;
pub mod schema;
pub mod seed;
pub mod stability;
pub mod time;
pub mod traits;
pub mod types;
pub mod user_journeys;
pub mod validators;
pub mod view;

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// ClankersSet
// ---------------------------------------------------------------------------

/// System sets defining the canonical execution order for Clankers simulation
/// stages within the Bevy `Update` schedule.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ClankersSet {
    Observe,
    Decide,
    Act,
    Simulate,
    BuildFrameTree,
    Update,
    Evaluate,
    Communicate,
}

// ---------------------------------------------------------------------------
// ClankersCorePlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that configures the core Clankers system sets and default
/// resources.
///
/// Adds the [`ClankersSet`] chain to the `Update` schedule and initializes
/// [`SimConfig`](config::SimConfig) and [`SimTime`](time::SimTime) resources.
pub struct ClankersCorePlugin;

impl Plugin for ClankersCorePlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            (
                ClankersSet::Observe,
                ClankersSet::Decide,
                ClankersSet::Act,
                ClankersSet::Simulate,
                ClankersSet::BuildFrameTree,
                ClankersSet::Update,
                ClankersSet::Evaluate,
                ClankersSet::Communicate,
            )
                .chain(),
        );
        // Also configure on FixedUpdate so viz can run physics at fixed rate.
        app.configure_sets(
            FixedUpdate,
            (
                ClankersSet::Observe,
                ClankersSet::Decide,
                ClankersSet::Act,
                ClankersSet::Simulate,
                ClankersSet::BuildFrameTree,
                ClankersSet::Update,
                ClankersSet::Evaluate,
                ClankersSet::Communicate,
            )
                .chain(),
        );
        app.init_resource::<config::SimConfig>();
        app.init_resource::<time::SimTime>();
        app.init_resource::<seed::SeedHierarchy>();
        app.init_resource::<types::RobotGroup>();
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        // Bevy integration
        ClankersCorePlugin,
        ClankersSet,
        // Config
        config::{ObjectConfig, RobotConfig, SceneConfig, Shape, SimConfig, TaskConfig},
        // Errors
        error::{
            ActionKindError, ClankersError, ConfigError, SimError, SpaceError, ValidationError,
        },
        // Layout (W1 + W2)
        layout::{JointKind, JointLayout, JointLayoutBuilder, JointSpec, JointSpecLimits},
        // Physics
        physics::{
            ContactData, EndEffectorState, ExternalForce, ImuData, LidarConfig, Mass,
            RaycastResult, SurfaceFriction,
        },
        // Schema (W1)
        schema::{
            ActionSchema, ActionSemantics, FrameSchema, ObservationSchema, ObservationSlot,
            RecorderSchema, SchemaMismatch,
        },
        // Seeds
        seed::{SeedHierarchy, derive_seed, derive_seed_indexed},
        // Time
        time::{Accumulator, Clock, SimTime},
        // Traits
        traits::{
            ActionApplicator, MultiRobotActionApplicator, ObservationSensor, Policy, Sensor,
            Simulation,
        },
        // Types
        types::{
            Action, ActionSpace, BatchResetResult, BatchStepResult, CompositeHandle, ContactEvent,
            EnvId, LayoutCompileError, MissingJoints, ObjectHandle, Observation, ObservationSpace,
            ResetInfo, ResetResult, RobotGroup, RobotHandle, RobotId, RobotInfo, SegmentationClass,
            SensorHandle, StepInfo, StepResult,
        },
        // View (W3)
        view::ObservationView,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clankers_core_plugin_registers_resources() {
        let mut app = App::new();
        app.add_plugins(ClankersCorePlugin);
        app.finish();
        app.cleanup();
        app.update();

        // Verify SimConfig resource exists with defaults.
        let sim_config = app.world().resource::<config::SimConfig>();
        assert!((sim_config.physics_dt - 0.001).abs() < f64::EPSILON);
        assert!((sim_config.control_dt - 0.02).abs() < f64::EPSILON);
        assert_eq!(sim_config.max_episode_steps, 1000);
        assert_eq!(sim_config.seed, 0);

        // Verify SimTime resource exists at zero.
        let sim_time = app.world().resource::<time::SimTime>();
        assert_eq!(sim_time.nanos(), 0);

        // Verify SeedHierarchy resource exists.
        let seeds = app.world().resource::<seed::SeedHierarchy>();
        assert_eq!(seeds.root(), 0);

        // Verify RobotGroup resource exists.
        let group = app.world().resource::<types::RobotGroup>();
        assert!(group.is_empty());
    }
}
