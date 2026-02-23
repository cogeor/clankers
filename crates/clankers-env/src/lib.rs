//! Environment state management, sensors, and episode lifecycle for Clankers.
//!
//! This crate provides the ECS integration layer for environment simulation:
//! episode lifecycle, observation collection, and sensor management.
//!
//! # Example
//!
//! ```no_run
//! use bevy::prelude::*;
//! use clankers_env::prelude::*;
//!
//! App::new()
//!     .add_plugins(clankers_core::ClankersCorePlugin)
//!     .add_plugins(ClankersEnvPlugin)
//!     .run();
//! ```

pub mod buffer;
pub mod episode;
pub mod sensors;
pub mod systems;
pub mod vec_buffer;
pub mod vec_env;
pub mod vec_episode;
pub mod vec_runner;

use bevy::prelude::*;
use clankers_core::ClankersSet;
use clankers_core::traits::ObservationSensor;

// ---------------------------------------------------------------------------
// SensorRegistry
// ---------------------------------------------------------------------------

/// Entry in the sensor registry: a boxed sensor paired with its buffer slot.
pub struct SensorEntry {
    pub sensor: Box<dyn ObservationSensor>,
    pub slot_index: usize,
}

/// Resource holding all registered observation sensors.
///
/// Sensors are registered with [`register`](Self::register), which also allocates
/// a slot in the [`ObservationBuffer`](buffer::ObservationBuffer).
#[derive(Resource, Default)]
pub struct SensorRegistry {
    pub(crate) entries: Vec<SensorEntry>,
}

impl SensorRegistry {
    /// Create an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a sensor, allocating a slot in the buffer. Returns the slot index.
    pub fn register(
        &mut self,
        sensor: Box<dyn ObservationSensor>,
        buffer: &mut buffer::ObservationBuffer,
    ) -> usize {
        let slot_index = buffer.register(sensor.name(), sensor.observation_dim());
        self.entries.push(SensorEntry { sensor, slot_index });
        slot_index
    }

    /// Number of registered sensors.
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ClankersEnvPlugin
// ---------------------------------------------------------------------------

/// Bevy plugin that initializes environment resources and adds observation
/// collection and episode lifecycle systems.
///
/// Adds:
/// - [`Episode`](episode::Episode) and [`EpisodeConfig`](episode::EpisodeConfig) resources
/// - [`ObservationBuffer`](buffer::ObservationBuffer) and [`SensorRegistry`] resources
/// - [`observe_system`](systems::observe_system) in [`ClankersSet::Observe`]
/// - [`episode_step_system`](systems::episode_step_system) in [`ClankersSet::Evaluate`]
pub struct ClankersEnvPlugin;

impl Plugin for ClankersEnvPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<episode::Episode>()
            .init_resource::<episode::EpisodeConfig>()
            .init_resource::<buffer::ObservationBuffer>()
            .init_resource::<SensorRegistry>()
            .add_systems(Update, systems::observe_system.in_set(ClankersSet::Observe))
            .add_systems(
                Update,
                systems::episode_step_system.in_set(ClankersSet::Evaluate),
            );
    }
}

// ---------------------------------------------------------------------------
// Prelude
// ---------------------------------------------------------------------------

pub mod prelude {
    pub use crate::{
        ClankersEnvPlugin, SensorRegistry,
        buffer::{ObservationBuffer, SensorSlot},
        episode::{Episode, EpisodeConfig, EpisodeState},
        sensors::{
            ContactSensor, ImuSensor, JointCommandSensor, JointStateSensor, JointTorqueSensor,
            NoisySensor, RaycastSensor, RobotContactSensor, RobotImuSensor,
            RobotJointCommandSensor, RobotJointStateSensor, RobotJointTorqueSensor,
            RobotRaycastSensor,
        },
        systems::{episode_step_system, observe_system},
        vec_buffer::{VecDoneBuffer, VecObsBuffer},
        vec_env::VecEnvConfig,
        vec_episode::{AutoResetMode, EnvEpisodeMap},
        vec_runner::{VecEnvInstance, VecEnvRunner},
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_builds_without_panic() {
        let mut app = App::new();
        app.add_plugins(clankers_core::ClankersCorePlugin);
        app.add_plugins(ClankersEnvPlugin);
        app.finish();
        app.cleanup();
        app.update();

        // Verify resources exist
        assert!(app.world().get_resource::<episode::Episode>().is_some());
        assert!(
            app.world()
                .get_resource::<episode::EpisodeConfig>()
                .is_some()
        );
        assert!(
            app.world()
                .get_resource::<buffer::ObservationBuffer>()
                .is_some()
        );
        assert!(app.world().get_resource::<SensorRegistry>().is_some());
    }

    #[test]
    fn sensor_registry_register_and_len() {
        let mut registry = SensorRegistry::new();
        let mut buffer = buffer::ObservationBuffer::new();
        assert!(registry.is_empty());

        let idx = registry.register(Box::new(sensors::JointStateSensor::new(3)), &mut buffer);
        assert_eq!(idx, 0);
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert_eq!(buffer.dim(), 6);
    }

    #[test]
    fn sensor_registry_multiple() {
        let mut registry = SensorRegistry::new();
        let mut buffer = buffer::ObservationBuffer::new();

        registry.register(Box::new(sensors::JointStateSensor::new(2)), &mut buffer);
        registry.register(Box::new(sensors::JointCommandSensor::new(2)), &mut buffer);

        assert_eq!(registry.len(), 2);
        assert_eq!(buffer.dim(), 6); // 4 + 2
    }
}
