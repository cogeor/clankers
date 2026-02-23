//! Mock implementations of core traits for testing.
//!
//! Provides lightweight stubs for sensors, reward functions, and termination
//! conditions that can be used in any crate's test suite.

use bevy::prelude::*;
use clankers_core::traits::{ObservationSensor, RewardFunction, Sensor, TerminationCondition};
use clankers_core::types::Observation;

// ---------------------------------------------------------------------------
// ConstantSensor
// ---------------------------------------------------------------------------

/// A sensor that always returns the same observation vector.
pub struct ConstantSensor {
    output: Vec<f32>,
}

impl ConstantSensor {
    /// Create a sensor that always returns `values`.
    pub const fn new(values: Vec<f32>) -> Self {
        Self { output: values }
    }

    /// Create a sensor that returns zeros of the given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            output: vec![0.0; dim],
        }
    }
}

impl Sensor for ConstantSensor {
    type Output = Observation;

    fn read(&self, _world: &mut World) -> Observation {
        Observation::new(self.output.clone())
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "ConstantSensor"
    }
}

impl ObservationSensor for ConstantSensor {
    fn observation_dim(&self) -> usize {
        self.output.len()
    }
}

// ---------------------------------------------------------------------------
// ConstantReward
// ---------------------------------------------------------------------------

/// A reward function that always returns a fixed value.
pub struct ConstantReward {
    value: f32,
    label: &'static str,
}

impl ConstantReward {
    /// Create a reward function that always returns `value`.
    pub const fn new(value: f32) -> Self {
        Self {
            value,
            label: "ConstantReward",
        }
    }

    /// Create a reward function with a custom label.
    pub const fn with_label(value: f32, label: &'static str) -> Self {
        Self { value, label }
    }
}

impl RewardFunction for ConstantReward {
    fn compute(&self, _world: &World) -> f32 {
        self.value
    }

    fn name(&self) -> &str {
        self.label
    }
}

// ---------------------------------------------------------------------------
// AlwaysTerminate / NeverTerminate
// ---------------------------------------------------------------------------

/// A termination condition that always signals termination.
pub struct AlwaysTerminate;

impl TerminationCondition for AlwaysTerminate {
    fn is_terminated(&self, _world: &World) -> bool {
        true
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "AlwaysTerminate"
    }
}

/// A termination condition that never signals termination.
pub struct NeverTerminate;

impl TerminationCondition for NeverTerminate {
    fn is_terminated(&self, _world: &World) -> bool {
        false
    }

    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "NeverTerminate"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_world() -> World {
        World::new()
    }

    // -- ConstantSensor --

    #[test]
    fn constant_sensor_returns_fixed_values() {
        let sensor = ConstantSensor::new(vec![1.0, 2.0, 3.0]);
        let mut world = test_world();
        let obs = sensor.read(&mut world);
        assert_eq!(obs.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn constant_sensor_zeros() {
        let sensor = ConstantSensor::zeros(4);
        let mut world = test_world();
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4);
        assert!(obs.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn constant_sensor_dim() {
        let sensor = ConstantSensor::new(vec![1.0, 2.0]);
        assert_eq!(sensor.observation_dim(), 2);
    }

    #[test]
    fn constant_sensor_name() {
        let sensor = ConstantSensor::zeros(1);
        assert_eq!(sensor.name(), "ConstantSensor");
    }

    // -- ConstantReward --

    #[test]
    fn constant_reward_returns_value() {
        let reward = ConstantReward::new(5.0);
        let world = test_world();
        assert!((reward.compute(&world) - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn constant_reward_default_name() {
        let reward = ConstantReward::new(0.0);
        assert_eq!(reward.name(), "ConstantReward");
    }

    #[test]
    fn constant_reward_custom_label() {
        let reward = ConstantReward::with_label(1.0, "distance");
        assert_eq!(reward.name(), "distance");
    }

    // -- AlwaysTerminate / NeverTerminate --

    #[test]
    fn always_terminate_returns_true() {
        let term = AlwaysTerminate;
        let world = test_world();
        assert!(term.is_terminated(&world));
    }

    #[test]
    fn never_terminate_returns_false() {
        let term = NeverTerminate;
        let world = test_world();
        assert!(!term.is_terminated(&world));
    }

    #[test]
    fn termination_names() {
        assert_eq!(AlwaysTerminate.name(), "AlwaysTerminate");
        assert_eq!(NeverTerminate.name(), "NeverTerminate");
    }

    // -- Send + Sync --

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn mock_types_are_send_sync() {
        assert_send_sync::<ConstantSensor>();
        assert_send_sync::<ConstantReward>();
        assert_send_sync::<AlwaysTerminate>();
        assert_send_sync::<NeverTerminate>();
    }
}
