//! Mock implementations of core traits for testing.
//!
//! Provides lightweight stubs for sensors that can be used in any crate's
//! test suite.

use bevy::prelude::*;
use clankers_core::traits::{ObservationSensor, Sensor};
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_world() -> World {
        World::new()
    }

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

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn mock_types_are_send_sync() {
        assert_send_sync::<ConstantSensor>();
    }
}
