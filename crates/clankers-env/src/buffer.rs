//! Pre-allocated observation buffer for efficient sensor data collection.
//!
//! Sensors write into the buffer at fixed offsets. The buffer can be
//! converted to an [`Observation`] once all sensors have written.

use bevy::prelude::*;
use clankers_core::types::Observation;

// ---------------------------------------------------------------------------
// SensorSlot
// ---------------------------------------------------------------------------

/// Metadata for a registered sensor's slot in the observation buffer.
#[derive(Clone, Debug)]
pub struct SensorSlot {
    /// Human-readable sensor name.
    pub name: String,
    /// Number of f32 values this sensor produces.
    pub dim: usize,
    /// Start offset into the observation buffer.
    pub offset: usize,
}

// ---------------------------------------------------------------------------
// ObservationBuffer
// ---------------------------------------------------------------------------

/// Pre-allocated buffer for collecting sensor observations.
///
/// Sensors are registered with a name and dimension.  Each sensor gets a
/// fixed slice `[offset..offset+dim)` in the buffer.  After all sensors
/// write, the buffer is read out as an [`Observation`].
#[derive(Resource, Clone, Debug)]
pub struct ObservationBuffer {
    /// Flat f32 storage.
    data: Vec<f32>,
    /// Registered sensor slots (in registration order).
    slots: Vec<SensorSlot>,
    /// Total dimension (sum of all sensor dims).
    total_dim: usize,
}

impl Default for ObservationBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationBuffer {
    /// Create an empty buffer with no sensors registered.
    pub const fn new() -> Self {
        Self {
            data: Vec::new(),
            slots: Vec::new(),
            total_dim: 0,
        }
    }

    /// Register a sensor and allocate its slot.  Returns the slot index.
    pub fn register(&mut self, name: impl Into<String>, dim: usize) -> usize {
        let offset = self.total_dim;
        let index = self.slots.len();
        self.slots.push(SensorSlot {
            name: name.into(),
            dim,
            offset,
        });
        self.total_dim += dim;
        self.data.resize(self.total_dim, 0.0);
        index
    }

    /// Total observation dimension.
    pub const fn dim(&self) -> usize {
        self.total_dim
    }

    /// Number of registered sensors.
    pub const fn sensor_count(&self) -> usize {
        self.slots.len()
    }

    /// Get the sensor slot metadata by index.
    pub fn slot(&self, index: usize) -> Option<&SensorSlot> {
        self.slots.get(index)
    }

    /// Get all sensor slots.
    pub fn slots(&self) -> &[SensorSlot] {
        &self.slots
    }

    /// Write values into a sensor's slot.  Panics if `values.len() != slot.dim`.
    pub fn write(&mut self, slot_index: usize, values: &[f32]) {
        let slot = &self.slots[slot_index];
        assert_eq!(
            values.len(),
            slot.dim,
            "sensor '{}': expected {} values, got {}",
            slot.name,
            slot.dim,
            values.len()
        );
        self.data[slot.offset..slot.offset + slot.dim].copy_from_slice(values);
    }

    /// Write a single scalar to a sensor slot at the given sub-index.
    pub fn write_scalar(&mut self, slot_index: usize, sub_index: usize, value: f32) {
        let slot = &self.slots[slot_index];
        assert!(
            sub_index < slot.dim,
            "sensor '{}': sub_index {} >= dim {}",
            slot.name,
            sub_index,
            slot.dim
        );
        self.data[slot.offset + sub_index] = value;
    }

    /// Read the values for a sensor slot.
    pub fn read(&self, slot_index: usize) -> &[f32] {
        let slot = &self.slots[slot_index];
        &self.data[slot.offset..slot.offset + slot.dim]
    }

    /// Convert the buffer to an [`Observation`].
    pub fn as_observation(&self) -> Observation {
        Observation::new(self.data.clone())
    }

    /// Clear the buffer (fill with zeros). Slots remain registered.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Raw data slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer() {
        let buf = ObservationBuffer::new();
        assert_eq!(buf.dim(), 0);
        assert_eq!(buf.sensor_count(), 0);
        assert!(buf.as_slice().is_empty());
    }

    #[test]
    fn register_single_sensor() {
        let mut buf = ObservationBuffer::new();
        let idx = buf.register("joint_pos", 3);
        assert_eq!(idx, 0);
        assert_eq!(buf.dim(), 3);
        assert_eq!(buf.sensor_count(), 1);
        assert_eq!(buf.slot(0).unwrap().name, "joint_pos");
        assert_eq!(buf.slot(0).unwrap().offset, 0);
    }

    #[test]
    fn register_multiple_sensors() {
        let mut buf = ObservationBuffer::new();
        let a = buf.register("positions", 4);
        let b = buf.register("velocities", 4);
        let c = buf.register("torques", 2);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(buf.dim(), 10);
        assert_eq!(buf.sensor_count(), 3);
        assert_eq!(buf.slot(1).unwrap().offset, 4);
        assert_eq!(buf.slot(2).unwrap().offset, 8);
    }

    #[test]
    fn write_and_read() {
        let mut buf = ObservationBuffer::new();
        let a = buf.register("pos", 3);
        let b = buf.register("vel", 2);
        buf.write(a, &[1.0, 2.0, 3.0]);
        buf.write(b, &[4.0, 5.0]);
        assert_eq!(buf.read(a), &[1.0, 2.0, 3.0]);
        assert_eq!(buf.read(b), &[4.0, 5.0]);
    }

    #[test]
    fn write_scalar() {
        let mut buf = ObservationBuffer::new();
        let idx = buf.register("test", 3);
        buf.write_scalar(idx, 0, 10.0);
        buf.write_scalar(idx, 2, 30.0);
        assert_eq!(buf.read(idx), &[10.0, 0.0, 30.0]);
    }

    #[test]
    fn as_observation() {
        let mut buf = ObservationBuffer::new();
        buf.register("a", 2);
        buf.register("b", 1);
        buf.write(0, &[1.0, 2.0]);
        buf.write(1, &[3.0]);
        let obs = buf.as_observation();
        assert_eq!(obs.len(), 3);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
        assert!((obs[1] - 2.0).abs() < f32::EPSILON);
        assert!((obs[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn clear_zeros_data() {
        let mut buf = ObservationBuffer::new();
        let idx = buf.register("test", 3);
        buf.write(idx, &[1.0, 2.0, 3.0]);
        buf.clear();
        assert_eq!(buf.read(idx), &[0.0, 0.0, 0.0]);
        // Slots remain
        assert_eq!(buf.sensor_count(), 1);
    }

    #[test]
    #[should_panic(expected = "expected 3 values, got 2")]
    fn write_wrong_dim_panics() {
        let mut buf = ObservationBuffer::new();
        let idx = buf.register("test", 3);
        buf.write(idx, &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "sub_index 3 >= dim 3")]
    fn write_scalar_oob_panics() {
        let mut buf = ObservationBuffer::new();
        let idx = buf.register("test", 3);
        buf.write_scalar(idx, 3, 1.0);
    }

    #[test]
    fn default_is_empty() {
        let buf = ObservationBuffer::default();
        assert_eq!(buf.dim(), 0);
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn buffer_is_send_sync() {
        assert_send_sync::<ObservationBuffer>();
    }
}
