//! Image sensor bridging the [`FrameBuffer`] to the observation pipeline.
//!
//! [`ImageSensor`] implements [`Sensor`] and [`ObservationSensor`] by reading
//! the current [`FrameBuffer`] and normalising pixel values to `[0.0, 1.0]`.

use bevy::prelude::*;

use clankers_core::traits::{ObservationSensor, Sensor};
use clankers_core::types::Observation;

use crate::buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// ImageSensor
// ---------------------------------------------------------------------------

/// Sensor that reads the [`FrameBuffer`] and produces a flat [`Observation`].
///
/// Each byte in the frame buffer is normalised to `[0.0, 1.0]` by dividing
/// by 255. The resulting observation vector has length
/// `width * height * channels`.
///
/// # Example
///
/// ```
/// use clankers_render::sensor::ImageSensor;
/// use clankers_core::traits::Sensor;
///
/// let sensor = ImageSensor::new("camera_front");
/// assert_eq!(sensor.name(), "camera_front");
/// ```
pub struct ImageSensor {
    label: String,
    rate: Option<f64>,
}

impl ImageSensor {
    /// Create an image sensor with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            label: name.into(),
            rate: None,
        }
    }

    /// Set the capture rate in Hz.
    #[must_use]
    pub const fn with_rate_hz(mut self, hz: f64) -> Self {
        self.rate = Some(hz);
        self
    }
}

impl Sensor for ImageSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Self::Output {
        let buf = world.resource::<FrameBuffer>();
        let data: Vec<f32> = buf.data().iter().map(|&b| f32::from(b) / 255.0).collect();
        Observation::new(data)
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn rate_hz(&self) -> Option<f64> {
        self.rate
    }
}

impl ObservationSensor for ImageSensor {
    fn observation_dim(&self) -> usize {
        // This is a dynamic sensor — dim depends on the frame buffer size.
        // Return 0 as the static dimension; actual dim comes from the observation.
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PixelFormat;

    #[test]
    fn image_sensor_name() {
        let sensor = ImageSensor::new("cam_0");
        assert_eq!(sensor.name(), "cam_0");
    }

    #[test]
    fn image_sensor_default_rate() {
        let sensor = ImageSensor::new("cam");
        assert!(sensor.rate_hz().is_none());
    }

    #[test]
    fn image_sensor_with_rate() {
        let sensor = ImageSensor::new("cam").with_rate_hz(30.0);
        assert!((sensor.rate_hz().unwrap() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn image_sensor_reads_frame_buffer() {
        let mut world = World::new();
        let mut buf = FrameBuffer::new(2, 1, PixelFormat::Rgb8);
        buf.write_frame(vec![0, 128, 255, 64, 192, 32]);
        world.insert_resource(buf);

        let sensor = ImageSensor::new("test");
        let obs = sensor.read(&mut world);

        assert_eq!(obs.len(), 6);
        assert!((obs[0] - 0.0).abs() < f32::EPSILON);
        assert!((obs[1] - f32::from(128_u8) / 255.0).abs() < 0.001);
        assert!((obs[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn image_sensor_normalises_to_unit_range() {
        let mut world = World::new();
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgb8);
        buf.write_frame(vec![0, 127, 255]);
        world.insert_resource(buf);

        let sensor = ImageSensor::new("test");
        let obs = sensor.read(&mut world);

        for val in obs.as_slice() {
            assert!(*val >= 0.0);
            assert!(*val <= 1.0);
        }
    }

    #[test]
    fn image_sensor_rgba() {
        let mut world = World::new();
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgba8);
        buf.write_frame(vec![255, 0, 128, 255]);
        world.insert_resource(buf);

        let sensor = ImageSensor::new("test");
        let obs = sensor.read(&mut world);

        assert_eq!(obs.len(), 4);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
        assert!((obs[1] - 0.0).abs() < f32::EPSILON);
        assert!((obs[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn image_sensor_observation_dim() {
        let sensor = ImageSensor::new("test");
        // Dynamic sensor returns 0 for static dim
        assert_eq!(sensor.observation_dim(), 0);
    }
}
