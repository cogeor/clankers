//! Image sensor bridging the [`CameraFrameBuffers`] to the observation pipeline.
//!
//! [`ImageSensor`] implements [`Sensor`] and [`ObservationSensor`] by reading
//! the current frame for its named camera from [`CameraFrameBuffers`] and
//! normalising pixel values to `[0.0, 1.0]`.

use bevy::prelude::*;

use clankers_core::traits::{ObservationSensor, Sensor};
use clankers_core::types::Observation;

use crate::buffer::CameraFrameBuffers;

// ---------------------------------------------------------------------------
// ImageSensor
// ---------------------------------------------------------------------------

/// Sensor that reads a named camera's frame from [`CameraFrameBuffers`] and
/// produces a flat [`Observation`].
///
/// Each byte in the frame buffer is normalised to `[0.0, 1.0]` by dividing
/// by 255. The resulting observation vector has length
/// `width * height * channels`.
///
/// # Example
///
/// ```
/// use clankers_render::sensor::ImageSensor;
/// use clankers_core::traits::{Sensor, ObservationSensor};
///
/// let sensor = ImageSensor::new("camera_front", 64, 64, 4);
/// assert_eq!(sensor.name(), "camera_front");
/// assert_eq!(sensor.observation_dim(), 64 * 64 * 4);
/// ```
pub struct ImageSensor {
    label: String,
    width: u32,
    height: u32,
    channels: u32,
    rate: Option<f64>,
}

impl ImageSensor {
    /// Create an image sensor with the given camera label and resolution.
    ///
    /// `channels` should match the pixel format of the [`FrameBuffer`]:
    /// 3 for RGB8, 4 for RGBA8.
    ///
    /// [`FrameBuffer`]: crate::buffer::FrameBuffer
    #[must_use]
    pub fn new(label: impl Into<String>, width: u32, height: u32, channels: u32) -> Self {
        Self {
            label: label.into(),
            width,
            height,
            channels,
            rate: None,
        }
    }

    /// Set the capture rate in Hz.
    #[must_use]
    pub const fn with_rate_hz(mut self, hz: f64) -> Self {
        self.rate = Some(hz);
        self
    }

    /// Width in pixels as configured.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels as configured.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Number of colour channels as configured.
    #[must_use]
    pub const fn channels(&self) -> u32 {
        self.channels
    }
}

impl Sensor for ImageSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Self::Output {
        // Look up the named camera in CameraFrameBuffers first.
        if let Some(bufs) = world.get_resource::<CameraFrameBuffers>() {
            if let Some(buf) = bufs.get(&self.label) {
                let data: Vec<f32> = buf.data().iter().map(|&b| f32::from(b) / 255.0).collect();
                return Observation::new(data);
            }
        }

        // Fall back to a zero observation of the declared dimension.
        Observation::new(vec![0.0; self.observation_dim()])
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
        (self.width * self.height * self.channels) as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{CameraFrameBuffers, FrameBuffer};
    use crate::config::PixelFormat;

    #[test]
    fn image_sensor_name() {
        let sensor = ImageSensor::new("cam_0", 64, 64, 3);
        assert_eq!(sensor.name(), "cam_0");
    }

    #[test]
    fn image_sensor_default_rate() {
        let sensor = ImageSensor::new("cam", 1, 1, 3);
        assert!(sensor.rate_hz().is_none());
    }

    #[test]
    fn image_sensor_with_rate() {
        let sensor = ImageSensor::new("cam", 1, 1, 3).with_rate_hz(30.0);
        assert!((sensor.rate_hz().unwrap() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn image_sensor_observation_dim_nonzero() {
        let sensor = ImageSensor::new("cam", 64, 48, 3);
        assert_eq!(sensor.observation_dim(), 64 * 48 * 3);
    }

    #[test]
    fn image_sensor_observation_dim_rgba() {
        let sensor = ImageSensor::new("cam", 32, 32, 4);
        assert_eq!(sensor.observation_dim(), 32 * 32 * 4);
    }

    #[test]
    fn image_sensor_reads_from_camera_frame_buffers() {
        let mut world = World::new();
        let mut bufs = CameraFrameBuffers::default();
        let mut buf = FrameBuffer::new(2, 1, PixelFormat::Rgb8);
        buf.write_frame(vec![0, 128, 255, 64, 192, 32]);
        bufs.insert("test".to_string(), buf);
        world.insert_resource(bufs);

        let sensor = ImageSensor::new("test", 2, 1, 3);
        let obs = sensor.read(&mut world);

        assert_eq!(obs.len(), 6);
        assert!((obs[0] - 0.0).abs() < f32::EPSILON);
        assert!((obs[1] - f32::from(128_u8) / 255.0).abs() < 0.001);
        assert!((obs[2] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn image_sensor_normalises_to_unit_range() {
        let mut world = World::new();
        let mut bufs = CameraFrameBuffers::default();
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgb8);
        buf.write_frame(vec![0, 127, 255]);
        bufs.insert("test".to_string(), buf);
        world.insert_resource(bufs);

        let sensor = ImageSensor::new("test", 1, 1, 3);
        let obs = sensor.read(&mut world);

        for val in obs.as_slice() {
            assert!(*val >= 0.0);
            assert!(*val <= 1.0);
        }
    }

    #[test]
    fn image_sensor_rgba() {
        let mut world = World::new();
        let mut bufs = CameraFrameBuffers::default();
        let mut buf = FrameBuffer::new(1, 1, PixelFormat::Rgba8);
        buf.write_frame(vec![255, 0, 128, 255]);
        bufs.insert("test".to_string(), buf);
        world.insert_resource(bufs);

        let sensor = ImageSensor::new("test", 1, 1, 4);
        let obs = sensor.read(&mut world);

        assert_eq!(obs.len(), 4);
        assert!((obs[0] - 1.0).abs() < f32::EPSILON);
        assert!((obs[1] - 0.0).abs() < f32::EPSILON);
        assert!((obs[3] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn image_sensor_fallback_when_no_buffer() {
        // When CameraFrameBuffers has no entry for the label, the sensor
        // should return a zero observation of the declared dimension.
        let mut world = World::new();
        world.insert_resource(CameraFrameBuffers::default());

        let sensor = ImageSensor::new("missing_cam", 4, 4, 3);
        let obs = sensor.read(&mut world);

        assert_eq!(obs.len(), 4 * 4 * 3);
        assert!(obs.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn image_sensor_fallback_when_no_resource() {
        // When CameraFrameBuffers is absent altogether, return zero obs.
        let mut world = World::new();
        let sensor = ImageSensor::new("cam", 2, 2, 3);
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2 * 2 * 3);
        assert!(obs.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn image_sensor_width_height_channels() {
        let sensor = ImageSensor::new("cam", 16, 8, 4);
        assert_eq!(sensor.width(), 16);
        assert_eq!(sensor.height(), 8);
        assert_eq!(sensor.channels(), 4);
    }
}
