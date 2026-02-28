//! Image sensor bridging the [`CameraFrameBuffers`] to the observation pipeline.
//!
//! [`ImageSensor`] implements [`Sensor`] and [`ObservationSensor`] by reading
//! the current frame for its named camera from [`CameraFrameBuffers`] and
//! normalising pixel values to `[0.0, 1.0]`.

use bevy::prelude::*;

use clankers_core::traits::{ObservationSensor, Sensor};
use clankers_core::types::Observation;

use crate::buffer::{CameraFrameBuffers, DepthFrameBuffer};

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
// DepthSensor
// ---------------------------------------------------------------------------

/// Sensor that reads from [`DepthFrameBuffer`] and returns linearised depth.
///
/// Raw depth values from the GPU are in a non-linear (hyperbolic) space.
/// [`DepthSensor::read`] applies the standard linearisation formula:
///
/// ```text
/// depth_m = (2 * near * far) / (far + near - raw * (far - near))
/// ```
///
/// The returned [`Observation`] contains one `f32` depth value (in metres)
/// per pixel, with length `width * height`.
///
/// # Example
///
/// ```
/// use clankers_render::sensor::DepthSensor;
/// use clankers_core::traits::{Sensor, ObservationSensor};
///
/// let sensor = DepthSensor {
///     label: "depth_front".to_string(),
///     width: 64,
///     height: 48,
///     near: 0.1,
///     far: 100.0,
/// };
/// assert_eq!(sensor.observation_dim(), 64 * 48);
/// ```
pub struct DepthSensor {
    /// Unique label identifying this depth sensor.
    pub label: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Near clipping plane distance in metres.
    pub near: f32,
    /// Far clipping plane distance in metres.
    pub far: f32,
}

impl DepthSensor {
    /// Linearise a raw GPU depth value to metres.
    ///
    /// Applies `(2 * near * far) / (far + near - raw * (far - near))`.
    #[must_use]
    pub fn linearise(&self, raw: f32) -> f32 {
        let near = self.near;
        let far = self.far;
        (2.0 * near * far) / (far + near - raw * (far - near))
    }
}

impl Sensor for DepthSensor {
    type Output = Observation;

    fn read(&self, world: &mut World) -> Self::Output {
        if let Some(buf) = world.get_resource::<DepthFrameBuffer>() {
            let depth: Vec<f32> = buf.data().iter().map(|&raw| self.linearise(raw)).collect();
            return Observation::new(depth);
        }

        // Fall back to a zero observation of the declared dimension.
        Observation::new(vec![0.0; self.observation_dim()])
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn rate_hz(&self) -> Option<f64> {
        None
    }
}

impl ObservationSensor for DepthSensor {
    fn observation_dim(&self) -> usize {
        (self.width * self.height) as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{CameraFrameBuffers, DepthFrameBuffer, FrameBuffer};
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

    // -----------------------------------------------------------------------
    // DepthSensor tests
    // -----------------------------------------------------------------------

    #[test]
    fn depth_sensor_observation_dim() {
        let sensor = DepthSensor {
            label: "depth".to_string(),
            width: 64,
            height: 48,
            near: 0.1,
            far: 100.0,
        };
        assert_eq!(sensor.observation_dim(), 64 * 48);
    }

    #[test]
    fn depth_sensor_name() {
        let sensor = DepthSensor {
            label: "depth_front".to_string(),
            width: 4,
            height: 4,
            near: 0.1,
            far: 50.0,
        };
        assert_eq!(sensor.name(), "depth_front");
    }

    #[test]
    fn depth_sensor_rate_is_none() {
        let sensor = DepthSensor {
            label: "d".to_string(),
            width: 1,
            height: 1,
            near: 0.1,
            far: 10.0,
        };
        assert!(sensor.rate_hz().is_none());
    }

    #[test]
    fn depth_sensor_linearise_known_values() {
        // With near=1.0 and far=10.0:
        //   raw=0.0 should give linear depth = 1.0 (near plane)
        //   raw=1.0 should give linear depth = 10.0 (far plane)
        //
        // Formula: (2*near*far) / (far + near - raw*(far-near))
        let sensor = DepthSensor {
            label: "d".to_string(),
            width: 1,
            height: 1,
            near: 1.0,
            far: 10.0,
        };

        // raw = 0.0: (2*1*10) / (10+1 - 0*(10-1)) = 20/11 ≈ 1.818
        let linearised_zero = sensor.linearise(0.0);
        // Formula at raw=0: 2*near*far / (far+near) = 20/11
        let expected_zero = 2.0 * 1.0 * 10.0 / (10.0 + 1.0);
        assert!(
            (linearised_zero - expected_zero).abs() < 1e-5,
            "linearise(0.0) = {linearised_zero}, expected {expected_zero}"
        );

        // raw = 1.0: (2*1*10) / (10+1 - 1*(10-1)) = 20/(11-9) = 20/2 = 10
        let linearised_one = sensor.linearise(1.0);
        assert!(
            (linearised_one - 10.0).abs() < 1e-5,
            "linearise(1.0) = {linearised_one}, expected 10.0"
        );
    }

    #[test]
    fn depth_sensor_reads_from_depth_frame_buffer() {
        let mut world = World::new();

        // Set up a 2x1 depth buffer with known raw values.
        let mut buf = DepthFrameBuffer::new(2, 1);
        buf.write_depth_frame(vec![0.0, 1.0]);
        world.insert_resource(buf);

        let sensor = DepthSensor {
            label: "depth".to_string(),
            width: 2,
            height: 1,
            near: 1.0,
            far: 10.0,
        };

        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 2);

        // Verify linearisation was applied.
        let expected_0 = sensor.linearise(0.0);
        let expected_1 = sensor.linearise(1.0);
        assert!((obs[0] - expected_0).abs() < 1e-5);
        assert!((obs[1] - expected_1).abs() < 1e-5);
    }

    #[test]
    fn depth_sensor_fallback_when_no_buffer() {
        let mut world = World::new();
        let sensor = DepthSensor {
            label: "d".to_string(),
            width: 4,
            height: 4,
            near: 0.1,
            far: 100.0,
        };
        let obs = sensor.read(&mut world);
        assert_eq!(obs.len(), 4 * 4);
        assert!(obs.as_slice().iter().all(|&v| v == 0.0));
    }
}
