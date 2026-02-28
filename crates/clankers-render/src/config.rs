//! Render configuration types.
//!
//! [`RenderConfig`] defines the resolution and pixel format for headless frame
//! capture. [`CameraConfig`] holds per-camera projection parameters.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// PixelFormat
// ---------------------------------------------------------------------------

/// Pixel storage format for captured frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PixelFormat {
    /// 3 bytes per pixel (red, green, blue).
    #[default]
    Rgb8,
    /// 4 bytes per pixel (red, green, blue, alpha).
    Rgba8,
}

impl PixelFormat {
    /// Number of bytes per pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }

    /// Number of colour channels.
    #[must_use]
    pub const fn channels(self) -> u32 {
        match self {
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// RenderConfig
// ---------------------------------------------------------------------------

/// Resource configuring headless frame capture.
///
/// # Example
///
/// ```
/// use clankers_render::RenderConfig;
/// use clankers_render::config::PixelFormat;
///
/// let config = RenderConfig::new(512, 512)
///     .with_format(PixelFormat::Rgba8);
///
/// assert_eq!(config.width, 512);
/// assert_eq!(config.height, 512);
/// assert_eq!(config.format, PixelFormat::Rgba8);
/// ```
#[derive(Resource, Clone, Debug)]
pub struct RenderConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel storage format.
    pub format: PixelFormat,
}

impl RenderConfig {
    /// Create a render config with the given resolution and default RGB8 format.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: PixelFormat::Rgb8,
        }
    }

    /// Set the pixel format.
    #[must_use]
    pub const fn with_format(mut self, format: PixelFormat) -> Self {
        self.format = format;
        self
    }

    /// Total number of bytes required for one frame.
    #[must_use]
    pub const fn frame_byte_count(&self) -> usize {
        (self.width * self.height * self.format.bytes_per_pixel()) as usize
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self::new(512, 512)
    }
}

// ---------------------------------------------------------------------------
// CameraConfig
// ---------------------------------------------------------------------------

/// Component for camera projection parameters.
///
/// Attached to camera entities to configure the viewpoint for frame capture.
/// The `label` field uniquely identifies the camera within [`CameraFrameBuffers`].
///
/// [`CameraFrameBuffers`]: crate::buffer::CameraFrameBuffers
#[derive(Component, Clone, Debug)]
pub struct CameraConfig {
    /// Unique label identifying this camera's frame buffer entry.
    pub label: String,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
}

impl CameraConfig {
    /// Create a camera config with typical defaults and an empty label.
    #[must_use]
    pub fn new() -> Self {
        Self {
            label: String::new(),
            fov_y: std::f32::consts::FRAC_PI_3,
            near: 0.01,
            far: 100.0,
        }
    }

    /// Set the camera label used to key into [`CameraFrameBuffers`].
    ///
    /// [`CameraFrameBuffers`]: crate::buffer::CameraFrameBuffers
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set the vertical field of view in radians.
    #[must_use]
    pub fn with_fov_y(mut self, fov_y: f32) -> Self {
        self.fov_y = fov_y;
        self
    }

    /// Set the near clipping plane.
    #[must_use]
    pub fn with_near(mut self, near: f32) -> Self {
        self.near = near;
        self
    }

    /// Set the far clipping plane.
    #[must_use]
    pub fn with_far(mut self, far: f32) -> Self {
        self.far = far;
        self
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgba8.bytes_per_pixel(), 4);
    }

    #[test]
    fn pixel_format_channels() {
        assert_eq!(PixelFormat::Rgb8.channels(), 3);
        assert_eq!(PixelFormat::Rgba8.channels(), 4);
    }

    #[test]
    fn pixel_format_default_is_rgb8() {
        assert_eq!(PixelFormat::default(), PixelFormat::Rgb8);
    }

    #[test]
    fn render_config_new() {
        let config = RenderConfig::new(640, 480);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.format, PixelFormat::Rgb8);
    }

    #[test]
    fn render_config_default() {
        let config = RenderConfig::default();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);
        assert_eq!(config.format, PixelFormat::Rgb8);
    }

    #[test]
    fn render_config_with_format() {
        let config = RenderConfig::new(64, 64).with_format(PixelFormat::Rgba8);
        assert_eq!(config.format, PixelFormat::Rgba8);
    }

    #[test]
    fn render_config_frame_byte_count_rgb() {
        let config = RenderConfig::new(4, 2);
        assert_eq!(config.frame_byte_count(), 4 * 2 * 3);
    }

    #[test]
    fn render_config_frame_byte_count_rgba() {
        let config = RenderConfig::new(4, 2).with_format(PixelFormat::Rgba8);
        assert_eq!(config.frame_byte_count(), 4 * 2 * 4);
    }

    #[test]
    fn camera_config_defaults() {
        let cam = CameraConfig::new();
        assert!((cam.fov_y - std::f32::consts::FRAC_PI_3).abs() < f32::EPSILON);
        assert!((cam.near - 0.01).abs() < f32::EPSILON);
        assert!((cam.far - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn camera_config_builder() {
        let cam = CameraConfig::new()
            .with_fov_y(1.5)
            .with_near(0.1)
            .with_far(50.0);
        assert!((cam.fov_y - 1.5).abs() < f32::EPSILON);
        assert!((cam.near - 0.1).abs() < f32::EPSILON);
        assert!((cam.far - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn camera_config_default_matches_new() {
        let a = CameraConfig::new();
        let b = CameraConfig::default();
        assert!((a.fov_y - b.fov_y).abs() < f32::EPSILON);
        assert!((a.near - b.near).abs() < f32::EPSILON);
        assert!((a.far - b.far).abs() < f32::EPSILON);
    }

    #[test]
    fn camera_config_default_label_is_empty() {
        let cam = CameraConfig::new();
        assert!(cam.label.is_empty());
    }

    #[test]
    fn camera_config_with_label() {
        let cam = CameraConfig::new().with_label("front_cam");
        assert_eq!(cam.label, "front_cam");
    }
}
